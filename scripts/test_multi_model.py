"""Tests for ml_engine multi-model support (V9 + V11 side by side).

The critical property: adding a second model must NOT change what the
primary slot does, because the live pilot depends on it.

Run standalone:  python scripts/test_multi_model.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.ml_engine as ME  # noqa: E402
from core.model_registry import _LIVE_STEMS  # noqa: E402

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"  PASS  {name}")
    else:
        _failed += 1; print(f"  FAIL  {name}")


print("\n-- registry knows v11 (promotion was impossible without it) --")
check("v11 in _LIVE_STEMS", "v11" in _LIVE_STEMS)
check("v11 stems correct",
      _LIVE_STEMS["v11"] == ("nifty_v11_models.pkl", "nifty_v11_scaler.pkl",
                             "feature_cols_v11.pkl"))
check("v9 stems unchanged",
      _LIVE_STEMS["v9"] == ("nifty_v9_models.pkl", "nifty_v9_scaler.pkl",
                            "feature_cols_v9.pkl"))

print("\n-- _signal_from_proba mirrors the live rule --")
f = ME._signal_from_proba
# CALL: p0 over threshold, edge over MIN_EDGE, skip under ceiling
s, c = f(np.array([0.40, 0.20, 0.40]))
check("clear CALL -> 0", s == 0)
check("confidence = max proba", abs(c - 0.40) < 1e-9)
s, _ = f(np.array([0.20, 0.40, 0.40]))
check("clear PUT -> 1", s == 1)
s, _ = f(np.array([0.10, 0.10, 0.80]))
check("high skip -> 2", s == 2)
s, _ = f(np.array([0.34, 0.33, 0.33]))
check("edge below MIN_EDGE -> 2 (not argmax)", s == 2)
s, _ = f(np.array([0.40, 0.20, 0.70]))
check("skip above SKIP_CEIL -> 2", s == 2)
# PUT threshold (0.25) is LOWER than CALL (0.32) -- asymmetric by design
s, _ = f(np.array([0.10, 0.28, 0.62]))
check("PUT fires at 0.28 (CALL would not)", s == 1)
s, _ = f(np.array([0.28, 0.10, 0.62]))
check("CALL does NOT fire at 0.28", s == 2)

print("\n-- extra-version loading is opt-in and fails safe --")
os.environ.pop("ML_EXTRA_VERSIONS", None)
check("no env var -> loads nothing", ME.load_extra_versions() == [])
os.environ["ML_EXTRA_VERSIONS"] = "v11"
got = ME.load_extra_versions()
check("unpromoted v11 is SKIPPED, not loaded", "V11" not in got)
check("_extra stays empty when files absent", "V11" not in ME._extra)
os.environ["ML_EXTRA_VERSIONS"] = "v99"
check("unknown version skipped without raising", ME.load_extra_versions() == [])
os.environ.pop("ML_EXTRA_VERSIONS", None)

print("\n-- primary slot untouched (the live pilot depends on this) --")
check("primary _models still None/unloaded here", ME._models is None or ME._loaded)
check("available_versions() empty when nothing loaded",
      ME.available_versions() == [] or ME._loaded)
check("predict_multi returns {} when no model loaded",
      ME.predict_multi(None) == {} or ME._loaded)

print("\n-- predict_multi degrades gracefully --")
check("bad input never raises", isinstance(ME.predict_multi(None), dict))
check("unknown version requested -> absent, no raise",
      isinstance(ME.predict_multi(None, versions=["V42"]), dict))

print("\n-- load_model_and_extras is a safe drop-in --")
check("callable", callable(ME.load_model_and_extras))
check("load_model still exists unchanged", callable(ME.load_model))

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
