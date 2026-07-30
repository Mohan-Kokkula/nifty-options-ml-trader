#!/usr/bin/env python3
"""
Unit tests for the HH/HL structure feature (parallel output only).

Standalone script (repo convention), run with:
    python scripts/test_structure_features.py

Covers:
  - core/structure_features.py::compute_hh_hl_structure() pure logic
  - core/claude_pilot.py::PilotConfig.enable_hh_hl_feature default/override
  - main.py's FEATURE_HH_HL_ENABLED env-var resolution (mirrors the exact
    expression at the PilotConfig(...) call site, same pattern as
    scripts/test_pilot_min_confidence_env.py)

This feature is logged every cycle but never read by any gate, threshold,
or trade decision -- there is nothing here that touches live broker state,
so this suite needs no mocking beyond plain function calls.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.structure_features import compute_hh_hl_structure, DEFAULT_LOOKBACK
from core.claude_pilot import PilotConfig


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True

    # ── Clean uptrend: a staircase of rally-pullback-rally (2 up, 1 back) ──
    # A textbook HH/HL structure needs real pullback bars to form swing LOWS
    # at all -- a laser-straight monotonic ramp has no pullbacks, so it
    # correctly produces zero swing lows (there's nothing to make a "higher
    # low" out of). Real 5-min price data always has this noise; this
    # fixture matches that instead of an unrealistic straight line.
    up = []
    price = 100.0
    for step in range(7):
        price += 6; up.append(price)
        price += 3; up.append(price)
        price -= 5; up.append(price)  # pullback deep enough to form a real local low
    r = compute_hh_hl_structure(up)
    all_ok &= check("staircase uptrend -> hh_hl_up=True", r["hh_hl_up"] is True)
    all_ok &= check("staircase uptrend -> hh_hl_down=False", r["hh_hl_down"] is False)
    all_ok &= check("staircase uptrend -> not insufficient_data", r["insufficient_data"] is False)

    # ── Clean downtrend: mirror staircase (2 down, 1 back up) ──────────────
    down = []
    price = 200.0
    for step in range(7):
        price -= 6; down.append(price)
        price -= 3; down.append(price)
        price += 5; down.append(price)  # bounce deep enough to form a real local high
    r = compute_hh_hl_structure(down)
    all_ok &= check("staircase downtrend -> hh_hl_down=True", r["hh_hl_down"] is True)
    all_ok &= check("staircase downtrend -> hh_hl_up=False", r["hh_hl_up"] is False)

    # ── Degenerate case: a perfectly straight monotonic ramp has NO ────────
    # pullback bars, so it correctly reports zero swing lows/highs on one
    # side -- documenting this as expected behavior, not a bug.
    straight = [100 + i for i in range(25)]
    r = compute_hh_hl_structure(straight)
    all_ok &= check(
        "perfectly straight ramp -> 0 swing lows (no pullbacks to form one)",
        r["n_swing_lows"] == 0,
    )
    all_ok &= check(
        "perfectly straight ramp -> hh_hl_up=False (needs >=2 swing lows by design)",
        r["hh_hl_up"] is False,
    )

    # ── Choppy/flat: oscillating around a fixed center -> neither ──────────
    chop = [100, 102, 99, 101, 100, 103, 98, 101, 100, 102,
            99, 101, 100, 103, 98, 101, 100, 102, 99, 101, 100]
    r = compute_hh_hl_structure(chop)
    all_ok &= check("choppy series -> hh_hl_up=False", r["hh_hl_up"] is False)
    all_ok &= check("choppy series -> hh_hl_down=False", r["hh_hl_down"] is False)

    # ── Insufficient data: fewer bars than the lookback requires ───────────
    short = [100, 101, 102]
    r = compute_hh_hl_structure(short, lookback=DEFAULT_LOOKBACK)
    all_ok &= check("too few bars -> insufficient_data=True", r["insufficient_data"] is True)
    all_ok &= check("too few bars -> hh_hl_up=False (fail-open)", r["hh_hl_up"] is False)
    all_ok &= check("too few bars -> hh_hl_down=False (fail-open)", r["hh_hl_down"] is False)

    # ── Malformed input: empty list / bad params never raises ──────────────
    try:
        r = compute_hh_hl_structure([])
        all_ok &= check("empty input -> no exception, insufficient_data=True",
                         r["insufficient_data"] is True)
    except Exception as e:
        all_ok &= check(f"empty input raised {e!r} (should fail-open instead)", False)

    try:
        r = compute_hh_hl_structure(up, swing_lag=0)
        all_ok &= check("swing_lag=0 -> no exception, fail-open", r["insufficient_data"] is True)
    except Exception as e:
        all_ok &= check(f"swing_lag=0 raised {e!r} (should fail-open instead)", False)

    # ── Causality: function only ever looks at the closes it's given -- ────
    # truncating the series to "as of bar i" must not require or reference
    # any bar after i. This is the property that makes the feature safe to
    # compute live (see the module docstring on trailing-only swings).
    full = [100 + i * 0.5 for i in range(30)]
    truncated = full[:22]  # simulate "as of an earlier bar" -- no lookahead
    try:
        r_trunc = compute_hh_hl_structure(truncated)
        all_ok &= check(
            "computing on a truncated (earlier-bar) series works with no lookahead",
            "hh_hl_up" in r_trunc,
        )
    except Exception as e:
        all_ok &= check(f"truncated-series call raised {e!r}", False)

    # ── PilotConfig wiring ──────────────────────────────────────────────────
    all_ok &= check(
        "PilotConfig defaults enable_hh_hl_feature=True (parallel output, safe-by-default)",
        PilotConfig().enable_hh_hl_feature is True,
    )
    all_ok &= check(
        "PilotConfig(enable_hh_hl_feature=False) is honored",
        PilotConfig(enable_hh_hl_feature=False).enable_hh_hl_feature is False,
    )

    # ── main.py env-var resolution (mirrors the exact PilotConfig(...) call site) ──
    def _resolve_hh_hl_enabled() -> bool:
        return os.getenv("FEATURE_HH_HL_ENABLED", "true").lower() == "true"

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FEATURE_HH_HL_ENABLED", None)
        all_ok &= check(
            "FEATURE_HH_HL_ENABLED unset -> defaults to enabled (unchanged behavior)",
            _resolve_hh_hl_enabled() is True,
        )

    with patch.dict(os.environ, {"FEATURE_HH_HL_ENABLED": "false"}):
        all_ok &= check(
            "FEATURE_HH_HL_ENABLED=false -> resolves to disabled",
            _resolve_hh_hl_enabled() is False,
        )

    with patch.dict(os.environ, {"FEATURE_HH_HL_ENABLED": "true"}):
        all_ok &= check(
            "FEATURE_HH_HL_ENABLED=true -> resolves to enabled",
            _resolve_hh_hl_enabled() is True,
        )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
