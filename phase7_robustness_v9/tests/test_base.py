"""Base module — constants, config, hash, exception hierarchy."""
from __future__ import annotations

from phase7_robustness_v9 import (
    CacheMismatchError, InvalidInputError, MissingManifestError,
    Phase7Config, Phase7Error, PRODUCTION_BASELINE, hash8,
    H_ROB_MIN_OUTPERFORM_FRAC, H_ROB_MAX_STABILITY_CV,
    H_ROB_CATASTROPHE_PF, H_ROB_CATASTROPHE_NET_MULT,
    SLIPPAGE_MULTIPLIERS, TCOST_MULTIPLIERS, EXEC_DELAYS_BARS,
)


def test_h_rob_constants_are_locked():
    assert H_ROB_MIN_OUTPERFORM_FRAC == 0.80
    assert H_ROB_MAX_STABILITY_CV == 0.50
    assert H_ROB_CATASTROPHE_PF == 0.80
    assert H_ROB_CATASTROPHE_NET_MULT == 2.0


def test_slippage_multipliers_exact():
    assert SLIPPAGE_MULTIPLIERS == (1.00, 1.25, 1.50, 1.75, 2.00)


def test_tcost_multipliers_exact():
    assert TCOST_MULTIPLIERS == (1.00, 1.25, 1.50, 1.75, 2.00)


def test_exec_delays_exact():
    assert EXEC_DELAYS_BARS == (0, 1, 2)


def test_production_baseline_matches_phase6():
    assert PRODUCTION_BASELINE.call_thr == 0.32
    assert PRODUCTION_BASELINE.put_thr == 0.25
    assert PRODUCTION_BASELINE.skip_ceil == 0.65
    assert PRODUCTION_BASELINE.min_edge == 0.05


def test_hash8_stable_and_short():
    a = hash8({"x": 1, "y": [2, 3]})
    b = hash8({"y": [2, 3], "x": 1})
    assert a == b
    assert len(a) == 8


def test_exception_hierarchy():
    assert issubclass(InvalidInputError, Phase7Error)
    assert issubclass(CacheMismatchError, Phase7Error)
    assert issubclass(MissingManifestError, Phase7Error)


def test_config_defaults():
    cfg = Phase7Config()
    assert cfg.seed == 42
    assert cfg.targets == ("mean", "stacking")
