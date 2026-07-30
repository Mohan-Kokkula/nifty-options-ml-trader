"""Bootstrap CI + paired-fold ΔPF bootstrap."""
from __future__ import annotations

import numpy as np

from phase7_robustness_v9 import (block_bootstrap_net, block_bootstrap_pf,
                                    paired_bootstrap_delta_pf,
                                    InvalidInputError)


def test_pf_ci_has_expected_shape(synth_pnl_by_fold):
    r = block_bootstrap_pf(synth_pnl_by_fold, seed=1, n_resamples=200)
    for k in ("point_estimate", "lower", "upper", "n_resamples",
              "n_blocks", "block_unit"):
        assert k in r


def test_net_ci_has_expected_shape(synth_pnl_by_fold):
    r = block_bootstrap_net(synth_pnl_by_fold, seed=1, n_resamples=200)
    for k in ("point_estimate", "lower", "upper", "n_blocks"):
        assert k in r


def test_paired_delta_pf_ci_shape(synth_pnl_by_fold, rng):
    a = synth_pnl_by_fold
    b = {f: v + rng.normal(-100, 50, size=len(v)) for f, v in a.items()}
    r = paired_bootstrap_delta_pf(a, b, seed=42, n_resamples=200)
    assert r["paired"] is True
    assert r["common_folds"] == sorted(a.keys())
    assert r["lower"] <= r["point_estimate"] + 1e-6
    assert r["upper"] >= r["point_estimate"] - 1e-6


def test_paired_reproducible_under_seed(synth_pnl_by_fold, rng):
    a = synth_pnl_by_fold
    b = {f: v + rng.normal(-100, 50, size=len(v)) for f, v in a.items()}
    r1 = paired_bootstrap_delta_pf(a, b, seed=7, n_resamples=200)
    r2 = paired_bootstrap_delta_pf(a, b, seed=7, n_resamples=200)
    assert r1["lower"] == r2["lower"]
    assert r1["upper"] == r2["upper"]


def test_empty_raises():
    import pytest
    with pytest.raises(InvalidInputError):
        block_bootstrap_pf({})
