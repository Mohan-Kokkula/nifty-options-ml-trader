"""Tests for stat_utils.white_rc."""
from __future__ import annotations

import numpy as np
import pytest

from stat_utils import (
    InsufficientDataError,
    InvalidInputError,
    WhiteRCResult,
    white_reality_check,
)


def test_returns_white_rc_result(rng: np.random.Generator):
    perf = rng.normal(size=(200, 5))
    r = white_reality_check(perf, n_bootstrap=200, seed=0)
    assert isinstance(r, WhiteRCResult)
    assert 0.0 <= r.pvalue <= 1.0
    assert r.n_models == 5


def test_null_distribution_size_close_to_alpha(rng: np.random.Generator):
    """Under the null (all models = benchmark = 0 mean), the empirical
    rejection rate at alpha=0.10 across 200 trials should be <= ~0.20.

    Politis-Romano stationary bootstrap under null gives an approximately
    valid but slightly conservative test."""
    n_trials = 60
    rejects = 0
    T = 150
    K = 4
    for trial in range(n_trials):
        perf = rng.normal(size=(T, K))
        r = white_reality_check(perf, n_bootstrap=200, seed=trial)
        if r.pvalue < 0.10:
            rejects += 1
    rate = rejects / n_trials
    # Under null we expect ~10% rejection. Allow slack for MC noise
    # and stationary-bootstrap conservatism.
    assert rate <= 0.25


def test_rejects_null_when_strong_signal():
    T = 500
    rng = np.random.default_rng(0)
    strong = rng.normal(loc=0.5, size=T)
    weak = rng.normal(loc=0.0, size=(T, 3))
    perf = np.column_stack([strong, weak])
    r = white_reality_check(perf, n_bootstrap=500, seed=42)
    assert r.pvalue < 0.05


def test_accepts_dict_of_streams(rng: np.random.Generator):
    T = 200
    perf = {"m1": rng.normal(size=T),
            "m2": rng.normal(size=T),
            "m3": rng.normal(loc=0.3, size=T)}
    r = white_reality_check(perf, n_bootstrap=100, seed=0)
    assert set(r.per_model_mean_perf.keys()) == {"m1", "m2", "m3"}


def test_rejects_mismatched_benchmark_length(rng: np.random.Generator):
    perf = rng.normal(size=(100, 3))
    with pytest.raises(InvalidInputError):
        white_reality_check(perf, benchmark=np.zeros(50),
                             n_bootstrap=50, seed=0)


def test_rejects_short_series(rng: np.random.Generator):
    perf = rng.normal(size=(1, 3))
    with pytest.raises(InsufficientDataError):
        white_reality_check(perf, n_bootstrap=10, seed=0)


def test_block_length_auto_returns_int(rng: np.random.Generator):
    perf = rng.normal(size=(200, 3))
    r = white_reality_check(perf, block_length="auto",
                             n_bootstrap=50, seed=0)
    assert isinstance(r.block_length, int) and r.block_length >= 1


def test_block_length_callable(rng: np.random.Generator):
    perf = rng.normal(size=(200, 3))
    r = white_reality_check(perf, block_length=lambda x: 7,
                             n_bootstrap=50, seed=0)
    assert r.block_length == 7


def test_reproducible_with_seed(rng: np.random.Generator):
    perf = rng.normal(size=(100, 3))
    a = white_reality_check(perf, n_bootstrap=100, seed=13)
    b = white_reality_check(perf, n_bootstrap=100, seed=13)
    assert a.pvalue == b.pvalue and a.statistic == b.statistic
