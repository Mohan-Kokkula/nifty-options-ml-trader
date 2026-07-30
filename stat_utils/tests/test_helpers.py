"""Tests for stat_utils.helpers."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from stat_utils import (
    InsufficientDataError,
    InvalidInputError,
    KSResult,
    KendallResult,
    LeveneResult,
    PermutationResult,
    kendall_tau,
    ks_2samp,
    levene,
    permutation_test,
)


def test_kendall_tau_matches_scipy(rng: np.random.Generator):
    x = rng.normal(size=100)
    y = 0.5 * x + rng.normal(size=100)
    r = kendall_tau(x, y)
    sp = sp_stats.kendalltau(x, y)
    assert r.statistic == pytest.approx(float(sp.statistic))
    assert r.pvalue == pytest.approx(float(sp.pvalue))


def test_kendall_rejects_mismatched():
    with pytest.raises(InvalidInputError):
        kendall_tau(np.arange(5.0), np.arange(6.0))


def test_levene_matches_scipy(rng: np.random.Generator):
    a = rng.normal(scale=1.0, size=200)
    b = rng.normal(scale=1.5, size=200)
    r = levene(a, b, center="median")
    sp = sp_stats.levene(a, b, center="median")
    assert r.statistic == pytest.approx(float(sp.statistic))
    assert r.pvalue == pytest.approx(float(sp.pvalue))
    assert isinstance(r, LeveneResult)
    assert r.n_groups == 2


def test_levene_needs_two_groups():
    with pytest.raises(InvalidInputError):
        levene(np.arange(10.0))


def test_ks_matches_scipy(rng: np.random.Generator):
    a = rng.normal(size=100)
    b = rng.normal(loc=0.5, size=100)
    r = ks_2samp(a, b)
    sp = sp_stats.ks_2samp(a, b)
    assert r.statistic == pytest.approx(float(sp.statistic))
    assert r.pvalue == pytest.approx(float(sp.pvalue))


def test_permutation_reproducible(rng: np.random.Generator):
    a = rng.normal(size=50)
    b = rng.normal(loc=0.5, size=50)
    def mean_diff(x, y):
        return float(x.mean() - y.mean())
    r1 = permutation_test(a, b, mean_diff, n_permutations=200, seed=1)
    r2 = permutation_test(a, b, mean_diff, n_permutations=200, seed=1)
    assert r1.pvalue == r2.pvalue


def test_permutation_null_uniformity_on_iid(rng: np.random.Generator):
    """Under exchangeability, the p-value distribution across independent
    trials should be roughly uniform. Test with a coarse tail check."""
    trials = 40
    high_p = 0
    for t in range(trials):
        a = rng.normal(size=40)
        b = rng.normal(size=40)
        def mean_diff(x, y):
            return float(x.mean() - y.mean())
        r = permutation_test(a, b, mean_diff, n_permutations=200,
                             alternative="two-sided", seed=t)
        if r.pvalue > 0.1:
            high_p += 1
    assert high_p >= trials * 0.55       # a lot more than we'd see if broken


def test_permutation_rejects_null_on_large_effect(rng: np.random.Generator):
    a = rng.normal(loc=1.0, size=200)
    b = rng.normal(loc=0.0, size=200)
    def mean_diff(x, y):
        return float(x.mean() - y.mean())
    r = permutation_test(a, b, mean_diff, n_permutations=500,
                          alternative="greater", seed=0)
    assert r.pvalue < 0.01


def test_permutation_returns_null_when_requested(rng: np.random.Generator):
    a = rng.normal(size=30)
    b = rng.normal(size=30)
    def mean_diff(x, y):
        return float(x.mean() - y.mean())
    r = permutation_test(a, b, mean_diff, n_permutations=100, seed=0,
                          return_null=True)
    assert r.null_distribution is not None
    assert r.null_distribution.size == 100


def test_json_serialisable(rng: np.random.Generator):
    r = kendall_tau(rng.normal(size=20), rng.normal(size=20))
    import json
    d = json.loads(r.to_json())
    assert "statistic" in d and "pvalue" in d
