"""Tests for stat_utils.dm (Diebold-Mariano)."""
from __future__ import annotations

import numpy as np
import pytest

from stat_utils import (
    DMResult,
    InsufficientDataError,
    InvalidInputError,
    diebold_mariano,
)


def test_identical_series_yields_zero_stat():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    r = diebold_mariano(x, x, lag=1)
    assert r.statistic == pytest.approx(0.0)
    assert r.pvalue == pytest.approx(1.0)


def test_returns_dm_result_type():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    r = diebold_mariano(x, x, lag=1)
    assert isinstance(r, DMResult)
    d = r.to_dict()
    for key in ("statistic", "pvalue", "ci_lower", "ci_upper"):
        assert key in d


def test_two_sided_and_one_sided_pvalues(rng: np.random.Generator):
    n = 500
    a = rng.normal(loc=0.1, size=n)          # A has higher mean
    b = rng.normal(loc=0.0, size=n)
    r_two = diebold_mariano(a, b, alternative="two-sided", lag=3)
    r_gt = diebold_mariano(a, b, alternative="greater", lag=3)
    r_lt = diebold_mariano(a, b, alternative="less", lag=3)
    # Two-sided p ~ 2 * one-sided p for the winning direction
    assert r_two.pvalue == pytest.approx(2.0 * r_gt.pvalue, abs=1e-9)
    assert r_gt.pvalue + r_lt.pvalue == pytest.approx(1.0, abs=1e-9)


def test_rejects_null_for_large_shift(rng: np.random.Generator):
    n = 500
    a = rng.normal(loc=0.5, size=n)
    b = rng.normal(loc=0.0, size=n)
    r = diebold_mariano(a, b, alternative="greater", lag=5)
    assert r.pvalue < 0.01


def test_ci_contains_mean_diff():
    n = 200
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.1, size=n)
    b = rng.normal(loc=0.0, size=n)
    r = diebold_mariano(a, b, lag=3, ci_level=0.90)
    assert r.ci_lower <= r.mean_loss_diff <= r.ci_upper


def test_rejects_mismatched_lengths():
    with pytest.raises(InvalidInputError):
        diebold_mariano(np.arange(5.0), np.arange(6.0))


def test_rejects_insufficient_length():
    with pytest.raises(InsufficientDataError):
        diebold_mariano(np.array([1.0, 2.0]), np.array([1.0, 2.0]), lag=5)


def test_rejects_bad_alternative():
    with pytest.raises(InvalidInputError):
        diebold_mariano(np.arange(10.0), np.arange(10.0),
                         alternative="not-real")


def test_json_roundtrip():
    r = diebold_mariano(np.arange(10.0), np.arange(10.0), lag=1)
    import json
    d = json.loads(r.to_json())
    assert d["statistic"] == 0.0
    assert d["pvalue"] == 1.0
