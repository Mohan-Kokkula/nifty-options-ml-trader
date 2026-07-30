"""Tests for stat_utils.hac."""
from __future__ import annotations

import numpy as np
import pytest

from stat_utils import InvalidInputError, newey_west_variance


def _hand_newey_west(x: np.ndarray, lag: int) -> float:
    """Reference implementation restated from scratch for cross-check."""
    T = len(x)
    d = x - x.mean()
    var = float(np.dot(d, d) / T)
    for k in range(1, lag + 1):
        gk = float(np.dot(d[k:], d[:-k]) / T)
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * gk
    return max(var, 0.0)


def test_lag_zero_equals_sample_variance():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = x - x.mean()
    expected = float(np.dot(d, d) / len(x))
    assert newey_west_variance(x, lag=0) == pytest.approx(expected)


def test_matches_hand_reference(rng: np.random.Generator):
    x = rng.normal(size=200)
    for L in (0, 1, 3, 5, 10):
        assert newey_west_variance(x, lag=L) == pytest.approx(
            _hand_newey_west(x, L), rel=1e-9, abs=1e-12)


def test_matches_statsmodels_if_available(block_correlated_series):
    """When statsmodels is installed, compare our estimator to
    statsmodels.stats.sandwich_covariance on shared input.
    """
    try:
        from statsmodels.stats.sandwich_covariance import S_hac_simple
    except ImportError:
        pytest.skip("statsmodels not installed")
    x = block_correlated_series
    T = len(x)
    # S_hac_simple expects centred residuals as a column vector
    resid = (x - x.mean()).reshape(-1, 1)
    sm = float(S_hac_simple(resid, nlags=5).ravel()[0] / T)
    ours = newey_west_variance(x, lag=5)
    assert ours == pytest.approx(sm, rel=1e-8, abs=1e-10)


def test_rejects_negative_lag():
    with pytest.raises(InvalidInputError):
        newey_west_variance(np.array([1.0, 2.0]), lag=-1)


def test_rejects_lag_ge_length():
    with pytest.raises(InvalidInputError):
        newey_west_variance(np.array([1.0, 2.0]), lag=2)


def test_returns_nonnegative(rng: np.random.Generator):
    x = rng.normal(size=50)
    for L in range(5):
        assert newey_west_variance(x, lag=L) >= 0.0
