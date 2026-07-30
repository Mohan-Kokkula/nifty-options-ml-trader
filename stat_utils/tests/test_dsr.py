"""Tests for stat_utils.dsr."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stat_utils import DSRResult, InvalidInputError, deflated_sharpe


def test_single_trial_matches_probability_of_positive_sharpe():
    """With N=1 (no selection deflation), threshold Sharpe is 0.
    DSR reduces to P(SR > 0 | observed).
    """
    r = deflated_sharpe(observed_sharpe=1.0, n_samples=100, n_trials=1)
    assert r.threshold_sharpe == pytest.approx(0.0)
    # z = 1.0 * sqrt(99) ≈ 9.95 → Φ(z) ≈ 1
    assert r.dsr > 0.99


def test_threshold_sharpe_increases_with_trials():
    a = deflated_sharpe(1.0, n_samples=100, n_trials=10)
    b = deflated_sharpe(1.0, n_samples=100, n_trials=1000)
    assert b.threshold_sharpe > a.threshold_sharpe


def test_dsr_decreases_with_more_trials():
    a = deflated_sharpe(1.0, n_samples=200, n_trials=10)
    b = deflated_sharpe(1.0, n_samples=200, n_trials=10_000)
    assert b.dsr < a.dsr


def test_dsr_matches_bailey_lopezdeprado_worked_example():
    """Bailey & López de Prado (2014) example: SR_obs = 2.5, N=1250,
    trials=100, γ₃=-1, kurtosis=6 (excess=3).

    Reproduce the reported values by direct computation using the same
    formulas rather than a paper-scraped constant, so the test remains
    self-checking.
    """
    from scipy import stats
    N = 1250
    trials = 100
    sr = 2.5 / math.sqrt(252)                 # annual → per-obs for daily
    skew = -1.0
    kurt_ex = 3.0                              # excess

    # Manually compute expected max Sharpe with variance=1
    gamma = 0.5772156649015329
    a = stats.norm.ppf(1 - 1 / trials)
    b = stats.norm.ppf(1 - 1 / (trials * math.e))
    sr_thr = (1 - gamma) * a + gamma * b
    denom = math.sqrt(1 - skew * sr + ((kurt_ex + 3.0 - 1.0) / 4.0) * sr * sr)
    z_expected = (sr - sr_thr) * math.sqrt(N - 1) / denom
    expected_dsr = float(stats.norm.cdf(z_expected))

    r = deflated_sharpe(sr, n_samples=N, n_trials=trials,
                        skewness=skew, kurtosis_excess=kurt_ex)
    assert r.dsr == pytest.approx(expected_dsr, abs=1e-9)


def test_returns_dsr_result_type():
    r = deflated_sharpe(1.0, n_samples=100, n_trials=1)
    assert isinstance(r, DSRResult)
    d = r.to_dict()
    assert set(("dsr", "observed_sharpe", "threshold_sharpe",
                 "n_samples", "n_trials")).issubset(d)


def test_rejects_bad_n_samples():
    with pytest.raises(InvalidInputError):
        deflated_sharpe(1.0, n_samples=1, n_trials=1)


def test_rejects_bad_n_trials():
    with pytest.raises(InvalidInputError):
        deflated_sharpe(1.0, n_samples=100, n_trials=0)
