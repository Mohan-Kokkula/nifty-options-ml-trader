"""Tests for stat_utils.metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stat_utils import (
    InsufficientDataError,
    InvalidInputError,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    win_rate,
)


def test_profit_factor_hand_computed():
    x = np.array([2.0, -1.0, 3.0, -2.0])   # gains=5, losses=3
    assert profit_factor(x) == pytest.approx(5.0 / 3.0)


def test_profit_factor_no_losses_returns_inf():
    x = np.array([1.0, 2.0, 3.0])
    assert profit_factor(x) == float("inf")


def test_profit_factor_no_trades_returns_nan():
    x = np.zeros(5)
    assert math.isnan(profit_factor(x))


def test_profit_factor_rejects_empty():
    with pytest.raises(InsufficientDataError):
        profit_factor(np.array([]))


def test_win_rate_hand_computed():
    x = np.array([1.0, -1.0, 2.0, 0.0])   # 2 wins of 4
    assert win_rate(x) == pytest.approx(0.5)


def test_expectancy_matches_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert expectancy(x) == pytest.approx(2.5)


def test_sharpe_matches_mean_over_std():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = x.mean() / x.std(ddof=1)
    assert sharpe(x) == pytest.approx(expected)


def test_sharpe_annualisation_multiplier():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base = sharpe(x)
    assert sharpe(x, annualization=math.sqrt(252)) == pytest.approx(
        base * math.sqrt(252))


def test_sharpe_zero_std_returns_nan():
    assert math.isnan(sharpe(np.ones(10)))


def test_sharpe_rejects_single_obs():
    with pytest.raises(InsufficientDataError):
        sharpe(np.array([1.0]))


def test_sortino_positive_only_returns_nan():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert math.isnan(sortino(x))


def test_sortino_matches_hand_formula():
    x = np.array([1.0, -1.0, 2.0, -2.0])
    excess = x - 0.0
    dd = np.sqrt(np.mean(excess[excess < 0] ** 2))
    expected = excess.mean() / dd
    assert sortino(x) == pytest.approx(expected)


def test_max_drawdown_zero_for_monotone():
    assert max_drawdown(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_max_drawdown_hand_computed():
    x = np.array([1.0, -3.0, 2.0])         # equity 1, -2, 0; peak 1; DD = 3
    assert max_drawdown(x) == pytest.approx(3.0)


def test_max_drawdown_nonnegative():
    x = np.array([-5.0, -3.0, 2.0, -1.0])
    assert max_drawdown(x) >= 0.0


def test_all_metrics_reject_2d():
    with pytest.raises(InvalidInputError):
        profit_factor(np.array([[1.0, 2.0]]))
