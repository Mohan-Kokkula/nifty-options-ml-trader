"""Trading performance metrics.

All metrics accept a 1-D array of per-observation P&L (typically per-trade
net rupees). None of them assume any time scaling unless documented.

Return type is always ``float``.
"""
from __future__ import annotations

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError
from ._validation import as_1d_float_array


def profit_factor(pnl: np.ndarray) -> float:
    """Ratio of gross winning P&L to gross losing P&L (absolute).

    Returns ``float('inf')`` when the losing sum is zero and winners exist,
    ``float('nan')`` when neither positive nor negative P&L exists.

    Parameters
    ----------
    pnl : array-like of float

    Notes
    -----
    Zeros are excluded from both sums (they carry no information).
    """
    arr = as_1d_float_array(pnl, "pnl")
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses <= 0 and gains <= 0:
        return float("nan")
    if losses <= 0:
        return float("inf")
    return float(gains / losses)


def win_rate(pnl: np.ndarray) -> float:
    """Fraction of strictly positive P&L observations, in [0, 1]."""
    arr = as_1d_float_array(pnl, "pnl")
    return float((arr > 0).mean())


def expectancy(pnl: np.ndarray) -> float:
    """Mean per-observation P&L."""
    arr = as_1d_float_array(pnl, "pnl")
    return float(arr.mean())


def sharpe(pnl: np.ndarray, annualization: float = 1.0) -> float:
    """Sample Sharpe ratio: mean / std, optionally annualised.

    Parameters
    ----------
    pnl : array-like of float
    annualization : float
        Multiplier applied to the mean/std ratio (e.g. ``sqrt(252)`` for
        daily observations targeting annual Sharpe). Default 1.0 leaves the
        ratio in the units of *pnl*.

    Returns
    -------
    float
        Sample Sharpe. Returns ``float('nan')`` when std is zero.
    """
    arr = as_1d_float_array(pnl, "pnl")
    if arr.size < 2:
        raise InsufficientDataError("sharpe: need at least 2 observations")
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(annualization * mu / sd)


def sortino(pnl: np.ndarray, annualization: float = 1.0, mar: float = 0.0
            ) -> float:
    """Sortino ratio: (mean - MAR) / downside deviation.

    Parameters
    ----------
    pnl : array-like of float
    annualization : float
        See :func:`sharpe`.
    mar : float
        Minimum acceptable return (per-observation). Default 0.

    Returns
    -------
    float
        Returns ``float('nan')`` when there is no downside deviation.
    """
    arr = as_1d_float_array(pnl, "pnl")
    if arr.size < 2:
        raise InsufficientDataError("sortino: need at least 2 observations")
    excess = arr - mar
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("nan")
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0.0:
        return float("nan")
    return float(annualization * excess.mean() / dd)


def max_drawdown(pnl: np.ndarray) -> float:
    """Maximum peak-to-trough decline of the cumulative P&L, as a
    non-negative float in the same units as *pnl*.

    Zero when the cumulative curve is monotonically non-decreasing.
    """
    arr = as_1d_float_array(pnl, "pnl")
    equity = np.cumsum(arr)
    peaks = np.maximum.accumulate(equity)
    dd = equity - peaks
    return float(-dd.min()) if dd.size else 0.0
