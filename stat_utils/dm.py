"""Diebold-Mariano test with HAC (Newey-West) variance.

The DM statistic tests whether two competing forecast-loss series have
equal expected loss. For our purposes we pass per-trade *P&L* rather than
losses; a positive mean loss differential ``d_t = a_t - b_t`` then means
"A pays more than B". The ``alternative`` argument selects the direction
of the alternative hypothesis.

Under the null of equal expected loss:

    DM = mean(d) / sqrt( LRV(d) / n )   ~  N(0, 1)      asymptotically

where ``LRV`` is the long-run variance (Newey-West).

References
----------
Diebold, F. X. and Mariano, R. S. (1995). "Comparing predictive accuracy."
Journal of Business & Economic Statistics 13(3), 253-263.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

from ._errors import InsufficientDataError, InvalidInputError
from ._types import DMResult
from ._validation import (
    as_1d_float_array,
    validate_alternative,
    validate_ci_level,
    validate_positive_int,
)
from .hac import newey_west_variance


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    lag: int = 5,
    variance_estimator: Literal["newey_west"] = "newey_west",
    ci_level: float = 0.90,
) -> DMResult:
    """Diebold-Mariano equal-predictive-accuracy test.

    Parameters
    ----------
    loss_a, loss_b : array-like of float
        Per-observation loss (or P&L; see module docstring) for A and B.
        Must have the same length.
    alternative : {"two-sided", "greater", "less"}
        Direction of the alternative.
        - ``"greater"``: mean(loss_a - loss_b) > 0 (A's loss exceeds B's)
        - ``"less"``:    mean(loss_a - loss_b) < 0
    lag : int
        Truncation lag for the HAC variance. Must satisfy
        ``0 <= lag < len(loss_a) - 1``.
    variance_estimator : {"newey_west"}
        Reserved for future variance estimators.
    ci_level : float
        Nominal coverage of the CI on the mean loss differential.

    Returns
    -------
    DMResult

    Raises
    ------
    InvalidInputError
        On structurally invalid input.
    InsufficientDataError
        If fewer than ``lag + 2`` observations are provided.
    """
    a = as_1d_float_array(loss_a, "loss_a")
    b = as_1d_float_array(loss_b, "loss_b")
    if a.size != b.size:
        raise InvalidInputError(
            f"loss_a and loss_b must be equal length; got {a.size} vs {b.size}")
    alternative = validate_alternative(alternative)
    lag = validate_positive_int(lag, "lag", minimum=0)
    ci_level = validate_ci_level(ci_level)
    if variance_estimator != "newey_west":
        raise InvalidInputError(
            f"variance_estimator: expected 'newey_west', got {variance_estimator!r}")

    d = a - b
    n = d.size
    if n < lag + 2:
        raise InsufficientDataError(
            f"diebold_mariano: need at least lag+2={lag+2} obs, got {n}")

    d_mean = float(d.mean())
    lrv = newey_west_variance(d, lag=lag)
    if lrv <= 0.0:
        # Degenerate: constant differential. Statistic is 0 if mean is 0,
        # otherwise infinite. Report as p == 1 or 0 accordingly.
        stat = 0.0 if d_mean == 0.0 else float("inf") * np.sign(d_mean)
        pval = 1.0 if d_mean == 0.0 else 0.0
        se = 0.0
        ci_lo, ci_hi = d_mean, d_mean
        return DMResult(
            statistic=stat, pvalue=pval, alternative=alternative,
            lag=lag, n=n, mean_loss_diff=d_mean, se_loss_diff=se,
            ci_lower=ci_lo, ci_upper=ci_hi, ci_level=ci_level,
            variance_estimator=variance_estimator,
        )

    se = float(np.sqrt(lrv / n))
    stat = d_mean / se

    if alternative == "two-sided":
        pval = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    elif alternative == "greater":
        pval = 1.0 - stats.norm.cdf(stat)
    else:  # less
        pval = stats.norm.cdf(stat)

    z_half = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    ci_lo = d_mean - z_half * se
    ci_hi = d_mean + z_half * se

    return DMResult(
        statistic=float(stat),
        pvalue=float(pval),
        alternative=alternative,
        lag=lag,
        n=n,
        mean_loss_diff=d_mean,
        se_loss_diff=se,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        ci_level=ci_level,
        variance_estimator=variance_estimator,
    )
