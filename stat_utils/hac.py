"""Newey-West heteroskedasticity- and autocorrelation-consistent variance.

For a mean-zero series :math:`\\{d_t\\}_{t=1}^{T}` with lag ``L``, the
long-run variance estimator is

.. math::

    \\hat{\\sigma}^2 = \\gamma_0 + 2 \\sum_{k=1}^{L} \\left(1 - \\frac{k}{L+1}\\right)\\, \\gamma_k

where :math:`\\gamma_k = T^{-1} \\sum_{t=k+1}^{T} d_t d_{t-k}` is the
sample autocovariance at lag ``k``. The kernel is the Bartlett (triangular)
kernel.

References
----------
Newey, W. K. and West, K. D. (1987). "A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix."
Econometrica 55(3), 703-708.
"""
from __future__ import annotations

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError
from ._validation import as_1d_float_array


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance estimator.

    Parameters
    ----------
    x : array-like of float
        Input series, treated as a scalar time series.
    lag : int
        Truncation lag (``L`` in the formula above). Must satisfy
        ``0 <= lag < len(x)``.

    Returns
    -------
    float
        Estimator of the long-run variance. Always non-negative.

    Raises
    ------
    InvalidInputError
        If ``lag`` is negative or ``>= len(x)``.
    InsufficientDataError
        If the input has fewer than 2 observations.

    Notes
    -----
    Complexity is :math:`O(T \\cdot L)`.
    """
    arr = as_1d_float_array(x, "x")
    T = arr.size
    if T < 2:
        raise InsufficientDataError("newey_west_variance: need >= 2 obs")
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool):
        raise InvalidInputError(f"lag: expected int, got {type(lag)}")
    L = int(lag)
    if L < 0:
        raise InvalidInputError(f"lag: must be >= 0, got {L}")
    if L >= T:
        raise InvalidInputError(
            f"lag: must be < len(x)={T}, got {L}")

    d = arr - arr.mean()
    gamma0 = float(np.dot(d, d) / T)
    var = gamma0
    for k in range(1, L + 1):
        gamma_k = float(np.dot(d[k:], d[:-k]) / T)
        weight = 1.0 - k / (L + 1.0)
        var += 2.0 * weight * gamma_k
    # Bartlett kernel is guaranteed PSD, but small numerical drift can push
    # the result slightly negative for near-constant series.
    return max(var, 0.0)
