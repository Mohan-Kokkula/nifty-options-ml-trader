"""Pluggable block-length selectors for stationary bootstrap.

The v1 implementation ships a single heuristic (``cbrt_block_length``) but
the public API accepts any callable satisfying :class:`BlockLengthSelector`,
so Politis-White or other selectors can be added later without changing
:func:`white_reality_check` / :func:`hansen_spa` signatures.

References
----------
Politis, D. N. and Romano, J. P. (1994). "The stationary bootstrap."
JASA 89(428), 1303-1313.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ._errors import InvalidInputError
from ._validation import as_1d_float_array


class BlockLengthSelector(Protocol):
    """Callable that maps a series to a mean block length for the
    stationary bootstrap.

    Implementations must return an ``int >= 1``.
    """

    def __call__(self, x: np.ndarray) -> int: ...  # pragma: no cover


def cbrt_block_length(x: np.ndarray) -> int:
    r"""Simple cube-root heuristic: ``ceil(4 * T^{1/3})``.

    Widely used as a default in empirical finance when no automatic
    selector is available. Documented as a heuristic, not a
    data-adaptive procedure.

    Parameters
    ----------
    x : array-like of float
        Series whose length ``T`` determines the block length.

    Returns
    -------
    int
        Mean block length, always ``>= 1``.
    """
    arr = as_1d_float_array(x, "x")
    T = arr.size
    return max(1, int(np.ceil(4.0 * T ** (1.0 / 3.0))))


def _resolve_block_length(
    block_length: int | BlockLengthSelector | str,
    x: np.ndarray,
) -> int:
    """Resolve *block_length* into an integer for the internal engine."""
    if isinstance(block_length, str):
        if block_length == "auto":
            return cbrt_block_length(x)
        raise InvalidInputError(
            f"block_length: unknown preset {block_length!r}")
    if callable(block_length):
        L = int(block_length(x))
        if L < 1:
            raise InvalidInputError(
                f"block_length selector returned {L}; must be >= 1")
        return L
    if isinstance(block_length, (int, np.integer)) and not isinstance(
            block_length, bool):
        L = int(block_length)
        if L < 1:
            raise InvalidInputError(
                f"block_length: must be >= 1, got {L}")
        return L
    raise InvalidInputError(
        f"block_length: unsupported type {type(block_length)}")
