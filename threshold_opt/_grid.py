"""Deterministic grid generator.

Every call with identical arguments yields the same candidate sequence
in the same order (``call_thr``, ``put_thr``, ``skip_ceil``,
``min_edge``). Candidates that fail
:class:`~threshold_opt._base.ThresholdCandidate` validation are
silently skipped rather than raised so the caller can iterate over a
naive cartesian product.
"""
from __future__ import annotations

from itertools import product
from typing import Iterator, Sequence

from ._base import InvalidThresholdError, ThresholdCandidate


# Approved production grid (from Phase-6 spec)
DEFAULT_CALL_RANGE: tuple[float, ...] = (0.20, 0.25, 0.30, 0.32, 0.35, 0.40)
DEFAULT_PUT_RANGE: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35)
DEFAULT_SKIP_RANGE: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75)
DEFAULT_EDGE_RANGE: tuple[float, ...] = (0.03, 0.05, 0.08)


def _norm(x: Sequence[float] | None,
           fallback: Sequence[float]) -> list[float]:
    return sorted(set(float(v) for v in (x if x is not None else fallback)))


def grid_generator(
    call: Sequence[float] | None = None,
    put: Sequence[float] | None = None,
    skip: Sequence[float] | None = None,
    edge: Sequence[float] | None = None,
) -> Iterator[ThresholdCandidate]:
    """Yield validated :class:`ThresholdCandidate` in deterministic order.

    Iteration order matches
    ``product(sorted(call), sorted(put), sorted(skip), sorted(edge))``.
    Invalid combinations (``min_edge >= threshold``) are skipped.
    """
    call_v = _norm(call, DEFAULT_CALL_RANGE)
    put_v = _norm(put, DEFAULT_PUT_RANGE)
    skip_v = _norm(skip, DEFAULT_SKIP_RANGE)
    edge_v = _norm(edge, DEFAULT_EDGE_RANGE)
    for c, p, s, e in product(call_v, put_v, skip_v, edge_v):
        try:
            yield ThresholdCandidate(call_thr=c, put_thr=p,
                                       skip_ceil=s, min_edge=e)
        except InvalidThresholdError:
            continue


def grid_size(
    call: Sequence[float] | None = None,
    put: Sequence[float] | None = None,
    skip: Sequence[float] | None = None,
    edge: Sequence[float] | None = None,
) -> int:
    """Count the number of VALID candidates the grid would yield."""
    return sum(1 for _ in grid_generator(call, put, skip, edge))
