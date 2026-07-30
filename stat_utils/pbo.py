"""Combinatorially Symmetric Cross-Validation (CSCV) / PBO.

Given ``T`` time periods and ``K`` competing models with a per-period
performance metric, we ask: how often does the model that is *best in
sample* (IS) end up *below the median* out of sample (OOS)?

Procedure (Bailey et al. 2014):
  1. Split the T rows into ``S`` non-overlapping contiguous chunks.
  2. Enumerate all C(S, S/2) partitions of the chunks into an IS half
     and an OOS half.
  3. On each partition:
       - compute the performance statistic for each of K models on IS
       - compute the performance statistic for each of K models on OOS
       - identify the model that maximises IS performance
       - compute the OOS rank ``r`` of that model, then the logit
         ``λ = log(w / (1 - w))`` where ``w = r / (K + 1)``.
  4. PBO = fraction of partitions with ``λ < 0`` (i.e. IS-best falls in
     the bottom half of OOS ranks).

References
----------
Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2014).
"The Probability of Backtest Overfitting."
Journal of Computational Finance.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Mapping

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError
from ._rng import as_generator, SeedLike
from ._types import PBOResult
from ._validation import (
    as_2d_float_array,
    validate_positive_int,
)
from .metrics import sharpe as _sharpe


def _prepare_perf(perf: np.ndarray | Mapping[str, np.ndarray]
                  ) -> tuple[np.ndarray, list[str]]:
    if isinstance(perf, Mapping):
        labels = list(perf.keys())
        if not labels:
            raise InsufficientDataError("perf: empty mapping")
        cols = [np.asarray(perf[k], dtype=np.float64) for k in labels]
        T = cols[0].size
        for k, c in zip(labels[1:], cols[1:]):
            if c.size != T:
                raise InvalidInputError(
                    f"perf[{k!r}]: length {c.size} != {T}")
        return np.column_stack(cols), labels
    mat = as_2d_float_array(perf, "perf")
    return mat, [f"model_{k}" for k in range(mat.shape[1])]


def probability_backtest_overfitting(
    perf: np.ndarray | Mapping[str, np.ndarray],
    *,
    S: int = 8,
    n_splits: int | str = "all",
    performance_statistic: Callable[[np.ndarray], float] = _sharpe,
    seed: SeedLike = None,
) -> PBOResult:
    """CSCV / Probability of Backtest Overfitting.

    Parameters
    ----------
    perf : (T, K) array or Mapping[label, (T,) array]
        Per-period performance of ``K`` competing configurations. Higher
        is better.
    S : int
        Number of contiguous chunks. Must be even. Default 8 (gives
        C(8,4)=70 partitions).
    n_splits : int or "all"
        If ``"all"``, enumerate every partition. If int, subsample
        ``n_splits`` partitions uniformly at random using ``seed``.
    performance_statistic : callable
        Per-column callable returning a float; higher is better. Default
        Sharpe.
    seed : SeedLike
        Only used when ``n_splits`` is an int.

    Returns
    -------
    PBOResult
    """
    mat, labels = _prepare_perf(perf)
    T, K = mat.shape
    if K < 2:
        raise InsufficientDataError(
            "PBO requires at least 2 competing models")
    S = validate_positive_int(S, "S", minimum=2)
    if S % 2 != 0:
        raise InvalidInputError(f"S: must be even, got {S}")
    if T < S:
        raise InsufficientDataError(
            f"T={T} < S={S}: not enough periods to form chunks")

    # Contiguous chunks by index slicing (last chunk absorbs remainder)
    chunk_edges = np.linspace(0, T, S + 1, dtype=int)
    chunk_slices = [slice(chunk_edges[i], chunk_edges[i + 1]) for i in range(S)]

    all_partitions = list(combinations(range(S), S // 2))
    if n_splits == "all":
        selected = all_partitions
    else:
        n_splits = validate_positive_int(n_splits, "n_splits")
        n_splits = min(n_splits, len(all_partitions))
        rng = as_generator(seed)
        pick = rng.choice(len(all_partitions), size=n_splits, replace=False)
        selected = [all_partitions[i] for i in pick]

    logits = np.empty(len(selected), dtype=np.float64)
    selected_counts: dict[int, int] = {}
    ranks_of_selected_in_oos: list[float] = []

    for i, is_chunks in enumerate(selected):
        is_mask = np.zeros(T, dtype=bool)
        for c in is_chunks:
            is_mask[chunk_slices[c]] = True
        oos_mask = ~is_mask

        # per-model IS and OOS statistic
        is_perf = np.array([performance_statistic(mat[is_mask, k])
                            for k in range(K)])
        oos_perf = np.array([performance_statistic(mat[oos_mask, k])
                             for k in range(K)])
        # Replace NaN with -inf so they never win selection
        is_perf = np.where(np.isnan(is_perf), -np.inf, is_perf)
        oos_perf = np.where(np.isnan(oos_perf), -np.inf, oos_perf)

        best = int(np.argmax(is_perf))
        selected_counts[best] = selected_counts.get(best, 0) + 1

        # rank of `best` in OOS (1 = worst, K = best) using dense rank
        oos_order = np.argsort(np.argsort(oos_perf))
        r = int(oos_order[best]) + 1
        ranks_of_selected_in_oos.append(r)
        w = r / (K + 1.0)
        w = min(max(w, 1e-9), 1.0 - 1e-9)
        logits[i] = np.log(w / (1.0 - w))

    pbo = float((logits < 0).mean())
    ranks_arr = np.array(ranks_of_selected_in_oos, dtype=np.float64)
    perf_degradation_median = float(np.median(ranks_arr) / K)

    return PBOResult(
        pbo=pbo,
        logits=logits,
        n_splits=len(selected),
        n_models=K,
        performance_degradation_median=perf_degradation_median,
        selected_model_counts=selected_counts,
        seed=int(seed) if isinstance(seed, int) else None,
    )
