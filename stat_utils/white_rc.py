"""White's Reality Check for data snooping.

Given ``K`` competing models and a benchmark, we ask whether the *best*
model's mean outperformance is statistically distinguishable from zero,
correcting for the fact that we picked the best of ``K``.

Implementation follows the bootstrap Reality Check of White (2000) with
Politis-Romano stationary bootstrap.

Test statistic:
    V = max_k  sqrt(T) * mean(f_k)
where ``f_k = perf_k - benchmark`` is the per-period outperformance of
model ``k``. Under the null (no model beats the benchmark), the p-value
is the fraction of bootstrap replicates of ``V*`` that exceed ``V``.

References
----------
White, H. (2000). "A Reality Check for data snooping." Econometrica
68(5), 1097-1126.
Politis, D. N. and Romano, J. P. (1994). "The stationary bootstrap."
JASA 89(428), 1303-1313.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError
from ._rng import as_generator, spawn_children, child_generator, SeedLike
from ._types import WhiteRCResult
from ._validation import (
    as_2d_float_array,
    validate_ci_level,
    validate_positive_int,
)
from .block_length import BlockLengthSelector, _resolve_block_length


def _stationary_bootstrap_indices(
    rng: np.random.Generator, T: int, mean_block: int
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric restart with
    probability ``p = 1 / mean_block``. Returns an int array of length T.
    """
    p = 1.0 / max(mean_block, 1)
    out = np.empty(T, dtype=np.int64)
    starts = rng.integers(0, T, size=T)  # candidate restart positions
    restarts = rng.random(T) < p
    restarts[0] = True  # always start from a random position
    pos = 0
    for t in range(T):
        if restarts[t]:
            pos = int(starts[t])
        else:
            pos = (pos + 1) % T
        out[t] = pos
    return out


def _prepare_perf_matrix(
    perf: np.ndarray | Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Return an (T, K) float matrix and a list of K model labels."""
    if isinstance(perf, Mapping):
        labels = list(perf.keys())
        if not labels:
            raise InsufficientDataError(
                "perf: empty mapping (no models supplied)")
        cols = [np.asarray(perf[k], dtype=np.float64) for k in labels]
        for k, c in zip(labels, cols):
            if c.ndim != 1:
                raise InvalidInputError(
                    f"perf[{k!r}]: expected 1-D, got shape {c.shape}")
        T = cols[0].size
        for k, c in zip(labels[1:], cols[1:]):
            if c.size != T:
                raise InvalidInputError(
                    f"perf[{k!r}]: length {c.size} != {T}")
        mat = np.column_stack(cols)
    else:
        mat = as_2d_float_array(perf, "perf")
        labels = [f"model_{k}" for k in range(mat.shape[1])]
    T, K = mat.shape
    if T < 2:
        raise InsufficientDataError(
            f"perf: need at least 2 time periods, got {T}")
    if K < 1:
        raise InsufficientDataError("perf: need at least 1 model")
    return mat, labels


def white_reality_check(
    perf: np.ndarray | Mapping[str, np.ndarray],
    benchmark: np.ndarray | None = None,
    *,
    n_bootstrap: int = 10_000,
    block_length: int | BlockLengthSelector | str = "auto",
    seed: SeedLike = None,
) -> WhiteRCResult:
    """White's Reality Check.

    Parameters
    ----------
    perf : (T, K) array or Mapping[label, (T,) array]
        Per-period performance of ``K`` competing models. Higher is better.
    benchmark : (T,) array or None
        Per-period benchmark. Defaults to zeros (test whether the best
        model has strictly positive mean performance).
    n_bootstrap : int
        Number of stationary-bootstrap replicates. Default 10 000.
    block_length : int, "auto", or BlockLengthSelector
        Mean geometric block length. ``"auto"`` uses
        :func:`stat_utils.block_length.cbrt_block_length` on the
        outperformance series of the leading model.
    seed : SeedLike

    Returns
    -------
    WhiteRCResult
    """
    mat, labels = _prepare_perf_matrix(perf)
    T, K = mat.shape
    if benchmark is None:
        bench = np.zeros(T)
    else:
        bench = np.asarray(benchmark, dtype=np.float64)
        if bench.ndim != 1 or bench.size != T:
            raise InvalidInputError(
                f"benchmark: expected 1-D length {T}, got shape {bench.shape}")
    n_bootstrap = validate_positive_int(n_bootstrap, "n_bootstrap")

    f = mat - bench[:, None]                # (T, K) outperformance
    f_mean = f.mean(axis=0)                 # (K,)
    L = _resolve_block_length(block_length, f[:, int(np.argmax(f_mean))])
    stat_v = float(np.sqrt(T) * f_mean.max())

    # Centre outperformance under H0
    centred = f - f_mean[None, :]

    children = spawn_children(seed, n_bootstrap)
    v_star = np.empty(n_bootstrap, dtype=np.float64)
    for i, child in enumerate(children):
        rng = child_generator(child)
        idx = _stationary_bootstrap_indices(rng, T, L)
        boot = centred[idx]                 # (T, K)
        boot_mean = boot.mean(axis=0) + f_mean
        # Reality-check statistic on the bootstrap sample uses the same
        # centring convention: sqrt(T) * max_k (boot_mean_k - f_mean_k)
        v_star[i] = float(np.sqrt(T) * (boot_mean - f_mean).max())

    pval = float((v_star >= stat_v).mean())
    return WhiteRCResult(
        pvalue=pval,
        statistic=stat_v,
        n_models=K,
        n_bootstrap=n_bootstrap,
        block_length=L,
        per_model_mean_perf={lbl: float(f_mean[i]) for i, lbl in enumerate(labels)},
        seed=int(seed) if isinstance(seed, int) else None,
    )
