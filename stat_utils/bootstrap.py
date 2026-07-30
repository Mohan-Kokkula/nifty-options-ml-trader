"""Block bootstrap for fold-structured P&L streams.

Two entry points:

  * :func:`block_bootstrap_ci` — single-arm CI for an arbitrary statistic
    computed on the pooled trades of a set of walk-forward folds.
  * :func:`paired_block_bootstrap_ci` — CI on a delta statistic computed
    on the pooled trades of two arms A and B under the SAME resampled
    fold selection (paired design).

Determinism guarantee
---------------------
Output is fully determined by ``(seed, n_resamples)`` regardless of
``n_jobs``. Internally we spawn one child :class:`SeedSequence` per
replicate up front, then farm the replicate axis out to worker threads.

Parallelism
-----------
``n_jobs=1`` runs in the calling thread. ``n_jobs>1`` uses
:class:`concurrent.futures.ThreadPoolExecutor`; the NumPy code path
releases the GIL during array operations so this gives a real speedup
without requiring pickle-able statistic functions.

Method
------
Only the percentile method is implemented in v1. The API accepts
``method="percentile"`` to leave room for future BCa or bias-corrected
alternatives without a signature change.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError
from ._rng import as_generator, spawn_children, child_generator, SeedLike
from ._types import BootstrapCI
from ._validation import (
    validate_ci_level,
    validate_fold_streams,
    validate_positive_int,
)

_METHODS = ("percentile",)

Statistic = Callable[[np.ndarray], float]
PairedStatistic = Callable[[np.ndarray, np.ndarray], float]


def _prepare_fold_arrays(
    streams: Mapping[Any, np.ndarray],
) -> tuple[list[Any], list[np.ndarray]]:
    """Return (fold_ids, per-fold arrays) in stable insertion order.

    Empty folds are retained so that resampling frequencies are unaffected
    when a fold happens to have zero trades.
    """
    ids = list(streams.keys())
    arrays = [streams[k] for k in ids]
    return ids, arrays


def _percentile_ci(dist: np.ndarray, ci_level: float
                   ) -> tuple[float, float]:
    lo = (1.0 - ci_level) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(dist, lo)), float(np.quantile(dist, hi))


def _run_replicates(
    child_ids: Sequence[np.random.SeedSequence],
    n_folds: int,
    evaluate: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Draw fold-index samples for each child seed and evaluate."""
    out = np.empty(len(child_ids), dtype=np.float64)
    for i, child in enumerate(child_ids):
        rng = child_generator(child)
        idx = rng.integers(0, n_folds, size=n_folds)
        out[i] = evaluate(idx)
    return out


def _chunked_run(
    children: list[np.random.SeedSequence],
    n_folds: int,
    evaluate: Callable[[np.ndarray], float],
    n_jobs: int,
    progress: bool,
) -> np.ndarray:
    """Split ``children`` into ``n_jobs`` chunks and run each in a thread.

    Determinism is preserved because each ``child`` produces a fixed
    stream of fold indices independent of the chunking.
    """
    if n_jobs == 1 or len(children) < n_jobs * 4:
        return _maybe_progress(
            _run_replicates(children, n_folds, evaluate),
            enabled=False,
        )
    chunks = np.array_split(np.arange(len(children)), n_jobs)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_run_replicates,
                             [children[j] for j in chunk],
                             n_folds, evaluate)
                   for chunk in chunks]
        parts = [f.result() for f in futures]
    result = np.concatenate(parts)
    return _maybe_progress(result, enabled=False)


def _maybe_progress(dist: np.ndarray, enabled: bool) -> np.ndarray:
    """Progress-bar hook. In v1 we simply pass through; the boolean is
    reserved so that a tqdm integration in a future minor version keeps
    the same call surface."""
    return dist


def _validate_method(method: str) -> str:
    if method not in _METHODS:
        raise InvalidInputError(
            f"method: expected one of {list(_METHODS)}, got {method!r}")
    return method


def _validate_statistic(statistic: Callable, name: str) -> None:
    if not callable(statistic):
        raise InvalidInputError(f"{name}: must be callable")


# ---------------------------------------------------------------------------
# Public: single-arm block bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    fold_streams: Mapping[Any, np.ndarray],
    statistic: Statistic,
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.90,
    seed: SeedLike = None,
    stat_name: str | None = None,
    method: Literal["percentile"] = "percentile",
    block_unit: str = "fold",
    n_jobs: int = 1,
    return_distribution: bool = False,
    progress: bool = False,
) -> BootstrapCI:
    """Block bootstrap confidence interval on the pooled trade stream.

    Parameters
    ----------
    fold_streams : Mapping[fold_id, ndarray]
        Per-fold trade P&L arrays. Fold-ids are the resampling blocks.
    statistic : callable
        Takes the pooled bootstrap trade array and returns a float.
    n_resamples : int
        Number of bootstrap replicates. Default 10 000.
    ci_level : float
        Nominal coverage in (0, 1). Default 0.90.
    seed : int, SeedSequence, Generator, or None
        Reproducibility seed. See :mod:`stat_utils._rng`.
    stat_name : str, optional
        Free-form label recorded in the result.
    method : {"percentile"}
        Only percentile CI is implemented in v1.
    block_unit : str
        Descriptive tag stored in the result (e.g. ``"fold"``).
    n_jobs : int
        Number of thread workers for the replicate loop. Determinism is
        preserved regardless of ``n_jobs``.
    return_distribution : bool
        If True, the ``bootstrap_distribution`` array is attached to the
        result. Adds ``8 * n_resamples`` bytes of memory.
    progress : bool
        Reserved for future progress-bar integration.

    Returns
    -------
    BootstrapCI

    Raises
    ------
    InvalidInputError
        On structurally invalid input.
    InsufficientDataError
        If the input has zero folds or every fold is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from stat_utils import block_bootstrap_ci, profit_factor
    >>> rng = np.random.default_rng(0)
    >>> streams = {k: rng.normal(size=200) for k in range(8)}
    >>> ci = block_bootstrap_ci(streams, profit_factor, seed=42)
    >>> ci.point_estimate  # doctest: +SKIP
    """
    streams = validate_fold_streams(fold_streams, "fold_streams")
    _validate_statistic(statistic, "statistic")
    n_resamples = validate_positive_int(n_resamples, "n_resamples")
    ci_level = validate_ci_level(ci_level)
    method = _validate_method(method)
    n_jobs = validate_positive_int(n_jobs, "n_jobs")

    fold_ids, arrays = _prepare_fold_arrays(streams)
    n_folds = len(arrays)
    pooled = np.concatenate(arrays) if any(a.size for a in arrays) else \
        np.empty(0, dtype=np.float64)
    if pooled.size == 0:
        raise InsufficientDataError(
            "block_bootstrap_ci: all folds are empty after validation")

    point_estimate = float(statistic(pooled))

    def evaluate(idx: np.ndarray) -> float:
        # Concatenating small arrays in a Python list-comprehension is
        # dominated by numpy allocation; this is intentionally simple.
        parts = [arrays[j] for j in idx]
        if all(p.size == 0 for p in parts):
            return float("nan")
        sample = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return float(statistic(sample))

    children = spawn_children(seed, n_resamples)
    dist = _chunked_run(children, n_folds, evaluate, n_jobs, progress)

    finite = np.isfinite(dist)
    n_valid = int(finite.sum())
    if n_valid == 0:
        raise InsufficientDataError(
            "block_bootstrap_ci: no finite bootstrap replicates")
    finite_dist = dist[finite]
    lo, hi = _percentile_ci(finite_dist, ci_level)

    return BootstrapCI(
        stat_name=stat_name or getattr(statistic, "__name__", "statistic"),
        point_estimate=point_estimate,
        lower=lo,
        upper=hi,
        ci_level=ci_level,
        n_resamples=n_resamples,
        n_valid_resamples=n_valid,
        n_blocks=n_folds,
        block_unit=block_unit,
        method=method,
        seed=_seed_to_int(seed),
        paired=False,
        bootstrap_distribution=finite_dist if return_distribution else None,
    )


# ---------------------------------------------------------------------------
# Public: paired block bootstrap
# ---------------------------------------------------------------------------
def paired_block_bootstrap_ci(
    fold_streams_a: Mapping[Any, np.ndarray],
    fold_streams_b: Mapping[Any, np.ndarray],
    statistic_delta: PairedStatistic,
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.90,
    seed: SeedLike = None,
    stat_name: str | None = None,
    method: Literal["percentile"] = "percentile",
    block_unit: str = "fold",
    n_jobs: int = 1,
    return_distribution: bool = False,
    progress: bool = False,
) -> BootstrapCI:
    """Paired block bootstrap CI on a delta statistic.

    Both arms are resampled under the SAME sequence of fold indices per
    replicate, so the resulting distribution properly reflects the
    joint variability of the two arms.

    ``fold_streams_a`` and ``fold_streams_b`` must share the same set of
    fold-ids. Per-fold arrays need NOT have equal lengths across arms.

    Parameters
    ----------
    fold_streams_a, fold_streams_b : Mapping[fold_id, ndarray]
    statistic_delta : callable(a_arr, b_arr) -> float
        Applied to the pooled resampled arrays of A and B respectively.

    Returns
    -------
    BootstrapCI
        ``paired=True``. ``point_estimate`` is the delta on the full
        (un-resampled) pooled arrays.
    """
    a = validate_fold_streams(fold_streams_a, "fold_streams_a")
    b = validate_fold_streams(fold_streams_b, "fold_streams_b")
    if set(a.keys()) != set(b.keys()):
        raise InvalidInputError(
            "fold_streams_a and fold_streams_b must share the same fold-ids")
    _validate_statistic(statistic_delta, "statistic_delta")
    n_resamples = validate_positive_int(n_resamples, "n_resamples")
    ci_level = validate_ci_level(ci_level)
    method = _validate_method(method)
    n_jobs = validate_positive_int(n_jobs, "n_jobs")

    fold_ids = list(a.keys())
    a_arrays = [a[k] for k in fold_ids]
    b_arrays = [b[k] for k in fold_ids]
    n_folds = len(fold_ids)

    a_pooled = np.concatenate(a_arrays) if any(x.size for x in a_arrays) \
        else np.empty(0)
    b_pooled = np.concatenate(b_arrays) if any(x.size for x in b_arrays) \
        else np.empty(0)
    if a_pooled.size == 0 or b_pooled.size == 0:
        raise InsufficientDataError(
            "paired_block_bootstrap_ci: at least one arm has zero trades")

    point_estimate = float(statistic_delta(a_pooled, b_pooled))

    def evaluate(idx: np.ndarray) -> float:
        a_parts = [a_arrays[j] for j in idx]
        b_parts = [b_arrays[j] for j in idx]
        if all(p.size == 0 for p in a_parts) or all(p.size == 0 for p in b_parts):
            return float("nan")
        a_sample = np.concatenate(a_parts) if len(a_parts) > 1 else a_parts[0]
        b_sample = np.concatenate(b_parts) if len(b_parts) > 1 else b_parts[0]
        return float(statistic_delta(a_sample, b_sample))

    children = spawn_children(seed, n_resamples)
    dist = _chunked_run(children, n_folds, evaluate, n_jobs, progress)

    finite = np.isfinite(dist)
    n_valid = int(finite.sum())
    if n_valid == 0:
        raise InsufficientDataError(
            "paired_block_bootstrap_ci: no finite replicates")
    finite_dist = dist[finite]
    lo, hi = _percentile_ci(finite_dist, ci_level)

    return BootstrapCI(
        stat_name=stat_name or getattr(statistic_delta, "__name__",
                                       "statistic_delta"),
        point_estimate=point_estimate,
        lower=lo,
        upper=hi,
        ci_level=ci_level,
        n_resamples=n_resamples,
        n_valid_resamples=n_valid,
        n_blocks=n_folds,
        block_unit=block_unit,
        method=method,
        seed=_seed_to_int(seed),
        paired=True,
        bootstrap_distribution=finite_dist if return_distribution else None,
    )


def _seed_to_int(seed: SeedLike) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    return None  # Generator / SeedSequence not representable as a scalar
