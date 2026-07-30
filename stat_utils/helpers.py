"""Wrappers around SciPy tests with typed return values.

These wrappers exist so downstream code (Phase 6 aggregator, publication
pack) can consume typed dataclasses rather than SciPy's positional
NamedTuples.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy import stats

from ._errors import InsufficientDataError, InvalidInputError
from ._rng import spawn_children, child_generator, SeedLike
from ._types import KendallResult, KSResult, LeveneResult, PermutationResult
from ._validation import (
    as_1d_float_array,
    validate_alternative,
    validate_positive_int,
)


def kendall_tau(x: np.ndarray, y: np.ndarray) -> KendallResult:
    """Kendall's tau-b rank correlation.

    Parameters
    ----------
    x, y : array-like of float

    Returns
    -------
    KendallResult
    """
    a = as_1d_float_array(x, "x")
    b = as_1d_float_array(y, "y")
    if a.size != b.size:
        raise InvalidInputError(
            f"x and y must have equal length; got {a.size} vs {b.size}")
    if a.size < 2:
        raise InsufficientDataError("kendall_tau: need >= 2 obs")
    r = stats.kendalltau(a, b)
    return KendallResult(statistic=float(r.statistic),
                          pvalue=float(r.pvalue),
                          n=int(a.size))


def levene(
    *groups: np.ndarray,
    center: Literal["mean", "median", "trimmed"] = "median",
) -> LeveneResult:
    """Levene / Brown-Forsythe test for equal variances.

    Parameters
    ----------
    *groups : sequence of array-like
        Two or more groups. Each must have >= 2 observations.
    center : {"mean", "median", "trimmed"}
        Brown-Forsythe (median) is the default.
    """
    if len(groups) < 2:
        raise InvalidInputError(
            f"levene: need >= 2 groups, got {len(groups)}")
    arrs = [as_1d_float_array(g, f"group_{i}") for i, g in enumerate(groups)]
    for i, a in enumerate(arrs):
        if a.size < 2:
            raise InsufficientDataError(
                f"levene: group {i} has {a.size} obs (< 2)")
    r = stats.levene(*arrs, center=center)
    return LeveneResult(statistic=float(r.statistic),
                        pvalue=float(r.pvalue),
                        center=center, n_groups=len(arrs))


def ks_2samp(x: np.ndarray, y: np.ndarray) -> KSResult:
    """Two-sample Kolmogorov-Smirnov test."""
    a = as_1d_float_array(x, "x")
    b = as_1d_float_array(y, "y")
    if a.size < 2 or b.size < 2:
        raise InsufficientDataError(
            f"ks_2samp: need >= 2 obs per sample; got {a.size}, {b.size}")
    r = stats.ks_2samp(a, b)
    return KSResult(statistic=float(r.statistic),
                    pvalue=float(r.pvalue),
                    n_x=int(a.size), n_y=int(b.size))


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_permutations: int = 10_000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    seed: SeedLike = None,
    return_null: bool = False,
) -> PermutationResult:
    """Two-sample permutation test.

    Under the null, ``x`` and ``y`` are exchangeable, i.e. drawn from the
    same distribution. Under each of ``n_permutations`` random relabelings
    of the pooled sample, the statistic is recomputed. The p-value is
    the (mid-p corrected) fraction of null replicates at least as extreme
    as the observed statistic.

    Parameters
    ----------
    x, y : array-like of float
        Independent samples.
    statistic : callable(x_perm, y_perm) -> float
    n_permutations : int
    alternative : {"two-sided", "greater", "less"}
    seed : SeedLike
    return_null : bool
        If True, the null distribution is attached to the result.

    Returns
    -------
    PermutationResult
    """
    a = as_1d_float_array(x, "x")
    b = as_1d_float_array(y, "y")
    if a.size < 2 or b.size < 2:
        raise InsufficientDataError(
            f"permutation_test: need >= 2 per sample")
    alternative = validate_alternative(alternative)
    n_permutations = validate_positive_int(n_permutations, "n_permutations")

    stat_obs = float(statistic(a, b))
    pooled = np.concatenate([a, b])
    n_a = a.size
    n_total = pooled.size

    children = spawn_children(seed, n_permutations)
    null = np.empty(n_permutations, dtype=np.float64)
    for i, child in enumerate(children):
        rng = child_generator(child)
        idx = rng.permutation(n_total)
        xp = pooled[idx[:n_a]]
        yp = pooled[idx[n_a:]]
        null[i] = float(statistic(xp, yp))

    if alternative == "two-sided":
        pval = float((np.abs(null) >= abs(stat_obs)).mean())
    elif alternative == "greater":
        pval = float((null >= stat_obs).mean())
    else:  # less
        pval = float((null <= stat_obs).mean())

    return PermutationResult(
        statistic=stat_obs,
        pvalue=pval,
        alternative=alternative,
        n_permutations=n_permutations,
        null_distribution=null if return_null else None,
    )
