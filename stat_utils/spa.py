"""Hansen's Superior Predictive Ability test.

SPA improves on White's Reality Check by studentising per-model outper-
formance and by handling models that are dominated ("poor" models) in a
way that yields a less conservative test. Hansen (2005) proposes three
p-values distinguished by how the null distribution's centring set is
determined:

    SPA_l  — lower  (conservative; all models kept)
    SPA_c  — consistent (recommended; models with strongly negative
             standardised mean are pushed toward the null)
    SPA_u  — upper  (liberal; only models with non-negative mean are kept)

Implementation follows Hansen (2005) Section 4 with Politis-Romano
stationary bootstrap.

References
----------
Hansen, P. R. (2005). "A test for Superior Predictive Ability."
Journal of Business & Economic Statistics 23(4), 365-380.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError
from ._rng import spawn_children, child_generator, SeedLike
from ._types import SPAResult
from ._validation import validate_positive_int
from .block_length import BlockLengthSelector, _resolve_block_length
from .hac import newey_west_variance
from .white_rc import _prepare_perf_matrix, _stationary_bootstrap_indices


def _long_run_variance_per_column(f: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West long-run variance for each column of ``f``.

    We use the Bartlett-kernel HAC estimator with truncation ``lag``
    (typically the mean block length of the stationary bootstrap). This
    replaces Hansen's stationary-bootstrap variance formula, which is
    algebraically equivalent in expectation but numerically unstable
    when accumulated over the full T-1 range on nearly-white series.
    """
    T, K = f.shape
    out = np.empty(K)
    L = min(lag, T - 1)
    for k in range(K):
        out[k] = newey_west_variance(f[:, k], lag=L)
    return out


def hansen_spa(
    perf: np.ndarray | Mapping[str, np.ndarray],
    benchmark: np.ndarray | None = None,
    *,
    n_bootstrap: int = 10_000,
    block_length: int | BlockLengthSelector | str = "auto",
    seed: SeedLike = None,
) -> SPAResult:
    """Hansen (2005) SPA test.

    Parameters
    ----------
    perf : (T, K) array or Mapping[label, (T,) array]
        Per-period performance (higher is better).
    benchmark : (T,) or None
        Per-period benchmark. Defaults to zeros.
    n_bootstrap : int
        Bootstrap replicates. Default 10 000.
    block_length : int, "auto", or BlockLengthSelector
        Mean geometric block length for the stationary bootstrap.
    seed : SeedLike

    Returns
    -------
    SPAResult
        Contains ``pvalue_lower``, ``pvalue_consistent`` (Hansen's
        recommended value), and ``pvalue_upper``.
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

    f = mat - bench[:, None]
    fbar = f.mean(axis=0)
    L = _resolve_block_length(block_length, f[:, int(np.argmax(fbar))])
    omega = _long_run_variance_per_column(f, L)
    # Guard against degenerate zero variance (constant column)
    scale = np.sqrt(np.maximum(omega, 1e-12))

    # Studentised statistic
    z = np.sqrt(T) * fbar / scale
    stat = float(z.max())

    # Recentring per Hansen (2005) Section 4. Naming refers to the
    # p-value bound each version produces:
    #   Lower  bound on p (least conservative): g_k = 0 for all k
    #   Consistent (recommended):               g_k = 0 iff sqrt(T)*fbar_k/omega_k < -A_n
    #                                           else g_k = fbar_k
    #   Upper  bound on p (most conservative):  g_k = fbar_k for all k
    A_n = np.sqrt(2.0 * np.log(np.log(max(T, 3))))
    lower_center = np.zeros_like(fbar)
    consistent_center = np.where(z < -A_n, 0.0, fbar)
    upper_center = fbar.copy()

    # Bootstrap
    children = spawn_children(seed, n_bootstrap)
    stats_l = np.empty(n_bootstrap)
    stats_c = np.empty(n_bootstrap)
    stats_u = np.empty(n_bootstrap)
    for i, child in enumerate(children):
        rng = child_generator(child)
        idx = _stationary_bootstrap_indices(rng, T, L)
        fb = f[idx].mean(axis=0)
        zb = np.sqrt(T) * (fb - fbar) / scale
        # Add back the (possibly shrunk) mean under each recentring
        z_l = zb + np.sqrt(T) * lower_center / scale
        z_c = zb + np.sqrt(T) * consistent_center / scale
        z_u = zb + np.sqrt(T) * upper_center / scale
        stats_l[i] = z_l.max()
        stats_c[i] = z_c.max()
        stats_u[i] = z_u.max()

    p_l = float((stats_l >= stat).mean())
    p_c = float((stats_c >= stat).mean())
    p_u = float((stats_u >= stat).mean())

    return SPAResult(
        pvalue_lower=p_l,
        pvalue_consistent=p_c,
        pvalue_upper=p_u,
        statistic=stat,
        n_models=K,
        n_bootstrap=n_bootstrap,
        block_length=L,
        per_model_mean_perf={lbl: float(fbar[i]) for i, lbl in enumerate(labels)},
        seed=int(seed) if isinstance(seed, int) else None,
    )
