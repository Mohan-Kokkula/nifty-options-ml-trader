"""Multiple hypothesis testing corrections.

Holm-Bonferroni (family-wise error rate, FWER)
    Sort p-values ascending; for the k-th smallest (1-indexed), reject if
    p_(k) <= alpha / (m - k + 1); stop at first non-rejection.

Benjamini-Hochberg (false discovery rate, FDR)
    Sort p-values ascending; find the largest k such that
    p_(k) <= k / m * alpha; reject all up to and including that k.

Both procedures return a keyed :class:`MHTResult` so downstream code can
index by hypothesis id.

References
----------
Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
Scandinavian Journal of Statistics 6(2), 65-70.

Benjamini, Y. and Hochberg, Y. (1995). "Controlling the false discovery
rate." JRSS-B 57(1), 289-300.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ._errors import InvalidInputError
from ._types import MHTResult
from ._validation import validate_alpha


def _validate_pvalues(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(pvalues, Mapping):
        raise InvalidInputError(
            f"pvalues: expected Mapping[str, float], got {type(pvalues)}")
    if not pvalues:
        raise InvalidInputError("pvalues: empty family")
    out: dict[str, float] = {}
    for k, v in pvalues.items():
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                f"pvalues[{k!r}]: cannot convert to float ({exc})") from exc
        if not np.isfinite(fv):
            raise InvalidInputError(
                f"pvalues[{k!r}]: must be finite, got {fv}")
        if not (0.0 <= fv <= 1.0):
            raise InvalidInputError(
                f"pvalues[{k!r}]: must lie in [0, 1], got {fv}")
        out[str(k)] = fv
    return out


def holm_bonferroni(
    pvalues: Mapping[str, float],
    alpha: float = 0.10,
) -> MHTResult:
    """Holm-Bonferroni step-down FWER-controlling procedure.

    Parameters
    ----------
    pvalues : Mapping[str, float]
        Family of raw p-values keyed by hypothesis id.
    alpha : float
        Target family-wise error rate. Default 0.10.

    Returns
    -------
    MHTResult
        ``decisions[key]`` contains ``rank`` (1 = smallest p), the
        ``adjusted_alpha`` at that rank, ``adjusted_pvalue`` (Holm), and
        ``reject`` (bool).
    """
    pvs = _validate_pvalues(pvalues)
    alpha = validate_alpha(alpha)
    m = len(pvs)

    order = sorted(pvs.items(), key=lambda kv: kv[1])
    # Adjusted p-values follow the step-down: p*_(k) = max_(j<=k) (m-j+1)*p_(j)
    running_max = 0.0
    holm_pvals: dict[str, float] = {}
    for rank, (key, p) in enumerate(order, start=1):
        adj = (m - rank + 1) * p
        running_max = max(running_max, adj)
        holm_pvals[key] = min(running_max, 1.0)

    decisions: dict[str, dict] = {}
    stopped = False
    for rank, (key, p) in enumerate(order, start=1):
        adj_alpha = alpha / (m - rank + 1)
        if not stopped and p <= adj_alpha:
            reject = True
        else:
            reject = False
            stopped = True
        decisions[key] = {
            "raw_pvalue": p,
            "adjusted_pvalue": holm_pvals[key],
            "adjusted_alpha": adj_alpha,
            "rank": rank,
            "reject": reject,
        }
    return MHTResult(method="holm_bonferroni", alpha=alpha, decisions=decisions)


def benjamini_hochberg(
    pvalues: Mapping[str, float],
    fdr: float = 0.10,
) -> MHTResult:
    """Benjamini-Hochberg FDR-controlling procedure.

    Parameters
    ----------
    pvalues : Mapping[str, float]
    fdr : float
        Target false discovery rate. Default 0.10.

    Returns
    -------
    MHTResult
        ``decisions[key]`` contains ``rank``, ``bh_threshold``,
        ``adjusted_pvalue`` (BH), and ``reject``.
    """
    pvs = _validate_pvalues(pvalues)
    fdr = validate_alpha(fdr)
    m = len(pvs)

    order = sorted(pvs.items(), key=lambda kv: kv[1])
    # BH adjusted p-values: q_(k) = min_(j>=k) (m/j) * p_(j)
    scaled = [(m / rank) * p for rank, (_, p) in enumerate(order, start=1)]
    running_min = np.inf
    bh_pvals: dict[str, float] = {}
    for rank in range(len(order), 0, -1):
        key = order[rank - 1][0]
        running_min = min(running_min, scaled[rank - 1])
        bh_pvals[key] = float(min(running_min, 1.0))

    # Find largest rank k with p_(k) <= k/m * fdr
    k_star = 0
    for rank, (_, p) in enumerate(order, start=1):
        if p <= (rank / m) * fdr:
            k_star = rank

    decisions: dict[str, dict] = {}
    for rank, (key, p) in enumerate(order, start=1):
        decisions[key] = {
            "raw_pvalue": p,
            "adjusted_pvalue": bh_pvals[key],
            "bh_threshold": (rank / m) * fdr,
            "rank": rank,
            "reject": rank <= k_star,
        }
    return MHTResult(method="benjamini_hochberg", alpha=fdr, decisions=decisions)
