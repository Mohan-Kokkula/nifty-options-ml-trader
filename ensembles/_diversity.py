"""Ensemble diversity metrics (descriptive only).

These metrics summarise how independently the base brains make errors,
which informs interpretation of the ensemble outputs — but this module
does NOT compute Profit Factor, Sharpe, or any performance evaluation.

References
----------
Yule, G. U. (1900). "On the association of attributes in statistics."
Philosophical Transactions of the Royal Society A, 194, 257-319.

Kuncheva, L. I. and Whitaker, C. J. (2003). "Measures of diversity in
classifier ensembles and their relationship with the ensemble
accuracy." Machine Learning 51, 181-207.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._validation import EnsembleInputError


# ---------------------------------------------------------------------------
def _pair_counts(pred_a: np.ndarray, pred_b: np.ndarray,
                  y: np.ndarray) -> tuple[int, int, int, int]:
    """Return (N11, N10, N01, N00) where:

    * N11 = both classifiers correct
    * N10 = A correct, B wrong
    * N01 = A wrong, B correct
    * N00 = both wrong
    """
    if pred_a.shape != pred_b.shape or pred_a.shape != y.shape:
        raise EnsembleInputError(
            f"pair_counts shape mismatch: pred_a={pred_a.shape} "
            f"pred_b={pred_b.shape} y={y.shape}")
    a_ok = (pred_a == y)
    b_ok = (pred_b == y)
    n11 = int((a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    n01 = int((~a_ok & b_ok).sum())
    n00 = int((~a_ok & ~b_ok).sum())
    return n11, n10, n01, n00


def q_statistic(pred_a: np.ndarray, pred_b: np.ndarray,
                 y: np.ndarray) -> float:
    """Yule's Q pairwise diversity.

    ``Q = (N11 * N00 - N10 * N01) / (N11 * N00 + N10 * N01)``

    Ranges in ``[-1, +1]``:
      * ``+1`` — classifiers make identical mistakes (redundant)
      * ``0`` — errors are independent
      * ``-1`` — errors are anti-correlated (maximally diverse)
    """
    n11, n10, n01, n00 = _pair_counts(pred_a, pred_b, y)
    denom = n11 * n00 + n10 * n01
    if denom == 0:
        return 0.0
    return float((n11 * n00 - n10 * n01) / denom)


def disagreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Fraction of samples on which the two classifiers disagree.

    Range ``[0, 1]``. Higher = more diverse.
    """
    if pred_a.shape != pred_b.shape:
        raise EnsembleInputError(
            f"disagreement shape mismatch: {pred_a.shape} vs {pred_b.shape}")
    if pred_a.size == 0:
        return 0.0
    return float((pred_a != pred_b).mean())


def double_fault(pred_a: np.ndarray, pred_b: np.ndarray,
                  y: np.ndarray) -> float:
    """Fraction of samples on which both classifiers are wrong.

    Range ``[0, 1]``. Lower is better for ensemble accuracy — high double
    fault means the two classifiers share the same failure modes.
    """
    _, _, _, n00 = _pair_counts(pred_a, pred_b, y)
    return float(n00 / pred_a.size) if pred_a.size else 0.0


def prediction_correlation(prob_a: np.ndarray,
                             prob_b: np.ndarray) -> float:
    """Pearson correlation of flattened probability outputs.

    Useful as a soft measure of diversity that uses the full probability
    output rather than just the argmax.
    """
    if prob_a.shape != prob_b.shape:
        raise EnsembleInputError(
            f"prediction_correlation shape mismatch: "
            f"{prob_a.shape} vs {prob_b.shape}")
    a = np.asarray(prob_a, dtype=np.float64).ravel()
    b = np.asarray(prob_b, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
def diversity_matrix(
    brain_argmax: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    metric: str,
    prob_map: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return a ``K x K`` matrix of the requested pairwise diversity metric.

    Parameters
    ----------
    brain_argmax : Mapping[str, ndarray]
        Per-brain argmax predictions ``(n,)``. Keys become the matrix
        row/column labels — no assumption about brain identity.
    y_true : ndarray
        Ground-truth labels ``(n,)``.
    metric : {"q_statistic", "disagreement", "double_fault", "correlation"}
    prob_map : Mapping[str, ndarray], optional
        Required when ``metric == 'correlation'`` (needs full probs, not
        argmax).

    Returns
    -------
    DataFrame indexed and columned by brain name.
    """
    if not brain_argmax or len(brain_argmax) < 2:
        raise EnsembleInputError(
            f"diversity_matrix: need >= 2 brains, got {len(brain_argmax)}")
    names = sorted(brain_argmax.keys())
    K = len(names)
    mat = np.zeros((K, K), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                mat[i, j] = 0.0
                continue
            if metric == "q_statistic":
                mat[i, j] = q_statistic(brain_argmax[a], brain_argmax[b],
                                          y_true)
            elif metric == "disagreement":
                mat[i, j] = disagreement(brain_argmax[a], brain_argmax[b])
            elif metric == "double_fault":
                mat[i, j] = double_fault(brain_argmax[a], brain_argmax[b],
                                          y_true)
            elif metric == "correlation":
                if prob_map is None:
                    raise EnsembleInputError(
                        "correlation metric requires prob_map")
                mat[i, j] = prediction_correlation(prob_map[a], prob_map[b])
            else:
                raise EnsembleInputError(
                    f"unknown diversity metric {metric!r}")
    return pd.DataFrame(mat, index=names, columns=names)


def average_diversity_across_folds(
    per_fold: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average and standard deviation across a list of diversity matrices.

    All input matrices must share the same brain-name axis. Missing
    brains across folds cause an :class:`EnsembleInputError`.
    """
    if not per_fold:
        raise EnsembleInputError("average_diversity_across_folds: empty list")
    ref_cols = list(per_fold[0].columns)
    for i, m in enumerate(per_fold[1:], start=1):
        if list(m.columns) != ref_cols or list(m.index) != ref_cols:
            raise EnsembleInputError(
                f"fold {i}: diversity matrix has inconsistent brain axis")
    stacked = np.stack([m.values for m in per_fold], axis=0)
    mean = pd.DataFrame(stacked.mean(axis=0), index=ref_cols, columns=ref_cols)
    std = pd.DataFrame(stacked.std(axis=0, ddof=0),
                        index=ref_cols, columns=ref_cols)
    return mean, std
