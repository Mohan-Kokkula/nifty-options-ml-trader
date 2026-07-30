"""R1 – walk-forward robustness variants.

Given a candidate's per-fold PnL streams (from the frozen Phase-5 window
split), Phase 7 reports the pooled metrics under three reuse-only
variations:

* **±1 fold shift** – recompute pooled metrics after dropping either the
  first or the last fold. This tests whether an edge-of-window fold is
  driving the aggregate. Retraining is out of scope (Phase 7 is
  read-only), so this is the strongest fold-shift check available.
* **Expanding window** – cumulative pools ``[fold_1..fold_k]`` for
  ``k = 1..8``. Shows how metrics stabilise as more folds accrue.
* **Rolling window** – 3-fold rolling pools ``[i, i+1, i+2]``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from threshold_opt._evaluate import _fold_metrics

from ._base import ROLLING_WINDOW


def _pool(pnl_by_fold: Mapping[int, np.ndarray], subset) -> dict:
    arrs = [pnl_by_fold[i] for i in subset if i in pnl_by_fold]
    if not arrs:
        return _fold_metrics(np.array([], dtype=np.float64))
    return _fold_metrics(np.concatenate(arrs))


def fold_shift_variants(pnl_by_fold: Mapping[int, np.ndarray]) -> dict:
    """Return metrics for {baseline, shift_-1, shift_+1}."""
    all_folds = sorted(pnl_by_fold.keys())
    result = {
        "baseline_all_folds": {
            "folds": all_folds,
            "metrics": _pool(pnl_by_fold, all_folds),
        },
        "shift_minus_1_drop_first": {
            "folds": all_folds[1:],
            "metrics": _pool(pnl_by_fold, all_folds[1:]),
        },
        "shift_plus_1_drop_last": {
            "folds": all_folds[:-1],
            "metrics": _pool(pnl_by_fold, all_folds[:-1]),
        },
    }
    return result


def expanding_window_variants(pnl_by_fold: Mapping[int, np.ndarray]
                                ) -> list[dict]:
    """Return one row per expanding pool [1..k] for k = 1..N."""
    folds = sorted(pnl_by_fold.keys())
    rows = []
    for k in range(1, len(folds) + 1):
        subset = folds[:k]
        rows.append({
            "up_to_fold": subset[-1],
            "n_folds": k,
            "folds": subset,
            "metrics": _pool(pnl_by_fold, subset),
        })
    return rows


def rolling_window_variants(pnl_by_fold: Mapping[int, np.ndarray],
                              w: int = ROLLING_WINDOW) -> list[dict]:
    """Return one row per rolling window of ``w`` consecutive folds."""
    folds = sorted(pnl_by_fold.keys())
    if len(folds) < w:
        return []
    rows = []
    for i in range(0, len(folds) - w + 1):
        subset = folds[i:i + w]
        rows.append({
            "start_fold": subset[0],
            "end_fold": subset[-1],
            "folds": subset,
            "metrics": _pool(pnl_by_fold, subset),
        })
    return rows


def walkforward_report(pnl_by_fold: Mapping[int, np.ndarray]) -> dict:
    return {
        "fold_shift":       fold_shift_variants(pnl_by_fold),
        "expanding_window": expanding_window_variants(pnl_by_fold),
        "rolling_window":   rolling_window_variants(pnl_by_fold),
    }
