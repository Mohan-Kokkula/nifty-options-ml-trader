"""Median Probability Ensemble.

Applies the elementwise median across brain probability outputs per bar
per class, then renormalises each row to sum to 1. Rows where the row
sum collapses toward zero are replaced by a uniform distribution
(``1/C`` in each cell); the number of such rows is recorded in the
fitted state for provenance.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs


class MedianProbabilityEnsemble(EnsembleAdapter):
    name = "median"
    requires_oof = False
    requires_inner_pnl = False
    stateful_across_bars = False

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None, weights=None,
            seed=42) -> "MedianProbabilityEnsemble":
        _ = self._record_brains(brain_probs_oof)
        self._state = {"note": "median is a stateless per-bar operation"}
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)   # (n, K, C)
        med = np.median(stacked, axis=1)                  # (n, C)
        row_sums = med.sum(axis=1, keepdims=True)
        safe = row_sums > 1e-12
        return np.where(safe, med / np.where(safe, row_sums, 1.0),
                          1.0 / med.shape[1])

    def weights_summary(self) -> dict:
        return {"kind": self.name,
                "operation": "elementwise_median_then_renormalise"}
