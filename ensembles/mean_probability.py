"""Mean Probability Ensemble — equal-weight arithmetic mean of brain probs."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture


class MeanProbabilityEnsemble(EnsembleAdapter):
    name = "mean"
    requires_oof = False
    requires_inner_pnl = False
    stateful_across_bars = False

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None, weights=None,
            seed=42) -> "MeanProbabilityEnsemble":
        _ = self._record_brains(brain_probs_oof)
        self._state = {"weights": {b: 1.0 / len(self._brains)
                                     for b in self._brains}}
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)
        K = len(self._brains)
        w = np.full(K, 1.0 / K)
        return weighted_mixture(stacked, w)

    def weights_summary(self) -> dict:
        return {"kind": self.name, "weights": self._state.get("weights", {})}
