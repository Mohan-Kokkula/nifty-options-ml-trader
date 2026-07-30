"""Uncertainty-Weighted Ensemble.

Per-bar weights are inversely proportional to each brain's Shannon
entropy over the class distribution. Low-entropy (peaked) predictions
receive higher weight; near-uniform (uncertain) predictions receive
lower weight.

    H_k(t)   = - sum_c p_k(t, c) * log(p_k(t, c) + eps)
    raw_k(t) = 1 / (H_k(t) + eps)
    w_k(t)   = raw_k(t) / sum_j raw_j(t)
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture


class UncertaintyWeightedEnsemble(EnsembleAdapter):
    name = "uncertainty"
    requires_oof = False
    requires_inner_pnl = False
    stateful_across_bars = True

    #: numerical floor to avoid log(0) and division by zero for one-hot
    epsilon: float = 1e-9

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None, weights=None,
            seed=42) -> "UncertaintyWeightedEnsemble":
        _ = self._record_brains(brain_probs_oof)
        self._state = {"epsilon": self.epsilon,
                        "brains": list(self._brains)}
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)          # (n, K, C)
        eps = float(self._state.get("epsilon", self.epsilon))
        log_p = np.log(stacked + eps)
        H = -np.sum(stacked * log_p, axis=2)                     # (n, K)
        raw = 1.0 / (H + eps)
        w = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)
        return weighted_mixture(stacked, w)

    def weights_summary(self) -> dict:
        return {"kind": self.name,
                "operation": "inverse_entropy per bar, softmax-normalised",
                "epsilon": self._state.get("epsilon", self.epsilon),
                "brains": list(self._brains)}
