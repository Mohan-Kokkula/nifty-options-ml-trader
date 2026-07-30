"""Confidence-Weighted Ensemble.

Per-bar weights are proportional to each brain's top-1 confidence
(``max_c p_brain_k(t, c)``). Brains that are more confident on a given
bar receive more weight on that bar.

Fitting records the participating brains and a ``temperature``
parameter (softmax over confidences); no OOF is required.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture


class ConfidenceWeightedEnsemble(EnsembleAdapter):
    name = "confidence"
    requires_oof = False
    requires_inner_pnl = False
    stateful_across_bars = True

    #: temperature parameter for the softmax over per-brain top-1 confidence
    temperature: float = 1.0

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None, weights=None,
            seed=42) -> "ConfidenceWeightedEnsemble":
        _ = self._record_brains(brain_probs_oof)
        self._state = {"temperature": self.temperature,
                        "brains": list(self._brains)}
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)   # (n, K, C)
        conf = stacked.max(axis=2)                       # (n, K)
        # Softmax across brains per bar
        logits = conf / max(float(self._state["temperature"]), 1e-9)
        logits = logits - logits.max(axis=1, keepdims=True)
        expo = np.exp(logits)
        w = expo / np.clip(expo.sum(axis=1, keepdims=True), 1e-12, None)
        return weighted_mixture(stacked, w)

    def weights_summary(self) -> dict:
        return {"kind": self.name,
                "operation": "softmax(top1_confidence / temperature) per bar",
                "temperature": self._state.get("temperature", self.temperature),
                "brains": list(self._brains)}
