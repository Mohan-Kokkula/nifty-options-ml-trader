"""Performance-Weighted Ensemble.

Weights derived from each brain's OOF multi-class log-loss:

    ll_k    = -mean_i log p_oof_brain_k(i, y_oof_i)
    w_k     = softmax(-ll_k / T)              # temperature T = 1.0

Lower log-loss → larger softmax → higher weight. Degenerate case
(any brain with log-loss = 0 exactly, or an all-NaN vector after
clipping) triggers an equal-weight fallback and records
``weights_fallback = "degenerate_log_loss"`` in the state.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture
from ._validation import EnsembleInputError, validate_labels


class PerformanceWeightedEnsemble(EnsembleAdapter):
    name = "performance"
    requires_oof = True
    requires_inner_pnl = False
    stateful_across_bars = False

    #: temperature parameter for the softmax over -log_loss
    temperature: float = 1.0

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None, weights=None,
            seed=42) -> "PerformanceWeightedEnsemble":
        _ = self._record_brains(brain_probs_oof)
        y = validate_labels(y_oof, f"{self.name}.fit(y_oof)")
        eps = 1e-12
        losses: dict[str, float] = {}
        for b in self._brains:
            p = np.asarray(brain_probs_oof[b], dtype=np.float64)
            if p.shape[0] != len(y):
                raise EnsembleInputError(
                    f"{self.name}: {b} oof rows={p.shape[0]} != y rows={len(y)}")
            picked = np.clip(p[np.arange(len(y)), y], eps, 1.0)
            losses[b] = float(-np.mean(np.log(picked)))

        finite = np.array([losses[b] for b in self._brains],
                            dtype=np.float64)
        fallback = False
        if not np.isfinite(finite).all():
            fallback = True
        else:
            logits = -finite / max(self.temperature, 1e-9)
            logits -= logits.max()
            expo = np.exp(logits)
            s = expo.sum()
            if s <= 0 or not np.isfinite(s):
                fallback = True
            else:
                w = expo / s
                if not np.isfinite(w).all():
                    fallback = True

        if fallback:
            w = np.full(len(self._brains), 1.0 / len(self._brains))
            self._state = {
                "weights": {b: float(x) for b, x in zip(self._brains, w)},
                "oof_log_loss": losses,
                "temperature": self.temperature,
                "weights_fallback": "degenerate_log_loss",
            }
        else:
            self._state = {
                "weights": {b: float(x) for b, x in zip(self._brains, w)},
                "oof_log_loss": losses,
                "temperature": self.temperature,
            }
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)
        w = np.array([self._state["weights"][b] for b in self._brains])
        return weighted_mixture(stacked, w)

    def weights_summary(self) -> dict:
        return {"kind": self.name,
                "weights": self._state.get("weights", {}),
                "oof_log_loss": self._state.get("oof_log_loss", {}),
                "temperature": self._state.get("temperature", self.temperature),
                "weights_fallback": self._state.get("weights_fallback")}
