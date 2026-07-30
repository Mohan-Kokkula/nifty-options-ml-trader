"""Weighted Probability Ensemble — user-specified fixed weights.

Weights can be supplied as a ``Mapping[str, float]`` to ``.fit()``.
When omitted, defaults to equal weights (equivalent to
:class:`MeanProbabilityEnsemble`). Weights are renormalised to sum to
1; supplying weights for an unknown brain (or missing weight for a
known brain) raises :class:`EnsembleInputError`.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture
from ._validation import EnsembleInputError


class WeightedProbabilityEnsemble(EnsembleAdapter):
    name = "weighted"
    requires_oof = False
    requires_inner_pnl = False
    stateful_across_bars = False

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None,
            weights: Mapping[str, float] | None = None,
            seed=42) -> "WeightedProbabilityEnsemble":
        _ = self._record_brains(brain_probs_oof)
        if weights is None:
            wdict = {b: 1.0 / len(self._brains) for b in self._brains}
        else:
            wanted = set(self._brains)
            got = set(weights)
            if wanted != got:
                extra = got - wanted
                missing = wanted - got
                raise EnsembleInputError(
                    f"{self.name}.fit(weights): brain-set mismatch "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})")
            raw = np.array([float(weights[b]) for b in self._brains],
                            dtype=np.float64)
            if (raw < 0).any():
                raise EnsembleInputError(
                    f"{self.name}.fit(weights): negative weights not "
                    f"allowed, got {raw.tolist()}")
            s = raw.sum()
            if s <= 0:
                raise EnsembleInputError(
                    f"{self.name}.fit(weights): weight sum must be > 0")
            normalised = raw / s
            wdict = {b: float(w) for b, w in zip(self._brains, normalised)}
        self._state = {"weights": wdict}
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)
        w = np.array([self._state["weights"][b] for b in self._brains])
        return weighted_mixture(stacked, w)

    def weights_summary(self) -> dict:
        return {"kind": self.name, "weights": self._state.get("weights", {})}
