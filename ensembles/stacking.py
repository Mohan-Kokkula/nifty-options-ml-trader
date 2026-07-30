"""Stacking Ensemble — multinomial logistic-regression meta-learner.

For each fold, the meta-learner is fitted on the concatenated OOF
probability vectors of every participating brain (in sorted order), and
at transform time it consumes the concatenated outer-test probability
vectors of the same brains in the same order.

Brain order is always the sorted brain-name tuple recorded at fit time —
this is what guarantees the concatenation index for column ``k * 3 + c``
always refers to brain ``sorted_brains[k]``'s class ``c``.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter
from ._meta import get_meta
from ._validation import EnsembleInputError, validate_labels


class StackingEnsemble(EnsembleAdapter):
    name = "stacking"
    requires_oof = True
    requires_inner_pnl = False
    stateful_across_bars = False

    def _concat(self, brain_probs: Mapping[str, np.ndarray]) -> np.ndarray:
        cols = [np.asarray(brain_probs[b], dtype=np.float64)
                 for b in self._brains]
        return np.concatenate(cols, axis=1)

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold=None, weights=None,
            seed=42) -> "StackingEnsemble":
        _ = self._record_brains(brain_probs_oof)
        y = validate_labels(y_oof, f"{self.name}.fit(y_oof)")
        X = self._concat(brain_probs_oof)
        if X.shape[0] != len(y):
            raise EnsembleInputError(
                f"{self.name}: X rows={X.shape[0]} != y rows={len(y)}")
        meta = get_meta("logistic", seed=seed)
        meta.fit(X, y)
        # Store the fitted meta-model + attribute snapshot used by
        # weights_summary. classes_ is stored so we can reorder into the
        # canonical [0, 1, 2] output layout at transform time even if
        # sklearn returned a different order (which can happen when the
        # OOF pool is missing a class label).
        self._state = {
            "meta": meta,
            "classes": [int(c) for c in meta.classes_],
            "n_meta_features": int(X.shape[1]),
            "coef_shape": [int(x) for x in np.shape(meta.coef_)],
        }
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        X = self._concat(vt)
        expected = self._state["n_meta_features"]
        if X.shape[1] != expected:
            raise EnsembleInputError(
                f"{self.name}.transform: test features={X.shape[1]} != "
                f"fit features={expected}")
        p = self._state["meta"].predict_proba(X)
        classes = self._state["classes"]
        if p.shape[1] == 3 and classes == [0, 1, 2]:
            out = p
        else:
            out = np.zeros((p.shape[0], 3), dtype=np.float64)
            for j, cls in enumerate(classes):
                if 0 <= cls <= 2:
                    out[:, cls] = p[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        safe = row_sums > 1e-12
        return np.where(safe, out / np.where(safe, row_sums, 1.0),
                          1.0 / 3.0)

    def weights_summary(self) -> dict:
        meta = self._state.get("meta")
        summary: dict = {
            "kind": self.name,
            "n_meta_features": self._state.get("n_meta_features"),
            "classes": self._state.get("classes"),
            "coef_shape": self._state.get("coef_shape"),
        }
        if meta is not None and hasattr(meta, "coef_"):
            # small subset of coefficients (per-class first three) so the
            # weights_summary stays JSON-safe and human-inspectable
            try:
                coef = np.asarray(meta.coef_)
                summary["coef_first_row_head"] = [
                    float(x) for x in coef[0, :min(6, coef.shape[1])]
                ]
                summary["intercept"] = [float(x)
                                          for x in np.asarray(meta.intercept_)]
            except Exception:
                pass
        return summary
