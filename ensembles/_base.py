"""EnsembleAdapter abstract base class + shared helpers.

Every ensemble family (mean, median, weighted, performance-weighted,
min-variance, stacking, confidence-weighted, uncertainty-weighted)
subclasses ``EnsembleAdapter`` and implements a small, stable interface.
No adapter has hardcoded knowledge of specific brain names — all
methods operate over ``Mapping[str, np.ndarray]`` where the keys ARE
the brain names in use for a given fit/transform call.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import (
    EnsembleInputError,
    check_shape_consistency,
    validate_brain_probs_mapping,
    validate_labels,
)


# ---------------------------------------------------------------------------
class EnsembleAdapter(ABC):
    """Abstract base for all Phase-5 ensemble adapters.

    Subclasses must implement :meth:`fit` and :meth:`transform` and
    override the four class attributes as appropriate for their method.
    """

    #: unique short identifier used as registry key and directory name.
    name: str = ""
    #: True iff .fit() consumes out-of-fold predictions.
    requires_oof: bool = False
    #: True iff .fit() also needs per-inner-fold trade P&L
    #: (only :class:`MinVarianceEnsemble` sets this).
    requires_inner_pnl: bool = False
    #: True iff weights depend on per-bar input (confidence /
    #: uncertainty-weighted). False for fixed-weight ensembles.
    stateful_across_bars: bool = False

    def __init__(self) -> None:
        self._fitted: bool = False
        self._brains: tuple[str, ...] = ()
        self._state: dict[str, Any] = {}

    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        brain_probs_oof: Mapping[str, np.ndarray],
        y_oof: np.ndarray,
        *,
        brain_trade_pnl_by_fold: Mapping[str, Mapping[int, np.ndarray]] | None = None,
        weights: Mapping[str, float] | None = None,
        seed: int = 42,
    ) -> "EnsembleAdapter":
        """Fit the ensemble on OUT-OF-FOLD data.

        Concrete subclasses may ignore parameters they don't use, but
        the signature is uniform so the orchestrator can call every
        adapter identically. Adapters must set ``self._brains`` to the
        sorted tuple of brain names they were fit on.
        """

    @abstractmethod
    def transform(
        self,
        brain_probs_test: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Return the ensemble's (n, 3) probability output.

        Rows must sum to 1. The mapping's keys are the brain names at
        test time; if a brain present at fit time is missing here (or
        vice versa), the adapter must raise
        :class:`EnsembleInputError`.
        """

    # ------------------------------------------------------------------
    def weights_summary(self) -> dict[str, Any]:
        """Return a JSON-safe dict describing the fitted weights.

        Default implementation returns ``{"kind": self.name}``; subclasses
        override with structured content.
        """
        return {"kind": self.name}

    def save(self, dir_path: Path,
              filename: str = "ensemble.pkl") -> Path:
        """Persist the fitted adapter via pickle."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}.save() called before fit()")
        dir_path.mkdir(parents=True, exist_ok=True)
        out = dir_path / filename
        with open(out, "wb") as fh:
            pickle.dump({"name": self.name,
                          "brains": self._brains,
                          "state": self._state}, fh)
        return out

    def load(self, dir_path: Path,
              filename: str = "ensemble.pkl") -> "EnsembleAdapter":
        """Restore a previously saved adapter."""
        with open(dir_path / filename, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("name") != self.name:
            raise ValueError(
                f"ensemble file was written by {payload.get('name')!r} "
                f"but load() called on {self.name!r}")
        self._brains = tuple(payload["brains"])
        self._state = payload["state"]
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Helpers used by every subclass ------------------------------------
    def _record_brains(self, brain_probs: Mapping[str, np.ndarray]) -> tuple[str, ...]:
        """Validate the mapping and freeze the sorted brain-name tuple."""
        validated = validate_brain_probs_mapping(brain_probs,
                                                    f"{self.name}.fit(brain_probs)")
        self._brains = tuple(sorted(validated))
        check_shape_consistency(validated)
        return self._brains

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}.transform() called before fit()")

    def _check_transform_brains(
        self, brain_probs_test: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Validate transform-time mapping and ensure it matches fit-time brains."""
        vt = validate_brain_probs_mapping(brain_probs_test,
                                            f"{self.name}.transform(brain_probs)")
        got = tuple(sorted(vt))
        if got != self._brains:
            raise EnsembleInputError(
                f"{self.name}: transform brains {got} do not match fit brains "
                f"{self._brains}")
        return vt


# ---------------------------------------------------------------------------
def stack_brain_probs(brain_probs: Mapping[str, np.ndarray],
                       brain_order: tuple[str, ...]) -> np.ndarray:
    """Return a ``(n, K, C)`` tensor from a brain-keyed probability mapping.

    ``brain_order`` fixes the axis-1 order so downstream weight vectors
    match brain identities regardless of dict iteration order.
    """
    arrays = [np.asarray(brain_probs[b], dtype=np.float64)
              for b in brain_order]
    return np.stack(arrays, axis=1)   # (n, K, C)


def weighted_mixture(
    stacked: np.ndarray,        # (n, K, C)
    weights: np.ndarray,        # (K,) or (n, K)
) -> np.ndarray:
    """Compute ``sum_k w_k * p_k`` producing an ``(n, C)`` matrix.

    Accepts either constant weights (shape ``(K,)``) or per-bar weights
    (shape ``(n, K)``). Rows are renormalised to sum to 1.
    """
    if stacked.ndim != 3:
        raise EnsembleInputError(
            f"stacked probs must be 3-D (n, K, C), got {stacked.shape}")
    n, k, c = stacked.shape
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        if w.shape[0] != k:
            raise EnsembleInputError(
                f"weight vector length {w.shape[0]} != K={k}")
        mixed = np.einsum("nkc,k->nc", stacked, w)
    elif w.ndim == 2:
        if w.shape != (n, k):
            raise EnsembleInputError(
                f"weight matrix shape {w.shape} != (n={n}, K={k})")
        mixed = np.einsum("nkc,nk->nc", stacked, w)
    else:
        raise EnsembleInputError(
            f"weights must be 1-D or 2-D, got shape {w.shape}")
    row_sums = mixed.sum(axis=1, keepdims=True)
    safe = row_sums > 1e-12
    mixed = np.where(safe, mixed / np.where(safe, row_sums, 1.0),
                      1.0 / c)
    return mixed
