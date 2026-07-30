"""BrainAdapter — abstract contract every model family implements.

Every concrete adapter (XGB, LGB, CAT, MLP, ...) subclasses BrainAdapter
and implements a small set of methods with a stable interface. The
orchestrator can then treat all brains identically.

Design principles
-----------------
* The adapter receives already-scaled numeric arrays. Scaling is the
  orchestrator's responsibility; the adapter's ``needs_scaling`` flag
  advertises whether scaling is required for correctness (True for MLP,
  False for tree-based).
* Sample weight handling is the adapter's problem. Adapters that support
  ``sample_weight`` natively pass it through. Adapters that don't (MLP)
  must oversample to reproduce the class-imbalance correction.
* Probabilities are returned as a 3-column array in the fixed order
  ``[CALL, PUT, SKIP]``. ``as_3class_proba`` normalises any classifier
  whose ``classes_`` attribute may be reordered or short.
* Persistence uses pickle (universal, sacrifices portability across
  Python versions). Every adapter writes to the same filename
  ``model.pkl`` in its fold directory.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BrainAdapter(ABC):
    """Abstract base class for all model-family adapters.

    Subclasses must set the four class attributes and implement the
    four abstract methods. ``optuna_search_space`` has a default
    (no tunables) that concrete classes typically override.
    """

    #: unique short identifier, used as directory name and registry key.
    name: str = ""
    #: True iff correctness requires standardised input features.
    needs_scaling: bool = False
    #: True iff ``fit`` accepts a ``sample_weight`` kwarg.
    supports_sample_weight: bool = True
    #: True iff ``fit`` supports early stopping on an eval set.
    supports_native_early_stopping: bool = True

    # ------------------------------------------------------------------
    @abstractmethod
    def default_params(self) -> dict:
        """Return the default hyperparameters used when HPO is off."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        params: dict | None = None,
        seed: int = 42,
    ) -> Any:
        """Fit and return a trained model instance.

        ``params`` fully overrides ``default_params`` (adapter merges).
        ``sample_weight`` may be ignored by adapters that don't support
        it — those adapters MUST reproduce equivalent behaviour via
        oversampling.
        """

    @abstractmethod
    def predict_proba_3class(
        self, model: Any, X: np.ndarray
    ) -> np.ndarray:
        """Return an ``(n, 3)`` probability array in column order
        ``[CALL=0, PUT=1, SKIP=2]``. Rows sum to 1 (up to fp)."""

    def save(self, model: Any, dir_path: Path) -> Path:
        """Persist ``model`` to ``dir_path/model.pkl`` and return the path."""
        dir_path.mkdir(parents=True, exist_ok=True)
        out = dir_path / "model.pkl"
        with open(out, "wb") as fh:
            pickle.dump(model, fh)
        return out

    def load(self, dir_path: Path) -> Any:
        """Load a model previously written by :meth:`save`."""
        with open(dir_path / "model.pkl", "rb") as fh:
            return pickle.load(fh)

    def optuna_search_space(self, trial) -> dict:
        """Return a dict of trial-sampled hyperparameters.

        Default implementation returns ``{}``, meaning HPO reduces to
        evaluating the default parameters repeatedly. Concrete adapters
        override to expose their tunable surface.
        """
        return {}


# ---------------------------------------------------------------------------
def as_3class_proba(p: np.ndarray, classes) -> np.ndarray:
    """Normalise a classifier's ``predict_proba`` output to ``(n, 3)``.

    Column order is fixed: ``[CALL=0, PUT=1, SKIP=2]``. Classes present
    in ``classes`` are copied into their slots; any class in
    ``{0, 1, 2}`` that is missing from the classifier is filled with
    zeros. Values outside ``{0, 1, 2}`` are silently discarded.

    ``classes`` may be a list, ndarray, or CatBoost-style nested list.
    """
    if p.ndim != 2:
        raise ValueError(f"predict_proba must be 2-D, got shape {p.shape}")

    # Normalise the classes container
    try:
        classes = list(classes)
    except TypeError:
        raise ValueError(f"cannot iterate classes: {classes!r}")
    if classes and hasattr(classes[0], "__iter__") and not isinstance(
            classes[0], (str, bytes)):
        # CatBoost sometimes returns nested [[0, 1, 2]]
        classes = list(classes[0])

    if p.shape[1] == 3 and [int(c) for c in classes] == [0, 1, 2]:
        return p

    out = np.zeros((p.shape[0], 3), dtype=float)
    for j, cls in enumerate(classes):
        try:
            c = int(cls)
        except (TypeError, ValueError):
            continue
        if 0 <= c <= 2 and j < p.shape[1]:
            out[:, c] = p[:, j]
    return out
