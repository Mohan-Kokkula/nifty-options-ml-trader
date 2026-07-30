"""CalibratorAdapter — abstract contract every calibration method implements.

Every calibrator receives (n, 3) uncalibrated probabilities (rows summing
to 1, in fixed column order [CALL, PUT, SKIP]) and returns (n, 3)
calibrated probabilities in the same shape.

Multi-class calibration uses one-vs-rest with post-hoc renormalisation:
for each class k, fit a 1-D calibrator on
``(p_uncal[:, k], (y == k).astype(int))``. On transform, apply each
per-class calibrator and renormalise so rows sum to 1.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ._metrics import validate_proba_input


class CalibratorAdapter(ABC):
    """Abstract base class for all calibrators.

    Subclasses must implement :meth:`fit` and :meth:`transform`.
    Persistence uses pickle by default; override :meth:`save` /
    :meth:`load` if a calibrator needs a different serialisation.
    """

    #: unique short identifier (e.g. "platt", "isotonic", "noop")
    name: str = ""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._per_class: list[Any] = []

    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        p_uncal: np.ndarray,
        y_true: np.ndarray,
        seed: int = 42,
    ) -> "CalibratorAdapter":
        """Fit the per-class calibrators.

        Parameters
        ----------
        p_uncal : (n, 3) ndarray
            Uncalibrated 3-class probabilities. Rows should sum to 1.
        y_true : (n,) ndarray of int in {0, 1, 2}
        seed : int
            Reproducibility seed.
        """

    @abstractmethod
    def transform(self, p_uncal: np.ndarray) -> np.ndarray:
        """Return calibrated (n, 3) probabilities with rows summing to 1.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """

    def save(self, dir_path: Path,
              filename: str = "calibrator.pkl") -> Path:
        """Persist the fitted calibrator to ``dir_path/filename``."""
        if not self._fitted:
            raise RuntimeError(
                "cannot save an unfitted calibrator; call fit first")
        dir_path.mkdir(parents=True, exist_ok=True)
        out = dir_path / filename
        with open(out, "wb") as fh:
            pickle.dump(
                {"name": self.name,
                 "per_class": self._per_class},
                fh)
        return out

    def load(self, dir_path: Path,
              filename: str = "calibrator.pkl") -> "CalibratorAdapter":
        """Restore a fitted calibrator previously written by :meth:`save`."""
        with open(dir_path / filename, "rb") as fh:
            state = pickle.load(fh)
        if state.get("name") != self.name:
            raise ValueError(
                f"calibrator file was written by {state.get('name')!r} "
                f"but load() called on {self.name!r}")
        self._per_class = state["per_class"]
        self._fitted = True
        return self


# ---------------------------------------------------------------------------
def renormalise_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalise a probability array so each row sums to 1.

    Rows whose sum is < eps are replaced by a uniform distribution
    (1/3 in each cell) rather than divided by ~0.
    """
    validate_proba_input(p, "p")
    row_sums = p.sum(axis=1, keepdims=True)
    safe = row_sums >= eps
    out = np.where(safe, p / np.where(safe, row_sums, 1.0), 1.0 / p.shape[1])
    return out


def validate_labels(y: np.ndarray, name: str, k: int = 3) -> np.ndarray:
    """Validate that ``y`` is a 1-D int array of class labels in [0, k)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(
            f"{name}: expected 1-D labels, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name}: labels array is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        try:
            arr = arr.astype(int)
        except Exception as e:
            raise ValueError(
                f"{name}: cannot coerce to int ({e})")
    if arr.min() < 0 or arr.max() >= k:
        raise ValueError(
            f"{name}: labels must lie in [0, {k}); got min={arr.min()}, "
            f"max={arr.max()}")
    return arr
