"""NoOp calibrator — identity mapping.

Used to represent the uncalibrated baseline in a uniform way so the
orchestrator can iterate over ``{"noop", "platt", "isotonic"}`` and
treat all three identically. Fitting stores nothing; transform is the
identity.
"""
from __future__ import annotations

import numpy as np

from ._base import CalibratorAdapter
from ._metrics import validate_proba_input


class NoOpCalibrator(CalibratorAdapter):
    name = "noop"

    def fit(self, p_uncal: np.ndarray, y_true: np.ndarray,
            seed: int = 42) -> "NoOpCalibrator":
        _ = validate_proba_input(p_uncal, "p_uncal")
        self._fitted = True
        return self

    def transform(self, p_uncal: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "NoOpCalibrator.transform called before fit")
        return validate_proba_input(p_uncal, "p_uncal").copy()
