"""Isotonic Regression calibration — one-vs-rest, non-parametric.

For each of the K classes, fit a monotonic non-decreasing regression
from ``p_uncal[:, k]`` to ``(y == k).astype(float)``. Predictions are
clipped to [0, 1] via ``out_of_bounds="clip"``.

Isotonic is strictly more flexible than Platt (piece-wise constant
non-decreasing function vs. sigmoid) but requires more calibration data
to avoid overfitting.

References
----------
Zadrozny, B. and Elkan, C. (2001). "Obtaining calibrated probability
estimates from decision trees and naive Bayesian classifiers." ICML 2001.
"""
from __future__ import annotations

import numpy as np

from ._base import CalibratorAdapter, renormalise_rows, validate_labels
from ._metrics import validate_proba_input


class IsotonicCalibrator(CalibratorAdapter):
    name = "isotonic"

    def fit(self, p_uncal: np.ndarray, y_true: np.ndarray,
            seed: int = 42) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        _ = seed  # isotonic is deterministic given input
        p = validate_proba_input(p_uncal, "p_uncal")
        y = validate_labels(y_true, "y_true", k=p.shape[1])

        self._per_class = []
        for k in range(p.shape[1]):
            y_k = (y == k).astype(float)
            if y_k.min() == y_k.max():
                self._per_class.append(
                    ("prior", float(y_k.mean())))
                continue
            iso = IsotonicRegression(
                y_min=0.0, y_max=1.0,
                out_of_bounds="clip",
                increasing=True,
            )
            iso.fit(p[:, k], y_k)
            self._per_class.append(("iso", iso))
        self._fitted = True
        return self

    def transform(self, p_uncal: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "IsotonicCalibrator.transform called before fit")
        p = validate_proba_input(p_uncal, "p_uncal")
        K = p.shape[1]
        if len(self._per_class) != K:
            raise ValueError(
                f"calibrator was fit on {len(self._per_class)} classes but "
                f"transform received {K}-column input")
        cal = np.empty_like(p)
        for k, entry in enumerate(self._per_class):
            kind, obj = entry
            if kind == "prior":
                cal[:, k] = float(obj)
            else:
                cal[:, k] = obj.predict(p[:, k])
        return renormalise_rows(cal)
