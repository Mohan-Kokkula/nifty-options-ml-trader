"""Platt Scaling — one-vs-rest logistic-regression calibration.

For each of the K classes, fit a scalar logistic regression on
``(p_uncal[:, k], (y == k))``. This is the standard sklearn-style
Platt calibration extended to multiclass via one-vs-rest.

References
----------
Platt, J. (1999). "Probabilistic outputs for support vector machines
and comparisons to regularized likelihood methods." Advances in Large
Margin Classifiers.
"""
from __future__ import annotations

import numpy as np

from ._base import CalibratorAdapter, renormalise_rows, validate_labels
from ._metrics import validate_proba_input


class PlattScalingCalibrator(CalibratorAdapter):
    name = "platt"

    def fit(self, p_uncal: np.ndarray, y_true: np.ndarray,
            seed: int = 42) -> "PlattScalingCalibrator":
        from sklearn.linear_model import LogisticRegression
        p = validate_proba_input(p_uncal, "p_uncal")
        y = validate_labels(y_true, "y_true", k=p.shape[1])
        eps = 1e-6
        p = np.clip(p, eps, 1.0 - eps)

        self._per_class = []
        for k in range(p.shape[1]):
            # Feature: uncalibrated probability for class k.
            # Target: binary "is class k".
            X_k = p[:, k].reshape(-1, 1)
            y_k = (y == k).astype(int)
            if y_k.min() == y_k.max():
                # Degenerate: only one class label present. Fall back to
                # a constant prior probability.
                self._per_class.append(
                    ("prior", float(y_k.mean())))
                continue
            lr = LogisticRegression(
                C=1e5,               # very weak regularisation → Platt-style
                solver="lbfgs",
                max_iter=200,
                random_state=seed,
            )
            lr.fit(X_k, y_k)
            self._per_class.append(("lr", lr))
        self._fitted = True
        return self

    def transform(self, p_uncal: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "PlattScalingCalibrator.transform called before fit")
        p = validate_proba_input(p_uncal, "p_uncal")
        K = p.shape[1]
        if len(self._per_class) != K:
            raise ValueError(
                f"calibrator was fit on {len(self._per_class)} classes but "
                f"transform received {K}-column input")
        eps = 1e-6
        p_in = np.clip(p, eps, 1.0 - eps)
        cal = np.empty_like(p)
        for k, entry in enumerate(self._per_class):
            kind, obj = entry
            if kind == "prior":
                cal[:, k] = float(obj)
            else:
                cal[:, k] = obj.predict_proba(
                    p_in[:, k].reshape(-1, 1))[:, 1]
        return renormalise_rows(cal)
