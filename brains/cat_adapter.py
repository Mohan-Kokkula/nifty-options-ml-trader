"""CatBoost adapter.

Uses ``iterations`` in place of ``n_estimators`` and ``depth`` in place
of ``max_depth`` per CatBoost's native API. Sample-weight is passed to
``fit`` directly; ``eval_set`` triggers early stopping via
``early_stopping_rounds``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BrainAdapter, as_3class_proba


class CatAdapter(BrainAdapter):
    name = "cat"
    needs_scaling = False
    supports_sample_weight = True
    supports_native_early_stopping = True

    def default_params(self) -> dict:
        return {
            "iterations": 700,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "loss_function": "MultiClass",
            "classes_count": 3,
            "verbose": False,
            "early_stopping_rounds": 50,
        }

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
        try:
            from catboost import CatBoostClassifier
        except ImportError:  # pragma: no cover
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip",
                                    "install", "catboost"])
            from catboost import CatBoostClassifier
        p = {**self.default_params(), **(params or {})}
        model = CatBoostClassifier(**p, random_state=seed)
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if X_eval is not None and y_eval is not None:
            fit_kwargs["eval_set"] = (X_eval, y_eval)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def predict_proba_3class(self, model, X):
        return as_3class_proba(model.predict_proba(X), model.classes_)

    def optuna_search_space(self, trial) -> dict:
        return {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.10, log=True),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 0.5, 10.0, log=True),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float(
                "random_strength", 0.0, 1.0),
            "border_count": trial.suggest_int(
                "border_count", 32, 254),
        }
