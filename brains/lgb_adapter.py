"""LightGBM adapter.

Defaults mirror the XGB production baseline (same regularization,
depth, learning rate, subsampling) so cross-family comparison is not
confounded by wildly different starting points.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BrainAdapter, as_3class_proba


class LGBAdapter(BrainAdapter):
    name = "lgb"
    needs_scaling = False
    supports_sample_weight = True
    supports_native_early_stopping = True

    def default_params(self) -> dict:
        return {
            "n_estimators": 700,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.75,
            "colsample_bytree": 0.5,
            "min_child_samples": 30,
            "reg_alpha": 1.5,
            "reg_lambda": 3.0,
            "num_leaves": 28,
            "objective": "multiclass",
            "num_class": 3,
            "verbose": -1,
            "n_jobs": -1,
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
        import lightgbm as lgb
        p = {**self.default_params(), **(params or {})}
        model = lgb.LGBMClassifier(**p, random_state=seed)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        callbacks = [lgb.log_evaluation(-1)]
        if X_eval is not None and y_eval is not None:
            fit_kwargs["eval_set"] = [(X_eval, y_eval)]
            callbacks.append(lgb.early_stopping(50, verbose=False))
        fit_kwargs["callbacks"] = callbacks
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def predict_proba_3class(self, model, X):
        return as_3class_proba(model.predict_proba(X), model.classes_)

    def optuna_search_space(self, trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.3, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.5, 10.0, log=True),
        }
