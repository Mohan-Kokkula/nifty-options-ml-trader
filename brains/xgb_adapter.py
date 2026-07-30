"""XGBoost adapter.

Defaults mirror ``backtest_threshold_sweep.train_fold`` verbatim so
downstream orchestration is a drop-in replacement for the existing XGB
code path. The search space matches Phase 2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BrainAdapter, as_3class_proba


class XGBAdapter(BrainAdapter):
    name = "xgb"
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
            "min_child_weight": 15,
            "gamma": 0.3,
            "reg_alpha": 1.5,
            "reg_lambda": 3.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 50,
            "verbosity": 0,
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
        import xgboost as xgb
        p = {**self.default_params(), **(params or {})}
        model = xgb.XGBClassifier(**p, random_state=seed)
        fit_kwargs = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if X_eval is not None and y_eval is not None:
            fit_kwargs["eval_set"] = [(X_eval, y_eval)]
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def predict_proba_3class(self, model, X):
        return as_3class_proba(model.predict_proba(X), model.classes_)

    def optuna_search_space(self, trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.10, log=True),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.3, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.5, 10.0, log=True),
        }
