"""MLP (sklearn) adapter.

Two departures from the tree adapters:

  * ``needs_scaling = True`` — MLPs require standardised inputs to
    converge in reasonable time.
  * ``supports_sample_weight = False`` — sklearn's ``MLPClassifier``
    does not accept a per-row weight. We reproduce the class-imbalance
    correction of the incumbent ``train_nn_fold`` via
    over-sampling of trade-class rows (labels 0 and 1) to match the
    prevalence of the skip class. This is deliberately a class-balance
    correction, not a general reweighting scheme.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import BrainAdapter, as_3class_proba


class MLPAdapter(BrainAdapter):
    name = "mlp"
    needs_scaling = True
    supports_sample_weight = False   # oversampled instead
    supports_native_early_stopping = True   # via early_stopping=True

    def default_params(self) -> dict:
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "verbose": False,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_eval: np.ndarray | None = None,   # unused; sklearn manages internal split
        y_eval: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        params: dict | None = None,
        seed: int = 42,
    ) -> Any:
        from sklearn.neural_network import MLPClassifier

        p = {**self.default_params(), **(params or {})}

        # Class-imbalance correction via oversampling, mirroring the
        # existing ``train_nn_fold`` behaviour. If sample_weight is
        # provided we still trigger oversampling (the presence of any
        # non-uniform weight signal is treated as "the caller wants
        # class balance").
        y_arr = np.asarray(y_train)
        if sample_weight is not None:
            rng = np.random.default_rng(seed)
            skip_pct = float((y_arr == 2).mean())
            trade_w = skip_pct / max(1.0 - skip_pct, 1e-9)
            trade_idx = np.where(y_arr != 2)[0]
            oversample_n = int(len(trade_idx) * (trade_w - 1))
            if oversample_n > 0 and len(trade_idx) > 0:
                extra = rng.choice(trade_idx, size=oversample_n,
                                     replace=True)
                idx = np.concatenate([np.arange(len(y_arr)), extra])
                rng.shuffle(idx)
                X_train = X_train[idx]
                y_arr = y_arr[idx]

        model = MLPClassifier(**p, random_state=seed)
        model.fit(X_train, y_arr)
        return model

    def predict_proba_3class(self, model, X):
        return as_3class_proba(model.predict_proba(X), model.classes_)

    def optuna_search_space(self, trial) -> dict:
        hs = trial.suggest_categorical(
            "hidden_layer_sizes",
            ["(64,)", "(128,)", "(128, 64)", "(256, 128)",
             "(128, 64, 32)"])
        return {
            "hidden_layer_sizes": eval(hs),   # string -> tuple; safe: literal only
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]),
        }
