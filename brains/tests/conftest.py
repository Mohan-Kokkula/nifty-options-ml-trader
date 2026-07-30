"""Shared fixtures for brains/tests.

Tests use tiny synthetic 3-class data with a mild signal so fits are
fast (well under 1 s per adapter for small-model overrides) and every
adapter can be exercised end-to-end without real data.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def synthetic_3class():
    """Return (X_train, y_train, X_eval, y_eval, X_test, w_train)
    with a mild but learnable signal.

    Shapes:
        X_*   (n, 8)   float64
        y_*   (n,)     int in {0, 1, 2}
        w_train (n,)   float64
    """
    rng = np.random.default_rng(2026)
    n_train, n_eval, n_test = 400, 80, 200
    n_feat = 8
    # Draw three centroids so the classes are separable but noisy.
    centres = np.array([
        [+1.0, -1.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0],
        [-1.0, +1.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    y_train = rng.integers(0, 3, size=n_train)
    y_eval = rng.integers(0, 3, size=n_eval)
    y_test = rng.integers(0, 3, size=n_test)
    X_train = rng.standard_normal((n_train, n_feat)) + centres[y_train]
    X_eval = rng.standard_normal((n_eval, n_feat)) + centres[y_eval]
    X_test = rng.standard_normal((n_test, n_feat)) + centres[y_test]
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_eval = sc.transform(X_eval)
    X_test = sc.transform(X_test)
    # Sample weights: down-weight skip (class 2)
    skip_pct = (y_train == 2).mean()
    trade_w = skip_pct / max(1 - skip_pct, 1e-9)
    w_train = np.where(y_train == 2, 1.0, trade_w).astype(float)
    return dict(X_train=X_train, y_train=y_train,
                 X_eval=X_eval, y_eval=y_eval,
                 X_test=X_test, y_test=y_test,
                 w_train=w_train)


@pytest.fixture
def small_params():
    """Tiny overrides so tree/NN fits complete in < 2 seconds each."""
    return {
        "xgb": {"n_estimators": 20, "max_depth": 3,
                "early_stopping_rounds": 5},
        "lgb": {"n_estimators": 20, "num_leaves": 8,
                "min_child_samples": 5},
        "cat": {"iterations": 20, "depth": 3,
                "early_stopping_rounds": 5},
        "mlp": {"hidden_layer_sizes": (16,), "max_iter": 30,
                "early_stopping": True, "validation_fraction": 0.1},
    }
