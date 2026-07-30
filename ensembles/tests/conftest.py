"""Shared fixtures for ensembles/tests.

All fixtures produce synthetic probabilities with arbitrary brain names
(``b0``, ``b1``, ``b2`` etc.) — tests never assume the presence of
"xgb"/"lgb"/"cat"/"mlp" or any other production brain name. This is
part of the brain-name-agnosticism contract of Phase 5.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20260710)


def _mk_prob(rng: np.random.Generator, n: int, alpha=1.0) -> np.ndarray:
    return rng.dirichlet(alpha * np.ones(3), size=n)


@pytest.fixture
def three_brains_random(rng):
    """Three synthetic brains with random Dirichlet-drawn probabilities."""
    n = 400
    p = {f"b{i}": _mk_prob(rng, n, alpha=0.5) for i in range(3)}
    y = rng.integers(0, 3, size=n)
    return p, y


@pytest.fixture
def three_brains_agreeing():
    """Three brains that all output the SAME probability every bar.

    Diversity metrics should indicate zero diversity here.
    """
    n = 100
    p_common = np.array([[0.5, 0.3, 0.2]] * n)
    p = {"b0": p_common.copy(), "b1": p_common.copy(), "b2": p_common.copy()}
    y = np.array([0] * n)
    return p, y


@pytest.fixture
def five_brains_separable(rng):
    """Five brains where one is CLEARLY better (higher accuracy).

    Used for performance_weighted / stacking tests.
    """
    n = 600
    y = rng.integers(0, 3, size=n)
    p: dict[str, np.ndarray] = {}
    for i in range(5):
        # Skill decreases as i increases; brain 0 is best.
        skill = 0.9 - 0.15 * i
        base = np.zeros((n, 3), dtype=np.float64)
        base[np.arange(n), y] = skill
        # Distribute remaining mass roughly uniformly across the other classes
        remaining = (1 - skill) / 2.0
        for c in range(3):
            base[y != c, c] = remaining
        # Add a bit of noise so probs aren't identical row-to-row
        noise = rng.uniform(-0.02, 0.02, size=base.shape)
        p_i = np.clip(base + noise, 1e-3, 1.0)
        p_i = p_i / p_i.sum(axis=1, keepdims=True)
        p[f"b{i}"] = p_i
    return p, y


@pytest.fixture
def two_brains_one_hot():
    """Two brains that emit one-hot probabilities (max entropy = 0)."""
    n = 50
    p_a = np.zeros((n, 3))
    p_a[:, 0] = 1.0
    p_b = np.zeros((n, 3))
    p_b[:, 1] = 1.0
    return {"a": p_a, "b": p_b}, np.zeros(n, dtype=int)


@pytest.fixture
def pnl_by_brain_and_fold(rng):
    """Synthetic per-inner-fold P&L series for min_variance tests.

    3 brains × 5 inner folds × ~30 trades per fold with different vols.
    """
    brains = ["b0", "b1", "b2"]
    folds = [1, 2, 3, 4, 5]
    scales = {"b0": 1.0, "b1": 1.5, "b2": 0.8}
    means = {"b0": 0.1, "b1": -0.05, "b2": 0.15}
    out: dict[str, dict[int, np.ndarray]] = {}
    for b in brains:
        out[b] = {}
        for k in folds:
            n_trades = int(rng.integers(20, 50))
            out[b][k] = rng.normal(loc=means[b], scale=scales[b],
                                     size=n_trades)
    return out
