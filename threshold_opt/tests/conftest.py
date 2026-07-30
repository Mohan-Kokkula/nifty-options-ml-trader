"""Shared synthetic fixtures for threshold_opt tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20260712)


@pytest.fixture
def synth_probs(rng):
    """(n=200, 3) probabilities with a mild CALL/PUT signal + noise."""
    n = 200
    # Truth: 30% CALL, 30% PUT, 40% SKIP
    y = np.zeros(n, dtype=int)
    y[:60] = 0
    y[60:120] = 1
    y[120:] = 2
    rng.shuffle(y)
    p = rng.dirichlet(np.ones(3), size=n)
    # Bias toward truth to make it interesting
    for i in range(n):
        p[i, y[i]] += 0.4
    p = p / p.sum(axis=1, keepdims=True)
    return p, y


@pytest.fixture
def synth_hand_probs():
    """Hand-verifiable 5-row probability set for signal checks."""
    return np.array([
        [0.60, 0.20, 0.20],   # bar 0: strong CALL
        [0.10, 0.70, 0.20],   # bar 1: strong PUT
        [0.30, 0.30, 0.40],   # bar 2: mixed → SKIP
        [0.40, 0.25, 0.35],   # bar 3: CALL by production defaults
        [0.20, 0.30, 0.50],   # bar 4: PUT by production defaults
    ])


@pytest.fixture
def synth_pnl_by_fold(rng):
    """Per-fold P&L for 4 synthetic folds — used for stats tests."""
    return {
        1: rng.normal(loc=+0.1, scale=1.0, size=25),
        2: rng.normal(loc=+0.2, scale=1.0, size=30),
        3: rng.normal(loc=-0.1, scale=1.0, size=20),
        4: rng.normal(loc=+0.15, scale=1.0, size=28),
    }
