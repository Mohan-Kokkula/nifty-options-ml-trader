"""Shared fixtures for stat_utils tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed generator so every test is reproducible."""
    return np.random.default_rng(20260708)


@pytest.fixture
def iid_gaussian_streams(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """8 fold streams of 200 iid N(0.05, 1) trades (mean edge = 5 bps)."""
    return {k: rng.normal(loc=0.05, scale=1.0, size=200) for k in range(1, 9)}


@pytest.fixture
def positive_edge_streams(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """8 fold streams of 200 trades with clearly positive edge (PF > 1)."""
    return {k: rng.normal(loc=0.20, scale=1.0, size=200) for k in range(1, 9)}


@pytest.fixture
def zero_edge_streams(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """8 fold streams of 200 trades with mean 0 (PF ≈ 1)."""
    return {k: rng.normal(loc=0.0, scale=1.0, size=200) for k in range(1, 9)}


@pytest.fixture
def block_correlated_series(rng: np.random.Generator) -> np.ndarray:
    """Length-1000 AR(1) series with rho=0.4 (non-trivial serial corr)."""
    n = 1000
    x = np.zeros(n)
    eps = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + eps[t]
    return x
