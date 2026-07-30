"""Shared fixtures for calibrators/tests."""
from __future__ import annotations

import numpy as np
import pytest


def _dirichlet_probs(rng, n, alpha):
    """Draw Dirichlet-distributed probs. alpha < 1 gives peaked distributions
    (overconfident); alpha > 1 gives flat distributions (underconfident)."""
    return rng.dirichlet(alpha * np.ones(3), size=n)


@pytest.fixture
def rng():
    return np.random.default_rng(20260709)


@pytest.fixture
def overconfident_uncal(rng):
    """3-class probabilities that are systematically overconfident:
    the classifier predicts a max class with confidence ~0.9 but is
    only correct ~0.4 of the time.
    """
    n = 1000
    y = rng.integers(0, 3, size=n)
    p = _dirichlet_probs(rng, n, alpha=0.15)   # peaked
    # 40% of predictions correct: 40% we align pred with truth, else scramble
    pred = p.argmax(axis=1)
    keep = rng.random(n) < 0.4
    # For scrambled rows, rotate the argmax to a different class
    other = (pred + 1 + rng.integers(0, 2, size=n)) % 3
    new_argmax = np.where(keep, y, other)
    # Roll each row so new_argmax is the largest column
    out = np.empty_like(p)
    for i in range(n):
        # Sort descending
        order = np.argsort(-p[i])
        # Place the largest value at position new_argmax[i]
        base = np.zeros(3)
        base[new_argmax[i]] = p[i, order[0]]
        # Distribute the remaining two into the other slots
        remaining = [c for c in range(3) if c != new_argmax[i]]
        base[remaining[0]] = p[i, order[1]]
        base[remaining[1]] = p[i, order[2]]
        out[i] = base
    # Renormalise
    out = out / out.sum(axis=1, keepdims=True)
    return out, y


@pytest.fixture
def calibrated_uncal(rng):
    """3-class probabilities that are already well-calibrated.
    Draws softmax logits, samples y from those probs, so predicted
    probability matches ground truth frequency by construction.
    """
    n = 500
    logits = rng.normal(size=(n, 3))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p[i]) for i in range(n)])
    return p, y


@pytest.fixture
def tiny_probs(rng):
    """Small hand-verifiable input for structural checks."""
    p = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
        [0.9, 0.05, 0.05],
    ])
    y = np.array([0, 1, 2, 0])
    return p, y
