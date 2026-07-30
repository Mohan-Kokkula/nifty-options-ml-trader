"""ConfidenceWeightedEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get


def test_confidence_rows_sum_to_one(three_brains_random):
    p, y = three_brains_random
    out = get("confidence").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_confident_brain_gets_more_weight_per_bar():
    """Two brains, one always outputs a peaked distribution, the other
    a flat distribution. The peaked one should dominate the mixture."""
    n = 30
    p_peaked = np.tile([[0.90, 0.05, 0.05]], (n, 1))
    p_flat = np.tile([[0.34, 0.33, 0.33]], (n, 1))
    p = {"a": p_peaked, "b": p_flat}
    y = np.zeros(n, dtype=int)
    out = get("confidence").fit(p, y).transform(p)
    # Ensemble output should be closer to p_peaked (peaked class 0) than the
    # unweighted mean.
    unweighted = (p_peaked + p_flat) / 2
    assert out[0, 0] > unweighted[0, 0]


def test_handles_one_hot_input():
    n = 10
    p_a = np.zeros((n, 3)); p_a[:, 0] = 1.0
    p_b = np.zeros((n, 3)); p_b[:, 1] = 1.0
    p = {"a": p_a, "b": p_b}
    y = np.zeros(n, dtype=int)
    out = get("confidence").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_deterministic(three_brains_random):
    p, y = three_brains_random
    a = get("confidence").fit(p, y).transform(p)
    b = get("confidence").fit(p, y).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_works_with_K_from_2_to_6(rng):
    n = 20
    for K in (2, 3, 4, 5, 6):
        p = {f"b{i}": rng.dirichlet(np.ones(3), size=n) for i in range(K)}
        y = rng.integers(0, 3, size=n)
        out = get("confidence").fit(p, y).transform(p)
        assert out.shape == (n, 3)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        get("confidence").transform({"a": np.array([[0.5, 0.3, 0.2]]),
                                        "b": np.array([[0.5, 0.3, 0.2]])})


def test_arbitrary_brain_names(rng):
    n = 20
    p = {"quark": rng.dirichlet(np.ones(3), size=n),
         "lepton": rng.dirichlet(np.ones(3), size=n)}
    y = rng.integers(0, 3, size=n)
    out = get("confidence").fit(p, y).transform(p)
    assert out.shape == (n, 3)
