"""UncertaintyWeightedEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get


def test_uncertainty_rows_sum_to_one(three_brains_random):
    p, y = three_brains_random
    out = get("uncertainty").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_low_entropy_brain_gets_higher_weight():
    n = 30
    p_peaked = np.tile([[0.90, 0.05, 0.05]], (n, 1))      # H ≈ 0.39
    p_flat   = np.tile([[0.34, 0.33, 0.33]], (n, 1))      # H ≈ 1.10 (max)
    p = {"a": p_peaked, "b": p_flat}
    y = np.zeros(n, dtype=int)
    out = get("uncertainty").fit(p, y).transform(p)
    # Same argument as confidence: peaked brain should dominate.
    unweighted = (p_peaked + p_flat) / 2
    assert out[0, 0] > unweighted[0, 0]


def test_handles_one_hot_without_div_by_zero(two_brains_one_hot):
    p, y = two_brains_one_hot
    out = get("uncertainty").fit(p, y).transform(p)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_deterministic(three_brains_random):
    p, y = three_brains_random
    a = get("uncertainty").fit(p, y).transform(p)
    b = get("uncertainty").fit(p, y).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_works_with_K_from_2_to_6(rng):
    n = 20
    for K in (2, 3, 4, 5, 6):
        p = {f"b{i}": rng.dirichlet(np.ones(3), size=n) for i in range(K)}
        y = rng.integers(0, 3, size=n)
        out = get("uncertainty").fit(p, y).transform(p)
        assert out.shape == (n, 3)


def test_epsilon_stored():
    from ensembles import UncertaintyWeightedEnsemble
    ens = UncertaintyWeightedEnsemble()
    ens.epsilon = 1e-6
    p = {"a": np.full((5, 3), [0.5, 0.3, 0.2]),
         "b": np.full((5, 3), [0.4, 0.4, 0.2])}
    y = np.zeros(5, dtype=int)
    ens.fit(p, y)
    assert ens.weights_summary()["epsilon"] == 1e-6


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        get("uncertainty").transform({"a": np.array([[0.5, 0.3, 0.2]]),
                                        "b": np.array([[0.5, 0.3, 0.2]])})
