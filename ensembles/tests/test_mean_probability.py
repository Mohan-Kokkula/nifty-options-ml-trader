"""MeanProbabilityEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get


def test_mean_output_is_arithmetic_mean(three_brains_random):
    p, y = three_brains_random
    ens = get("mean").fit(p, y)
    out = ens.transform(p)
    expected = np.mean(np.stack([p[b] for b in sorted(p)], axis=0), axis=0)
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_mean_rows_sum_to_one(three_brains_random):
    p, y = three_brains_random
    out = get("mean").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_mean_weights_summary_equal(three_brains_random):
    p, y = three_brains_random
    ens = get("mean").fit(p, y)
    ws = ens.weights_summary()
    assert ws["kind"] == "mean"
    for b, w in ws["weights"].items():
        assert w == pytest.approx(1.0 / 3.0)


def test_mean_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        get("mean").transform({"a": np.array([[0.5, 0.3, 0.2]]),
                                "b": np.array([[0.4, 0.3, 0.3]])})


def test_mean_arbitrary_brain_names():
    n = 20
    p = {"zebra": np.full((n, 3), [0.5, 0.3, 0.2]),
         "yak":    np.full((n, 3), [0.1, 0.6, 0.3])}
    y = np.zeros(n, dtype=int)
    out = get("mean").fit(p, y).transform(p)
    np.testing.assert_allclose(out[0], [0.3, 0.45, 0.25], atol=1e-9)


def test_mean_works_with_2_to_6_brains(rng):
    for K in (2, 3, 4, 5, 6):
        n = 30
        p = {f"b{i}": rng.dirichlet(np.ones(3), size=n) for i in range(K)}
        y = rng.integers(0, 3, size=n)
        out = get("mean").fit(p, y).transform(p)
        assert out.shape == (n, 3)
