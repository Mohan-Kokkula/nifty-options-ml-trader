"""MedianProbabilityEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get


def test_median_matches_numpy_median_after_renormalisation():
    p = {
        "b0": np.array([[0.6, 0.3, 0.1]]),
        "b1": np.array([[0.3, 0.4, 0.3]]),
        "b2": np.array([[0.2, 0.5, 0.3]]),
    }
    y = np.array([0])
    out = get("median").fit(p, y).transform(p)   # (1, 3)
    raw = np.median(np.stack([p[b] for b in sorted(p)]), axis=0)   # (1, 3)
    expected = raw / raw.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_median_rows_sum_to_one(three_brains_random):
    p, y = three_brains_random
    out = get("median").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_median_output_in_unit_interval(three_brains_random):
    p, y = three_brains_random
    out = get("median").fit(p, y).transform(p)
    assert (out >= 0).all() and (out <= 1).all()


def test_median_before_fit_raises():
    with pytest.raises(RuntimeError):
        get("median").transform({"a": np.array([[0.5, 0.3, 0.2]]),
                                    "b": np.array([[0.4, 0.3, 0.3]])})


def test_median_arbitrary_brain_names(rng):
    n = 20
    p = {"cat": rng.dirichlet(np.ones(3), size=n),
         "dog": rng.dirichlet(np.ones(3), size=n),
         "hamster": rng.dirichlet(np.ones(3), size=n)}
    y = rng.integers(0, 3, size=n)
    out = get("median").fit(p, y).transform(p)
    assert out.shape == (n, 3)


def test_median_deterministic(three_brains_random):
    p, y = three_brains_random
    a = get("median").fit(p, y).transform(p)
    b = get("median").fit(p, y).transform(p)
    np.testing.assert_array_equal(a, b)


def test_median_works_with_K_from_2_to_6(rng):
    n = 20
    for K in (2, 3, 4, 5, 6):
        p = {f"b{i}": rng.dirichlet(np.ones(3), size=n) for i in range(K)}
        y = rng.integers(0, 3, size=n)
        out = get("median").fit(p, y).transform(p)
        assert out.shape == (n, 3)


def test_median_save_load_roundtrip(tmp_path, three_brains_random):
    p, y = three_brains_random
    ens = get("median").fit(p, y)
    p1 = ens.transform(p)
    ens.save(tmp_path)
    ens2 = get("median").load(tmp_path)
    p2 = ens2.transform(p)
    np.testing.assert_allclose(p1, p2, atol=1e-9)
