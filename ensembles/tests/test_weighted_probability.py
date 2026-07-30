"""WeightedProbabilityEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import EnsembleInputError, get


def test_default_weights_are_equal(three_brains_random):
    p, y = three_brains_random
    ens = get("weighted").fit(p, y)
    ws = ens.weights_summary()["weights"]
    for _, w in ws.items():
        assert w == pytest.approx(1 / 3)


def test_user_weights_honoured():
    n = 10
    p = {"a": np.full((n, 3), [0.1, 0.1, 0.8]),
         "b": np.full((n, 3), [0.7, 0.2, 0.1])}
    y = np.zeros(n, dtype=int)
    ens = get("weighted").fit(p, y, weights={"a": 3.0, "b": 1.0})
    ws = ens.weights_summary()["weights"]
    assert ws["a"] == pytest.approx(0.75)
    assert ws["b"] == pytest.approx(0.25)
    out = ens.transform(p)
    expected = 0.75 * np.array([0.1, 0.1, 0.8]) + 0.25 * np.array([0.7, 0.2, 0.1])
    np.testing.assert_allclose(out[0], expected / expected.sum(), atol=1e-9)


def test_weights_renormalise_to_one():
    n = 5
    p = {"a": np.full((n, 3), [0.5, 0.3, 0.2]),
         "b": np.full((n, 3), [0.4, 0.4, 0.2])}
    y = np.zeros(n, dtype=int)
    ens = get("weighted").fit(p, y, weights={"a": 4.0, "b": 4.0})
    ws = ens.weights_summary()["weights"]
    assert ws["a"] == pytest.approx(0.5)
    assert ws["b"] == pytest.approx(0.5)


def test_missing_weight_raises(three_brains_random):
    p, y = three_brains_random
    with pytest.raises(EnsembleInputError, match="brain-set mismatch"):
        get("weighted").fit(p, y, weights={"b0": 1.0, "b1": 1.0})   # missing b2


def test_unknown_weight_key_raises(three_brains_random):
    p, y = three_brains_random
    with pytest.raises(EnsembleInputError, match="brain-set mismatch"):
        get("weighted").fit(p, y,
                              weights={"b0": 1.0, "b1": 1.0, "b2": 1.0,
                                       "phantom_brain": 5.0})


def test_negative_weights_rejected():
    n = 5
    p = {"a": np.full((n, 3), [0.5, 0.3, 0.2]),
         "b": np.full((n, 3), [0.4, 0.4, 0.2])}
    y = np.zeros(n, dtype=int)
    with pytest.raises(EnsembleInputError, match="negative"):
        get("weighted").fit(p, y, weights={"a": 1.0, "b": -0.5})


def test_zero_weights_rejected():
    n = 5
    p = {"a": np.full((n, 3), [0.5, 0.3, 0.2]),
         "b": np.full((n, 3), [0.4, 0.4, 0.2])}
    y = np.zeros(n, dtype=int)
    with pytest.raises(EnsembleInputError, match="> 0"):
        get("weighted").fit(p, y, weights={"a": 0.0, "b": 0.0})


def test_transform_rows_sum_to_one(three_brains_random):
    p, y = three_brains_random
    out = get("weighted").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
