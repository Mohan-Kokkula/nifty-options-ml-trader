"""StackingEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get


def test_stacking_meta_beats_mean_on_separable(five_brains_separable):
    p, y = five_brains_separable
    stack = get("stacking").fit(p, y, seed=1).transform(p)
    mean = get("mean").fit(p, y).transform(p)
    stack_acc = (stack.argmax(axis=1) == y).mean()
    mean_acc = (mean.argmax(axis=1) == y).mean()
    assert stack_acc >= mean_acc


def test_stacking_rows_sum_to_one(five_brains_separable):
    p, y = five_brains_separable
    out = get("stacking").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_stacking_deterministic(five_brains_separable):
    p, y = five_brains_separable
    a = get("stacking").fit(p, y, seed=13).transform(p)
    b = get("stacking").fit(p, y, seed=13).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_stacking_save_load_roundtrip(tmp_path, five_brains_separable):
    p, y = five_brains_separable
    ens = get("stacking").fit(p, y, seed=1)
    p1 = ens.transform(p)
    ens.save(tmp_path)
    ens2 = get("stacking").load(tmp_path)
    p2 = ens2.transform(p)
    np.testing.assert_allclose(p1, p2, atol=1e-9)


def test_stacking_brain_order_invariant(five_brains_separable):
    """Feeding brains in a different dict order still produces the same
    output because stacking internally sorts the brain names."""
    p, y = five_brains_separable
    forward = get("stacking").fit(p, y, seed=1).transform(p)
    # Same fit, but reverse the input dict at transform time
    reversed_dict = {k: p[k] for k in sorted(p, reverse=True)}
    ens2 = get("stacking").fit(p, y, seed=1)
    reverse_out = ens2.transform(reversed_dict)
    np.testing.assert_allclose(forward, reverse_out, atol=1e-9)


def test_stacking_works_across_K(rng):
    n = 300
    for K in (2, 3, 5):
        p = {f"b{i}": rng.dirichlet(np.ones(3), size=n) for i in range(K)}
        y = rng.integers(0, 3, size=n)
        out = get("stacking").fit(p, y, seed=1).transform(p)
        assert out.shape == (n, 3)


def test_stacking_weights_summary_has_meta_shape(five_brains_separable):
    p, y = five_brains_separable
    ws = get("stacking").fit(p, y, seed=1).weights_summary()
    assert ws["kind"] == "stacking"
    assert ws["n_meta_features"] == 5 * 3


def test_stacking_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        get("stacking").transform({"a": np.array([[0.5, 0.3, 0.2]]),
                                     "b": np.array([[0.5, 0.3, 0.2]])})
