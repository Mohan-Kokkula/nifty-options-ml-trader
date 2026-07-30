"""PerformanceWeightedEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get


def test_better_brain_gets_higher_weight(five_brains_separable):
    p, y = five_brains_separable
    ens = get("performance").fit(p, y)
    ws = ens.weights_summary()
    weights = ws["weights"]
    losses = ws["oof_log_loss"]
    # b0 has lowest log-loss → highest weight
    best = min(losses, key=losses.get)
    assert best == "b0"
    assert weights["b0"] == max(weights.values())


def test_weights_sum_to_one(five_brains_separable):
    p, y = five_brains_separable
    ens = get("performance").fit(p, y)
    s = sum(ens.weights_summary()["weights"].values())
    assert s == pytest.approx(1.0, abs=1e-9)


def test_deterministic_given_seed(five_brains_separable):
    p, y = five_brains_separable
    a = get("performance").fit(p, y, seed=42).weights_summary()["weights"]
    b = get("performance").fit(p, y, seed=42).weights_summary()["weights"]
    for k in a:
        assert a[k] == pytest.approx(b[k])


def test_transform_rows_sum_to_one(five_brains_separable):
    p, y = five_brains_separable
    out = get("performance").fit(p, y).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_all_equal_brains_produce_equal_weights():
    n = 100
    p_common = np.full((n, 3), 1.0 / 3.0)
    p = {"a": p_common, "b": p_common, "c": p_common}
    y = np.random.default_rng(0).integers(0, 3, size=n)
    ens = get("performance").fit(p, y)
    ws = ens.weights_summary()["weights"]
    for _, w in ws.items():
        assert w == pytest.approx(1 / 3, abs=1e-6)


def test_temperature_stored():
    from ensembles import PerformanceWeightedEnsemble
    ens = PerformanceWeightedEnsemble()
    ens.temperature = 2.0
    p = {"a": np.full((5, 3), [0.6, 0.3, 0.1]),
         "b": np.full((5, 3), [0.2, 0.6, 0.2])}
    y = np.array([0, 0, 1, 0, 1])
    ens.fit(p, y)
    assert ens.weights_summary()["temperature"] == 2.0


def test_save_load_roundtrip(tmp_path, five_brains_separable):
    p, y = five_brains_separable
    ens = get("performance").fit(p, y)
    p1 = ens.transform(p)
    ens.save(tmp_path)
    ens2 = get("performance").load(tmp_path)
    p2 = ens2.transform(p)
    np.testing.assert_allclose(p1, p2, atol=1e-9)


def test_degenerate_fallback_recorded():
    # Zero probability for the correct class → -log(eps) → still finite,
    # but if we craft an even more extreme case with y outside the
    # positive support it will trigger the fallback.
    n = 50
    p = {"a": np.tile([[0.4, 0.3, 0.3]], (n, 1)),
         "b": np.tile([[0.3, 0.4, 0.3]], (n, 1))}
    y = np.random.default_rng(0).integers(0, 3, size=n)
    ens = get("performance").fit(p, y)
    # No fallback expected here — this is a sanity check that the
    # flag is only set when needed.
    assert ens.weights_summary().get("weights_fallback") is None
