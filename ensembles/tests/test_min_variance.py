"""MinVarianceEnsemble tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import EnsembleInputError, get


def test_valid_qp_produces_nonnegative_weights_summing_to_one(
        three_brains_random, pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get("min_variance").fit(
        p, y, brain_trade_pnl_by_fold=pnl_by_brain_and_fold)
    ws = ens.weights_summary()
    weights = ws["weights"]
    for _, w in weights.items():
        assert w >= -1e-9
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_requires_inner_pnl(three_brains_random):
    p, y = three_brains_random
    with pytest.raises(EnsembleInputError, match="requires"):
        get("min_variance").fit(p, y)


def test_missing_pnl_for_brain_raises(three_brains_random):
    p, y = three_brains_random
    # Only supply pnl for two of the three brains
    partial = {"b0": {1: np.array([0.1, 0.2, -0.1])},
               "b1": {1: np.array([0.05, 0.1, -0.05])}}
    with pytest.raises(EnsembleInputError, match="missing pnl"):
        get("min_variance").fit(p, y, brain_trade_pnl_by_fold=partial)


def test_inconsistent_fold_ids_raise(three_brains_random):
    p, y = three_brains_random
    pnl = {
        "b0": {1: np.array([0.1, 0.2]),  2: np.array([0.1])},
        "b1": {1: np.array([0.1]),        3: np.array([0.05])},   # different fold ids
        "b2": {1: np.array([0.1]),        2: np.array([0.05])},
    }
    with pytest.raises(EnsembleInputError, match="inconsistent inner-fold"):
        get("min_variance").fit(p, y, brain_trade_pnl_by_fold=pnl)


def test_single_inner_fold_falls_back_to_equal_weights(three_brains_random):
    p, y = three_brains_random
    pnl = {b: {1: np.array([0.1, -0.05, 0.2])} for b in ("b0", "b1", "b2")}
    ens = get("min_variance").fit(p, y, brain_trade_pnl_by_fold=pnl)
    ws = ens.weights_summary()
    assert ws["weights_fallback"] == "insufficient_inner_folds"
    for _, w in ws["weights"].items():
        assert w == pytest.approx(1 / 3, abs=1e-6)


def test_transform_rows_sum_to_one(three_brains_random,
                                     pnl_by_brain_and_fold):
    p, y = three_brains_random
    out = get("min_variance").fit(
        p, y,
        brain_trade_pnl_by_fold=pnl_by_brain_and_fold).transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_shrinkage_recorded(three_brains_random, pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get("min_variance").fit(
        p, y, brain_trade_pnl_by_fold=pnl_by_brain_and_fold)
    ws = ens.weights_summary()
    assert "shrinkage" in ws
    assert 0.0 <= ws["shrinkage"] <= 1.0


def test_deterministic_result(three_brains_random, pnl_by_brain_and_fold):
    p, y = three_brains_random
    a = get("min_variance").fit(
        p, y, brain_trade_pnl_by_fold=pnl_by_brain_and_fold
    ).weights_summary()["weights"]
    b = get("min_variance").fit(
        p, y, brain_trade_pnl_by_fold=pnl_by_brain_and_fold
    ).weights_summary()["weights"]
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9)
