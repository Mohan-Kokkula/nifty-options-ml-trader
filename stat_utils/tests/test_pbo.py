"""Tests for stat_utils.pbo (CSCV / Probability of Backtest Overfitting)."""
from __future__ import annotations

import numpy as np
import pytest

from stat_utils import (
    InsufficientDataError,
    InvalidInputError,
    PBOResult,
    probability_backtest_overfitting,
)


def test_returns_pbo_result(rng: np.random.Generator):
    perf = rng.normal(size=(400, 5))
    r = probability_backtest_overfitting(perf, S=4, n_splits="all", seed=0)
    assert isinstance(r, PBOResult)
    assert 0.0 <= r.pbo <= 1.0
    assert r.n_models == 5


def test_no_true_edge_gives_pbo_near_half(rng: np.random.Generator):
    """When there is no true edge, the *expected* PBO across random draws
    of the (T x K) performance matrix is 0.5. Any single draw exhibits
    Monte-Carlo variability; test the average across several draws.
    """
    pbos = []
    for trial in range(6):
        perf = rng.normal(size=(480, 20))
        r = probability_backtest_overfitting(perf, S=8, n_splits="all",
                                              seed=trial)
        pbos.append(r.pbo)
    mean_pbo = float(np.mean(pbos))
    # E[PBO] = 0.5 under no edge; averaged across 6 trials the observed
    # mean tightens toward 0.5 substantially. Allow generous slack.
    assert 0.30 <= mean_pbo <= 0.70


def test_one_dominant_model_gives_low_pbo(rng: np.random.Generator):
    T = 480
    K = 20
    perf = rng.normal(size=(T, K))
    perf[:, 0] += 1.0                          # first model dominates
    r = probability_backtest_overfitting(perf, S=8, n_splits="all", seed=0)
    assert r.pbo < 0.20


def test_dominant_model_gets_selected_most_often(rng: np.random.Generator):
    T = 480
    perf = rng.normal(size=(T, 10))
    perf[:, 3] += 1.5
    r = probability_backtest_overfitting(perf, S=8, n_splits="all", seed=0)
    assert max(r.selected_model_counts, key=r.selected_model_counts.get) == 3


def test_rejects_odd_S(rng: np.random.Generator):
    with pytest.raises(InvalidInputError):
        probability_backtest_overfitting(rng.normal(size=(100, 5)),
                                          S=7, seed=0)


def test_rejects_too_few_periods(rng: np.random.Generator):
    with pytest.raises(InsufficientDataError):
        probability_backtest_overfitting(rng.normal(size=(4, 5)),
                                          S=8, seed=0)


def test_rejects_single_model(rng: np.random.Generator):
    with pytest.raises(InsufficientDataError):
        probability_backtest_overfitting(rng.normal(size=(100, 1)),
                                          S=8, seed=0)


def test_subsample_n_splits(rng: np.random.Generator):
    perf = rng.normal(size=(200, 8))
    r = probability_backtest_overfitting(perf, S=6, n_splits=10, seed=0)
    assert r.n_splits == 10


def test_json_roundtrip(rng: np.random.Generator):
    perf = rng.normal(size=(200, 5))
    r = probability_backtest_overfitting(perf, S=4, n_splits="all", seed=0)
    import json
    d = json.loads(r.to_json())
    assert "pbo" in d and "logits" in d and "n_splits" in d
