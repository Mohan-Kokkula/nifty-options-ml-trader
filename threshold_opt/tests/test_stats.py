"""Statistical comparison tests — using synthetic P&L streams."""
from __future__ import annotations

import numpy as np
import pytest

from threshold_opt import compare_to_baseline, top_k_comparison


def test_compare_returns_expected_shape(synth_pnl_by_fold):
    winner = synth_pnl_by_fold
    baseline = {k: v * 0.5 for k, v in synth_pnl_by_fold.items()}
    r = compare_to_baseline(winner, baseline, seed=1)
    assert "paired_ci_90" in r
    assert "diebold_mariano" in r
    assert "common_folds" in r
    assert r["common_folds"] == [1, 2, 3, 4]


def test_compare_paired_ci_finite(synth_pnl_by_fold):
    winner = synth_pnl_by_fold
    baseline = {k: v * 0.8 for k, v in synth_pnl_by_fold.items()}
    r = compare_to_baseline(winner, baseline, seed=42)
    ci = r["paired_ci_90"]
    assert np.isfinite(ci.get("lower", 0)) or ci.get("lower") is None


def test_top_k_comparison_returns_spa_wrc(synth_pnl_by_fold, rng):
    baseline = synth_pnl_by_fold
    top_k = {}
    for i in range(4):
        top_k[f"cand_{i}"] = {k: v + rng.normal(0.05, 0.5, size=len(v))
                                for k, v in baseline.items()}
    r = top_k_comparison(top_k, baseline, seed=1)
    assert "hansen_spa" in r
    assert "white_reality_check" in r
    assert "dm_pvalues" in r
    assert "holm_bonferroni" in r


def test_top_k_comparison_empty_returns_note():
    r = top_k_comparison({}, {}, seed=1)
    assert "note" in r


def test_top_k_comparison_insufficient_folds():
    r = top_k_comparison(
        {"cand_a": {1: np.array([1.0])}},
        {1: np.array([0.5])},
        seed=1,
    )
    assert "insufficient" in r.get("note", "")


def test_dm_pvalues_are_in_unit_interval(synth_pnl_by_fold, rng):
    baseline = synth_pnl_by_fold
    top_k = {}
    for i in range(3):
        top_k[f"cand_{i}"] = {k: v + rng.normal(0.05, 0.5, size=len(v))
                                for k, v in baseline.items()}
    r = top_k_comparison(top_k, baseline, seed=1)
    for name, p in r.get("dm_pvalues", {}).items():
        assert 0.0 <= p <= 1.0, f"{name}: p={p}"
