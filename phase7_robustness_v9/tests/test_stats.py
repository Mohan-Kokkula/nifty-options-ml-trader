"""DM/SPA/WRC wire-up on Phase 7 aggregations."""
from __future__ import annotations

import numpy as np

from phase7_robustness_v9 import (dm_winner_vs_baseline_by_fold,
                                    spa_and_wrc_over_variants)


def test_dm_returns_pvalue_in_unit_interval(synth_pnl_by_fold, rng):
    a = synth_pnl_by_fold
    b = {f: v * 0.5 for f, v in a.items()}
    r = dm_winner_vs_baseline_by_fold(a, b)
    assert 0.0 <= r["pvalue"] <= 1.0


def test_spa_wrc_returns_expected_keys(synth_pnl_by_fold, rng):
    baseline = synth_pnl_by_fold
    variants = {
        f"v{i}": {f: v + rng.normal(0, 100, size=len(v))
                    for f, v in baseline.items()}
        for i in range(3)
    }
    r = spa_and_wrc_over_variants(variants, baseline, seed=1)
    assert "hansen_spa" in r
    assert "white_reality_check" in r


def test_spa_wrc_empty_variants_returns_note():
    baseline = {1: np.array([100.0, -50.0])}
    r = spa_and_wrc_over_variants({}, baseline, seed=1)
    assert "note" in r
