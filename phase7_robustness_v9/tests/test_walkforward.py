"""Walk-forward variant arithmetic."""
from __future__ import annotations

import numpy as np

from phase7_robustness_v9 import (
    expanding_window_variants, fold_shift_variants,
    rolling_window_variants, walkforward_report,
)


def test_fold_shift_drops_edges(synth_pnl_by_fold):
    r = fold_shift_variants(synth_pnl_by_fold)
    assert r["baseline_all_folds"]["folds"] == list(range(1, 9))
    assert r["shift_minus_1_drop_first"]["folds"] == list(range(2, 9))
    assert r["shift_plus_1_drop_last"]["folds"] == list(range(1, 8))


def test_expanding_window_grows_monotone(synth_pnl_by_fold):
    r = expanding_window_variants(synth_pnl_by_fold)
    ns = [row["n_folds"] for row in r]
    assert ns == list(range(1, 9))
    assert r[-1]["up_to_fold"] == 8


def test_rolling_window_length(synth_pnl_by_fold):
    r = rolling_window_variants(synth_pnl_by_fold, w=3)
    assert len(r) == 8 - 3 + 1
    assert r[0]["folds"] == [1, 2, 3]
    assert r[-1]["folds"] == [6, 7, 8]


def test_rolling_window_too_wide_returns_empty(synth_pnl_by_fold):
    assert rolling_window_variants(synth_pnl_by_fold, w=99) == []


def test_walkforward_report_has_all_sections(synth_pnl_by_fold):
    r = walkforward_report(synth_pnl_by_fold)
    assert set(r.keys()) == {"fold_shift", "expanding_window", "rolling_window"}
