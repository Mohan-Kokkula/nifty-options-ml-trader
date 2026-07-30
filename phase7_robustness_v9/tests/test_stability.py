"""Stability diagnostics."""
from __future__ import annotations

import numpy as np

from phase7_robustness_v9 import (flag_unstable_folds, fold_variance,
                                    rolling_report, stability_report)


def test_fold_variance_reports_per_fold(synth_pnl_by_fold):
    r = fold_variance(synth_pnl_by_fold)
    assert set(r["per_fold_pf"].keys()) == set(synth_pnl_by_fold.keys())


def test_cv_finite_when_positive_pf(synth_pnl_by_fold):
    r = fold_variance(synth_pnl_by_fold)
    # If mean PF is finite and non-zero, CV should be finite
    if np.isfinite(r["mean_per_fold_pf"]) and r["mean_per_fold_pf"] != 0:
        assert np.isfinite(r["cv_per_fold_pf"])


def test_rolling_window_row_count(synth_pnl_by_fold):
    r = rolling_report(synth_pnl_by_fold, w=3)
    assert len(r) == 8 - 3 + 1


def test_flag_unstable_finds_outliers():
    """The flagger consumes per-fold PF, so build folds where fold 5's PnL
    stream produces a finite but far-out PF outlier (not +inf)."""
    rng = np.random.default_rng(0)
    fold = {}
    # 8 folds with roughly PF=1.0-ish
    for i in range(1, 9):
        wins = rng.uniform(90, 110, size=15)
        losses = -rng.uniform(90, 110, size=15)
        arr = np.concatenate([wins, losses])
        rng.shuffle(arr)
        fold[i] = arr.astype(float)
    # fold 5: enormous wins, tiny losses → finite PF outlier
    wins = rng.uniform(9000, 11000, size=15)
    losses = -rng.uniform(90, 110, size=15)
    arr = np.concatenate([wins, losses])
    rng.shuffle(arr)
    fold[5] = arr.astype(float)
    flagged = flag_unstable_folds(fold, z_thresh=2.0)
    assert 5 in flagged


def test_stability_report_has_sections(synth_pnl_by_fold):
    r = stability_report(synth_pnl_by_fold)
    assert set(r.keys()) == {"fold_variance", "rolling_metrics",
                              "unstable_folds_z_gt_2"}
