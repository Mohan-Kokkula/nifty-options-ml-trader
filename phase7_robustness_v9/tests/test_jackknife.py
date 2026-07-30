"""Leave-one-fold-out jackknife."""
from __future__ import annotations

import numpy as np

from phase7_robustness_v9 import leave_one_out


def test_loo_produces_one_row_per_fold(synth_pnl_by_fold):
    r = leave_one_out(synth_pnl_by_fold)
    assert set(r["per_dropped_fold"].keys()) == set(synth_pnl_by_fold.keys())


def test_loo_dependence_classification(synth_pnl_by_fold):
    r = leave_one_out(synth_pnl_by_fold)
    assert r["single_fold_dependence"] in ("LOW", "MODERATE", "HIGH", "UNKNOWN")


def test_loo_delta_pf_finite_when_pnl_present(synth_pnl_by_fold):
    r = leave_one_out(synth_pnl_by_fold)
    finite = [v["delta_pf_vs_all"]
              for v in r["per_dropped_fold"].values()
              if v["delta_pf_vs_all"] is not None
              and np.isfinite(v["delta_pf_vs_all"])]
    assert len(finite) >= 1


def test_loo_pooled_all_matches_pf_from_metrics(synth_pnl_by_fold):
    r = leave_one_out(synth_pnl_by_fold)
    from threshold_opt._evaluate import _fold_metrics
    all_pnl = np.concatenate([synth_pnl_by_fold[f]
                                for f in sorted(synth_pnl_by_fold)])
    m = _fold_metrics(all_pnl)
    assert abs(r["pooled_all_folds"]["pf"] - m["pf"]) < 1e-9
