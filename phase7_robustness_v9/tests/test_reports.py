"""Report aggregation + H_rob verdict logic + JSON writers."""
from __future__ import annotations

import json

import numpy as np

from phase7_robustness_v9 import (collect_scenarios,
                                    compute_h_rob_verdict,
                                    dump_json, tornado_ranking,
                                    write_reports, render_phase7_md)


def _mk_target_result(baseline_pf=1.5, baseline_net=10000, winner_pf=2.0,
                        winner_net=25000, cv=0.3, ci_lb=0.10):
    """Minimal stub for compute_h_rob_verdict / write_reports."""
    per_fold_pf = {i: 1.5 + 0.1 * (i - 4) for i in range(1, 9)}
    per_fold_net = {i: 3000.0 for i in range(1, 9)}
    per_fold_dd = {i: 1000.0 for i in range(1, 9)}
    return {
        "regime": {"per_fold": {i: {"regime": "SIDEWAYS"}
                                  for i in range(1, 9)},
                     "counts": {"SIDEWAYS": 8}},
        "baseline_pooled_pf": baseline_pf,
        "baseline_pooled_net": baseline_net,
        "walkforward": {
            "fold_shift": {
                "baseline_all_folds": {
                    "folds": list(range(1, 9)),
                    "metrics": {"pf": winner_pf, "net": winner_net,
                                  "dd": 5000.0, "n": 60, "wr": 0.5,
                                  "sharpe": 0.5, "sortino": 0.5},
                },
                "shift_minus_1_drop_first": {
                    "folds": list(range(2, 9)),
                    "metrics": {"pf": winner_pf * 0.95, "net": winner_net * 0.9,
                                  "dd": 5000.0, "n": 55}},
                "shift_plus_1_drop_last": {
                    "folds": list(range(1, 8)),
                    "metrics": {"pf": winner_pf * 0.98, "net": winner_net * 0.95,
                                  "dd": 5000.0, "n": 58}},
            },
            "expanding_window": [
                {"up_to_fold": i, "n_folds": i,
                    "folds": list(range(1, i + 1)),
                    "metrics": {"pf": winner_pf, "net": winner_net,
                                  "dd": 4000.0, "n": 30}}
                for i in range(1, 9)
            ],
            "rolling_window": [
                {"start_fold": i, "end_fold": i + 2,
                    "folds": list(range(i, i + 3)),
                    "metrics": {"pf": winner_pf, "net": winner_net,
                                  "dd": 3000.0, "n": 20}}
                for i in range(1, 7)
            ],
        },
        "jackknife": {"per_dropped_fold": {i: {"pooled_pf": winner_pf,
                                                  "pooled_net": winner_net,
                                                  "delta_pf_vs_all": 0.01}
                                              for i in range(1, 9)},
                        "influence": {}, "single_fold_dependence": "LOW"},
        "slippage": {"curve": {m: {"pooled_pf": winner_pf - 0.1 * (m - 1),
                                     "pooled_net": winner_net,
                                     "pooled_max_dd": 5000.0,
                                     "pooled_trade_count": 60,
                                     "pooled_wr": 0.5,
                                     "pooled_sharpe": 0.5,
                                     "pooled_sortino": 0.5}
                                 for m in (1.0, 1.25, 1.5, 1.75, 2.0)},
                       "per_fold_by_mult": {}},
        "tcost": {"curve": {m: {"pooled_pf": winner_pf - 0.05 * (m - 1),
                                  "pooled_net": winner_net,
                                  "pooled_max_dd": 5000.0,
                                  "pooled_trade_count": 60,
                                  "pooled_wr": 0.5}
                              for m in (1.0, 1.25, 1.5, 1.75, 2.0)},
                    "break_even_cost_multiplier": 3.4,
                    "break_even_slippage_multiplier": 2.8},
        "exec_delay": {"curve": {d: {"pooled_pf": winner_pf - 0.05 * d,
                                        "pooled_net": winner_net,
                                        "pooled_max_dd": 5000.0,
                                        "pooled_trade_count": 60,
                                        "pooled_wr": 0.5,
                                        "per_fold_pf": {i: 1.5 for i in range(1, 9)}}
                                    for d in (0, 1, 2)},
                         "per_fold_pnl_by_delay": {}},
        "bootstrap": {
            "pf_ci": {"lower": winner_pf * 0.9, "upper": winner_pf * 1.1,
                        "point_estimate": winner_pf},
            "net_ci": {"lower": winner_net * 0.8, "upper": winner_net * 1.2,
                         "point_estimate": winner_net},
            "paired_delta_pf": {"lower": ci_lb, "upper": winner_pf - baseline_pf + 0.1,
                                  "point_estimate": winner_pf - baseline_pf,
                                  "paired": True},
        },
        "stability": {"fold_variance": {"cv_per_fold_pf": cv,
                                          "std_per_fold_pf": 0.2,
                                          "mean_per_fold_pf": winner_pf,
                                          "per_fold_pf": per_fold_pf,
                                          "per_fold_net": per_fold_net,
                                          "per_fold_max_dd": per_fold_dd,
                                          "n_folds_with_finite_pf": 8},
                        "rolling_metrics": [{"start_fold": i, "end_fold": i+2,
                                                "pf": winner_pf, "net": winner_net,
                                                "max_dd": 3000.0, "n_trades": 20}
                                              for i in range(1, 7)],
                        "unstable_folds_z_gt_2": []},
        "stats": {},
        "tornado": [{"factor": "slippage_x2.00",
                      "pooled_pf": winner_pf - 0.2,
                      "delta_pf_vs_baseline": -0.2}],
        "h_rob": None,
    }


def test_collect_scenarios_reports_multiple_families():
    tr = _mk_target_result()
    rows = collect_scenarios(tr)
    fams = {r["family"] for r in rows}
    assert {"walkforward", "jackknife", "slippage", "tcost", "delay"} <= fams


def test_h_rob_accept_when_all_criteria_pass():
    tr = _mk_target_result(baseline_pf=1.0, winner_pf=2.0, cv=0.3, ci_lb=0.5)
    v = compute_h_rob_verdict(tr, baseline_pf=1.0, baseline_net=10000)
    assert v["verdict"] == "ACCEPT"


def test_h_rob_reject_when_most_criteria_fail():
    tr = _mk_target_result(baseline_pf=2.0, winner_pf=0.5, cv=1.5, ci_lb=-0.5)
    v = compute_h_rob_verdict(tr, baseline_pf=2.0, baseline_net=10000)
    assert v["verdict"] == "REJECT"


def test_h_rob_undecided_between():
    tr = _mk_target_result(baseline_pf=1.5, winner_pf=1.7, cv=0.6, ci_lb=-0.1)
    v = compute_h_rob_verdict(tr, baseline_pf=1.5, baseline_net=10000)
    assert v["verdict"] in ("UNDECIDED", "ACCEPT")


def test_tornado_sorted_by_delta():
    tr = _mk_target_result()
    rows = tornado_ranking(tr, baseline_pf=1.5)
    deltas = [r["delta_pf_vs_baseline"] for r in rows]
    assert deltas == sorted(deltas)


def test_write_reports_creates_all_files(tmp_path):
    tr = _mk_target_result(baseline_pf=1.0, winner_pf=2.0, cv=0.3, ci_lb=0.5)
    tr["h_rob"] = compute_h_rob_verdict(tr, 1.0, 10000)
    tr["tornado"] = tornado_ranking(tr, baseline_pf=1.0)
    paths = write_reports(tmp_path, {"mean": tr},
                              {"seed": 42, "targets": ["mean"]})
    for k in ("summary", "robustness", "stress", "jackknife", "bootstrap"):
        assert paths[k].exists(), k


def test_render_md_writes_file(tmp_path):
    tr = _mk_target_result(baseline_pf=1.0, winner_pf=2.0, cv=0.3, ci_lb=0.5)
    tr["h_rob"] = compute_h_rob_verdict(tr, 1.0, 10000)
    tr["tornado"] = tornado_ranking(tr, baseline_pf=1.0)
    p = render_phase7_md(tmp_path, {"mean": tr}, {"seed": 42})
    assert p.exists()
    assert "H_rob" in p.read_text() or "Phase 7" in p.read_text()
