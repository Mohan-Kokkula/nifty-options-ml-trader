"""Visualization tests — chart data schema (PNGs are optional)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from threshold_opt import (CandidateResult, ThresholdCandidate,
                             build_chart_data, save_chart_data,
                             write_chart_pngs)


def _synth_result(call, put, skip, edge, pf, net, n_trades):
    from threshold_opt._evaluate import _fold_metrics
    cand = ThresholdCandidate(call, put, skip, edge)
    per_fold = {}
    trade_pnl_by_fold = {}
    for i in range(1, 5):
        pnl = np.array([+2.0] * (n_trades // 4) + [-1.0] * (n_trades // 4))
        per_fold[i] = _fold_metrics(pnl)
        trade_pnl_by_fold[i] = pnl
    pooled = {"n": n_trades, "pf": pf, "net": net}
    return CandidateResult(candidate=cand, per_fold=per_fold,
                             pooled=pooled, passes_min_trades=True,
                             trade_pnl_by_fold=trade_pnl_by_fold)


def test_chart_data_schema_has_all_panels():
    results = [
        _synth_result(0.30, 0.20, 0.65, 0.05, 1.5, 100, 20),
        _synth_result(0.35, 0.25, 0.65, 0.05, 1.2, 80, 15),
        _synth_result(0.40, 0.30, 0.65, 0.05, 0.9, -20, 10),
    ]
    d = build_chart_data(results, hold_skip=0.65, hold_edge=0.05)
    for k in ("hold", "call_vs_pf", "put_vs_pf",
                "call_x_put_pf", "call_x_put_trade_count"):
        assert k in d, k


def test_chart_data_respects_hold_values():
    results = [
        _synth_result(0.30, 0.20, 0.65, 0.05, 1.5, 100, 20),
        _synth_result(0.30, 0.20, 0.70, 0.05, 1.0, 50, 10),   # skip=0.70 → filtered
    ]
    d = build_chart_data(results, hold_skip=0.65, hold_edge=0.05)
    assert d["hold"]["skip_ceil"] == 0.65
    assert d["hold"]["n_candidates_in_slice"] == 1


def test_chart_data_empty_slice():
    results = [_synth_result(0.30, 0.20, 0.65, 0.03, 1.0, 0, 1)]
    d = build_chart_data(results, hold_skip=0.65, hold_edge=0.05)
    # slice for edge=0.05 has zero rows
    assert d["hold"]["n_candidates_in_slice"] == 0
    assert d["call_vs_pf"] == {}


def test_save_chart_data_writes_json(tmp_path):
    results = [_synth_result(0.30, 0.20, 0.65, 0.05, 1.5, 100, 20)]
    d = build_chart_data(results)
    p = save_chart_data(d, tmp_path)
    assert p.exists()
    d2 = json.loads(p.read_text())
    assert d2["hold"]["skip_ceil"] == 0.65


def test_pivot_data_shape():
    results = [
        _synth_result(0.30, 0.20, 0.65, 0.05, 1.5, 100, 20),
        _synth_result(0.30, 0.25, 0.65, 0.05, 1.2, 80, 15),
        _synth_result(0.35, 0.20, 0.65, 0.05, 1.0, 40, 12),
        _synth_result(0.35, 0.25, 0.65, 0.05, 0.9, 20, 10),
    ]
    d = build_chart_data(results)
    piv = d["call_x_put_pf"]
    assert set(piv["index"]) == {0.30, 0.35}
    assert set(piv["columns"]) == {0.20, 0.25}


def test_write_pngs_skips_silently_when_matplotlib_absent(tmp_path, monkeypatch):
    """If matplotlib import fails, write_chart_pngs returns []."""
    import sys
    # Force ImportError
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    d = build_chart_data([_synth_result(0.30, 0.20, 0.65, 0.05, 1.0, 0, 1)])
    out = write_chart_pngs(d, tmp_path)
    assert isinstance(out, list)
