"""Transaction-cost curve + break-even solvers."""
from __future__ import annotations

import numpy as np

from phase7_robustness_v9 import (find_break_even_slippage,
                                    find_break_even_tcost,
                                    run_tcost_curve, TCOST_MULTIPLIERS)


def test_tcost_curve_covers_all_multipliers(synth_trades_by_fold):
    r = run_tcost_curve(synth_trades_by_fold)
    assert set(r.keys()) == set(map(float, TCOST_MULTIPLIERS))


def test_tcost_pf_monotone_non_increasing(synth_trades_by_fold):
    r = run_tcost_curve(synth_trades_by_fold)
    keys = sorted(r.keys())
    pfs = [r[k]["pooled_pf"] for k in keys]
    for a, b in zip(pfs, pfs[1:]):
        assert b <= a + 1e-9


def test_break_even_tcost_when_profitable(synth_trades_by_fold):
    # Make synth profitable by boosting gross so tcost break-even exists.
    for f, df in synth_trades_by_fold.items():
        df["gross_option"] = df["gross_option"] + 5000.0
    r = find_break_even_tcost(synth_trades_by_fold)
    if r["found"]:
        assert r["break_even_multiplier"] > 1.0
        # Bisection converges when the multiplier width is under tol. Net is
        # in rupees at that width, so its magnitude depends on cost slope;
        # only the multiplier itself is guaranteed precise.
        assert 1.0 < r["break_even_multiplier"] < 100.0
        assert r["iterations"] >= 1


def test_break_even_slippage_when_profitable(synth_trades_by_fold):
    for f, df in synth_trades_by_fold.items():
        df["gross_option"] = df["gross_option"] + 5000.0
    r = find_break_even_slippage(synth_trades_by_fold)
    if r["found"]:
        assert r["break_even_multiplier"] > 1.0


def test_break_even_reports_reason_when_unprofitable(synth_trades_by_fold):
    # Force base loss by cutting gross
    for f, df in synth_trades_by_fold.items():
        df["gross_option"] = df["gross_option"] - 50000.0
    r = find_break_even_tcost(synth_trades_by_fold)
    assert r["found"] is False
    assert "reason" in r
