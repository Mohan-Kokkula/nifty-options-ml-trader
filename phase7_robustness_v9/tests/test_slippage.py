"""Slippage stress curve."""
from __future__ import annotations

from phase7_robustness_v9 import (SLIPPAGE_MULTIPLIERS,
                                    run_slippage_curve)


def test_curve_covers_all_multipliers(synth_trades_by_fold):
    r = run_slippage_curve(synth_trades_by_fold)
    assert set(r["curve"].keys()) == set(map(float, SLIPPAGE_MULTIPLIERS))


def test_pf_monotone_non_increasing(synth_trades_by_fold):
    r = run_slippage_curve(synth_trades_by_fold)
    keys = sorted(r["curve"].keys())
    pfs = [r["curve"][k]["pooled_pf"] for k in keys]
    # non-increasing (may equal on tiny synthetic data)
    for a, b in zip(pfs, pfs[1:]):
        assert b <= a + 1e-9, f"PF non-monotone: {pfs}"


def test_per_fold_streams_length_match(synth_trades_by_fold):
    r = run_slippage_curve(synth_trades_by_fold)
    for pf_by_fold in r["per_fold_by_mult"].values():
        for f, arr in pf_by_fold.items():
            assert len(arr) == len(synth_trades_by_fold[f])
