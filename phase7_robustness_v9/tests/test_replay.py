"""Trade replay + cost-stress arithmetic identity vs. production baseline."""
from __future__ import annotations

import numpy as np
import pytest

from phase7_robustness_v9 import (
    apply_cost_stress, non_spread_cost_component,
    spread_cost_component, InvalidInputError,
)


def test_cost_stress_default_equals_original_net(synth_trades_df):
    """At slippage=1 and tcost=1 the recomputed net must match the recorded
    net_option in a formula-consistent way — i.e. reproduces gross - cost."""
    net = apply_cost_stress(synth_trades_df, slippage_mult=1.0, tcost_mult=1.0)
    # Reconstruct the same-formula cost and confirm match
    from backtest_options import round_trip_cost, LOT_SIZE
    expected_cost = np.array([
        round_trip_cost(pe, px, LOT_SIZE)
        for pe, px in zip(synth_trades_df["prem_entry"],
                             synth_trades_df["prem_exit"])
    ])
    expected_net = synth_trades_df["gross_option"].values - expected_cost
    assert np.allclose(net, expected_net, atol=1e-6)


def test_slippage_scale_only_touches_spread(synth_trades_df):
    # spread part times 2, tcost part unchanged → net drops by exactly spread
    base_net = apply_cost_stress(
        synth_trades_df, slippage_mult=1.0, tcost_mult=1.0)
    stressed = apply_cost_stress(
        synth_trades_df, slippage_mult=2.0, tcost_mult=1.0)
    from backtest_options import LOT_SIZE
    spread_only = np.array([
        spread_cost_component(pe, LOT_SIZE)
        for pe in synth_trades_df["prem_entry"]])
    diff = base_net - stressed
    assert np.allclose(diff, spread_only, atol=1e-6)


def test_tcost_scale_only_touches_non_spread(synth_trades_df):
    from backtest_options import LOT_SIZE
    base_net = apply_cost_stress(
        synth_trades_df, slippage_mult=1.0, tcost_mult=1.0)
    stressed = apply_cost_stress(
        synth_trades_df, slippage_mult=1.0, tcost_mult=2.0)
    non_spread = np.array([
        non_spread_cost_component(pe, px, LOT_SIZE)
        for pe, px in zip(synth_trades_df["prem_entry"],
                             synth_trades_df["prem_exit"])])
    diff = base_net - stressed
    assert np.allclose(diff, non_spread, atol=1e-6)


def test_apply_cost_stress_missing_columns_raises(synth_trades_df):
    bad = synth_trades_df.drop(columns=["prem_entry"])
    with pytest.raises(InvalidInputError):
        apply_cost_stress(bad)


def test_apply_cost_stress_empty_returns_empty_array():
    import pandas as pd
    empty = pd.DataFrame(columns=["prem_entry", "prem_exit", "gross_option"])
    out = apply_cost_stress(empty)
    assert isinstance(out, np.ndarray)
    assert out.size == 0
