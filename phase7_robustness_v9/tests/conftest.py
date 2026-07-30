"""Synthetic fixtures for Phase 7 unit tests — no filesystem, no models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


@pytest.fixture
def synth_pnl_by_fold(rng):
    """8 folds, each with 20-40 trades of mixed win/loss."""
    out = {}
    for f in range(1, 9):
        n = 20 + f * 3
        wins = rng.uniform(500, 3000, size=n // 2)
        losses = -rng.uniform(300, 1500, size=n - n // 2)
        arr = np.concatenate([wins, losses])
        rng.shuffle(arr)
        out[f] = arr.astype(np.float64)
    return out


@pytest.fixture
def synth_trades_df(rng):
    """Per-trade DataFrame with columns replay code expects."""
    n = 80
    prem_entry = rng.uniform(80.0, 220.0, size=n)
    prem_exit  = prem_entry + rng.normal(0.0, 30.0, size=n)
    gross_option = (prem_exit - prem_entry) * 65
    cost = np.abs(gross_option) * 0.02 + 200.0
    net_option = gross_option - cost
    return pd.DataFrame({
        "prem_entry": prem_entry,
        "prem_exit":  prem_exit,
        "gross_option": gross_option,
        "cost": cost,
        "net_option": net_option,
        "dir": ["CALL"] * n,
    })


@pytest.fixture
def synth_trades_by_fold(synth_trades_df):
    return {f: synth_trades_df.copy() for f in range(1, 9)}
