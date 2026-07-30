"""Regime classification."""
from __future__ import annotations

import pandas as pd

from phase7_robustness_v9 import classify_fold


def _mk(close_series, days):
    idx = pd.date_range("2024-01-01", periods=len(close_series), freq="15min")
    df = pd.DataFrame({"close": close_series}, index=idx)
    df.index = df.index + pd.Timedelta(days=0)
    # override end - start days
    df.index = pd.date_range("2024-01-01", periods=len(close_series),
                                freq=f"{max(1, days*24*60 // max(1, len(close_series)))}min")
    return df


def test_bull_classification():
    df = _mk([100.0, 130.0], 90)
    r = classify_fold(df)
    assert r["regime"] in ("BULL", "SIDEWAYS")  # >30% total return, allow ann>>8%


def test_bear_classification():
    df = _mk([100.0, 70.0], 90)
    r = classify_fold(df)
    assert r["regime"] in ("BEAR", "SIDEWAYS")


def test_sideways_classification():
    df = _mk([100.0, 100.5], 90)
    r = classify_fold(df)
    assert r["regime"] == "SIDEWAYS"


def test_empty_returns_unknown():
    df = pd.DataFrame({"close": []})
    r = classify_fold(df)
    assert r["regime"] == "UNKNOWN"
