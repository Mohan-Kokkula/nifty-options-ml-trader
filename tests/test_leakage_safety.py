"""
test_leakage_safety.py — causality regression tests for the feature pipeline.

Principle: perturb FUTURE data; assert PRESENT feature values do not move.
These tests lock in the 2026-06-11 Phase-0 fixes — if anyone reintroduces a
same-day daily merge or a partial-HTF-bar merge, these fail.

Run:  python tests/test_leakage_safety.py   (also pytest-compatible)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    features_daily, merge_htf, merge_vix_onto_5min,
)

CLOSE_DERIVED = ["day_rsi", "day_ema9_bull", "day_ema_bull", "day_ema9_dist",
                 "day_ema_dist", "day_ema50_dist", "day_week_pos",
                 "day_atr_pct"]


def _synth_daily(n=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = 23000 + np.cumsum(rng.normal(0, 80, n))
    o = c + rng.normal(0, 30, n)
    h = np.maximum(o, c) + abs(rng.normal(20, 10, n))
    l = np.minimum(o, c) - abs(rng.normal(20, 10, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_daily_features_do_not_see_todays_close():
    df = _synth_daily()
    base = features_daily(df)
    pert = df.copy()
    pert.iloc[-1, pert.columns.get_loc("close")] += 500.0   # mutate TODAY's close
    pert.iloc[-1, pert.columns.get_loc("high")] += 500.0
    after = features_daily(pert)
    for col in CLOSE_DERIVED:
        a, b = base[col].iloc[-1], after[col].iloc[-1]
        same = (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, equal_nan=True)
        assert same, (
            f"LEAK: {col} on day D changed when day D's close was perturbed "
            f"({a} → {b}) — same-day close leakage reintroduced")
    print("  PASS daily features blind to today's close")


def test_daily_gap_features_still_use_todays_open():
    # gap features legitimately use today's OPEN — they must still respond
    df = _synth_daily()
    base = features_daily(df)
    pert = df.copy()
    pert.iloc[-1, pert.columns.get_loc("open")] += 300.0
    after = features_daily(pert)
    assert not np.isclose(base["day_gap_pct"].iloc[-1],
                          after["day_gap_pct"].iloc[-1]), \
        "day_gap_pct stopped responding to today's open — over-shifted"
    print("  PASS gap features still see today's open (not over-shifted)")


def test_merge_htf_uses_only_completed_bars():
    # 5m bars 09:15..10:30; HTF frame: one row per 15m bar START whose value
    # encodes the bar's start minute. After the fix, the 5m bar at 10:05
    # (closes 10:10) may see at most the 09:45-10:00 bar — never 10:00-10:15.
    idx5 = pd.date_range("2026-06-12 09:15", periods=16, freq="5min")
    df5 = pd.DataFrame({"close": np.arange(16.0)}, index=idx5)
    idx15 = pd.date_range("2026-06-12 09:15", periods=6, freq="15min")
    htf = pd.DataFrame({"tf15_marker": idx15.hour * 60 + idx15.minute},
                       index=idx15)
    merged = merge_htf(df5, htf, "15m")
    val = merged.loc[pd.Timestamp("2026-06-12 10:05"), "tf15_marker"]
    bar_0945 = 9 * 60 + 45
    bar_1000 = 10 * 60 + 0
    assert val != bar_1000, (
        "LEAK: 5m bar 10:05 sees the in-progress 10:00-10:15 HTF bar — "
        "partial-bar leakage reintroduced")
    assert val == bar_0945, f"unexpected HTF mapping: {val}"
    print("  PASS merge_htf serves only completed HTF bars")


def test_vix_merge_is_shifted():
    idx5 = pd.date_range("2026-06-11 09:15", periods=5, freq="5min")
    df5 = pd.DataFrame({"close": np.ones(5)}, index=idx5)
    vix = pd.DataFrame(
        {"vix_level": [10.0, 20.0]},
        index=pd.to_datetime(["2026-06-10", "2026-06-11"]))
    merged = merge_vix_onto_5min(df5, vix)
    assert merged["vix_level"].iloc[0] == 10.0, (
        "LEAK: intraday bars see TODAY's daily VIX instead of yesterday's")
    print("  PASS VIX merge uses yesterday's value")


def test_option_context_merge_is_shifted():
    from scripts.train_model_v10 import merge_option_context_onto_5min
    idx5 = pd.date_range("2026-06-11 09:15", periods=5, freq="5min")
    df5 = pd.DataFrame({"close": np.ones(5)}, index=idx5)
    ctx = pd.DataFrame(
        {"atm_iv": [0.10, 0.99]},
        index=pd.to_datetime(["2026-06-10", "2026-06-11"]))
    merged = merge_option_context_onto_5min(df5, ctx)
    assert merged["atm_iv"].iloc[0] == 0.10, (
        "LEAK: intraday bars see TODAY's option context instead of yesterday's")
    print("  PASS option-context merge uses yesterday's value")


def test_oi_archiver_writes_and_never_raises():
    import core.openalgo_client as oc
    with tempfile.TemporaryDirectory() as td:
        old_dir = oc._ARCHIVE_DIR
        old_ts = oc._last_archived_ts
        oc._ARCHIVE_DIR = td
        try:
            snap = {"chain": [{"strike": 23300,
                               "ce": {"oi": 1, "ltp": 2.0, "iv": 3.0},
                               "pe": {"oi": 4, "ltp": 5.0, "iv": 6.0}}],
                    "underlying_ltp": 23280.0, "atm_strike": 23300,
                    "pcr": 1.01}
            oc._last_archived_ts = None
            oc._archive_snapshot(snap, "16JUN26")
            # Same-second dedup guard: an immediate repeat of the identical
            # snapshot must be skipped, not appended as a second row.
            oc._archive_snapshot(snap, "16JUN26")
            files = list(Path(td).glob("oi_*.csv"))
            assert len(files) == 1
            lines = files[0].read_text().strip().splitlines()
            assert len(lines) == 2 and lines[0].startswith("snapshot_ts")
            # A genuinely new snapshot (different timestamp) must append.
            oc._last_archived_ts = None
            oc._archive_snapshot(snap, "16JUN26")
            lines = files[0].read_text().strip().splitlines()
            assert len(lines) == 3
            # garbage inputs must be swallowed silently
            oc._archive_snapshot({}, "X")
            oc._archive_snapshot(None, "X")
            oc._archive_snapshot({"chain": [{"bad": 1}]}, "X")
        finally:
            oc._ARCHIVE_DIR = old_dir
            oc._last_archived_ts = old_ts
    print("  PASS OI archiver writes correctly and never raises")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} leakage-safety tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
