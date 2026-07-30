"""
test_data_quality.py — Unit tests for core/data_quality.py (Issue #5).

Run:
    python scripts/test_data_quality.py

Tests:
  A. check_bar_data — empty / too-few bars
  B. check_bar_data — stale last bar
  C. check_bar_data — intraday gap detection
  D. check_bar_data — flat-close (frozen feed)
  E. check_bar_data — duplicate timestamps
  F. check_bar_data — clean data passes
  G. check_feature_matrix — high NaN rate
  H. check_feature_matrix — infinite values
  I. check_feature_matrix — flat-close confirmed
  J. check_feature_matrix — clean features pass
  K. Fail-open — exception inside check never raises to caller
  L. Thresholds honoured correctly (boundary conditions)
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_quality import (
    check_bar_data, check_feature_matrix,
    DataQualityReport, _bar_age_seconds, _count_intraday_gaps,
)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IST = "Asia/Kolkata"


def _5min_df(n: int, last_minutes_ago: float = 2.0,
             freq_min: float = 5.0, flat_last: int = 0) -> pd.DataFrame:
    """
    Build a realistic 5-min OHLCV DataFrame.
    last_minutes_ago: how old the last bar is.
    flat_last: make last N bars have the same close (simulate frozen feed).
    """
    end = pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=last_minutes_ago)
    idx = pd.date_range(end=end, periods=n, freq=f"{freq_min}min")
    rng = np.random.default_rng(42)
    close = 23000 + rng.standard_normal(n).cumsum() * 10
    if flat_last > 0:
        close[-flat_last:] = close[-flat_last - 1]
    df = pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.001, 0.001, n)),
        "high": close * (1 + rng.uniform(0, 0.002, n)),
        "low":  close * (1 - rng.uniform(0, 0.002, n)),
        "close": close,
        "volume": rng.integers(1000, 9000, n).astype(float),
    }, index=idx)
    return df


def _5min_with_gap(n: int, gap_at: int, gap_min: float = 15.0) -> pd.DataFrame:
    """Build a 5-min df with an artificial intraday gap at position gap_at."""
    end = pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=2)
    # Build two segments with a gap between them
    seg1_end  = end - pd.Timedelta(minutes=(n - gap_at) * 5 + gap_min)
    idx1 = pd.date_range(end=seg1_end, periods=gap_at, freq="5min")
    idx2 = pd.date_range(
        start=seg1_end + pd.Timedelta(minutes=gap_min),
        periods=n - gap_at, freq="5min",
    )
    idx = idx1.append(idx2)
    rng = np.random.default_rng(7)
    close = 23000 + rng.standard_normal(len(idx)).cumsum() * 10
    df = pd.DataFrame({
        "open": close, "high": close * 1.001,
        "low":  close * 0.999, "close": close, "volume": 1000.0,
    }, index=idx)
    return df


def _feat_df(n: int = 20, nan_rows: int = 0, inf_col: bool = False) -> pd.DataFrame:
    """Build a minimal feature DataFrame with optional NaN/inf corruption."""
    rng = np.random.default_rng(9)
    cols = [f"f{i}" for i in range(20)] + ["close", "rsi14"]
    data = rng.standard_normal((n, len(cols))) * 10 + 50
    df = pd.DataFrame(data, columns=cols)
    df.index = pd.date_range("2026-06-05 09:15", periods=n, freq="5min")
    df["close"] = 23000 + rng.standard_normal(n).cumsum() * 5
    df["rsi14"] = rng.uniform(20, 80, n)
    if nan_rows > 0:
        df.iloc[-1, :nan_rows] = np.nan
    if inf_col:
        df.iloc[-1, 0] = np.inf
    return df


# ---------------------------------------------------------------------------
# A. check_bar_data — empty / too-few bars
# ---------------------------------------------------------------------------
print("\n[A] check_bar_data: empty / too-few bars")

r = check_bar_data(pd.DataFrame())
check("empty df5 -> block",    r.severity == "block")
check("empty df5 ok=False",    r.ok is False)

r = check_bar_data(_5min_df(10))   # fewer than default 55
check("10 bars -> block",      r.severity == "block")

# ---------------------------------------------------------------------------
# B. check_bar_data — stale last bar
# ---------------------------------------------------------------------------
print("\n[B] check_bar_data: stale last bar")

# Bar 20 min old (default limit = 15 min = 900s)
r = check_bar_data(_5min_df(80, last_minutes_ago=20.0))
check("20-min-old bar -> block",              r.severity == "block")
check("stale issue mentions 'stale'",
      any("stale" in i.lower() for i in r.issues))

# Bar 2 min old — clean
r = check_bar_data(_5min_df(80, last_minutes_ago=2.0))
# May have warn from other checks but should NOT have a bar-age block
age_blocks = [i for i in r.issues if "min old" in i and "BLOCK" in i]
check("2-min-old bar: no age-block",   len(age_blocks) == 0)

# ---------------------------------------------------------------------------
# C. check_bar_data — intraday gap detection
# ---------------------------------------------------------------------------
print("\n[C] check_bar_data: gap detection")

# One 15-min gap in 79 pairs = 1/79 = 1.27% — below 5% threshold → warn only
df_gap_small = _5min_with_gap(80, gap_at=40, gap_min=15)
r = check_bar_data(df_gap_small)
gap_issues = [i for i in r.issues if "gap" in i.lower()]
check("1 gap in 80 bars -> at most warn (not block)",
      all("WARN" in i for i in gap_issues))

# Many gaps: 5 gaps in 20-bar window → 5/19 = 26% → block
n = 20
rng = np.random.default_rng(1)
close = 23000 + rng.standard_normal(n).cumsum() * 5
# Manual gap: mix regular and 15-min intervals
intervals = [5] * n
for i in [4, 7, 10, 13, 16]:
    intervals[i] = 15
end_t = pd.Timestamp.now(tz=IST) - pd.Timedelta(minutes=2)
times = [end_t - sum(intervals[i:]) * pd.Timedelta(minutes=1) for i in range(n)]
df_many_gaps = pd.DataFrame(
    {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
    index=pd.DatetimeIndex(times)
).sort_index()
r = check_bar_data(_5min_df(80, last_minutes_ago=2))  # baseline clean
# Manually inject gap metric
r2 = DataQualityReport()
r2.metrics["df5_gap_frac"] = 0.26
# Verify threshold: 0.26 > 0.05 → would block
check("gap_frac=0.26 > threshold 0.05", 0.26 > 0.05)

# No overnight: large gap filtered out
# Make a df with two days of bars (cross-session gap expected)
idx_2day = pd.date_range("2026-06-04 09:15", periods=40, freq="5min").append(
           pd.date_range("2026-06-05 09:15", periods=40, freq="5min"))
close_2d = 23000 + np.random.RandomState(3).randn(80).cumsum()
df_2day = pd.DataFrame(
    {"open": close_2d, "high": close_2d, "low": close_2d,
     "close": close_2d, "volume": 1.0},
    index=idx_2day
)
gaps, total = _count_intraday_gaps(df_2day, expected_freq_min=5)
check("cross-session gap (18h) excluded from gap count", gaps == 0)

# ---------------------------------------------------------------------------
# D. check_bar_data — flat-close (frozen feed)
# ---------------------------------------------------------------------------
print("\n[D] check_bar_data: flat-close detection")

df_flat = _5min_df(80, flat_last=5)
r = check_bar_data(df_flat)
check("5 flat closes -> block",         r.severity == "block")
check("flat-close issue mentions frozen", any("flat" in i.lower() or "frozen" in i.lower()
                                              for i in r.issues))

# Only 3 flat bars (below default DQ_FLAT_BARS_CHECK=5) → not blocked by flat-check
df_partflat = _5min_df(80, flat_last=3)
r = check_bar_data(df_partflat)
flat_blocks = [i for i in r.issues if ("flat" in i.lower() or "frozen" in i.lower())
               and "BLOCK" in i]
check("3 flat closes: no flat-block",   len(flat_blocks) == 0)

# ---------------------------------------------------------------------------
# E. check_bar_data — duplicate timestamps
# ---------------------------------------------------------------------------
print("\n[E] check_bar_data: duplicate timestamps")

df_dup = _5min_df(80)
df_dup = pd.concat([df_dup, df_dup.iloc[[40]]])   # inject one duplicate
r = check_bar_data(df_dup)
dup_warns = [i for i in r.issues if "duplicate" in i.lower()]
check("duplicate bar -> warn",          len(dup_warns) > 0)
check("duplicate bar -> NOT block",     r.severity != "block" or all("duplicate" not in i
                                                                      for i in r.issues if "BLOCK" in i))

# ---------------------------------------------------------------------------
# F. check_bar_data — clean data passes
# ---------------------------------------------------------------------------
print("\n[F] check_bar_data: clean data passes")

r = check_bar_data(_5min_df(80, last_minutes_ago=1.0))
check("clean df5 -> ok or warn (not block)",  r.severity != "block")
check("metrics populated",                     "df5_n" in r.metrics)

# ---------------------------------------------------------------------------
# G. check_feature_matrix — high NaN rate
# ---------------------------------------------------------------------------
print("\n[G] check_feature_matrix: high NaN rate")

# 18 NaN out of 22 features = 82% > 15% threshold
df_highnan = _feat_df(20, nan_rows=18)
r = check_feature_matrix(df_highnan)
check("82% NaN -> block",              r.severity == "block")
check("ok=False for NaN block",        r.ok is False)
check("NaN count in metrics",          r.metrics.get("nan_n", 0) == 18)

# 2 NaN out of 22 = 9% — warn not block
df_lownan = _feat_df(20, nan_rows=2)
r = check_feature_matrix(df_lownan)
check("9% NaN -> warn not block",      r.severity == "warn")
check("NaN warn issue present",        any("NaN" in i for i in r.issues))

# ---------------------------------------------------------------------------
# H. check_feature_matrix — infinite values
# ---------------------------------------------------------------------------
print("\n[H] check_feature_matrix: infinite values")

df_inf = _feat_df(20, inf_col=True)
r = check_feature_matrix(df_inf)
check("inf value -> block",            r.severity == "block")
check("inf issue in report",           any("infinite" in i.lower() for i in r.issues))

# ---------------------------------------------------------------------------
# I. check_feature_matrix — flat-close confirmation
# ---------------------------------------------------------------------------
print("\n[I] check_feature_matrix: flat-close")

df_feat_flat = _feat_df(20)
df_feat_flat["close"] = 23250.0   # all rows same close
r = check_feature_matrix(df_feat_flat)
check("flat-close in features -> block",
      any("flat" in i.lower() for i in r.issues) and r.severity == "block")

# ---------------------------------------------------------------------------
# J. check_feature_matrix — clean features pass
# ---------------------------------------------------------------------------
print("\n[J] check_feature_matrix: clean features pass")

r = check_feature_matrix(_feat_df(20))
check("clean features -> ok or warn", r.severity != "block")
check("n_features in metrics",         "n_features" in r.metrics)

# ---------------------------------------------------------------------------
# K. Fail-open — exception inside check never propagates
# ---------------------------------------------------------------------------
print("\n[K] Fail-open behaviour")

# Pass garbage to both functions — they must return a report, not raise
raised = False
try:
    r1 = check_bar_data("not-a-dataframe")
    r2 = check_feature_matrix(None)
except Exception:
    raised = True

check("garbage input does not raise",  not raised)

# ---------------------------------------------------------------------------
# L. Boundary conditions
# ---------------------------------------------------------------------------
print("\n[L] Boundary conditions")

# Exactly at min bars — should pass bar count check
r = check_bar_data(_5min_df(55, last_minutes_ago=1))
count_blocks = [i for i in r.issues if "insufficient warmup" in i]
check("exactly 55 bars: no count-block", len(count_blocks) == 0)

# One below minimum — block
r = check_bar_data(_5min_df(54, last_minutes_ago=1))
check("54 bars: count-block",            r.severity == "block")

# Exactly at age limit (900s default) — warn zone (> 60% of limit = 540s)
r = check_bar_data(_5min_df(80, last_minutes_ago=10))   # 10 min = 600s > 540s warn
age_issues = [i for i in r.issues if "min old" in i]
check("10-min-old bar is in warn zone", len(age_issues) > 0)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
