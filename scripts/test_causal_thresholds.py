"""
test_causal_thresholds.py -- Unit tests for compute_thresholds_causal (Issue #3).

Run:
    python scripts/test_causal_thresholds.py

Tests:
  A. No future data used  -- the core property
  B. Monotone non-decrease for expanding window
  C. Session isolation    -- AM bars do not see MID/PM past returns
  D. Rolling window       -- shorter history, regime-adaptive
  E. min_periods fallback -- early bars get 0.001
  F. Positivity + finite  -- all thresholds are in (0, 1]
  G. Consistency          -- calling twice gives same result (pure function)
  H. create_labels smoke  -- full label pipeline runs without error
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model_v9 import (
    FWD_BARS, compute_thresholds, compute_thresholds_causal, create_labels,
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

def _make_df(n_days=60, seed=42):
    """Build a realistic synthetic 5-min DataFrame with market-hour coverage."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rows = []
    for d in dates:
        for h, m in [(9, 15), (9, 20), (9, 25), (9, 30), (9, 35), (9, 40),
                     (9, 45), (9, 50), (9, 55), (10, 0), (10, 5), (10, 10),
                     (10, 15), (10, 20), (10, 25), (10, 30),
                     (10, 35), (10, 40), (10, 45), (10, 50), (10, 55),
                     (11, 0), (11, 5), (11, 30), (11, 55), (12, 30),
                     (13, 5), (13, 30),
                     (13, 35), (14, 0), (14, 30), (15, 0)]:
            ts = pd.Timestamp(d.year, d.month, d.day, h, m)
            rows.append(ts)
    idx = pd.DatetimeIndex(rows)
    close = 20000 + rng.standard_normal(len(idx)).cumsum() * 10
    df = pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.001, 0.001, len(idx))),
        "high": close * (1 + rng.uniform(0.000, 0.002, len(idx))),
        "low":  close * (1 - rng.uniform(0.000, 0.002, len(idx))),
        "close": close,
        "volume": rng.integers(1000, 9000, len(idx)).astype(float),
    }, index=idx)
    return df


# ---------------------------------------------------------------------------
# A. No future data used
# ---------------------------------------------------------------------------
print("\n[A] No future data used")

df = _make_df(n_days=120)
bar_thresh = compute_thresholds_causal(df, min_periods=10)

# The threshold at bar i must equal the threshold we'd get if we truncated the
# dataset to bars 0..i. We verify this for a random sample of positions.
rng = np.random.default_rng(0)
positions = rng.integers(20, len(df) - FWD_BARS - 2, size=15)

leak_found = False
for pos in positions:
    # Compute threshold using only data up to pos (inclusive)
    df_past = df.iloc[:pos + 1]
    thresh_past = compute_thresholds_causal(df_past, min_periods=10)
    t_past = thresh_past.iloc[-1]   # threshold at the last (= pos) bar

    # Threshold at pos from the full dataset run
    t_full = bar_thresh.iloc[pos]

    if abs(t_past - t_full) > 1e-9:
        leak_found = True
        print(f"    LEAK at pos={pos}: past={t_past:.7f} full={t_full:.7f}")
        break

check("no future leak: truncated == full at every sampled position", not leak_found)

# ---------------------------------------------------------------------------
# B. Quantile value is arithmetically correct (spot-check vs manual)
# ---------------------------------------------------------------------------
print("\n[B] Quantile value correct at sampled positions")

ts = df.index.strftime("%H:%M")
am_mask = (ts >= "09:15") & (ts <= "10:30")

# The expanding 70th-percentile CAN decrease when a new low-vol bar is added —
# that is mathematically correct. We test the VALUE is right, not the direction.
fwd_full = df["close"].shift(-FWD_BARS) / df["close"] - 1
fwd_real  = fwd_full.shift(FWD_BARS)
sess_fwd_am = fwd_real[am_mask].abs()

# Pick 5 positions (skipping NaN at start) and verify manual == causal
am_indices  = np.where(am_mask)[0]
test_positions = am_indices[20:25]   # well past min_periods=10

errors = []
for pos in test_positions:
    # Manual: all past AM realized values up to and including pos
    past_am_vals = fwd_real.iloc[:pos + 1][am_mask[:pos + 1]].dropna().abs()
    if len(past_am_vals) == 0:
        continue
    manual_q = float(np.quantile(past_am_vals.values, 0.70))
    causal_q = float(bar_thresh.iloc[pos])
    if abs(manual_q - causal_q) > 1e-9:
        errors.append((pos, manual_q, causal_q))

check("causal quantile matches manual calculation at 5 sampled AM positions",
      len(errors) == 0)

# ---------------------------------------------------------------------------
# C. Regime adaptation -- thresholds reflect PAST regime, not future
# ---------------------------------------------------------------------------
print("\n[C] Regime adaptation (future high-vol does not inflate past thresholds)")

# Build a df where the first half is low-vol and second half is high-vol.
# Causal thresholds in the first half must reflect ONLY the low-vol past.
rng2 = np.random.default_rng(7)
n_days_half = 30
dates = pd.bdate_range("2020-01-02", periods=n_days_half * 2)
rows, closes = [], []
vol = 2.0   # low-vol first half
for i, d in enumerate(dates):
    if i == n_days_half:
        vol = 20.0  # regime change: 10x volatility
    for h, m in [(9, 15), (9, 30), (9, 45), (10, 0), (10, 15), (10, 30)]:
        rows.append(pd.Timestamp(d.year, d.month, d.day, h, m))
        if len(closes) == 0:
            closes.append(20000.0)
        else:
            closes.append(closes[-1] + rng2.standard_normal() * vol)

idx2 = pd.DatetimeIndex(rows)
c2   = np.array(closes)
df2  = pd.DataFrame({"open": c2, "high": c2 * 1.001, "low": c2 * 0.999,
                     "close": c2, "volume": 1000.0}, index=idx2)

t2 = compute_thresholds_causal(df2, min_periods=10)
ts2 = df2.index.strftime("%H:%M")
am2 = (ts2 >= "09:15") & (ts2 <= "10:30")
am_t2 = t2[am2].values

# Threshold at the LAST bar of the first half should be LOW-vol driven
n_first_half_am = am2[:len(am2)//2].sum()
thresh_end_low  = am_t2[n_first_half_am - 1]
# Threshold at the END of the second half should be higher (high-vol incorporated)
thresh_end_high = am_t2[-1]

check("causal threshold at end of high-vol period > end of low-vol period",
      thresh_end_high > thresh_end_low)

# ---------------------------------------------------------------------------
# D. Rolling window gives different (shorter-lookback) result
# ---------------------------------------------------------------------------
print("\n[D] Rolling window")

thresh_exp  = compute_thresholds_causal(df, min_periods=10, lookback_bars=None)
thresh_roll = compute_thresholds_causal(df, min_periods=10, lookback_bars=30)

# The two should differ (rolling forgets distant data)
n_differ = (thresh_exp.values != thresh_roll.values).sum()
check("expanding vs rolling give different results for same data",
      n_differ > 0)

# Rolling must also be non-negative
check("rolling thresholds >= 0", (thresh_roll.values >= 0).all())

# ---------------------------------------------------------------------------
# E. min_periods fallback -- early bars get 0.001 before enough samples
# ---------------------------------------------------------------------------
print("\n[E] min_periods fallback")

# With min_periods=500 and a 60-day dataset (~15 AM bars/day = 900 AM bars),
# the first ~33 trading days of AM bars should get 0.001 (fallback)
df_small = _make_df(n_days=60)
thresh_small = compute_thresholds_causal(df_small, min_periods=500)

ts_s = df_small.index.strftime("%H:%M")
am_s = (ts_s >= "09:15") & (ts_s <= "10:30")
am_vals = thresh_small[am_s].values

n_fallback = (am_vals == 0.001).sum()
check("early AM bars get fallback 0.001 when < min_periods samples",
      n_fallback > 0)

# Later AM bars (once enough data) should be data-driven (!=0.001)
check("later AM bars are data-driven (!=0.001) once min_periods met",
      (am_vals != 0.001).any())

# ---------------------------------------------------------------------------
# F. Positivity + finite
# ---------------------------------------------------------------------------
print("\n[F] Positivity + finite")

check("all thresholds > 0",          (bar_thresh.values > 0).all())
check("all thresholds finite",        np.isfinite(bar_thresh.values).all())
check("all thresholds <= 1.0",        (bar_thresh.values <= 1.0).all())

# ---------------------------------------------------------------------------
# G. Consistency (pure function, no hidden state)
# ---------------------------------------------------------------------------
print("\n[G] Consistency")

t1 = compute_thresholds_causal(df, min_periods=10)
t2 = compute_thresholds_causal(df, min_periods=10)
check("calling twice gives identical results", (t1.values == t2.values).all())

# ---------------------------------------------------------------------------
# H. create_labels smoke test (full pipeline)
# ---------------------------------------------------------------------------
print("\n[H] create_labels smoke")

from scripts.train_model_v9 import features_5min, create_labels

df_feat = features_5min(df.copy())
df_feat = create_labels(df_feat)

check("create_labels runs without error", "label" in df_feat.columns)
valid_labels = set(df_feat["label"].dropna().unique())
check("labels are only 0/1/2 (CALL/PUT/SKIP)", valid_labels.issubset({0, 1, 2}))
check("some CALL labels exist",  (df_feat["label"] == 0).any())
check("some PUT labels exist",   (df_feat["label"] == 1).any())
check("some SKIP labels exist",  (df_feat["label"] == 2).any())

# Verify causal threshold is stricter than global for early bars and adaptive later
old_am, old_mid, old_pm = compute_thresholds(df_feat)
ts_f = df_feat.index.strftime("%H:%M")
am_f = (ts_f >= "09:15") & (ts_f <= "10:30")
causal_am = compute_thresholds_causal(df_feat, min_periods=10)[am_f]
# The FINAL causal threshold should converge toward global (same data)
final_causal = causal_am.iloc[-1]
check("final causal AM threshold within 20% of global (converges)",
      abs(final_causal - old_am) / max(old_am, 1e-6) < 0.20)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
