"""
test_banknifty_leadlag.py — preliminary BankNifty futures→spot lead-lag test.
=============================================================================
Built 2026-06-20. Pulls 1-min BankNifty spot (NSE:BANKNIFTY) and continuous
front-month future (NSE:BANKNIFTY1!) from TradingView and tests whether
future returns lead spot returns at 1, 2, 3, 5, and 10 minute lags.

Hypothesis: corr(fut_ret[t], spot_ret[t+k]) > 0 for k>0 means futures lead.
If significant (p<0.01) at k=1-3 min: lead-lag exists at retail latency.
If only significant at k=0 (contemporaneous): no exploitable lead at 1-min.

PRELIMINARY: ~5-6 sessions of data. A pass here is a green light to deploy
the multi-session archiver; a fail closes the avenue with minimal effort.
"""
import warnings, os, sys; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
import logging
logging.getLogger("tvDatafeed.main").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "config" / "settings.env")
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed(username=os.getenv("TV_USERNAME", ""),
                password=os.getenv("TV_PASSWORD", ""))

print("Fetching BankNifty spot + futures 1-min...")
spot = tv.get_hist("BANKNIFTY", "NSE", interval=Interval.in_1_minute, n_bars=2000)
fut  = tv.get_hist("BANKNIFTY1!", "NSE", interval=Interval.in_1_minute, n_bars=2000)
print(f"  spot: {len(spot)} bars {spot.index[0]}..{spot.index[-1]}")
print(f"  fut:  {len(fut)} bars  {fut.index[0]}..{fut.index[-1]}")

# Align on common timestamps
df = pd.DataFrame({"spot": spot["close"], "fut": fut["close"]}).dropna()
df["spot_ret"] = df["spot"].pct_change()
df["fut_ret"]  = df["fut"].pct_change()
df["sess"]     = df.index.normalize()

# Restrict to NSE session minutes 09:15-15:30 to avoid overnight gaps
hm = df.index.hour * 60 + df.index.minute
df = df[(hm >= 9*60+15) & (hm <= 15*60+30)]
# Drop overnight return rows (spurious diff across day boundary)
df = df[df["sess"] == df["sess"].shift(1)]

print(f"\nAligned in-session bars: {len(df)}  across {df['sess'].nunique()} sessions")
print(f"  spot ret std: {df['spot_ret'].std()*100:.3f}%/min")
print(f"  fut ret std:  {df['fut_ret'].std()*100:.3f}%/min")

# Lagged correlations: corr(fut_ret[t], spot_ret[t+k])
# k>0 → futures lead; k<0 → spot leads; k=0 → contemporaneous
print("\n" + "="*78)
print("  LAGGED CORRELATION: corr(fut_ret[t], spot_ret[t+k])")
print("  k>0 means futures LEAD spot.  k<0 means spot leads futures.")
print("="*78)
print(f"  {'lag (min)':>10} {'pearson':>10} {'p-value':>10} {'n':>8}  interpretation")
print("  " + "-"*72)
for k in range(-3, 11):
    a = df["fut_ret"].values[:-k] if k > 0 else df["fut_ret"].values[-k:] if k < 0 else df["fut_ret"].values
    b = df["spot_ret"].values[k:] if k > 0 else df["spot_ret"].values[:k] if k < 0 else df["spot_ret"].values
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 50: continue
    r, p = stats.pearsonr(a[m], b[m])
    flag = ""
    if k > 0 and p < 0.01: flag = " *** futures LEAD (tradeable)" if abs(r) > 0.05 else " * weak lead"
    if k == 0 and p < 0.01: flag = " (contemporaneous — no edge)"
    if k < 0 and p < 0.01: flag = " (spot leads — opposite direction)"
    print(f"  {k:>10} {r:>+10.4f} {p:>10.4f} {m.sum():>8}{flag}")

print("\n" + "="*78)
print("  VERDICT")
print("="*78)
# Test specifically the 1-2-3 min lead lags
verdicts = []
for k in (1, 2, 3):
    a = df["fut_ret"].values[:-k]
    b = df["spot_ret"].values[k:]
    m = ~np.isnan(a) & ~np.isnan(b)
    r, p = stats.pearsonr(a[m], b[m])
    verdicts.append((k, r, p))

tradeable = any(p < 0.01 and r > 0.05 for k, r, p in verdicts)
print("  Lead-lag at 1-3 min cadence (the retail-tradeable window):")
for k, r, p in verdicts:
    print(f"    k={k}min: r={r:+.4f}, p={p:.4f}  "
          f"{'TRADEABLE' if (p<0.01 and r>0.05) else 'not tradeable'}")
print()
if tradeable:
    print("  [PASS] Significant lead-lag at retail cadence → deploy archiver,")
    print("         build futures-led BankNifty option signal.")
else:
    print("  [FAIL] No significant tradeable lead at 1-min cadence.")
    print("         Either lead has compressed sub-minute (HFT-arbitraged),")
    print("         or doesn't exist at this granularity. Close the avenue.")
print("="*78)
df.to_csv(ROOT / "logs/banknifty_leadlag_data.csv")
