"""
validate_fixes_pessimistic.py — execution-friction stress on the validation.

Adjustments per trade (applied to the cached trade lists):
  - slippage+spread : entry 1.5 + exit 1.5 spot-pts adverse (≈0.75 premium pt/side)
  - delayed fill    : +1.0 spot-pt adverse (signal→fill latency on moving market)
  - extra txn costs : +₹80 (₹180 total vs ₹100 modeled)
  - theta drag      : ₹200 flat (ATM weekly long option, ~2h avg hold)
  - missed fills    : 7% of winning trades removed (fast moves run away)

Outputs base vs pessimistic PF / expectancy / maxDD for the frozen test
window and full period, plus bootstrap 95% CI on PF.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data/validation_results"
DELTA, LOT = 0.50, 65
FRICTION_PTS = 4.0          # 1.5+1.5 slip/spread + 1.0 delayed fill
EXTRA_COSTS = 80.0
THETA = 200.0
MISS_WIN_FRAC = 0.07
TEST_START = pd.Timestamp("2025-07-07")
RNG = np.random.default_rng(7)


def pf_ci(pnl, n=10_000):
    pfs = np.empty(n)
    arr = pnl.values
    for k in range(n):
        s = arr[RNG.integers(0, len(arr), len(arr))]
        w = s[s > 0].sum()
        l = abs(s[s <= 0].sum())
        pfs[k] = w / max(l, 1e-9)
    return np.percentile(pfs, [2.5, 50, 97.5])


def stats(tr, label):
    pnl = tr["pnl"]
    w = pnl[pnl > 0]; l = pnl[pnl <= 0]
    daily = tr.groupby(pd.to_datetime(tr["ts"]).dt.date)["pnl"].sum()
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    pf = w.sum() / max(abs(l.sum()), 1e-9)
    lo, mid, hi = pf_ci(pnl)
    print(f"  {label:34s} n={len(tr):4d} wr={len(w)/len(tr):.1%} "
          f"PF={pf:.2f} [CI {lo:.2f}–{hi:.2f}] exp={pnl.mean():+7.0f} "
          f"maxDD={dd:>9,.0f} net={pnl.sum():>10,.0f}")
    return pf, pnl.mean(), dd


def pessimize(tr):
    out = tr.copy()
    out["pnl"] = out["pnl"] - FRICTION_PTS * DELTA * LOT - EXTRA_COSTS - THETA
    winners = out.index[out["pnl"] > 0].to_numpy()
    drop = RNG.choice(winners, size=int(len(winners) * MISS_WIN_FRAC),
                      replace=False) if len(winners) > 10 else []
    return out.drop(index=drop)


import sys
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""

for name in ("NEW", "OLD", "PRE_JUN10"):
    tr = pd.read_csv(RES / f"trades_{name}{SUFFIX}.csv", parse_dates=["ts"])
    test = tr[tr["ts"] > TEST_START]
    print(f"\n=== {name} ===")
    print(f"  trading days, full period : {tr['ts'].dt.date.nunique()} days w/ trades; "
          f"test window: {test['ts'].dt.date.nunique()} days w/ trades")
    stats(test, "TEST window — as reported")
    stats(pessimize(test), "TEST window — PESSIMISTIC")
    stats(tr, "Full 11.4y — as reported")
    stats(pessimize(tr), "Full 11.4y — PESSIMISTIC")

# processed days overall (from cache)
cache = pd.read_pickle(ROOT / "data/validation_predcache.pkl")
days = pd.Series(cache.index.date).nunique()
tdays = pd.Series(cache.index.date)[cache.index > TEST_START].nunique()
print(f"\nProcessed trading days: {days} total; {tdays} in frozen test window")
print(f"Bars: {len(cache):,}")
