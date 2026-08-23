"""
walkforward_filter_refit.py — re-estimate the range filter every year.

WHY
    find_losing_conditions.py fitted day_range_atr >= 3.25 once, on
    2017-2022, and applied it to 2023+. That is one held-out period and one
    threshold. If the threshold only works because 3.25 happened to suit
    2023-2026, a walk-forward refit will expose it: fit on everything
    before year Y, apply to year Y, never look forward.

    It also asks the question the confound analysis raised. The filter
    correlates +0.785 with time of day and does nothing after 11:00, so it
    may be the 11:30 clock in disguise. Both are run side by side here --
    if a fixed "trade after 11:00" rule matches the refit filter, the range
    measurement is adding nothing and the simpler rule wins.

INPUT
    data/.wf_trades.csv, written by find_losing_conditions.py. Every row is
    one walk-forward trade with its ex-ante entry conditions and realised
    P&L, so nothing is retrained here.

USAGE
    python scripts/walkforward_filter_refit.py
    python scripts/walkforward_filter_refit.py --col atr_pts
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / ".wf_trades.csv"
LOT = 65


def fit_threshold(early: pd.DataFrame, col: str, q: int = 5):
    """Lowest quintile edge whose bucket expectancy is positive."""
    try:
        b = pd.qcut(early[col], q, duplicates="drop")
    except ValueError:
        return None
    tab = early.groupby(b, observed=True)["pnl"].mean()
    pos = tab[tab > 0]
    if len(pos) == 0 or len(pos) == len(tab):
        return None
    return float(min(iv.left for iv in pos.index))


def agg(a: pd.Series) -> str:
    if len(a) == 0:
        return "     -"
    return f"{a.mean():+6.2f}"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="day_range_atr")
    ap.add_argument("--min-train", type=int, default=600)
    a = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing — run find_losing_conditions.py first")
    t = pd.read_csv(CACHE)
    col = a.col
    print(f"{len(t):,} cached walk-forward trades | filter column: {col}")
    print("Threshold re-estimated from scratch before every year.\n")

    print(f"{'year':<6}{'thr':>7}{'n raw':>7}{'raw':>8}{'n filt':>8}{'filt':>8}"
          f"{'delta':>8}{'  |':>4}{'hr>=11':>8}{'n':>6}")
    print("-" * 74)
    rows, keep_f, keep_r, keep_h = [], [], [], []
    for Y in sorted(t.year.unique()):
        early, cur = t[t.year < Y], t[t.year == Y]
        if len(early) < a.min_train or len(cur) == 0:
            continue
        thr = fit_threshold(early, col)
        if thr is None:
            continue
        f = cur[cur[col] >= thr]
        h = cur[cur.hour >= 11]
        rows.append((Y, thr, cur, f, h))
        keep_r.append(cur.pnl.values)
        keep_f.append(f.pnl.values)
        keep_h.append(h.pnl.values)
        d = (f.pnl.mean() - cur.pnl.mean()) if len(f) else float("nan")
        print(f"{Y:<6}{thr:>7.2f}{len(cur):>7}{agg(cur.pnl):>8}{len(f):>8}"
              f"{agg(f.pnl):>8}{d:>+8.2f}{'  |':>4}{agg(h.pnl):>8}{len(h):>6}")

    if not rows:
        print("not enough history to refit")
        return
    R = np.concatenate(keep_r)
    F = np.concatenate([x for x in keep_f if len(x)])
    H = np.concatenate([x for x in keep_h if len(x)])
    print("-" * 74)

    def line(tag, arr):
        w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
        pf = w / l if l > 0 else float("inf")
        eq = np.cumsum(arr)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        print(f"  {tag:<22}n={len(arr):>5}  {arr.mean():+6.2f} pts  "
              f"PF {pf:5.3f}  {arr.sum()*LOT:>+11,.0f} INR  maxDD {dd:>6.0f}")

    print("AGGREGATE over the refit years")
    line("unfiltered", R)
    line(f"{col} refit", F)
    line("hour >= 11 (fixed)", H)

    thrs = [r[1] for r in rows]
    print(f"\nthreshold stability: {min(thrs):.2f}-{max(thrs):.2f}  "
          f"median {np.median(thrs):.2f}  std {np.std(thrs):.2f}")
    better = sum(1 for _, _, c, f, _ in rows if len(f) and f.pnl.mean() > c.pnl.mean())
    print(f"filter improved the year in {better}/{len(rows)} refits")
    vs_h = sum(1 for _, _, _, f, h in rows
               if len(f) and len(h) and f.pnl.mean() > h.pnl.mean())
    print(f"range filter beat the fixed 11:00 clock in {vs_h}/{len(rows)} years")

    print("\nVERDICT")
    if F.mean() <= R.mean():
        print("  REJECT — the refit filter does not beat unfiltered.")
    elif F.mean() <= H.mean():
        print(f"  USE THE CLOCK — refit filter {F.mean():+.2f} does not beat")
        print(f"  the fixed hour>=11 rule {H.mean():+.2f}. The range measurement")
        print("  is a proxy for time of day and adds nothing over it.")
    else:
        print(f"  RANGE FILTER WINS — {F.mean():+.2f} vs clock {H.mean():+.2f}")
        print(f"  vs unfiltered {R.mean():+.2f}. Carries information beyond")
        print("  time of day; worth a forward test.")


if __name__ == "__main__":
    main()
