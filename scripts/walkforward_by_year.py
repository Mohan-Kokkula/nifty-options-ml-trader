"""
walkforward_by_year.py — true out-of-sample accuracy and P&L, year by year.

WHY THIS AND NOT THE USUAL SPLIT
    The purged 85/7.5/7.5 split yields ONE validation year and ONE test
    year. Every model number in this project rests on those two windows,
    which is why VAL and TEST have disagreed so often -- n is around 230
    trades per window and that cannot separate edge from luck.

    This retrains from scratch for every year: train on everything strictly
    before year Y (minus the purge gap), predict year Y, keep nothing.
    Ten independent out-of-sample years instead of one.

HELD CONSTANT
    features   V11's 26
    labels     create_labels()'s fixed 3-bar forward return
    decision   the live rule (CONFIDENCE_CALL/PUT, MIN_EDGE, SKIP_CEIL)
    exit       4xATR stop, ride to close
    costs      5.9 pts round trip, futures
    sizing     1 lot = 65, for the rupee column only

    No promoted pickle is loaded. Every year trains its own ensemble.

WHAT dir_acc MEANS HERE
    Accuracy over bars the model FIRED on whose true label was directional.
    It is not "next-bar direction accuracy" -- that sits near 50% at every
    timeframe. Reported because it was asked for, but tonight produced two
    separate demonstrations that accuracy and profit move independently
    (conformal at 100% accuracy lost 275 pts; trend-scanning labels lost
    both at once). Read the PF column first.

USAGE
    python scripts/walkforward_by_year.py
    python scripts/walkforward_by_year.py --start 2017
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_model_v9 as V9  # noqa: E402
import scripts.train_model_v11 as V11  # noqa: E402

FRICTION = 5.9
ATR_STOP = 4.0
LOT = 65
EOD, ENTRY_START, ENTRY_END = "15:15", "09:20", "15:00"
CALL, PUT, SKIP = 0, 1, 2


def simulate(df, rows, sig):
    sub = df.iloc[rows]
    H, L, C = sub["high"].values, sub["low"].values, sub["close"].values
    atr = (sub["a14"] * sub["close"]).values
    hhmm = sub.index.strftime("%H:%M").values
    days = sub.index.normalize().values
    trades, pos, cur = [], None, None
    for i in range(len(sub)):
        if days[i] != cur:
            cur, pos = days[i], None
        if pos is not None:
            d, e, sl = pos
            hit = None
            if d == CALL and L[i] <= sl:
                hit = sl - e
            elif d == PUT and H[i] >= sl:
                hit = e - sl
            if hit is None and hhmm[i] >= EOD:
                hit = (C[i] - e) if d == CALL else (e - C[i])
            if hit is not None:
                trades.append(hit - FRICTION)
                pos = None
            continue
        if sig[i] == SKIP or not (ENTRY_START <= hhmm[i] < ENTRY_END):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        e = C[i]
        pos = (int(sig[i]), e, e - a * ATR_STOP if sig[i] == CALL
               else e + a * ATR_STOP)
    return np.array(trades, dtype=float)


def stats(a):
    if len(a) == 0:
        return dict(n=0, pf=0.0, avg=0.0, win=0.0, total=0.0, dd=0.0)
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    eq = np.cumsum(a)
    return dict(n=len(a), pf=float(w / l) if l > 0 else float("inf"),
                avg=float(a.mean()), win=float((a > 0).mean() * 100),
                total=float(a.sum()),
                dd=float((np.maximum.accumulate(eq) - eq).max()))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2017)
    a = ap.parse_args()
    t0 = time.time()

    print("Walk-forward: each year is predicted by a model that never saw it.")
    print("V11 features, fixed 3-bar labels, live rule, 4xATR/EOD, 5.9pt costs.\n")

    df = V11.build_dataset(V9.CSV_5M, V9.CSV_15M, V9.CSV_30M, V9.CSV_60M,
                           V9.CSV_DAY, V9.CSV_VIX, V9.CSV_FUT)
    X = df[list(V11.V11_FEATURES)].replace([np.inf, -np.inf], 0.0)\
                                  .fillna(0.0).values
    y = df["label"].values.astype(int)
    yr = df.index.year.values
    gap = V9.FWD_BARS + V9.EMBARGO_BARS
    years = [int(v) for v in sorted(set(yr)) if v >= a.start]

    from sklearn.preprocessing import StandardScaler
    rows, allt = [], []
    print(f"{'year':<6}{'trades':>7}{'win%':>7}{'PF':>7}{'dir_acc':>9}"
          f"{'avg':>8}{'pts':>8}{'INR':>10}{'maxDD':>7}{'trainN':>9}")
    print("-" * 80)
    for Y in years:
        te = np.where(yr == Y)[0]
        tr = np.where(yr < Y)[0]
        if len(tr) < 20000 or len(te) < 500:
            continue
        tr = tr[: max(0, len(tr) - gap)]                # purge the boundary
        if len(np.unique(y[tr])) < 3:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        cut = int(len(tr) * 0.9)
        models = V9._fit_ensemble(Xtr[:cut], y[tr][:cut],
                                  Xtr[cut:], y[tr][cut:],
                                  V11.build_sample_weights(df, tr, y[tr])[:cut])
        p = np.mean([m.predict_proba(sc.transform(X[te]))
                     for m in models.values()], axis=0)
        sig = V9.live_rule_signals(p)
        m = (sig != SKIP) & (y[te] != SKIP)
        acc = float((sig[m] == y[te][m]).mean()) if m.any() else float("nan")
        t = simulate(df, te, sig)
        allt.append(t)
        s = stats(t)
        rows.append((Y, s, acc))
        pf = "  inf " if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
        print(f"{Y:<6}{s['n']:>7}{s['win']:>6.1f}%{pf}{acc:>8.1%}"
              f"{s['avg']:>+8.2f}{s['total']:>+8.0f}{s['total']*LOT:>+10,.0f}"
              f"{s['dd']:>7.0f}{len(tr):>9,}")

    if not rows:
        print("no evaluable years")
        return
    a_all = np.concatenate(allt)
    S = stats(a_all)
    accs = [r[2] for r in rows if np.isfinite(r[2])]
    print("-" * 80)
    pf = "  inf " if S["pf"] == float("inf") else f"{S['pf']:6.3f}"
    print(f"{'ALL':<6}{S['n']:>7}{S['win']:>6.1f}%{pf}"
          f"{np.mean(accs):>8.1%}{S['avg']:>+8.2f}{S['total']:>+8.0f}"
          f"{S['total']*LOT:>+10,.0f}{S['dd']:>7.0f}")
    pos = sum(1 for _, s, _ in rows if s["total"] > 0)
    print(f"\nprofitable years: {pos}/{len(rows)}   "
          f"mean dir_acc {np.mean(accs):.1%} (range {min(accs):.1%}-{max(accs):.1%})")
    print(f"accuracy vs PF rank correlation: "
          f"{pd.Series(accs).corr(pd.Series([r[1]['pf'] for r in rows if np.isfinite(r[2])]), method='spearman'):+.3f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
