"""
test_trend_scanning_labels.py — adaptive-horizon labels vs the fixed 3-bar label.

WHAT CHANGES AND WHAT DOES NOT
    Features:  V11's 26, unchanged.
    Exit:      4xATR stop, ride to close, 5.9pt friction, unchanged.
    Split:     purged chronological, unchanged in shape.
    ONLY the LABEL differs. Everything else is held constant so the
    comparison attributes the difference to labelling and nothing else.

    baseline   fwd = close.shift(-3)/close - 1, thresholded per session
               (train_model_v9.create_labels) -- every bar judged over the
               same 15 minutes regardless of what the market was doing.

    candidate  trend scanning: fit OLS over every horizon in span, keep the
               one with the largest |t|, label by its sign. Each bar gets
               graded on the trend it actually belongs to.

NO PRE-TRAINED MODEL IS LOADED. Both arms train a fresh ensemble from
scratch on the same proper-train rows, so nothing inherited from the
promoted V9/V11 pickles can flatter either side.

PURGE CORRECTNESS -- the detail that makes or breaks this
    Trend-scanning labels look up to max(span) bars ahead, so the split
    must purge at least max(span), not FWD_BARS=3. Using the baseline's
    78-bar gap would leak the label across the boundary and hand the
    candidate a fake win. The candidate arm therefore gets its own,
    larger gap and the script prints both.

PRE-REGISTERED before the run
    selection metric = VAL profit factor.
    Adopt only if trend-scanning beats the fixed label on VAL PF *and*
    holds on TEST. dir_acc is reported but is NOT the criterion -- the
    conformal test already showed 100% accuracy losing 275 points, so
    accuracy is known to be the wrong target here.

USAGE
    python scripts/test_trend_scanning_labels.py
    python scripts/test_trend_scanning_labels.py --tspan 5,21 --tthr 2.0,2.5
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
from core.trend_scanning import trend_scan_labels  # noqa: E402

FRICTION = 5.9
ATR_STOP = 4.0
EOD, ENTRY_START, ENTRY_END = "15:15", "09:20", "15:00"
CALL, PUT, SKIP = 0, 1, 2


def purged_split(n: int, gap: int, train_frac=0.85, val_frac=0.075):
    """Chronological 3-way split with an explicit purge gap."""
    tr_end = int(n * train_frac)
    va_end = int(n * (train_frac + val_frac))
    tr = np.arange(0, tr_end - gap)
    va = np.arange(tr_end, va_end - gap)
    te = np.arange(va_end, n - gap)
    return tr, va, te


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
    arr = np.array(trades, dtype=float)
    if len(arr) == 0:
        return {"pf": 0.0, "n": 0, "avg": 0.0, "total": 0.0}
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return {"pf": float(w / l) if l > 0 else float("inf"), "n": len(arr),
            "avg": float(arr.mean()), "total": float(arr.sum())}


def fit_eval(df, X, y, tr, va, te, tag):
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y[tr])) < 3:
        print(f"  {tag}: only {len(np.unique(y[tr]))} classes in train — skip")
        return None
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    models = V9._fit_ensemble(Xtr, y[tr], sc.transform(X[va]), y[va],
                              V11.build_sample_weights(df, tr, y[tr]))
    out = {}
    for name, idx in (("VAL", va), ("TEST", te)):
        p = np.mean([m.predict_proba(sc.transform(X[idx]))
                     for m in models.values()], axis=0)
        s = V9.live_rule_signals(p)
        st = simulate(df, idx, s)
        m = (s != SKIP) & (y[idx] != SKIP)
        st["acc"] = float((s[m] == y[idx][m]).mean()) if m.any() else np.nan
        out[name] = st
    return out


def row(tag, s):
    pf = "  inf " if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{tag:<32}{pf}{s['n']:>7}{s['avg']:>+9.2f}"
            f"{s['total']:>+9.0f}{s['acc']:>9.1%}")


HDR = f"{'labelling / split':<32}{'PF':>6}{'n':>7}{'avg':>9}{'total':>9}{'dir_acc':>9}"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--tspan", default="5,21")
    ap.add_argument("--tthr", default="2.0,2.5,3.0")
    a = ap.parse_args()
    lo, hi = [int(x) for x in a.tspan.split(",")]
    t0 = time.time()

    print("PRE-REGISTERED: selection metric = VAL profit factor.")
    print("Only the LABEL changes. Features, exit, costs held constant.")
    print("No pre-trained model is loaded; both arms train from scratch.\n")

    df = V11.build_dataset(V9.CSV_5M, V9.CSV_15M, V9.CSV_30M, V9.CSV_60M,
                           V9.CSV_DAY, V9.CSV_VIX, V9.CSV_FUT)
    X = df[list(V11.V11_FEATURES)].replace([np.inf, -np.inf], 0.0)\
                                  .fillna(0.0).values
    n = len(df)

    base_gap = V9.FWD_BARS + V9.EMBARGO_BARS
    ts_gap = hi + V9.EMBARGO_BARS
    print(f"rows {n:,} | baseline purge {base_gap} (fwd {V9.FWD_BARS}) "
          f"| trend-scan purge {ts_gap} (max span {hi})\n")

    print("=" * 76)
    print("BASELINE — fixed 3-bar forward label (what V9/V11 use today)")
    print("=" * 76)
    yb = df["label"].values.astype(int)
    tr, va, te = purged_split(n, base_gap)
    bb = fit_eval(df, X, yb, tr, va, te, "baseline")
    if bb is None:
        return
    print(HDR)
    for k in ("VAL", "TEST"):
        print(row(f"fixed-3bar  {k}", bb[k]))
    bal = np.bincount(yb, minlength=3)
    print(f"\nlabel mix  CALL {bal[0]:,}  PUT {bal[1]:,}  SKIP {bal[2]:,}")

    print(f"\n{'='*76}")
    print("CANDIDATE — trend scanning, adaptive horizon (own larger purge)")
    print("=" * 76)
    print(HDR)
    trs, vas, tes = purged_split(n, ts_gap)
    best = None
    for thr in [float(x) for x in a.tthr.split(",")]:
        lab = trend_scan_labels(df["close"], span=(lo, hi), t_threshold=thr)
        yt = lab["label"].values.astype(int)
        mix = np.bincount(yt, minlength=3)
        res = fit_eval(df, X, yt, trs, vas, tes, f"t>={thr}")
        if res is None:
            continue
        for k in ("VAL", "TEST"):
            print(row(f"trendscan t>={thr}  {k}", res[k]))
        print(f"    label mix  CALL {mix[0]:,}  PUT {mix[1]:,}  SKIP {mix[2]:,}")
        if res["VAL"]["n"] >= 30 and (best is None
                                      or res["VAL"]["pf"] > best[0]["VAL"]["pf"]):
            best = (res, thr)
        print("-" * 76)

    print("\n" + "=" * 76)
    print("COMPARISON")
    print("=" * 76)
    if best is None:
        print("  No trend-scanning threshold produced enough VAL trades.")
        return
    br, thr = best
    dv = br["VAL"]["pf"] - bb["VAL"]["pf"]
    dt = br["TEST"]["pf"] - bb["TEST"]["pf"]
    print(f"  VAL picked t>={thr}")
    print(f"  VAL   trendscan {br['VAL']['pf']:.3f}  vs fixed "
          f"{bb['VAL']['pf']:.3f}   delta {dv:+.3f}")
    print(f"  TEST  trendscan {br['TEST']['pf']:.3f}  vs fixed "
          f"{bb['TEST']['pf']:.3f}   delta {dt:+.3f}")
    print("\nVERDICT")
    if dv <= 0:
        print(f"  REJECT — trend scanning does not beat the fixed label on VAL ({dv:+.3f}).")
    elif dt <= 0:
        print(f"  REJECT — VAL improved ({dv:+.3f}) but TEST did not ({dt:+.3f}).")
        print("  Same VAL-only pattern that has failed repeatedly here.")
    else:
        print(f"  ADOPT-CANDIDATE — VAL {dv:+.3f} and TEST {dt:+.3f} both positive.")
        print("  Needs walk-forward confirmation before replacing create_labels().")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
