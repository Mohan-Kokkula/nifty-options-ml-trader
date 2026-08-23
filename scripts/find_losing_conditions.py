"""
find_losing_conditions.py — what separates the model's winning trades from
its losing ones, using only information available BEFORE entry.

THE REQUEST AND THE TRAP
    "Find the pattern in the losing years and build a strategy for it."
    Taken literally that is selecting on the outcome: 5 losing years and 5
    winning years is 10 data points, and any rule fitted to which ones lost
    describes that sample and nothing else. Tonight already produced four
    VAL-only mirages from exactly this shape of reasoning.

    The defensible version asks the same question where the data actually
    is. The walk-forward produced 2,810 trades. Each one carries conditions
    that were KNOWN AT ENTRY -- volatility, VIX, trend strength, position
    in the day's range, time of day, model confidence. If losing trades
    cluster in measurable conditions, that is a filter you could have
    applied at the time. If they do not, the losses are noise and no
    strategy can target them.

METHOD
    1. Re-run the walk-forward, recording per-trade entry conditions.
    2. Bucket each condition into quintiles; report expectancy per bucket.
       A condition that matters shows a monotone gradient, not one odd cell.
    3. Fit a filter on the EARLY years only, then apply it -- untouched --
       to the LATE years. Fitting and testing on the same trades would
       reproduce the mirage this script exists to avoid.

    Ex-ante only. Nothing that requires knowing how the trade turned out.

USAGE
    python scripts/find_losing_conditions.py
    python scripts/find_losing_conditions.py --split-year 2023
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

FRICTION, ATR_STOP, LOT = 5.9, 4.0, 65
EOD, ENTRY_START, ENTRY_END = "15:15", "09:20", "15:00"
CALL, PUT, SKIP = 0, 1, 2

# every one of these is computable at the entry bar
COND = ["atr_pts", "vix", "conf", "adx", "rsi", "dist_vwap_atr",
        "day_range_atr", "close_pos", "hour", "dir"]


def simulate_with_conditions(df, rows, sig, proba):
    """Same exit as every other test tonight; also records entry state."""
    sub = df.iloc[rows]
    H, L, C = sub["high"].values, sub["low"].values, sub["close"].values
    atr = (sub["a14"] * sub["close"]).values
    hhmm = sub.index.strftime("%H:%M").values
    hour = sub.index.hour.values + sub.index.minute.values / 60.0
    days = sub.index.normalize().values
    vix = sub["vix"].values if "vix" in sub.columns else np.full(len(sub), np.nan)
    adx = sub["tf15_adx"].values if "tf15_adx" in sub.columns else np.full(len(sub), np.nan)
    rsi = sub["rsi14"].values if "rsi14" in sub.columns else np.full(len(sub), np.nan)
    dvw = sub["vwap_dist_atr"].values if "vwap_dist_atr" in sub.columns else np.full(len(sub), np.nan)
    cpos = sub["close_pos"].values if "close_pos" in sub.columns else np.full(len(sub), np.nan)

    out, pos, cur, dhi, dlo = [], None, None, np.nan, np.nan
    for i in range(len(sub)):
        if days[i] != cur:
            cur, pos = days[i], None
            dhi, dlo = H[i], L[i]
        else:
            dhi, dlo = max(dhi, H[i]), min(dlo, L[i])
        if pos is not None:
            d, e, sl, rec = pos
            hit = None
            if d == CALL and L[i] <= sl:
                hit = sl - e
            elif d == PUT and H[i] >= sl:
                hit = e - sl
            if hit is None and hhmm[i] >= EOD:
                hit = (C[i] - e) if d == CALL else (e - C[i])
            if hit is not None:
                rec["pnl"] = hit - FRICTION
                out.append(rec)
                pos = None
            continue
        if sig[i] == SKIP or not (ENTRY_START <= hhmm[i] < ENTRY_END):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        rec = {"atr_pts": a, "vix": vix[i], "conf": float(proba[i].max()),
               "adx": adx[i], "rsi": rsi[i], "dist_vwap_atr": dvw[i],
               "day_range_atr": (dhi - dlo) / a, "close_pos": cpos[i],
               "hour": hour[i], "dir": float(sig[i])}
        e = C[i]
        pos = (int(sig[i]), e, e - a * ATR_STOP if sig[i] == CALL
               else e + a * ATR_STOP, rec)
    return out


def expectancy_table(t: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    x = t[col]
    if x.nunique() < q:
        g = t.groupby(x.round(2))
    else:
        try:
            g = t.groupby(pd.qcut(x, q, duplicates="drop"))
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame({"n": g["pnl"].size(), "avg": g["pnl"].mean(),
                         "total": g["pnl"].sum()})


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-year", type=int, default=2023)
    ap.add_argument("--start", type=int, default=2017)
    a = ap.parse_args()
    t0 = time.time()

    print("Conditions are recorded at ENTRY only. Filter is fitted on the")
    print(f"early years and applied untouched to {a.split_year}+.\n")

    df = V11.build_dataset(V9.CSV_5M, V9.CSV_15M, V9.CSV_30M, V9.CSV_60M,
                           V9.CSV_DAY, V9.CSV_VIX, V9.CSV_FUT)
    X = df[list(V11.V11_FEATURES)].replace([np.inf, -np.inf], 0.0)\
                                  .fillna(0.0).values
    y = df["label"].values.astype(int)
    yr = df.index.year.values
    gap = V9.FWD_BARS + V9.EMBARGO_BARS
    from sklearn.preprocessing import StandardScaler

    recs = []
    for Y in [int(v) for v in sorted(set(yr)) if v >= a.start]:
        te, tr = np.where(yr == Y)[0], np.where(yr < Y)[0]
        if len(tr) < 20000 or len(te) < 500:
            continue
        tr = tr[: max(0, len(tr) - gap)]
        if len(np.unique(y[tr])) < 3:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        cut = int(len(tr) * 0.9)
        models = V9._fit_ensemble(Xtr[:cut], y[tr][:cut], Xtr[cut:],
                                  y[tr][cut:],
                                  V11.build_sample_weights(df, tr, y[tr])[:cut])
        p = np.mean([m.predict_proba(sc.transform(X[te]))
                     for m in models.values()], axis=0)
        for r in simulate_with_conditions(df, te, V9.live_rule_signals(p), p):
            r["year"] = Y
            recs.append(r)
        print(f"  {Y}: {sum(1 for r in recs if r['year']==Y)} trades")

    CACHE = ROOT / "data" / ".wf_trades.csv"
    t = pd.DataFrame(recs).dropna(subset=["pnl"])
    t.to_csv(CACHE, index=False)
    print(f"  cached trades -> {CACHE.name}")
    print(f"\n{len(t):,} trades | overall {t.pnl.mean():+.2f} pts/trade\n")

    print("=" * 74)
    print("EXPECTANCY BY ENTRY CONDITION (all years, exploratory)")
    print("=" * 74)
    for c in COND:
        tab = expectancy_table(t, c)
        if tab.empty or len(tab) < 3:
            continue
        cells = " ".join(f"{v:+6.2f}" for v in tab["avg"])
        spread = tab["avg"].max() - tab["avg"].min()
        print(f"{c:<15}{cells}   spread {spread:5.2f}")
    print("\n(each cell is mean pts/trade in that quintile, low -> high)")

    # ── fit a filter on early years only ────────────────────────────────
    early, late = t[t.year < a.split_year], t[t.year >= a.split_year]
    print(f"\n{'='*74}")
    print(f"FILTER — fitted on {early.year.min()}-{a.split_year-1} "
          f"({len(early):,} trades), applied to {a.split_year}+ ({len(late):,})")
    print("=" * 74)

    # Rank conditions by MONOTONICITY on the early years, not by spread.
    # A real effect grades smoothly; one hot bucket among five is noise, and
    # intersecting every condition's positive buckets (the first version of
    # this script) over-constrains to nothing.
    scored = []
    for c in COND:
        tab = expectancy_table(early, c)
        if tab.empty or len(tab) < 4:
            continue
        v = tab["avg"].values
        rho = float(pd.Series(v).corr(pd.Series(range(len(v))), method="spearman"))
        scored.append((abs(rho), rho, c, tab))
    scored.sort(reverse=True)
    print("  monotonicity on the early years (|rho| over quintiles):")
    for ar, rho, c, _ in scored:
        print(f"    {c:<16} rho {rho:+.2f}")
    keep = {}
    if scored and scored[0][0] >= 0.7:
        _, rho, c, tab = scored[0]
        keep[c] = list(tab[tab["avg"] > 0].index)
        print(f"\n  using the single most monotone condition: "
              f"{c} (rho {rho:+.2f})")
    else:
        print("\n  no condition reaches |rho| >= 0.7 — "
              "nothing gradual enough to trust")

    if not keep:
        print("  No condition split the early years into a positive subset.")
    else:
        def apply_filter(frame):
            m = pd.Series(True, index=frame.index)
            for c, buckets in keep.items():
                try:
                    b = pd.qcut(frame[c], 5, duplicates="drop")
                except ValueError:
                    continue
                m &= b.isin(buckets)
            return frame[m]

        for name, frame in (("EARLY (fitted)", early), ("LATE (held out)", late)):
            f = apply_filter(frame)
            base = frame.pnl
            print(f"\n{name}")
            print(f"  unfiltered  n={len(frame):>5}  {base.mean():+6.2f} pts  "
                  f"total {base.sum()*LOT:>+10,.0f} INR")
            if len(f) == 0:
                print("  filtered    nothing survives")
            else:
                print(f"  filtered    n={len(f):>5}  {f.pnl.mean():+6.2f} pts  "
                      f"total {f.pnl.sum()*LOT:>+10,.0f} INR"
                      f"   kept {len(f)/len(frame):.0%}")
        print(f"\n  conditions used: {list(keep)}")

    print(f"\n{'='*74}\nYEAR CONTEXT (n=10 — underpowered, shown for orientation only)")
    print("=" * 74)
    yg = t.groupby("year").agg(n=("pnl", "size"), avg=("pnl", "mean"),
                               atr=("atr_pts", "mean"), vix=("vix", "mean"),
                               adx=("adx", "mean"))
    yg["result"] = np.where(yg["avg"] > 0, "WIN ", "lose")
    print(yg.round(2).to_string())
    w = yg[yg.avg > 0]
    l = yg[yg.avg <= 0]
    print(f"\nwinning yrs: ATR {w.atr.mean():.1f}  VIX {w.vix.mean():.1f}  "
          f"ADX {w.adx.mean():.1f}")
    print(f"losing  yrs: ATR {l.atr.mean():.1f}  VIX {l.vix.mean():.1f}  "
          f"ADX {l.adx.mean():.1f}")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
