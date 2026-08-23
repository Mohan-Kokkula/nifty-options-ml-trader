"""
audit_feature_leakage.py — find look-ahead by feeding the pipeline pure noise.

THE IDEA (from MQL5 art. 23659, "Trend-Scanning Features", Feature
Engineering for ML Part 13)
    Run the feature builders on a synthetic RANDOM WALK. There is nothing to
    predict in a random walk, by construction. So if a feature computed at
    bar t correlates with the return from t onward, that correlation cannot
    be skill -- it is the feature reading data it should not have.

    That article measured trend-scanning's forward window at a 56-61% hit
    rate on random walks against 49.9-50.1% for the causal window. Same
    function, same data, one argument different.

WHY THIS PROJECT NEEDS IT
    V9 once reported 92.9% direction accuracy. After the leakage purge it
    was 58.7%. That discovery cost real work; this test would have caught
    it in seconds, because noise has no future to leak.

    Nothing in this repo has ever been audited this way. 170 V9 features,
    26 in V11, 7 profile features added this week -- all unexamined against
    a null where the correct answer is known exactly.

METHOD
    1. Generate synthetic 5-minute OHLC as a driftless random walk, scaled
       to NIFTY's realised volatility, in real session shape (09:15-15:30).
    2. Resample to 15/30/60-min and daily; synthesise a mean-reverting VIX.
    3. Run the PRODUCTION builders on it -- V11.build_dataset, i.e. exactly
       what training uses. Not a reimplementation, or the audit would be
       testing the wrong code.
    4. For every feature, Spearman-rank IC against the forward return at
       +1 bar and at +FWD_BARS (the label horizon).
    5. Repeat over seeds. On noise the true IC is 0; anything consistently
       outside the noise band is reading the future.

READING THE OUTPUT
    |IC| inside the band  -> clean, as expected
    |IC| outside, one seed -> probably chance, look at the mean
    |IC| outside on the mean across seeds -> LEAK. Investigate that feature.

    A feature can also be legitimately non-zero at +1 bar if it is a pure
    function of the current bar's close (e.g. close-vs-open position) --
    that is autocorrelation of the bar itself, not look-ahead. The report
    flags magnitude; you still have to think about the mechanism.

USAGE
    python scripts/audit_feature_leakage.py
    python scripts/audit_feature_leakage.py --seeds 5 --days 600
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_model_v9 as V9  # noqa: E402
import scripts.train_model_v11 as V11  # noqa: E402

BARS_PER_DAY = 75          # 09:15 -> 15:30 inclusive of 5-min bars
SIGMA_5M = 0.00055         # ~13.3pt ATR at 24,000 -- NIFTY's actual scale


def synth_5min(days: int, seed: int) -> pd.DataFrame:
    """Driftless random walk shaped like a NIFTY 5-minute session."""
    rng = np.random.default_rng(seed)
    n = days * BARS_PER_DAY
    ret = rng.normal(0.0, SIGMA_5M, n)          # no drift, no memory
    close = 24000.0 * np.exp(np.cumsum(ret))
    # intrabar extremes from an independent draw so H/L carry no future info
    wig = np.abs(rng.normal(0.0, SIGMA_5M * 0.6, n)) * close
    op = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(op, close) + wig
    low = np.minimum(op, close) - wig

    idx = []
    d0 = pd.Timestamp("2016-01-04 09:15")
    day = 0
    while len(idx) < n:
        base = d0 + pd.Timedelta(days=day)
        if base.weekday() < 5:
            idx.extend(base + pd.Timedelta(minutes=5 * k)
                       for k in range(BARS_PER_DAY))
        day += 1
    idx = pd.DatetimeIndex(idx[:n])
    return pd.DataFrame({"open": op, "high": high, "low": low,
                         "close": close, "volume": 0.0}, index=idx)


def write_inputs(df5: pd.DataFrame, tmp: Path, seed: int) -> dict:
    """Materialise the whole CSV set the production pipeline reads."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    paths = {}
    p5 = tmp / "s5.csv"
    df5.to_csv(p5, index_label="date")
    paths["csv5"] = str(p5)
    for name, rule in (("csv15", "15min"), ("csv30", "30min"),
                       ("csv60", "60min")):
        r = df5.resample(rule).agg(agg).dropna()
        p = tmp / f"{name}.csv"
        r.to_csv(p, index_label="date")
        paths[name] = str(p)
    dayf = df5.resample("1D").agg(agg).dropna()
    pd_ = tmp / "sday.csv"
    dayf.to_csv(pd_, index_label="date")
    paths["csv_day"] = str(pd_)

    rng = np.random.default_rng(seed + 999)
    v = 13.0 + np.cumsum(rng.normal(0, 0.25, len(dayf)))
    v = 13.0 + (v - v.mean()) * 0.5          # keep it in a sane band
    vix = pd.DataFrame({"vix": np.clip(v, 8, 35)}, index=dayf.index)
    pv = tmp / "svix.csv"
    vix.to_csv(pv, index_label="date")
    paths["csv_vix"] = str(pv)
    paths["csv_fut"] = str(tmp / "nonexistent_fut.csv")   # force spot proxy
    return paths


def rank_ic(x: pd.Series, fwd: pd.Series) -> float:
    m = x.notna() & fwd.notna() & np.isfinite(x) & np.isfinite(fwd)
    if m.sum() < 200 or x[m].nunique() < 5:
        return float("nan")
    return float(x[m].rank().corr(fwd[m].rank()))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--days", type=int, default=600)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    print("Feeding the PRODUCTION feature builders a driftless random walk.")
    print("True IC is 0 for every feature. Anything else is look-ahead.\n")

    ic1, ic3, ns = {}, {}, []
    for seed in range(a.seeds):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            df5 = synth_5min(a.days, seed)
            paths = write_inputs(df5, tmp, seed)
            print(f"  seed {seed}: {len(df5):,} synthetic bars -> building...")
            feat = V11.build_dataset(**paths)
            if feat.empty:
                print("    empty frame, skipping")
                continue
            c = feat["close"]
            f1 = c.shift(-1) / c - 1
            f3 = c.shift(-V9.FWD_BARS) / c - 1
            ns.append(len(feat))
            skip = {"open", "high", "low", "close", "volume", "label"}
            for col in feat.columns:
                if col in skip or not np.issubdtype(feat[col].dtype, np.number):
                    continue
                ic1.setdefault(col, []).append(rank_ic(feat[col], f1))
                ic3.setdefault(col, []).append(rank_ic(feat[col], f3))

    if not ns:
        print("No synthetic frames built — cannot audit.")
        return
    n = int(np.mean(ns))
    band = 1.96 / np.sqrt(n)
    print(f"\n{len(ic1)} features | ~{n:,} rows/seed | "
          f"95% noise band |IC| < {band:.4f}\n")

    rows = []
    for k in ic1:
        m1 = np.nanmean(ic1[k])
        m3 = np.nanmean(ic3[k])
        if np.isnan(m1) and np.isnan(m3):
            continue
        rows.append((k, m1, m3, max(abs(np.nan_to_num(m1)),
                                    abs(np.nan_to_num(m3)))))
    rows.sort(key=lambda r: -r[3])

    v11 = set(V11.V11_FEATURES)
    print(f"{'feature':<26}{'IC(+1)':>10}{'IC(+3)':>10}{'verdict':>12}  in V11")
    print("-" * 72)
    flagged = 0
    for k, m1, m3, mx in rows[:a.top]:
        bad = mx > band
        flagged += bad
        print(f"{k:<26}{m1:>+10.4f}{m3:>+10.4f}"
              f"{('LEAK?' if bad else 'clean'):>12}  {'yes' if k in v11 else ''}")

    print("-" * 72)
    over = [r for r in rows if r[3] > band]
    print(f"{len(over)} of {len(rows)} features exceed the noise band")
    v11_over = [r[0] for r in over if r[0] in v11]
    print(f"of those, {len(v11_over)} are in V11's live feature set: "
          f"{v11_over if v11_over else 'none'}")
    print("\nNOTE: exceeding the band is a flag, not a conviction. A feature")
    print("that is a pure function of the CURRENT bar can show a small +1 IC")
    print("from bar autocorrelation. Judge the mechanism, not just the number.")


if __name__ == "__main__":
    main()
