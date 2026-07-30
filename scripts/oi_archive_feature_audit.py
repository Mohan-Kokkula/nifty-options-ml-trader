"""
oi_archive_feature_audit.py -- audit the newly-collected data/oi_archive/
full option-chain snapshots against the production feature frame.

NOT a repeat of any prior study: this is the first time the live archiver's
own data (449k rows, 29 trading days, Jun 12 - Jul 22 2026) has been
touched. The earlier intraday_oi_kaggle_ic.csv study used a third-party
historical dataset instead. Reuses build_frame() (unmodified) as the join
target; no new model is trained (permutation importance / SHAP need a
fitted model and are deliberately NOT computed here -- flagged, not faked).

Causal-only feature engineering: every feature at bar t uses only OI
snapshots with snapshot_ts <= t's bar-close, and rolling windows are
strictly backward-looking (no centered windows, no future leakage).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import build_frame


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        k = n - rank
        prev = min(prev, p[idx] * n / k)
        q[idx] = prev
    return q


def main():
    files = sorted(glob.glob(str(ROOT / "data/oi_archive/oi_2026-*.csv")))
    print(f"Loading {len(files)} OI archive files ...")
    raw = pd.concat([pd.read_csv(f, parse_dates=["snapshot_ts"]) for f in files],
                     ignore_index=True)
    print(f"  raw rows: {len(raw):,}  days: {raw.snapshot_ts.dt.date.nunique()}")

    # ---- ATM-band series: ATM strike CE/PE OI per snapshot ----
    atm = raw[raw["strike"] == raw["atm_strike"]].copy()
    atm = atm.sort_values("snapshot_ts").drop_duplicates("snapshot_ts", keep="last")
    print(f"  ATM-strike snapshots: {len(atm):,}")

    # causal resample to the production 5-min bar grid: last snapshot <= bar close
    atm["bar5"] = atm["snapshot_ts"].dt.ceil("5min")  # snapshot belongs to the bar it CLOSES within
    bar = atm.groupby("bar5").last()[["spot", "atm_strike", "ce_oi", "pe_oi",
                                        "ce_iv", "pe_iv", "pcr_total"]]
    bar = bar.sort_index()
    print(f"  bars after causal resample: {len(bar):,}")

    # ---- Engineer the 12 requested causal features ----
    f = pd.DataFrame(index=bar.index)
    f["atm_ce_oi_chg_pct"] = bar["ce_oi"].pct_change()
    f["atm_pe_oi_chg_pct"] = bar["pe_oi"].pct_change()
    f["net_oi_change"] = (bar["ce_oi"] - bar["pe_oi"]).diff()
    f["oi_momentum_3"] = (bar["ce_oi"] - bar["pe_oi"]).diff(3)
    f["oi_acceleration"] = f["net_oi_change"].diff()
    f["ce_pe_oi_ratio"] = bar["ce_oi"] / bar["pe_oi"].replace(0, np.nan)
    f["pcr"] = bar["pcr_total"]
    ce_roll_mean = bar["ce_oi"].rolling(20, min_periods=10).mean()
    ce_roll_std = bar["ce_oi"].rolling(20, min_periods=10).std()
    f["oi_zscore"] = (bar["ce_oi"] - ce_roll_mean) / ce_roll_std.replace(0, np.nan)
    f["oi_rolling_vol"] = bar["ce_oi"].pct_change().rolling(10, min_periods=5).std()
    price_ret = bar["spot"].pct_change()
    oi_ret = (bar["ce_oi"] - bar["pe_oi"]).pct_change()
    f["oi_price_divergence"] = np.sign(price_ret) != np.sign(oi_ret)
    oi_net = bar["ce_oi"] - bar["pe_oi"]
    oi_band = oi_net.rolling(20, min_periods=10)
    f["oi_breakout"] = (oi_net - oi_band.mean()) / oi_band.std().replace(0, np.nan)
    f["oi_trend_persistence"] = np.sign(f["net_oi_change"]).rolling(5, min_periods=3).apply(
        lambda x: (x == x.iloc[-1]).sum(), raw=False)

    oi_cols = list(f.columns)
    print(f"  engineered {len(oi_cols)} causal OI features")

    # ---- Join to production frame (unmodified build_frame) ----
    print("\nBuilding production frame ...")
    feat, fcols = build_frame()
    joined = feat.join(f, how="inner")
    join_rate = len(joined) / len(feat[(feat.index >= f.index.min()) & (feat.index <= f.index.max())]) \
        if len(feat[(feat.index >= f.index.min()) & (feat.index <= f.index.max())]) else float("nan")

    print(f"\nOI-covered calendar span: {f.index.min()} -> {f.index.max()}")
    print(f"Production bars in that span: "
          f"{len(feat[(feat.index >= f.index.min()) & (feat.index <= f.index.max())]):,}")
    print(f"Bars with a successful join (exact 5-min timestamp match): {len(joined):,}")
    print(f"Join success rate (of production bars in OI's calendar span): {join_rate*100:.1f}%")
    print(f"Usable trading days after join: {joined.index.date if len(joined) else []}")
    days = sorted(set(joined.index.date)) if len(joined) else []
    print(f"Usable trading days: {len(days)} -> {days}")
    missing_pct = 100 * (1 - len(joined) / len(f)) if len(f) else float("nan")
    print(f"Missing % (OI bars that found no production-frame row): {missing_pct:.1f}%")

    if len(joined) < 30:
        print(f"\n*** Only {len(joined)} joined bars. Too few for any of the requested "
              "gate statistics (IC/MI/permutation-importance/SHAP/monthly-or-fold "
              "stability/multiple-testing correction) to be meaningful. Reporting "
              "raw exploratory numbers ONLY, with explicit underpowering caveats. ***\n")

    # ---- Exploratory (NOT gate-quality) correlation / MI on whatever joined ----
    if len(joined) >= 10:
        fwd_ret = feat["close"].pct_change().shift(-3)  # matches CURRENT_FWD_BARS=3 convention
        y_cont = fwd_ret.loc[joined.index]
        from sklearn.feature_selection import mutual_info_regression
        results = []
        for c in oi_cols:
            x = joined[c].astype(float)
            m = x.notna() & y_cont.notna() & np.isfinite(x) & np.isfinite(y_cont)
            n = int(m.sum())
            if n < 10:
                results.append(dict(feature=c, n=n, spearman=np.nan, p=np.nan, mi=np.nan))
                continue
            sr, sp = stats.spearmanr(x[m], y_cont[m])
            try:
                mi = float(mutual_info_regression(x[m].values.reshape(-1, 1), y_cont[m].values,
                                                    random_state=42)[0])
            except Exception:
                mi = np.nan
            results.append(dict(feature=c, n=n, spearman=sr, p=sp, mi=mi))
        res_df = pd.DataFrame(results)
        pvals = res_df["p"].fillna(1.0).values
        res_df["q_bh"] = bh_fdr(pvals)
        print("\nExploratory Spearman / MI vs 3-bar forward return (NOT a validated gate result):")
        print(res_df.to_string(index=False))
        res_df.to_csv(ROOT / "logs/oi_archive_feature_audit_exploratory.csv", index=False)
    else:
        print("\nFewer than 10 joined bars -- not even exploratory correlation is meaningful.")

    print("\n--- Data quality notes ---")
    print(f"ATM-strike snapshot cadence (median): "
          f"{atm['snapshot_ts'].sort_values().diff().dt.total_seconds().median():.0f}s")
    dup_ts = raw.duplicated(subset=["snapshot_ts", "strike"]).sum()
    print(f"Duplicate (snapshot_ts, strike) rows in raw archive: {dup_ts}")


if __name__ == "__main__":
    main()
