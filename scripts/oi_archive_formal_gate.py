"""
oi_archive_formal_gate.py -- apply the SAME formal significance gate used
in cross_asset_ic.csv / futidx_basis_ic.csv / gex_kaggle_ic.csv to the
newly-joinable OI archive features (28 days, 2,054 bars, now that the
nifty_5min.csv price-feed gap is fixed).

Gate (reused verbatim from validate_gex_kaggle.py): q_bh < 0.01 AND
captured_pct > FRICTION_FLOOR_PCT (0.12), where
captured_pct = |spearman| * std(forward_return) * 100.

Stability: split the 28 joined days into two chronological sub-periods
(June vs July -- the only split this sample size supports; NOT the
production's 3-month quarterly folds, which don't fit inside 28 days)
and report sign/significance agreement, matching the "stab: x/2"
convention already used in futidx_basis_ic.csv for similarly thin data.

Permutation importance / SHAP: not computed -- both need a fitted model,
and the task instructs not to train a new production model yet.
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

FRICTION_FLOOR_PCT = 0.12  # same constant as validate_gex_kaggle.py
Q_GATE = 0.01


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


def build_oi_features():
    files = sorted(glob.glob(str(ROOT / "data/oi_archive/oi_2026-*.csv")))
    raw = pd.concat([pd.read_csv(f, parse_dates=["snapshot_ts"]) for f in files], ignore_index=True)
    atm = raw[raw["strike"] == raw["atm_strike"]].copy()
    atm = atm.sort_values("snapshot_ts").drop_duplicates("snapshot_ts", keep="last")
    atm["bar5"] = atm["snapshot_ts"].dt.ceil("5min")
    bar = atm.groupby("bar5").last()[["spot", "atm_strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "pcr_total"]]
    bar = bar.sort_index()

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
    f["oi_price_divergence"] = (np.sign(price_ret) != np.sign(oi_ret)).astype(float)
    oi_net = bar["ce_oi"] - bar["pe_oi"]
    oi_band = oi_net.rolling(20, min_periods=10)
    f["oi_breakout"] = (oi_net - oi_band.mean()) / oi_band.std().replace(0, np.nan)
    f["oi_trend_persistence"] = np.sign(f["net_oi_change"]).rolling(5, min_periods=3).apply(
        lambda x: (x == x.iloc[-1]).sum(), raw=False)
    return f


def ic_test(x, y):
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 20:
        return dict(n=n, spearman=np.nan, p=1.0)
    sr, sp = stats.spearmanr(x[m], y[m])
    return dict(n=n, spearman=sr, p=sp)


def main():
    print("Building OI features (28-day archive) ...")
    f = build_oi_features()
    oi_cols = list(f.columns)

    print("Building production frame ...")
    feat, fcols = build_frame()
    joined = feat.join(f, how="inner")
    fwd_ret = feat["close"].pct_change().shift(-3)
    y_all = fwd_ret.loc[joined.index]
    dstd = float(np.nanstd(y_all))

    joined["_month"] = joined.index.month
    months = sorted(joined["_month"].unique())
    print(f"Joined bars: {len(joined)}  months present: {months}")

    rows = []
    for c in oi_cols:
        full = ic_test(joined[c], y_all)
        captured = abs(full["spearman"]) * dstd * 100 if np.isfinite(full["spearman"]) else np.nan

        # stability across the two available sub-periods (June / July)
        sub_results = []
        for mo in months:
            sub_mask = joined["_month"] == mo
            r = ic_test(joined.loc[sub_mask, c], y_all.loc[sub_mask])
            sub_results.append(r)
        same_sign = [np.sign(r["spearman"]) == np.sign(full["spearman"])
                     for r in sub_results if np.isfinite(r["spearman"])]
        stab = f"{sum(same_sign)}/{len(months)}"

        rows.append(dict(feature=c, n=full["n"], spearman=full["spearman"], p=full["p"],
                          captured_pct=captured, stab=stab,
                          n_jun=sub_results[0]["n"] if len(sub_results) > 0 else None,
                          spear_jun=sub_results[0]["spearman"] if len(sub_results) > 0 else None,
                          n_jul=sub_results[1]["n"] if len(sub_results) > 1 else None,
                          spear_jul=sub_results[1]["spearman"] if len(sub_results) > 1 else None))

    df = pd.DataFrame(rows)
    df["q_bh"] = bh_fdr(df["p"].fillna(1.0).values)
    df["clears"] = (df["q_bh"] < Q_GATE) & (df["captured_pct"] > FRICTION_FLOOR_PCT)
    df = df.sort_values("q_bh")

    pd.set_option("display.width", 160)
    print("\n" + "=" * 100)
    print(f"  FORMAL GATE (reused from validate_gex_kaggle.py): q_bh < {Q_GATE}  AND  captured_pct > {FRICTION_FLOOR_PCT}")
    print("=" * 100)
    print(df[["feature", "n", "spearman", "p", "q_bh", "captured_pct", "stab", "clears"]].to_string(index=False))
    print()
    print(f"Features clearing the gate: {int(df['clears'].sum())} / {len(df)}")
    df.to_csv(ROOT / "logs/oi_archive_formal_gate.csv", index=False)
    print("Wrote logs/oi_archive_formal_gate.csv")


if __name__ == "__main__":
    main()
