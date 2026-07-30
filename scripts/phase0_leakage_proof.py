"""
phase0_leakage_proof.py — PROOF that the accuracy collapse is caused by
leakage removal, not a coding mistake.

Method:
  1. Build the feature matrix with the FIXED pipeline (current code).
  2. Build it again with locally re-implemented ORIGINAL (leaked) versions of
     features_daily and merge_htf — byte-for-byte the pre-fix logic.
  3. Diff all 170 model features column-by-column. If the fix is correct and
     surgical, EXACTLY the flagged leaking columns differ and every other
     column is bit-identical.
  4. Show before/after values for a concrete bar.
  5. Print top-20 feature importances of the leaked production model vs the
     clean retrained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    load_5min, load_higher, load_vix,
    features_5min, features_15min, features_30min, features_60min,
    features_vix, resample_from_5min, merge_daily_onto_5min,
    merge_vix_onto_5min, add_intraday_context, merge_htf,
)

DATA = ROOT / "data"


# ── ORIGINAL (leaked) implementations, reproduced verbatim ────────────
def leaked_features_daily(df_day):
    c, h, l, o = df_day["close"], df_day["high"], df_day["low"], df_day["open"]
    out = pd.DataFrame(index=df_day.index)
    out["day_prev_high"] = h.shift(1)
    out["day_prev_low"] = l.shift(1)
    out["day_prev_close"] = c.shift(1)
    out["week_high"] = h.rolling(5).max().shift(1)
    out["week_low"] = l.rolling(5).min().shift(1)
    out["day_gap_pct"] = (o - c.shift(1)) / c.shift(1)
    out["day_gap_up"] = out["day_gap_pct"].clip(lower=0)
    out["day_gap_down"] = out["day_gap_pct"].clip(upper=0).abs()
    out["day_big_gap"] = (out["day_gap_pct"].abs() > 0.005).astype(int)
    e9 = c.ewm(span=9, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    out["day_ema9_bull"] = (e9 > e20).astype(int)          # LEAK: today's close
    out["day_ema_bull"] = (e20 > e50).astype(int)          # LEAK
    out["day_ema9_dist"] = (c - e9) / c                    # LEAK
    out["day_ema_dist"] = (c - e20) / c                    # LEAK
    out["day_ema50_dist"] = (c - e50) / c                  # LEAK
    out["day_week_pos"] = (c - out["week_low"]) / (
        out["week_high"] - out["week_low"] + 1e-9)         # LEAK
    d2 = c.diff()
    g = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["day_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))  # LEAK
    tr = pd.concat([h - l, (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    out["day_atr_pct"] = tr.rolling(14).mean() / c         # LEAK
    return out


def leaked_merge_htf(df5, htf_feat, prefix):
    # original: as-of-backward on bar-START labels (partial-bar lookahead)
    htf_reset = htf_feat.reset_index()
    htf_reset.columns = ["ts"] + list(htf_feat.columns)
    df5_reset = df5.reset_index()
    df5_reset = df5_reset.rename(columns={df5_reset.columns[0]: "ts"})
    merged = pd.merge_asof(df5_reset.sort_values("ts"),
                           htf_reset.sort_values("ts"),
                           on="ts", direction="backward")
    merged = merged.set_index("ts")
    result = df5.copy()
    for col in [c for c in merged.columns if c not in df5.columns]:
        result[col] = merged[col].values
    return result


def build(daily_fn, htf_fn):
    df5 = load_5min(str(DATA / "nifty_5min.csv"))
    df15 = load_higher(str(DATA / "nifty_15min.csv"), "15-min")
    delta = df5[df5.index > df15.index[-1]]
    re15 = resample_from_5min(delta, "15min")
    if not re15.empty:
        df15 = pd.concat([df15, re15]).sort_index()
    df30 = load_higher(str(DATA / "nifty_30min.csv"), "30-min")
    df60 = load_higher(str(DATA / "nifty_60min.csv"), "60min")
    df_day = load_higher(str(DATA / "nifty_day.csv"), "Daily")
    df_day = df_day[~df_day.index.duplicated(keep="last")]
    feat15, feat30 = features_15min(df15), features_30min(df30)
    feat60, feat_day = features_60min(df60), daily_fn(df_day)
    df_vix = load_vix(str(DATA / "india_vix.csv"))
    feat_vix = features_vix(df_vix) if not df_vix.empty else pd.DataFrame()
    m = features_5min(df5)
    m = htf_fn(m, feat15, "15m")
    m = htf_fn(m, feat30, "30m")
    m = htf_fn(m, feat60, "60m")
    m = merge_daily_onto_5min(m, feat_day)
    if not feat_vix.empty:
        m = merge_vix_onto_5min(m, feat_vix)
    m = add_intraday_context(m, has_real_futures=False)
    return m


def main():
    from scripts.train_model_v9 import features_daily as fixed_daily
    print("Building FIXED matrix (current pipeline)...")
    fixed = build(fixed_daily, merge_htf)
    print("Building LEAKED matrix (original pipeline, reproduced)...")
    leaked = build(leaked_features_daily, leaked_merge_htf)

    fcols = joblib.load(ROOT / "models/feature_cols_v9.pkl")
    idx = fixed.index.intersection(leaked.index)
    f, k = fixed.loc[idx], leaked.loc[idx]

    print(f"\n=== COLUMN DIFF over {len(idx):,} bars, {len(fcols)} model features ===")
    changed, unchanged = [], []
    for c in fcols:
        if c not in f.columns or c not in k.columns:
            changed.append((c, 1.0, np.nan)); continue
        a, b = f[c].values.astype(float), k[c].values.astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = ~both_nan & ~np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=False)
        frac = diff.mean()
        if frac > 0:
            with np.errstate(invalid="ignore"):
                mx = np.nanmax(np.abs(np.where(diff, a - b, 0)))
            changed.append((c, frac, mx))
        else:
            unchanged.append(c)
    print(f"  CHANGED by the fix : {len(changed)} features")
    print(f"  BIT-IDENTICAL      : {len(unchanged)} features")
    print(f"\n  {'feature':22s} {'%bars differing':>16s} {'max |delta|':>12s}")
    for c, fr, mx in sorted(changed, key=lambda x: -x[1]):
        print(f"  {c:22s} {fr:>15.1%} {mx:>12.4g}")

    rep = pd.DataFrame(changed, columns=["feature", "frac_bars_differing", "max_abs_delta"])
    rep.to_csv(DATA / "validation_results/leakage_proof_diff.csv", index=False)

    # ── concrete example bar ──────────────────────────────────────────
    ts = pd.Timestamp("2026-06-03 10:00:00")
    if ts not in idx:
        ts = idx[idx.get_indexer([ts], method="nearest")[0]]
    show = ["day_rsi", "day_ema_dist", "day_atr_pct", "day_week_pos",
            "tf15_adx", "tf60_adx"]
    show = [c for c in show if c in f.columns]
    print(f"\n=== EXAMPLE BAR: {ts} (NIFTY close={f.loc[ts,'close']:.2f}) ===")
    print(f"  {'feature':14s} {'LEAKED (before fix)':>20s} {'FIXED (after fix)':>20s}")
    for c in show:
        print(f"  {c:14s} {k.loc[ts, c]:>20.6f} {f.loc[ts, c]:>20.6f}")

    # ── feature importances: leaked prod model vs clean model ─────────
    LEAK_DAILY = {"day_rsi", "day_ema9_bull", "day_ema_bull", "day_ema9_dist",
                  "day_ema_dist", "day_ema50_dist", "day_week_pos", "day_atr_pct"}

    def top20(models_path, fcols_path, label):
        models = joblib.load(models_path)
        fc = joblib.load(fcols_path)
        m = models.get("xgb") or list(models.values())[0]
        imp = m.feature_importances_
        order = np.argsort(imp)[::-1][:20]
        print(f"\n=== TOP 20 FEATURE IMPORTANCES — {label} ===")
        for r, i in enumerate(order, 1):
            name = fc[i]
            tag = ("  <-- LEAKED-daily" if name in LEAK_DAILY else
                   ("  <-- was partial-bar HTF" if name.startswith(("tf15_", "tf30_", "tf60_")) else ""))
            print(f"  {r:2d}. {name:22s} {imp[i]:.4f}{tag}")
        leak_mass = sum(imp[i] for i, n in enumerate(fc) if n in LEAK_DAILY)
        htf_mass = sum(imp[i] for i, n in enumerate(fc)
                       if n.startswith(("tf15_", "tf30_", "tf60_")))
        print(f"  total importance mass: leaked-daily={leak_mass:.1%} "
              f"HTF={htf_mass:.1%}")

    top20(ROOT / "models/nifty_v9_models.pkl",
          ROOT / "models/feature_cols_v9.pkl", "LEAKED production model")
    top20(ROOT / "models/sandbox_clean/models.pkl",
          ROOT / "models/sandbox_clean/feature_cols.pkl", "CLEAN retrained model")


if __name__ == "__main__":
    main()
