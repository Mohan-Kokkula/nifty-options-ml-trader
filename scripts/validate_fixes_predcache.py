"""
validate_fixes_predcache.py — Step 1 of the 10-year fix validation.

Builds the EXACT V10 feature matrix over the full 2015-2026 5-min history
(same pipeline as train_model_v10.main steps 1-4), runs the DEPLOYED V10
ensemble over every bar, and caches per-bar probabilities + the auxiliary
columns the gate simulator needs.

Output: data/validation_predcache.parquet
    ts, open, high, low, close, call_p, put_p, skip_p, rsi14
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("predcache")

from scripts.train_model_v9 import (                          # noqa: E402
    load_5min, load_higher, load_vix, load_futures,
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix, features_futures,
    resample_from_5min, merge_htf, merge_daily_onto_5min,
    merge_vix_onto_5min, merge_futures_onto_5min, add_intraday_context,
)
from scripts.train_model_v10 import (                         # noqa: E402
    load_option_features, compute_daily_option_context,
    merge_option_context_onto_5min,
)

CSV5   = ROOT / "data/nifty_5min.csv"
CSV15  = ROOT / "data/nifty_15min.csv"
CSV30  = ROOT / "data/nifty_30min.csv"
CSV60  = ROOT / "data/nifty_60min.csv"
CSVDAY = ROOT / "data/nifty_day.csv"
CSVVIX = ROOT / "data/india_vix.csv"
CSVFUT = ROOT / "data/nifty_fut_5min.csv"   # absent → spot proxy (same as live)
V10CSV = ROOT / "data/v10_training_features.csv"

OUT = ROOT / "data/validation_predcache.parquet"


def main():
    log.info("[1/4] Option chain context (zero/NaN before 2024-05, as in prod)...")
    bhav = load_option_features(str(V10CSV))
    opt_ctx = compute_daily_option_context(bhav) if not bhav.empty else pd.DataFrame()

    log.info("[2/4] Spot data + V9 features...")
    df_fut = load_futures(str(CSVFUT)) if CSVFUT.exists() else pd.DataFrame()
    feat_fut = features_futures(df_fut) if not df_fut.empty else pd.DataFrame()
    use_real_vwap = not feat_fut.empty
    log.info(f"  VWAP source: {'futures' if use_real_vwap else 'SPOT PROXY (matches live)'}")

    df5 = load_5min(str(CSV5))
    df15 = load_higher(str(CSV15), "15-min") if CSV15.exists() else resample_from_5min(df5, "15min")
    # extend stored 15m with any newer 5m bars (same as training)
    if CSV15.exists():
        delta = df5[df5.index > df15.index[-1]]
        re15 = resample_from_5min(delta, "15min")
        if not re15.empty:
            df15 = pd.concat([df15, re15]).sort_index()
    df30 = load_higher(str(CSV30), "30-min") if CSV30.exists() else resample_from_5min(df5, "30min")
    df60 = load_higher(str(CSV60), "60min") if CSV60.exists() else resample_from_5min(df5, "60min")
    if CSVDAY.exists():
        df_day = load_higher(str(CSVDAY), "Daily")
        df_day = df_day[~df_day.index.duplicated(keep="last")]
    else:
        df_day = resample_from_5min(df5, "D")

    feat15, feat30 = features_15min(df15), features_30min(df30)
    feat60, feat_day = features_60min(df60), features_daily(df_day)
    df_vix = load_vix(str(CSVVIX))
    feat_vix = features_vix(df_vix) if not df_vix.empty else pd.DataFrame()

    df_feat = features_5min(df5)
    df_feat = merge_htf(df_feat, feat15, "15m")
    df_feat = merge_htf(df_feat, feat30, "30m")
    df_feat = merge_htf(df_feat, feat60, "60m")
    df_feat = merge_daily_onto_5min(df_feat, feat_day)
    if not feat_vix.empty:
        df_feat = merge_vix_onto_5min(df_feat, feat_vix)
    if use_real_vwap:
        df_feat = merge_futures_onto_5min(df_feat, feat_fut)
    df_feat = add_intraday_context(df_feat, has_real_futures=use_real_vwap)
    df_feat = merge_option_context_onto_5min(df_feat, opt_ctx)
    log.info(f"  Feature matrix: {df_feat.shape}")

    log.info("[3/4] Predicting: V9 over full history (primary) + V10 over "
             "its native 2024-05+ window (secondary)...")

    def _predict(tag, mp, sp, fp):
        models = joblib.load(ROOT / mp)
        scaler = joblib.load(ROOT / sp)
        fcols = joblib.load(ROOT / fp)
        for c in [c for c in fcols if c not in df_feat.columns]:
            df_feat[c] = 0.0
        sub = df_feat.dropna(subset=fcols)
        log.info(f"  [{tag}] rows with full features: {len(sub):,} / {len(df_feat):,} "
                 f"({sub.index[0].date()} → {sub.index[-1].date()})")
        X = sub[fcols].values
        probs = np.zeros((len(sub), 3))
        CH = 50_000
        for i in range(0, len(sub), CH):
            Xs = scaler.transform(X[i:i + CH])
            probs[i:i + CH] = np.mean(
                [m.predict_proba(Xs) for m in models.values()], axis=0)
            log.info(f"  [{tag}] predicted {min(i + CH, len(sub)):,}/{len(sub):,}")
        return pd.DataFrame(
            {f"call_{tag}": probs[:, 0], f"put_{tag}": probs[:, 1],
             f"skip_{tag}": probs[:, 2]}, index=sub.index)

    p9 = _predict("v9", "models/nifty_v9_models.pkl",
                  "models/nifty_v9_scaler.pkl", "models/feature_cols_v9.pkl")
    p10 = _predict("v10", "models/nifty_v10_models.pkl",
                   "models/nifty_v10_scaler.pkl", "models/feature_cols_v10.pkl")

    out = df_feat[["open", "high", "low", "close"]].copy()
    out["rsi14"] = df_feat["rsi14"] if "rsi14" in df_feat.columns else np.nan
    out = out.join(p9, how="inner").join(p10, how="left")
    out.index.name = "ts"

    log.info("[4/4] Saving cache (pickle)...")
    out.to_pickle(str(OUT).replace(".parquet", ".pkl"))
    log.info(f"Saved {len(out):,} rows → {OUT}")
    log.info(f"Date range: {out.index[0]} → {out.index[-1]}")
    sig = ((out.call_v9 >= 0.25) | (out.put_v9 >= 0.25)).mean()
    log.info(f"Sanity: V9 bars with any dir prob ≥0.25: {sig:.1%}")


if __name__ == "__main__":
    main()
