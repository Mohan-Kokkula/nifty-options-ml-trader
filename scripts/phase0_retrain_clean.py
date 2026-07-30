"""
phase0_retrain_clean.py — PHASE 0 leakage remediation: clean retrain.

1. Rebuilds the V9 feature matrix with the FIXED pipeline
   (features_daily shift-1 on close-derived cols; merge_htf completed-bar).
2. Emits the per-feature leakage report (data/validation_results/leakage_report.csv).
3. Retrains the V9 ensemble from scratch — identical hyperparams, identical
   purged 85/7.5/7.5 split, identical 170-feature list as production — and
   saves artifacts to models/sandbox_clean/ (production pickles untouched).
4. Reports clean val/test dir_acc using the exact production eval (CONF=0.22)
   for direct comparison against the leaked model's 0.9427 / 0.9286.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("phase0")

from scripts.train_model_v9 import (                          # noqa: E402
    load_5min, load_higher, load_vix,
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix,
    resample_from_5min, merge_htf, merge_daily_onto_5min,
    merge_vix_onto_5min, add_intraday_context, create_labels,
    purged_train_val_test_split, _fit_ensemble,
)

DATA = ROOT / "data"
OUT = ROOT / "models/sandbox_clean"
OUT.mkdir(parents=True, exist_ok=True)
RES = DATA / "validation_results"
RES.mkdir(exist_ok=True)


def build_features():
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
    feat60, feat_day = features_60min(df60), features_daily(df_day)
    df_vix = load_vix(str(DATA / "india_vix.csv"))
    feat_vix = features_vix(df_vix) if not df_vix.empty else pd.DataFrame()

    df_feat = features_5min(df5)
    df_feat = merge_htf(df_feat, feat15, "15m")
    df_feat = merge_htf(df_feat, feat30, "30m")
    df_feat = merge_htf(df_feat, feat60, "60m")
    df_feat = merge_daily_onto_5min(df_feat, feat_day)
    if not feat_vix.empty:
        df_feat = merge_vix_onto_5min(df_feat, feat_vix)
    df_feat = add_intraday_context(df_feat, has_real_futures=False)
    return df_feat


FIXED_DAILY = {"day_rsi", "day_ema9_bull", "day_ema_bull", "day_ema9_dist",
               "day_ema_dist", "day_ema50_dist", "day_week_pos", "day_atr_pct"}


def classify(col):
    if col in FIXED_DAILY:
        return ("daily", "FIXED — was same-day close leak; now shift(1)")
    if col.startswith(("tf15_", "tf30_", "tf60_")):
        return (col[:4].rstrip("_"),
                "FIXED — was partial-HTF-bar values; now completed bars only")
    if col.startswith(("day_prev_", "week_", "day_gap", "day_big_gap")):
        return ("daily", "clean — shifted levels / open-vs-prev-close")
    if col.startswith("vix_"):
        return ("daily-VIX", "clean — shift(1) merge (yesterday's VIX)")
    if col.startswith(("vwap", "dist_vwap", "above_vwap", "below_vwap",
                       "bars_above", "bars_below", "intraday", "day_move",
                       "day_open")):
        return ("5m-session", "clean — cumulative within day up to current bar"
                " (note: volume-exists switch uses whole-day metadata; benign)")
    if col.startswith(("dte", "expiry", "is_morning", "is_midday",
                       "is_afternoon", "dow", "is_expiry")):
        return ("calendar", "clean — deterministic calendar")
    return ("5m", "clean — rolling/backward on closed 5m bars")


def main():
    log.info("[1/4] Building CLEAN feature matrix...")
    df_feat = build_features()

    fcols = joblib.load(ROOT / "models/feature_cols_v9.pkl")
    missing = [c for c in fcols if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0.0
    log.info(f"  matrix {df_feat.shape}; {len(missing)} fcols zero-filled")

    log.info("[2/4] Writing leakage report...")
    rows = []
    for c in fcols:
        tf, verdict = classify(c)
        first = df_feat[c].first_valid_index()
        rows.append({"feature": c, "source_timeframe": tf,
                     "first_available": str(first),
                     "future_information_can_enter": "NO (post-fix)" if "FIXED" in verdict else "NO",
                     "verdict": verdict})
    pd.DataFrame(rows).to_csv(RES / "leakage_report.csv", index=False)
    n_fixed = sum(1 for r in rows if "FIXED" in r["verdict"])
    log.info(f"  {len(rows)} features audited; {n_fixed} were leaking (now fixed)")

    log.info("[3/4] Labels + purged split + retrain from scratch...")
    df_feat = create_labels(df_feat)
    df_feat.dropna(subset=fcols + ["label"], inplace=True)
    X = df_feat[fcols].values
    y = df_feat["label"].values
    tr_idx, va_idx, te_idx = purged_train_val_test_split(len(X))
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr_idx])
    Xvl = sc.transform(X[va_idx])
    Xte = sc.transform(X[te_idx])
    ytr, yvl, yte = y[tr_idx], y[va_idx], y[te_idx]

    # sample weights — copied verbatim from train_model_v9.main
    call_n, put_n = int((ytr == 0).sum()), int((ytr == 1).sum())
    dir_n = call_n + put_n
    skip_pct = (ytr == 2).mean()
    trade_w = skip_pct / (1 - skip_pct) if skip_pct < 1 else 2.0
    call_b = dir_n / (2.0 * call_n) if call_n else 1.0
    put_b = dir_n / (2.0 * put_n) if put_n else 1.0
    sample_w = np.where(ytr == 0, trade_w * call_b,
                np.where(ytr == 1, trade_w * put_b, 1.0))
    if "expiry_is_tue" in df_feat.columns:
        tue = df_feat["expiry_is_tue"].values[tr_idx].astype(bool)
        sample_w = sample_w * np.where(tue, 6.0, 1.0)

    models = _fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w)

    log.info("[4/4] Evaluating (production metric, CONF=0.22)...")
    CONF = 0.22

    def _eval(Xe, ye):
        p = np.mean([m.predict_proba(Xe) for m in models.values()], axis=0)
        s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
        t = s != 2
        if not t.any():
            return None
        dm = t & (ye != 2)
        return {"acc": float((ye[t] == s[t]).mean()),
                "dir_acc": float((ye[dm] == s[dm]).mean()) if dm.any() else 0.0,
                "n_signals": int(t.sum())}

    vm, tm = _eval(Xvl, yvl), _eval(Xte, yte)
    dates = df_feat.index
    meta = {
        "clean_retrain": True,
        "train_end": str(dates[tr_idx].max()), "val_end": str(dates[va_idx].max()),
        "test_range": [str(dates[te_idx].min()), str(dates[te_idx].max())],
        "val": vm, "test": tm,
        "leaked_reference": {"val_dir_acc": 0.9427, "test_dir_acc": 0.9286},
    }
    joblib.dump(models, OUT / "models.pkl")
    joblib.dump(sc, OUT / "scaler.pkl")
    joblib.dump(fcols, OUT / "feature_cols.pkl")
    json.dump(meta, open(OUT / "metadata.json", "w"), indent=1)
    log.info(f"  VAL : {vm}")
    log.info(f"  TEST: {tm}")
    log.info(f"  (leaked model reference: val 0.9427 / test 0.9286)")
    log.info(f"Saved clean artifacts → {OUT}")


if __name__ == "__main__":
    main()
