"""
forensic_fp_fn_audit.py -- Missing-information audit, forensic pass.

Reuses the EXISTING production pipeline only (build_frame/train_fold/_proba3/
signals_from_probas from backtest_threshold_sweep, simulate_trades/build_iv_map
from backtest_options, PRODUCTION_BASELINE thresholds from threshold_opt, and
label_atr_barrier/scan_forward from label_redesign_study -- the same frozen-
exit barrier scanner already validated in the label study). NO new model
architecture, NO new labels are introduced here -- this only APPLIES already-
built, already-validated code to classify errors and inspect their raw
context, per the explicit instruction "Do not propose new models."

For 4 spread walk-forward folds (1, 3, 5, 7 of the standard 8):
  - Train the production XGB+LGB ensemble (identical config) on each fold.
  - Classify every traded bar as TP (net_option>0) or FP (net_option<=0).
  - Classify every SKIP bar using the ATR-barrier oracle (frozen live exit
    TP=2xATR10/SL=6xATR10/max_hold=7, scan_forward) as TN (oracle also says
    no clean opportunity) or FN (oracle shows a clean, fast, favorable
    barrier touch the model never saw / gated out).
  - For each bucket, pull already-existing raw context columns (VIX level/
    regime/change, gap%, realized vol at multiple lookbacks, volume vs its
    own rolling mean, day_move_atr, dte, bar-of-day, expiry-day flag) and
    compare distributions FP vs TP and FN vs TN.

Output: logs/forensic_fp_fn_audit.json
"""
from __future__ import annotations
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import (
    build_frame, train_fold, _proba3, signals_from_probas, EMBARGO_DAYS,
)
from backtest_options import build_iv_map, simulate_trades
from threshold_opt import PRODUCTION_BASELINE as PB
from scripts.label_redesign_study import label_atr_barrier

CALL_THR, PUT_THR, SKIP_CEIL = PB.call_thr, PB.put_thr, PB.skip_ceil

FOLDS_ALL = [
    (date(2024, 7, 1), date(2024, 10, 1)),
    (date(2024, 10, 1), date(2025, 1, 1)),
    (date(2025, 1, 1), date(2025, 4, 1)),
    (date(2025, 4, 1), date(2025, 7, 1)),
    (date(2025, 7, 1), date(2025, 10, 1)),
    (date(2025, 10, 1), date(2026, 1, 1)),
    (date(2026, 1, 1), date(2026, 4, 1)),
    (date(2026, 4, 1), date(2026, 5, 1)),
]
USE_FOLD_IDX = [0, 2, 4, 6]  # folds 1, 3, 5, 7 (1-indexed) -- spread across regimes

CONTEXT_COLS_CANDIDATES = [
    "vix_level", "vix_regime", "vix_change1", "vix_change5", "vix_spike",
    "vix_danger", "vix_zscore_20d", "day_gap_pct", "day_move_atr",
    "day_move_pct", "rv5", "rv10", "rv20", "volume", "dte", "is_expiry",
    "tf15_bb_squeeze", "bbw", "intraday_range_atr", "_atr14_pts",
    "is_morning", "is_midday", "is_afternoon",
]


def describe(df, cols):
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) == 0:
            continue
        out[c] = {"mean": float(x.mean()), "median": float(x.median()),
                   "p90": float(x.quantile(0.9)), "n": int(len(x))}
    return out


def main():
    t_start = time.time()
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    closes = feat["close"].values.astype(np.float64)
    highs = feat["high"].values.astype(np.float64)
    lows = feat["low"].values.astype(np.float64)
    prev_close = np.roll(closes, 1); prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr10 = pd.Series(tr, index=feat.index).rolling(10).mean().values
    atr_result = label_atr_barrier(feat, closes, highs, lows, atr10)
    oracle_label = atr_result["labels"]          # 0=CALL 1=PUT 2=SKIP, oracle ground truth
    print(f"Oracle (ATR-barrier) label balance: "
          f"{pd.Series(oracle_label).value_counts(normalize=True).sort_index().to_dict()}")

    # volume z-score vs its own trailing 20-bar mean/std (leak-safe: rolling, shifted)
    vol = feat["volume"].astype(float) if "volume" in feat.columns else None
    if vol is not None:
        vmean = vol.rolling(20).mean().shift(1)
        vstd = vol.rolling(20).std().shift(1)
        feat = feat.copy()
        feat["volume_zscore_20"] = ((vol - vmean) / vstd.replace(0, np.nan)).values
        CONTEXT_COLS = CONTEXT_COLS_CANDIDATES + ["volume_zscore_20"]
    else:
        CONTEXT_COLS = CONTEXT_COLS_CANDIDATES

    per_fold = {}
    fp_rows, tp_rows, fn_rows, tn_rows = [], [], [], []

    for fi in USE_FOLD_IDX:
        a, b = FOLDS_ALL[fi]
        k = fi + 1
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = (feat.index.date < cutoff)
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        print(f"[Fold {k}/8] train<{cutoff} ({n_tr:,}) -> test {a}..{b} ({n_te:,})")
        if n_tr < 5000 or n_te < 50:
            print("  skipped"); continue

        test = feat[te_mask]
        t0 = time.time()
        models, sc = train_fold(feat, fcols, tr_mask)
        if models is None:
            print("  skipped (train_fold None)"); continue
        Xte = sc.transform(test[fcols].values)
        p_xgb = _proba3(models["xgb"], Xte)
        p_lgb = _proba3(models["lgb"], Xte)
        p_ens = (p_xgb + p_lgb) / 2
        print(f"  fit={time.time()-t0:.1f}s")

        sig = signals_from_probas(p_ens, CALL_THR, PUT_THR, SKIP_CEIL)
        tdf = simulate_trades(test, sig, p_ens, iv, exp)
        pnl_map = dict(zip(tdf["entry_time"], tdf["net_option"])) if len(tdf) and "entry_time" in tdf.columns else {}

        oracle_te = pd.Series(oracle_label, index=feat.index).loc[test.index].values
        traded_mask = (sig != 2)
        n_fp = n_tp = n_fn = n_tn = 0
        for i, ts in enumerate(test.index):
            row_ctx = test.iloc[i]
            if traded_mask[i]:
                pnl = pnl_map.get(ts)
                if pnl is None:
                    continue
                if pnl > 0:
                    n_tp += 1; tp_rows.append(row_ctx)
                else:
                    n_fp += 1; fp_rows.append(row_ctx)
            else:
                if oracle_te[i] != 2:
                    n_fn += 1; fn_rows.append(row_ctx)
                else:
                    n_tn += 1; tn_rows.append(row_ctx)

        per_fold[k] = dict(test_range=[str(a), str(b)], n_test=n_te,
                            n_traded=int(traded_mask.sum()), n_tp=n_tp, n_fp=n_fp,
                            n_fn=n_fn, n_tn=n_tn,
                            fp_rate=(n_fp / (n_tp + n_fp)) if (n_tp + n_fp) else None,
                            fn_capture_rate=(n_fn / (n_fn + n_tn)) if (n_fn + n_tn) else None)
        print(f"  TP={n_tp} FP={n_fp} FN={n_fn} TN={n_tn}")

    fp_df = pd.DataFrame(fp_rows); tp_df = pd.DataFrame(tp_rows)
    fn_df = pd.DataFrame(fn_rows); tn_df = pd.DataFrame(tn_rows)

    result = dict(
        wall_time_s=round(time.time() - t_start, 1),
        folds_used=[fi + 1 for fi in USE_FOLD_IDX],
        thresholds=dict(call_thr=CALL_THR, put_thr=PUT_THR, skip_ceil=SKIP_CEIL),
        per_fold=per_fold,
        totals=dict(n_tp=len(tp_df), n_fp=len(fp_df), n_fn=len(fn_df), n_tn=len(tn_df)),
        context_fp=describe(fp_df, CONTEXT_COLS),
        context_tp=describe(tp_df, CONTEXT_COLS),
        context_fn=describe(fn_df, CONTEXT_COLS),
        context_tn=describe(tn_df, CONTEXT_COLS),
    )
    with open("logs/forensic_fp_fn_audit.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDone in {result['wall_time_s']}s. Wrote logs/forensic_fp_fn_audit.json")
    print(f"Totals: TP={len(tp_df)} FP={len(fp_df)} FN={len(fn_df)} TN={len(tn_df)}")


if __name__ == "__main__":
    main()
