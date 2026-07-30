"""
calibration_walkforward_validation.py -- isotonic-calibration validation.

Reuses the EXISTING, ALREADY-IMPLEMENTED isotonic calibrator
(calibrators.get("isotonic") -> calibrators/isotonic.py) as a pure
post-hoc probability transform on top of the UNMODIFIED production
pipeline. Nothing else changes:

  - Features:      backtest_threshold_sweep.build_frame()      (unmodified)
  - Labels:        the same production create_labels() output  (unmodified)
  - Architecture:  backtest_threshold_sweep.train_fold()        (unmodified
                    XGB+LGB, simple mean ensemble -- identical to every
                    other experiment in this research arc and to the live
                    core/ml_engine.py ensembling)
  - Thresholds:    threshold_opt.PRODUCTION_BASELINE             (unmodified,
                    applied identically to raw AND calibrated probabilities)
  - Hyperparameters: untouched (train_fold's fixed config)

Calibrator fitting methodology (also reused, not invented): the
calibrator's own docstring specifies fit(p_oof, y_oof) -- out-of-fold
probabilities, never the outer test set. This repo already has an
established, tested procedure for generating that OOF pool:
brains._hpo.build_inner_folds(outer_tr_dates, k_inner=3), the exact
purged-embargo inner-CV boundary function used by phase4_run_single_fold.py
(the repo's own calibration harness). For each of the 3 inner folds within
an outer fold's training window, train_fold() is invoked UNCHANGED on the
inner-train mask and used to predict the inner-val chunk; the 3 chunks are
concatenated into the OOF pool. The isotonic calibrator is fit on that
pool, then .transform() is applied ONLY to the outer test-fold's already-
computed probabilities (from the outer model trained on the full outer
training window, exactly as in every prior experiment). The outer test
set is never touched during fitting.

Same 8-fold quarterly walk-forward (2024-07-01 -> 2026-05-01, 3-day
embargo) as every other experiment in this research arc.

Output: logs/calibration_walkforward_validation.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
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
    build_frame, train_fold, _proba3, signals_from_probas,
    metrics as trade_metrics, EMBARGO_DAYS, add_months,
)
from backtest_options import build_iv_map, simulate_trades
from threshold_opt import PRODUCTION_BASELINE
from brains._hpo import build_inner_folds
from calibrators import get as get_calibrator
from calibrators import top1_ece, multiclass_brier, reliability_bins

CALL_THR = PRODUCTION_BASELINE.call_thr
PUT_THR = PRODUCTION_BASELINE.put_thr
SKIP_CEIL = PRODUCTION_BASELINE.skip_ceil
K_INNER = 3
SEED = 42


def classification_metrics(y_true, p) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    pred = p.argmax(axis=1)
    out = dict(
        precision=float(precision_score(y_true, pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, pred, average="macro", zero_division=0)),
    )
    try:
        y_bin = np.zeros((len(y_true), 3))
        y_bin[np.arange(len(y_true)), y_true] = 1
        out["auc"] = float(roc_auc_score(y_bin, p, average="macro", multi_class="ovr"))
    except Exception:
        out["auc"] = float("nan")
    return out


def paired_bootstrap(a_by_fold: dict, b_by_fold: dict, n_resamples=10_000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    folds = sorted(set(a_by_fold) & set(b_by_fold))
    diffs = np.array([float(np.sum(a_by_fold[k])) - float(np.sum(b_by_fold[k])) for k in folds])
    point = float(diffs.mean())
    n = len(folds)
    if n == 0:
        return dict(n_folds=0, point_estimate=None, p_value=None)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_resamples)])
    p_pos = float((boots <= 0).mean())
    p_neg = float((boots >= 0).mean())
    p_value = float(2 * min(p_pos, p_neg))
    p_value = min(p_value, 1.0)
    return dict(n_folds=n, point_estimate=point,
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                p_value=p_value)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b01 = int(np.sum((correct_a == 0) & (correct_b == 1)))
    b10 = int(np.sum((correct_a == 1) & (correct_b == 0)))
    n = b01 + b10
    if n == 0:
        return dict(n_pairs=0, b01=0, b10=0, statistic=0.0, p_value=1.0)
    stat = ((abs(b01 - b10) - 1) ** 2) / n
    try:
        from scipy import stats as sstats
        p_value = float(1 - sstats.chi2.cdf(stat, df=1))
    except Exception:
        import math
        z = math.sqrt(max(stat, 0.0))
        p_value = float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
    return dict(n_pairs=n, b01=b01, b10=b10, statistic=float(stat), p_value=p_value)


def main():
    t_start = time.time()
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    folds = []
    f0, f_end = date(2024, 7, 1), date(2026, 5, 1)
    while f0 < f_end:
        f1 = min(add_months(f0, 3), f_end)
        folds.append((f0, f1))
        f0 = f1
    print(f"\n{len(folds)} folds (identical production walk-forward)\n")
    print(f"Thresholds (unchanged): call={CALL_THR} put={PUT_THR} skip_ceil={SKIP_CEIL}\n")

    per_fold = {}
    net_raw_by_fold, net_cal_by_fold = {}, {}
    paired_correct_raw, paired_correct_cal = [], []
    all_p_raw, all_p_cal, all_y = [], [], []

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = (feat.index.date < cutoff)
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        print(f"[Fold {k}/{len(folds)}] train<{cutoff} ({n_tr:,}) -> test {a}..{b} ({n_te:,})")
        if n_tr < 5000 or n_te < 50:
            print("  skipped"); continue

        test = feat[te_mask]

        # ---- Outer model: IDENTICAL to every other experiment this arc ----
        t0 = time.time()
        models, sc = train_fold(feat, fcols, tr_mask)
        fit_time = time.time() - t0
        if models is None:
            print("  skipped (train_fold None)"); continue
        Xte = sc.transform(test[fcols].values)
        p_xgb = _proba3(models["xgb"], Xte)
        p_lgb = _proba3(models["lgb"], Xte)
        p_raw = (p_xgb + p_lgb) / 2
        y_true = feat.loc[test.index, "label"].values

        # ---- OOF pool for calibration fitting (repo's own k_inner=3 methodology) ----
        outer_tr_dates = sorted(set(feat.index[tr_mask].date))
        inner_folds = build_inner_folds(outer_tr_dates, K_INNER)
        p_oof_pieces, y_oof_pieces = [], []
        for ii, (train_end, val_start, val_end) in enumerate(inner_folds, 1):
            inner_tr_mask = tr_mask & (feat.index.date < train_end)
            inner_val_mask = (feat.index.date >= val_start) & (feat.index.date < val_end)
            n_itr, n_ival = int(inner_tr_mask.sum()), int(inner_val_mask.sum())
            if n_itr < 5000 or n_ival < 50:
                print(f"    inner {ii}: insufficient ({n_itr}/{n_ival}), skipped")
                continue
            im, isc = train_fold(feat, fcols, inner_tr_mask)
            if im is None:
                continue
            inner_val = feat[inner_val_mask]
            Xival = isc.transform(inner_val[fcols].values)
            pv = (_proba3(im["xgb"], Xival) + _proba3(im["lgb"], Xival)) / 2
            p_oof_pieces.append(pv)
            y_oof_pieces.append(inner_val["label"].values)
        if not p_oof_pieces:
            print("  skipped (no OOF pool produced)"); continue
        p_oof = np.concatenate(p_oof_pieces, axis=0)
        y_oof = np.concatenate(y_oof_pieces, axis=0)
        print(f"  fit={fit_time:.1f}s  OOF pool n={len(y_oof):,} (k_inner={K_INNER})")

        # ---- Fit isotonic calibrator on OOF ONLY, transform outer test ----
        cal = get_calibrator("isotonic")
        cal.fit(p_oof, y_oof, seed=SEED)
        p_cal = cal.transform(p_raw)

        cls_raw = classification_metrics(y_true, p_raw)
        cls_cal = classification_metrics(y_true, p_cal)
        ece_raw = top1_ece(y_true, p_raw)
        ece_cal = top1_ece(y_true, p_cal)
        brier_raw = multiclass_brier(y_true, p_raw)
        brier_cal = multiclass_brier(y_true, p_cal)

        # ---- Signals: SAME unchanged thresholds applied to both ----
        sig_raw = signals_from_probas(p_raw, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_cal = signals_from_probas(p_cal, CALL_THR, PUT_THR, SKIP_CEIL)

        tdf_raw = simulate_trades(test, sig_raw, p_raw, iv, exp)
        tdf_cal = simulate_trades(test, sig_cal, p_cal, iv, exp)
        m_raw = trade_metrics(tdf_raw) or dict(trades=0, pf=float("nan"), win_rate=float("nan"), ev=0.0, max_dd=0.0, net=0.0)
        m_cal = trade_metrics(tdf_cal) or dict(trades=0, pf=float("nan"), win_rate=float("nan"), ev=0.0, max_dd=0.0, net=0.0)

        net_raw_by_fold[k] = float(m_raw.get("net", 0.0))
        net_cal_by_fold[k] = float(m_cal.get("net", 0.0))

        # ---- Paired correctness for McNemar (bars where BOTH fired) ----
        both_traded = (sig_raw != 2) & (sig_cal != 2)
        if both_traded.any() and len(tdf_raw) and len(tdf_cal) \
                and "entry_time" in tdf_raw.columns and "entry_time" in tdf_cal.columns:
            pnl_map_raw = dict(zip(tdf_raw["entry_time"], tdf_raw["net_option"]))
            pnl_map_cal = dict(zip(tdf_cal["entry_time"], tdf_cal["net_option"]))
            idx_pos = np.where(both_traded)[0]
            for pos in idx_pos:
                ts = test.index[pos]
                if ts in pnl_map_raw and ts in pnl_map_cal:
                    paired_correct_raw.append(1 if pnl_map_raw[ts] > 0 else 0)
                    paired_correct_cal.append(1 if pnl_map_cal[ts] > 0 else 0)

        all_p_raw.append(p_raw); all_p_cal.append(p_cal); all_y.append(y_true)

        per_fold[k] = dict(
            test_range=[str(a), str(b)], n_test=n_te, oof_pool_n=int(len(y_oof)),
            raw=dict(pf=m_raw.get("pf"), expectancy=m_raw.get("ev"), net=m_raw.get("net"),
                      win_rate=m_raw.get("win_rate"), max_dd=m_raw.get("max_dd"),
                      trades=m_raw.get("trades"), **cls_raw, ece=ece_raw, brier=brier_raw),
            calibrated=dict(pf=m_cal.get("pf"), expectancy=m_cal.get("ev"), net=m_cal.get("net"),
                             win_rate=m_cal.get("win_rate"), max_dd=m_cal.get("max_dd"),
                             trades=m_cal.get("trades"), **cls_cal, ece=ece_cal, brier=brier_cal),
        )
        print(f"  RAW: PF={m_raw.get('pf'):.3f} Net={m_raw.get('net'):+,.0f} "
              f"WR={m_raw.get('win_rate'):.3f} trades={m_raw.get('trades')} "
              f"ECE={ece_raw:.4f} Brier={brier_raw:.4f}")
        print(f"  CAL: PF={m_cal.get('pf'):.3f} Net={m_cal.get('net'):+,.0f} "
              f"WR={m_cal.get('win_rate'):.3f} trades={m_cal.get('trades')} "
              f"ECE={ece_cal:.4f} Brier={brier_cal:.4f}")

    # ---- Pooled reliability curves (before/after) ----
    p_raw_all = np.concatenate(all_p_raw, axis=0)
    p_cal_all = np.concatenate(all_p_cal, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    reliability_before = reliability_bins(y_all, p_raw_all, n_bins=10)
    reliability_after = reliability_bins(y_all, p_cal_all, n_bins=10)
    pooled_ece_before = top1_ece(y_all, p_raw_all)
    pooled_ece_after = top1_ece(y_all, p_cal_all)
    pooled_brier_before = multiclass_brier(y_all, p_raw_all)
    pooled_brier_after = multiclass_brier(y_all, p_cal_all)

    # ---- Significance tests ----
    boot = paired_bootstrap(net_cal_by_fold, net_raw_by_fold)
    mcn = mcnemar_test(np.array(paired_correct_raw), np.array(paired_correct_cal)) \
        if paired_correct_raw else dict(n_pairs=0, b01=0, b10=0, statistic=0.0, p_value=1.0)

    pf_folds_raw = [per_fold[k]["raw"]["pf"] for k in per_fold if per_fold[k]["raw"]["pf"] is not None]
    pf_folds_cal = [per_fold[k]["calibrated"]["pf"] for k in per_fold if per_fold[k]["calibrated"]["pf"] is not None]
    n_folds_cal_better_pf = sum(
        1 for k in per_fold
        if np.isfinite(per_fold[k]["calibrated"]["pf"]) and np.isfinite(per_fold[k]["raw"]["pf"])
        and per_fold[k]["calibrated"]["pf"] > per_fold[k]["raw"]["pf"])
    n_folds_total = len(per_fold)

    ece_improves = pooled_ece_after < pooled_ece_before
    brier_improves = pooled_brier_after < pooled_brier_before
    trading_significant = (boot.get("p_value") is not None and boot["p_value"] < 0.05
                            and boot.get("point_estimate", 0) > 0
                            and n_folds_cal_better_pf >= (n_folds_total * 0.5))

    if (ece_improves or brier_improves) and not trading_significant:
        verdict = ("Calibration improves probability quality (ECE/Brier) but does NOT "
                    "produce a statistically significant improvement in trading performance. "
                    "Per the pre-specified rule, calibration is NOT a production improvement.")
    elif trading_significant:
        verdict = ("Calibration produces a statistically significant improvement in trading "
                    "performance across most folds. Recommend further validation before deployment.")
    else:
        verdict = "Calibration improves neither probability quality nor trading performance."

    result = dict(
        wall_time_s=round(time.time() - t_start, 1),
        thresholds_used=dict(call_thr=CALL_THR, put_thr=PUT_THR, skip_ceil=SKIP_CEIL),
        k_inner=K_INNER, seed=SEED,
        per_fold=per_fold,
        pooled_calibration=dict(
            ece_before=pooled_ece_before, ece_after=pooled_ece_after,
            brier_before=pooled_brier_before, brier_after=pooled_brier_after,
            reliability_before=reliability_before, reliability_after=reliability_after,
        ),
        paired_bootstrap_net_pnl_cal_minus_raw=boot,
        mcnemar_paired_correctness=mcn,
        n_folds_calibrated_pf_better=n_folds_cal_better_pf,
        n_folds_total=n_folds_total,
        verdict=verdict,
    )
    with open(ROOT / "logs/calibration_walkforward_validation.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDone in {result['wall_time_s']}s.")
    print(f"Pooled ECE: {pooled_ece_before:.4f} -> {pooled_ece_after:.4f}")
    print(f"Pooled Brier: {pooled_brier_before:.4f} -> {pooled_brier_after:.4f}")
    print(f"Paired bootstrap (cal-raw net P&L): point={boot.get('point_estimate')} p={boot.get('p_value')}")
    print(f"McNemar: {mcn}")
    print(f"Folds where calibrated PF > raw PF: {n_folds_cal_better_pf}/{n_folds_total}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
