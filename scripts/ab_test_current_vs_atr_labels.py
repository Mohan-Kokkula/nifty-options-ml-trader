"""
ab_test_current_vs_atr_labels.py — Isolated A/B experiment: does the
ATR-based dynamic barrier label actually improve the ML system, holding
everything else fixed?

READ-ONLY / EXPERIMENT-ONLY. Does not modify core/ml_engine.py, does not
touch models/nifty_v9_*.pkl, does not change create_labels() in
scripts/train_model_v9.py. Trains two throwaway model sets in-memory only.

Model A: production XGBoost+LightGBM ensemble (backtest_threshold_sweep.
         train_fold, unmodified hyperparameters) trained on the CURRENT
         production label (feat['label'] as build_frame() already computes
         it via create_labels()).
Model B: the IDENTICAL ensemble code (same train_fold call, same
         hyperparameters, same sample-weighting logic, which is internal
         to train_fold and auto-adapts to whatever label column it's given)
         trained on the ATR-based dynamic barrier label — reused BY IMPORT
         from scripts/label_redesign_study.py::label_atr_barrier(), so it
         is the exact same label definition already diagnosed in that
         study, not a re-derivation.

Everything else is held fixed and reused unmodified:
  - Features: backtest_threshold_sweep.build_frame() (leak-fixed V9 frame,
    full 209,076-row dataset)
  - Train/val/test splits: identical 8-fold quarterly walk-forward
    (2024-07-01 -> 2026-05-01, 3-day embargo) — same FOLDS as the CatBoost
    comparison and phase6_threshold_optimizer.py
  - Hyperparameters: backtest_threshold_sweep.train_fold's XGB/LGB configs,
    untouched
  - Thresholds: threshold_opt.PRODUCTION_BASELINE (call=0.32, put=0.25,
    skip_ceil=0.65, min_edge=0.05)
  - Option-premium simulator: backtest_options.simulate_trades /
    build_iv_map, untouched

Output: per-fold + aggregate AUC/Precision/Recall/F1/PF/Expectancy/
WinRate/MaxDD/TradeCount/TrainingTime, paired bootstrap on fold P&L,
McNemar on realized-trade correctness, calibration (ECE/Brier), class
balance, label stability, and feature-importance stability (Jaccard
overlap + Spearman rank correlation of XGBoost gain-importances across
folds).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path
from datetime import date, timedelta
from itertools import combinations

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
from scripts.label_redesign_study import label_atr_barrier

CALL_THR = PRODUCTION_BASELINE.call_thr
PUT_THR = PRODUCTION_BASELINE.put_thr
SKIP_CEIL = PRODUCTION_BASELINE.skip_ceil

OUT_JSON = ROOT / "logs" / "ab_current_vs_atr_labels.json"


def classification_metrics(y_true, probas):
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    y_pred = probas.argmax(axis=1)
    try:
        auc = roc_auc_score(y_true, probas, multi_class="ovr", average="macro",
                             labels=[0, 1, 2])
    except Exception:
        auc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return dict(auc=float(auc), precision=float(p), recall=float(r), f1=float(f1))


def brier_score_multiclass(y_true, probas):
    onehot = np.zeros_like(probas)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probas - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, probas, n_bins=10):
    conf = probas.max(axis=1)
    pred = probas.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return dict(mean_diff=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    p_value=float("nan"), n=0)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_pos = (boot_means <= 0).mean()
    p_neg = (boot_means >= 0).mean()
    p_value = float(min(2 * min(p_pos, p_neg), 1.0))
    return dict(mean_diff=mean_diff, ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                p_value=p_value, n=n)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    n = len(correct_a)
    if n == 0:
        return dict(n_pairs=0, b01=0, b10=0, statistic=float("nan"), p_value=float("nan"))
    b01 = int(((~correct_a) & correct_b).sum())
    b10 = int((correct_a & (~correct_b)).sum())
    if b01 + b10 == 0:
        return dict(n_pairs=n, b01=b01, b10=b10, statistic=0.0, p_value=1.0)
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    try:
        from scipy.stats import chi2
        p_value = float(1 - chi2.cdf(stat, df=1))
    except ImportError:
        import math
        z = math.sqrt(max(stat, 0.0))
        p_value = float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
    return dict(n_pairs=n, b01=b01, b10=b10, statistic=float(stat), p_value=p_value)


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    t_start = time.time()
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    # ---- ATR-barrier labels: reused verbatim from the prior study ----
    closes = feat["close"].values.astype(np.float64)
    highs = feat["high"].values.astype(np.float64)
    lows = feat["low"].values.astype(np.float64)
    prev_close = np.roll(closes, 1); prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr10 = pd.Series(tr, index=feat.index).rolling(10).mean().values
    atr_result = label_atr_barrier(feat, closes, highs, lows, atr10)
    atr_labels = atr_result["labels"]

    feat_current = feat                       # already has the production label
    feat_atr = feat.copy()
    feat_atr["label"] = atr_labels            # ONLY the label column differs

    print(f"Current label balance: {pd.Series(feat_current['label']).value_counts(normalize=True).sort_index().to_dict()}")
    print(f"ATR-barrier label balance: {pd.Series(feat_atr['label']).value_counts(normalize=True).sort_index().to_dict()}")

    folds = []
    f0, f_end = date(2024, 7, 1), date(2026, 5, 1)
    while f0 < f_end:
        f1 = min(add_months(f0, 3), f_end)
        folds.append((f0, f1))
        f0 = f1
    print(f"\n{len(folds)} folds (identical to the CatBoost comparison / phase6_threshold_optimizer.FOLDS)\n")

    per_fold = {}
    net_a_by_fold, net_b_by_fold = {}, {}
    paired_correct_a, paired_correct_b = [], []
    all_probas_a, all_y_a = [], []
    all_probas_b, all_y_b = [], []
    top20_a_by_fold, top20_b_by_fold = {}, {}
    full_importance_a_by_fold, full_importance_b_by_fold = {}, {}

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = (feat.index.date < cutoff)
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        print(f"[Fold {k}/{len(folds)}] train<{cutoff} ({n_tr:,}) -> test {a}..{b} ({n_te:,})")
        if n_tr < 5000 or n_te < 50:
            print("  skipped"); continue

        test = feat[te_mask]   # OHLC/IV-relevant columns identical for both variants

        # ---- Model A: current label ----
        t0 = time.time()
        models_a, sc_a = train_fold(feat_current, fcols, tr_mask)
        fit_time_a = time.time() - t0
        if models_a is None:
            print("  skipped (train_fold None)"); continue
        Xte_a = sc_a.transform(test[fcols].values)
        p_xgb_a = _proba3(models_a["xgb"], Xte_a)
        p_lgb_a = _proba3(models_a["lgb"], Xte_a)
        p_ens_a = (p_xgb_a + p_lgb_a) / 2
        y_true_a = feat_current.loc[test.index, "label"].values

        # ---- Model B: ATR-barrier label ----
        t0 = time.time()
        models_b, sc_b = train_fold(feat_atr, fcols, tr_mask)
        fit_time_b = time.time() - t0
        if models_b is None:
            print("  skipped (train_fold None, ATR variant)"); continue
        Xte_b = sc_b.transform(test[fcols].values)
        p_xgb_b = _proba3(models_b["xgb"], Xte_b)
        p_lgb_b = _proba3(models_b["lgb"], Xte_b)
        p_ens_b = (p_xgb_b + p_lgb_b) / 2
        y_true_b = feat_atr.loc[test.index, "label"].values

        print(f"  A(current): fit={fit_time_a:.1f}s   B(atr): fit={fit_time_b:.1f}s")

        cls_a = classification_metrics(y_true_a, p_ens_a)
        cls_b = classification_metrics(y_true_b, p_ens_b)

        sig_a = signals_from_probas(p_ens_a, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_b = signals_from_probas(p_ens_b, CALL_THR, PUT_THR, SKIP_CEIL)

        tdf_a = simulate_trades(test, sig_a, p_ens_a, iv, exp)
        tdf_b = simulate_trades(test, sig_b, p_ens_b, iv, exp)
        m_a = trade_metrics(tdf_a) or dict(trades=0, pf=float("nan"), win_rate=float("nan"), ev=0.0, max_dd=0.0, net=0.0)
        m_b = trade_metrics(tdf_b) or dict(trades=0, pf=float("nan"), win_rate=float("nan"), ev=0.0, max_dd=0.0, net=0.0)

        net_a_by_fold[k] = float(m_a.get("net", 0.0))
        net_b_by_fold[k] = float(m_b.get("net", 0.0))

        # ---- Paired correctness for McNemar: bars where BOTH fired, judged
        # against the REAL, OBJECTIVE realized P&L (net_option > 0), not
        # against either model's own possibly-flawed label -- this is the
        # only fair, shared ground truth between two models trained on
        # different targets. ----
        both_traded = (sig_a != 2) & (sig_b != 2)
        if both_traded.any():
            idx_pos = np.where(both_traded)[0]
            test_idx = test.index[idx_pos]
            # tdf_a/tdf_b are trade-level (not full-bar) frames. 'time' is
            # the EXIT timestamp (see backtest_options._close_position) --
            # the correct join key back to the entry bar is 'entry_time'.
            if len(tdf_a) and "entry_time" in tdf_a.columns:
                pnl_map_a = dict(zip(tdf_a["entry_time"], tdf_a["net_option"]))
            else:
                pnl_map_a = {}
            if len(tdf_b) and "entry_time" in tdf_b.columns:
                pnl_map_b = dict(zip(tdf_b["entry_time"], tdf_b["net_option"]))
            else:
                pnl_map_b = {}
            c_a, c_b = [], []
            for ts in test_idx:
                if ts in pnl_map_a and ts in pnl_map_b:
                    c_a.append(pnl_map_a[ts] > 0)
                    c_b.append(pnl_map_b[ts] > 0)
            if c_a:
                paired_correct_a.append(np.array(c_a))
                paired_correct_b.append(np.array(c_b))

        all_probas_a.append(p_ens_a); all_y_a.append(y_true_a)
        all_probas_b.append(p_ens_b); all_y_b.append(y_true_b)

        # ---- Feature importance (XGBoost gain-based) ----
        imp_a = models_a["xgb"].feature_importances_
        imp_b = models_b["xgb"].feature_importances_
        order_a = np.argsort(imp_a)[::-1][:20]
        order_b = np.argsort(imp_b)[::-1][:20]
        top20_a_by_fold[k] = [fcols[i] for i in order_a]
        top20_b_by_fold[k] = [fcols[i] for i in order_b]
        full_importance_a_by_fold[k] = imp_a
        full_importance_b_by_fold[k] = imp_b

        fold_key = f"fold{k}"
        per_fold[fold_key] = {
            "test_start": a.isoformat(), "test_end": b.isoformat(),
            "n_train": n_tr, "n_test": n_te,
            "A_CurrentLabel": {
                **cls_a, "trade_count": int(m_a.get("trades", 0)),
                "win_rate": float(m_a.get("win_rate", float("nan"))) / 100.0 if m_a.get("win_rate") is not None else float("nan"),
                "pf": float(m_a.get("pf", float("nan"))),
                "expectancy": float(m_a.get("ev", 0.0)),
                "max_drawdown_pts": float(m_a.get("max_dd", 0.0)),
                "net_pts": float(m_a.get("net", 0.0)),
                "fit_time_s": round(fit_time_a, 2),
            },
            "B_ATRBarrierLabel": {
                **cls_b, "trade_count": int(m_b.get("trades", 0)),
                "win_rate": float(m_b.get("win_rate", float("nan"))) / 100.0 if m_b.get("win_rate") is not None else float("nan"),
                "pf": float(m_b.get("pf", float("nan"))),
                "expectancy": float(m_b.get("ev", 0.0)),
                "max_drawdown_pts": float(m_b.get("max_dd", 0.0)),
                "net_pts": float(m_b.get("net", 0.0)),
                "fit_time_s": round(fit_time_b, 2),
            },
        }
        with open(OUT_JSON, "w") as fh:
            json.dump({"per_fold": per_fold, "status": "in_progress", "folds_done": k},
                       fh, indent=2, default=str)

    # -----------------------------------------------------------------
    def agg_stats(metric_key, model_key, higher_is_better=True):
        vals = []
        for fk, fv in per_fold.items():
            v = fv[model_key].get(metric_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append((fk, v))
        if not vals:
            return dict(mean=float("nan"), std=float("nan"), worst=None, best=None)
        arr = np.array([v for _, v in vals], dtype=np.float64)
        order = np.argsort(arr) if higher_is_better else np.argsort(-arr)
        worst_fk, worst_v = vals[order[0]]
        best_fk, best_v = vals[order[-1]]
        return dict(mean=float(arr.mean()), std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    worst={"fold": worst_fk, "value": float(worst_v)},
                    best={"fold": best_fk, "value": float(best_v)})

    metric_defs = [
        ("auc", True), ("precision", True), ("recall", True), ("f1", True),
        ("pf", True), ("expectancy", True), ("win_rate", True),
        ("max_drawdown_pts", False), ("trade_count", True), ("fit_time_s", False),
    ]
    summary = {"A_CurrentLabel": {}, "B_ATRBarrierLabel": {}}
    for mk, higher in metric_defs:
        summary["A_CurrentLabel"][mk] = agg_stats(mk, "A_CurrentLabel", higher)
        summary["B_ATRBarrierLabel"][mk] = agg_stats(mk, "B_ATRBarrierLabel", higher)

    # ---- Statistical significance ----
    common_folds = sorted(set(net_a_by_fold) & set(net_b_by_fold))
    diffs = np.array([net_b_by_fold[f] - net_a_by_fold[f] for f in common_folds])
    bootstrap_result = paired_bootstrap(diffs)

    if paired_correct_a:
        mcnemar_result = mcnemar_test(np.concatenate(paired_correct_a), np.concatenate(paired_correct_b))
    else:
        mcnemar_result = dict(n_pairs=0, b01=0, b10=0, statistic=float("nan"),
                               p_value=float("nan"), note="no paired same-bar trades found")

    n_folds_b_better = int((diffs > 0).sum())
    n_folds_total = len(diffs)

    # ---- Calibration (pooled across folds, each model vs its OWN label) ----
    probas_a_pool = np.concatenate(all_probas_a); y_a_pool = np.concatenate(all_y_a)
    probas_b_pool = np.concatenate(all_probas_b); y_b_pool = np.concatenate(all_y_b)
    calibration = {
        "A_CurrentLabel": {
            "brier_score": brier_score_multiclass(y_a_pool, probas_a_pool),
            "ece": expected_calibration_error(y_a_pool, probas_a_pool),
        },
        "B_ATRBarrierLabel": {
            "brier_score": brier_score_multiclass(y_b_pool, probas_b_pool),
            "ece": expected_calibration_error(y_b_pool, probas_b_pool),
        },
    }

    # ---- Class balance (pooled test-set) ----
    class_balance = {
        "A_CurrentLabel": pd.Series(y_a_pool).value_counts(normalize=True).sort_index().to_dict(),
        "B_ATRBarrierLabel": pd.Series(y_b_pool).value_counts(normalize=True).sort_index().to_dict(),
    }

    # ---- Label stability (bar-to-bar flip rate, full dataset) ----
    label_stability = {
        "A_CurrentLabel_flip_rate": float((feat_current["label"].values[1:] != feat_current["label"].values[:-1]).mean()),
        "B_ATRBarrierLabel_flip_rate": float((atr_labels[1:] != atr_labels[:-1]).mean()),
    }

    # ---- Feature importance stability across folds ----
    def importance_stability(top20_by_fold, full_imp_by_fold):
        fold_ids = sorted(top20_by_fold.keys())
        jaccards = [jaccard(top20_by_fold[i], top20_by_fold[j])
                    for i, j in combinations(fold_ids, 2)]
        spearmans = []
        for i, j in combinations(fold_ids, 2):
            from scipy.stats import spearmanr
            rho, _ = spearmanr(full_imp_by_fold[i], full_imp_by_fold[j])
            if not np.isnan(rho):
                spearmans.append(rho)
        return dict(
            mean_top20_jaccard=float(np.mean(jaccards)) if jaccards else float("nan"),
            mean_spearman_rank_corr=float(np.mean(spearmans)) if spearmans else float("nan"),
        )

    feat_importance_stability = {
        "A_CurrentLabel": importance_stability(top20_a_by_fold, full_importance_a_by_fold),
        "B_ATRBarrierLabel": importance_stability(top20_b_by_fold, full_importance_b_by_fold),
    }

    verdict = {
        "folds_where_B_net_pnl_higher": n_folds_b_better,
        "folds_total": n_folds_total,
        "bootstrap_p_value": bootstrap_result["p_value"],
        "mcnemar_p_value": mcnemar_result["p_value"],
        "statistically_significant_at_0.05": bool(
            bootstrap_result["p_value"] < 0.05 or mcnemar_result["p_value"] < 0.05
        ),
        "consistent_majority_improvement": bool(
            n_folds_total > 0 and n_folds_b_better > n_folds_total / 2
        ),
    }
    verdict["recommend_replacing_current_label"] = bool(
        verdict["consistent_majority_improvement"] and verdict["statistically_significant_at_0.05"]
    )

    final = {
        "status": "complete",
        "thresholds_used": PRODUCTION_BASELINE.to_dict(),
        "folds": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in folds],
        "per_fold": per_fold,
        "summary": summary,
        "paired_bootstrap_net_pnl": bootstrap_result,
        "mcnemar_paired_correctness_vs_realized_pnl": mcnemar_result,
        "calibration": calibration,
        "class_balance": class_balance,
        "label_stability": label_stability,
        "feature_importance_stability": feat_importance_stability,
        "verdict": verdict,
        "total_wall_time_s": round(time.time() - t_start, 1),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(final, fh, indent=2, default=str)

    print_report(final)


def print_report(final):
    print("\n" + "=" * 100)
    print("  A/B EXPERIMENT: CURRENT LABEL vs ATR-BASED DYNAMIC BARRIER LABEL")
    print("  (identical model, features, splits, hyperparameters, thresholds, simulator)")
    print("=" * 100)
    for fk, fv in final["per_fold"].items():
        print(f"\n--- {fk} ({fv['test_start']}..{fv['test_end']}, train={fv['n_train']:,} test={fv['n_test']:,}) ---")
        for model in ("A_CurrentLabel", "B_ATRBarrierLabel"):
            m = fv[model]
            print(f"  {model:<18} AUC={m['auc']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} | "
                  f"PF={m['pf']:.3f} Exp={m['expectancy']:+.2f} WR={m['win_rate']*100:.1f}% "
                  f"DD={m['max_drawdown_pts']:.0f} N={m['trade_count']} | fit={m['fit_time_s']:.1f}s")

    print("\n" + "=" * 100 + "\n  SUMMARY (mean / std / worst / best)\n" + "=" * 100)
    for model in ("A_CurrentLabel", "B_ATRBarrierLabel"):
        print(f"\n{model}:")
        for mk in ("auc", "precision", "recall", "f1", "pf", "expectancy", "win_rate",
                   "max_drawdown_pts", "trade_count", "fit_time_s"):
            s = final["summary"][model][mk]
            if s["worst"] is None:
                print(f"  {mk:<20} no valid folds"); continue
            print(f"  {mk:<20} mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"worst={s['worst']['value']:.4f} ({s['worst']['fold']})  best={s['best']['value']:.4f} ({s['best']['fold']})")

    print("\n" + "=" * 100 + "\n  STATISTICAL SIGNIFICANCE\n" + "=" * 100)
    bs = final["paired_bootstrap_net_pnl"]
    mn = final["mcnemar_paired_correctness_vs_realized_pnl"]
    print(f"Paired bootstrap (10,000 resamples) on per-fold net P&L (B - A):")
    print(f"  mean diff = {bs['mean_diff']:+.2f} pts  95% CI [{bs['ci_lo']:+.2f}, {bs['ci_hi']:+.2f}]  p = {bs['p_value']:.4f}  (n={bs['n']})")
    print(f"\nMcNemar's test (paired same-bar trades, correctness = realized P&L > 0):")
    print(f"  n_pairs={mn.get('n_pairs',0)}  b01(A wrong/B right)={mn.get('b01',0)}  "
          f"b10(A right/B wrong)={mn.get('b10',0)}  statistic={mn.get('statistic',float('nan')):.4f}  p={mn.get('p_value',float('nan')):.4f}")

    print("\n" + "=" * 100 + "\n  CALIBRATION / CLASS BALANCE / STABILITY\n" + "=" * 100)
    for model in ("A_CurrentLabel", "B_ATRBarrierLabel"):
        c = final["calibration"][model]
        print(f"{model}: Brier={c['brier_score']:.4f}  ECE={c['ece']:.4f}  "
              f"class_balance={final['class_balance'][model]}")
    print(f"\nLabel bar-to-bar flip rate: A={final['label_stability']['A_CurrentLabel_flip_rate']*100:.1f}%  "
          f"B={final['label_stability']['B_ATRBarrierLabel_flip_rate']*100:.1f}%")
    for model in ("A_CurrentLabel", "B_ATRBarrierLabel"):
        fis = final["feature_importance_stability"][model]
        print(f"Feature-importance stability [{model}]: "
              f"mean_top20_Jaccard={fis['mean_top20_jaccard']:.3f}  "
              f"mean_Spearman_rank_corr={fis['mean_spearman_rank_corr']:.3f}")

    v = final["verdict"]
    print("\n" + "=" * 100 + "\n  VERDICT\n" + "=" * 100)
    print(f"Folds where B (ATR label) net P&L > A (current label): {v['folds_where_B_net_pnl_higher']}/{v['folds_total']}")
    print(f"Statistically significant at alpha=0.05: {v['statistically_significant_at_0.05']} "
          f"(bootstrap p={bs['p_value']:.4f}, McNemar p={mn.get('p_value', float('nan')):.4f})")
    print(f"Consistent majority improvement across folds: {v['consistent_majority_improvement']}")
    print(f"\n>>> RECOMMEND REPLACING CURRENT LABEL WITH ATR-BARRIER LABEL: "
          f"{v['recommend_replacing_current_label']} <<<")
    if not v["recommend_replacing_current_label"]:
        print("    (Per instructions: do not recommend replacement based on cleaner labels alone -- "
              "only on statistically significant trading-metric improvement across most folds.)")
    print(f"\nTotal wall time: {final['total_wall_time_s']:.0f}s")
    print(f"Full results: {OUT_JSON}")


if __name__ == "__main__":
    main()
