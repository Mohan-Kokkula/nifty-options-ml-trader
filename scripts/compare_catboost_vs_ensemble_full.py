"""
compare_catboost_vs_ensemble_full.py — Full-dataset, production-methodology
comparison: CatBoost vs. the current XGBoost+LightGBM ensemble.

READ-ONLY / ANALYSIS-ONLY. Does not modify any production file, does not
change core/ml_engine.py's live model, does not touch models/nifty_v9_*.pkl.

Reuses, by import, EXACTLY the production/canonical research pipeline:
  - backtest_threshold_sweep.build_frame()        leak-fixed V9 feature+label
                                                    frame, FULL data/nifty_5min.csv
                                                    (209,734 rows, 2015-01..2026-06),
                                                    NOT the 60k-row subsample.
  - backtest_threshold_sweep.train_fold()          production XGB+LGB per-fold
                                                    training, unchanged hyperparams
                                                    (n_estimators=700, max_depth=5,
                                                    lr=0.02, early_stopping=50, ...).
  - backtest_threshold_sweep._proba3 / EMBARGO_DAYS(=3) / add_months
  - backtest_options.simulate_trades / build_iv_map   identical option-P&L
                                                        simulation (theta, spread,
                                                        STT, brokerage, GST) — TP/SL
                                                        behavior embedded, unchanged.
  - Same 8-fold quarterly walk-forward as phase6_threshold_optimizer.FOLDS /
    phase3_catboost.py: 2024-07-01 -> 2026-05-01, 3-month steps, 3-day embargo.
  - Same production gate thresholds as threshold_opt.PRODUCTION_BASELINE /
    backtest_threshold_sweep.CONFIGS[0] "A_current":
        call_thr=0.32, put_thr=0.25, skip_ceil=0.65, min_edge=0.05
    (independently confirmed identical across all three reference sources).
  - Same ensembling method as the live core/ml_engine.py: unweighted mean of
    each model's class probabilities (np.mean(individual_probas, axis=0)).

Adds ONLY:
  - A CatBoost fold-trainer with configuration identical to the pre-existing
    phase3_catboost.py::train_cat_fold (iterations=700, depth=5, lr=0.03,
    l2_leaf_reg=3.0, early_stopping_rounds=50, MultiClass, same 95/5
    train/eval cut and identical _sample_weights logic) — no hyperparameter
    tuning performed for either model.
  - AUC / Precision / Recall / F1 classification metrics (macro, one-vs-rest
    for AUC) against the true 3-class label on each fold's test set.
  - Wall-clock training and inference timing per fold per model.
  - Paired bootstrap significance test (10,000 resamples) on the per-fold
    net-P&L difference (CatBoost - Existing_Ensemble).
  - McNemar's test on paired win/loss outcomes, restricted to bars where
    BOTH models fired a non-SKIP signal (the only bars where a genuine
    paired comparison of trade correctness exists).

Output: JSON with full per-fold detail + printed report matching the
requested format.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path
from datetime import date, timedelta

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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise SystemExit("catboost not installed — run: pip install catboost")

CALL_THR = PRODUCTION_BASELINE.call_thr
PUT_THR = PRODUCTION_BASELINE.put_thr
SKIP_CEIL = PRODUCTION_BASELINE.skip_ceil
MIN_EDGE = PRODUCTION_BASELINE.min_edge
print(f"Thresholds (unchanged, from threshold_opt.PRODUCTION_BASELINE): "
      f"call={CALL_THR} put={PUT_THR} skip_ceil={SKIP_CEIL} min_edge={MIN_EDGE}")

OUT_JSON = ROOT / "logs" / "catboost_vs_ensemble_full_wf.json"


# ---------------------------------------------------------------------------
# CatBoost fold trainer — identical config to phase3_catboost.py::train_cat_fold
# ---------------------------------------------------------------------------
def _sample_weights(sub):
    """Verbatim copy of backtest_threshold_sweep._sample_weights (not
    importable — it's a module-private helper — reproduced exactly, byte
    for byte, so CatBoost gets identical sample weighting to XGB/LGB)."""
    y = sub["label"].values
    skip_pct = (y == 2).mean()
    trade_pct = 1 - skip_pct
    trade_w = skip_pct / trade_pct if trade_pct > 0 else 2.0
    w = np.where(y == 2, 1.0, trade_w)
    try:
        if "expiry_is_tue" in sub.columns:
            tue = sub["expiry_is_tue"].values.astype(bool)
            w = w * np.where(tue, 6.0, 1.0)
    except Exception:
        pass
    try:
        bd = sub.index.date
        d_open = sub.groupby(bd)["open"].first()
        d_close = sub.groupby(bd)["close"].last()
        d_move = (d_close - d_open).abs()
        trend_days = set(d_move[d_move > 150.0].index)
        tmask = np.array([d in trend_days for d in bd], dtype=bool)
        w = w * np.where(tmask, 3.0, 1.0)
    except Exception:
        pass
    return w


def train_cat_fold(feat, fcols, train_mask):
    """Identical to phase3_catboost.py::train_cat_fold — same iterations,
    depth, lr, l2_leaf_reg, early_stopping_rounds, MultiClass loss, 95/5
    train/eval cut, StandardScaler, sample weights."""
    sub = feat[train_mask]
    if len(sub) < 5000:
        return None, None
    cut = int(len(sub) * 0.95)
    tr, ev = sub.iloc[:cut], sub.iloc[cut:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    Xev = sc.transform(ev[fcols].values)
    ytr, yev = tr["label"].values, ev["label"].values
    w = _sample_weights(tr)
    m = CatBoostClassifier(
        iterations=700, depth=5, learning_rate=0.03,
        loss_function="MultiClass", classes_count=3,
        l2_leaf_reg=3.0, random_state=42, verbose=False,
        early_stopping_rounds=50,
    )
    m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xev, yev), verbose=False)
    return m, sc


# ---------------------------------------------------------------------------
def classification_metrics(y_true, probas):
    """Macro one-vs-rest AUC + macro precision/recall/F1 (argmax predicted
    class vs true 3-class label). Returns NaNs if a class is entirely
    absent from y_true for this fold (AUC undefined in that case)."""
    y_pred = probas.argmax(axis=1)
    try:
        auc = roc_auc_score(y_true, probas, multi_class="ovr", average="macro",
                             labels=[0, 1, 2])
    except Exception:
        auc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return dict(auc=float(auc), precision=float(p), recall=float(r), f1=float(f1))


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42):
    """Paired bootstrap on an array of per-fold (CatBoost - Ensemble)
    differences. Returns (mean_diff, ci_lo, ci_hi, p_value_two_sided)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return dict(mean_diff=float("nan"), ci_lo=float("nan"),
                     ci_hi=float("nan"), p_value=float("nan"), n=0)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    # two-sided p-value: fraction of bootstrap means on the opposite side of 0
    p_pos = (boot_means <= 0).mean()
    p_neg = (boot_means >= 0).mean()
    p_value = float(2 * min(p_pos, p_neg))
    p_value = min(p_value, 1.0)
    return dict(mean_diff=mean_diff, ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                p_value=p_value, n=n)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    """McNemar's test (with continuity correction) on paired binary outcomes
    (correct/incorrect) for the SAME bars under model A vs model B."""
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    n = len(correct_a)
    if n == 0:
        return dict(n_pairs=0, b01=0, b10=0, statistic=float("nan"), p_value=float("nan"))
    b01 = int(((~correct_a) & correct_b).sum())   # A wrong, B right
    b10 = int((correct_a & (~correct_b)).sum())   # A right, B wrong
    if b01 + b10 == 0:
        return dict(n_pairs=n, b01=b01, b10=b10, statistic=0.0, p_value=1.0)
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)   # continuity-corrected chi2, df=1
    try:
        from scipy.stats import chi2
        p_value = float(1 - chi2.cdf(stat, df=1))
    except ImportError:
        # Fallback: normal approximation to chi2(1) survival function
        import math
        z = math.sqrt(max(stat, 0.0))
        p_value = float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
    return dict(n_pairs=n, b01=b01, b10=b10, statistic=float(stat), p_value=p_value)


# ---------------------------------------------------------------------------
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
    print(f"\n{len(folds)} folds (identical to phase6_threshold_optimizer.FOLDS "
          f"/ phase3_catboost.py): {folds}\n")

    per_fold_results = {}
    ens_net_by_fold = {}
    cat_net_by_fold = {}
    paired_correct_ens = []
    paired_correct_cat = []

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
        print(f"[Fold {k}/{len(folds)}] train<{cutoff} ({n_tr:,} rows) "
              f"-> test {a}..{b} ({n_te:,} rows)")
        if n_tr < 5000 or n_te < 50:
            print("  skipped (insufficient rows)")
            continue

        test = feat[te_mask]
        y_true = test["label"].values

        # ---- Existing production ensemble: XGBoost + LightGBM ----
        t0 = time.time()
        models, sc_xl = train_fold(feat, fcols, tr_mask)
        fit_time_ens = time.time() - t0
        if models is None:
            print("  skipped (train_fold returned None)")
            continue
        Xte_xl = sc_xl.transform(test[fcols].values)
        t0 = time.time()
        p_xgb = _proba3(models["xgb"], Xte_xl)
        p_lgb = _proba3(models["lgb"], Xte_xl)
        p_ens = (p_xgb + p_lgb) / 2   # matches core/ml_engine.py: np.mean(individual_probas, axis=0)
        inf_time_ens = time.time() - t0
        print(f"  Existing_Ensemble: fit={fit_time_ens:.1f}s infer={inf_time_ens*1000:.1f}ms")

        # ---- CatBoost ----
        t0 = time.time()
        cat, sc_cat = train_cat_fold(feat, fcols, tr_mask)
        fit_time_cat = time.time() - t0
        if cat is None:
            print("  CatBoost skipped (insufficient rows)")
            continue
        Xte_cat = sc_cat.transform(test[fcols].values)
        t0 = time.time()
        p_cat = _proba3(cat, Xte_cat)
        inf_time_cat = time.time() - t0
        print(f"  CatBoost:          fit={fit_time_cat:.1f}s infer={inf_time_cat*1000:.1f}ms")

        # ---- Classification metrics (unchanged labels) ----
        cls_ens = classification_metrics(y_true, p_ens)
        cls_cat = classification_metrics(y_true, p_cat)

        # ---- Trading signals under UNCHANGED thresholds ----
        sig_ens = signals_from_probas(p_ens, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_cat = signals_from_probas(p_cat, CALL_THR, PUT_THR, SKIP_CEIL)

        tdf_ens = simulate_trades(test, sig_ens, p_ens, iv, exp)
        tdf_cat = simulate_trades(test, sig_cat, p_cat, iv, exp)

        m_ens = trade_metrics(tdf_ens) or dict(trades=0, pf=float("nan"),
                                                win_rate=float("nan"), ev=0.0,
                                                max_dd=0.0, net=0.0)
        m_cat = trade_metrics(tdf_cat) or dict(trades=0, pf=float("nan"),
                                                win_rate=float("nan"), ev=0.0,
                                                max_dd=0.0, net=0.0)

        ens_net_by_fold[k] = float(m_ens.get("net", 0.0))
        cat_net_by_fold[k] = float(m_cat.get("net", 0.0))

        # ---- Paired-bar correctness for McNemar (only bars where BOTH
        # models fired a non-SKIP signal on the SAME bar -- a genuine
        # paired comparison) ----
        both_traded = (sig_ens != 2) & (sig_cat != 2)
        if both_traded.any():
            correct_ens_here = (sig_ens[both_traded] == y_true[both_traded])
            correct_cat_here = (sig_cat[both_traded] == y_true[both_traded])
            paired_correct_ens.append(correct_ens_here)
            paired_correct_cat.append(correct_cat_here)

        fold_key = f"fold{k}"
        per_fold_results[fold_key] = {
            "test_start": a.isoformat(), "test_end": b.isoformat(),
            "n_train": n_tr, "n_test": n_te,
            "Existing_Ensemble": {
                **cls_ens,
                "trade_count": int(m_ens.get("trades", 0)),
                "win_rate": float(m_ens.get("win_rate", float("nan"))) / 100.0
                            if m_ens.get("win_rate") is not None else float("nan"),
                "pf": float(m_ens.get("pf", float("nan"))),
                "expectancy": float(m_ens.get("ev", 0.0)),
                "max_drawdown_pts": float(m_ens.get("max_dd", 0.0)),
                "net_pts": float(m_ens.get("net", 0.0)),
                "fit_time_s": round(fit_time_ens, 2),
                "inference_time_ms": round(inf_time_ens * 1000, 2),
            },
            "CatBoost": {
                **cls_cat,
                "trade_count": int(m_cat.get("trades", 0)),
                "win_rate": float(m_cat.get("win_rate", float("nan"))) / 100.0
                            if m_cat.get("win_rate") is not None else float("nan"),
                "pf": float(m_cat.get("pf", float("nan"))),
                "expectancy": float(m_cat.get("ev", 0.0)),
                "max_drawdown_pts": float(m_cat.get("max_dd", 0.0)),
                "net_pts": float(m_cat.get("net", 0.0)),
                "fit_time_s": round(fit_time_cat, 2),
                "inference_time_ms": round(inf_time_cat * 1000, 2),
            },
        }
        # Checkpoint after every fold so a long run's progress is never lost.
        with open(OUT_JSON, "w") as fh:
            json.dump({"per_fold": per_fold_results, "status": "in_progress",
                       "folds_done": k}, fh, indent=2, default=str)

    # -----------------------------------------------------------------
    # Aggregate: mean / std / worst / best per metric per model
    # -----------------------------------------------------------------
    def agg_stats(metric_key, model_key, higher_is_better=True):
        vals = []
        for fk, fv in per_fold_results.items():
            v = fv[model_key].get(metric_key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append((fk, v))
        if not vals:
            return dict(mean=float("nan"), std=float("nan"),
                        worst=None, best=None)
        arr = np.array([v for _, v in vals], dtype=np.float64)
        order = np.argsort(arr) if higher_is_better else np.argsort(-arr)
        worst_fk, worst_v = vals[order[0]]
        best_fk, best_v = vals[order[-1]]
        return dict(mean=float(arr.mean()),
                    std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    worst={"fold": worst_fk, "value": float(worst_v)},
                    best={"fold": best_fk, "value": float(best_v)})

    metric_defs = [
        ("auc", True), ("precision", True), ("recall", True), ("f1", True),
        ("pf", True), ("expectancy", True), ("win_rate", True),
        ("max_drawdown_pts", False), ("trade_count", True),
        ("fit_time_s", False), ("inference_time_ms", False),
    ]
    summary = {"Existing_Ensemble": {}, "CatBoost": {}}
    for mk, higher in metric_defs:
        summary["Existing_Ensemble"][mk] = agg_stats(mk, "Existing_Ensemble", higher)
        summary["CatBoost"][mk] = agg_stats(mk, "CatBoost", higher)

    # -----------------------------------------------------------------
    # Statistical significance
    # -----------------------------------------------------------------
    common_folds = sorted(set(ens_net_by_fold) & set(cat_net_by_fold))
    diffs = np.array([cat_net_by_fold[f] - ens_net_by_fold[f] for f in common_folds])
    bootstrap_result = paired_bootstrap(diffs)

    if paired_correct_ens:
        all_correct_ens = np.concatenate(paired_correct_ens)
        all_correct_cat = np.concatenate(paired_correct_cat)
        mcnemar_result = mcnemar_test(all_correct_ens, all_correct_cat)
    else:
        mcnemar_result = dict(n_pairs=0, b01=0, b10=0, statistic=float("nan"),
                               p_value=float("nan"), note="no paired same-bar trades found")

    n_folds_cat_better = int((diffs > 0).sum())
    n_folds_total = len(diffs)

    verdict = {
        "folds_where_catboost_net_pnl_higher": n_folds_cat_better,
        "folds_total": n_folds_total,
        "bootstrap_p_value": bootstrap_result["p_value"],
        "mcnemar_p_value": mcnemar_result["p_value"],
        "statistically_significant_at_0.05": bool(
            bootstrap_result["p_value"] < 0.05 or mcnemar_result["p_value"] < 0.05
        ),
        "consistent_majority_improvement": bool(
            n_folds_total > 0 and n_folds_cat_better > n_folds_total / 2
        ),
    }
    verdict["recommend_replacing_production_model"] = bool(
        verdict["consistent_majority_improvement"]
        and verdict["statistically_significant_at_0.05"]
    )

    final = {
        "status": "complete",
        "thresholds_used": PRODUCTION_BASELINE.to_dict(),
        "folds": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in folds],
        "per_fold": per_fold_results,
        "summary": summary,
        "paired_bootstrap_net_pnl": bootstrap_result,
        "mcnemar_paired_correctness": mcnemar_result,
        "verdict": verdict,
        "total_wall_time_s": round(time.time() - t_start, 1),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(final, fh, indent=2, default=str)

    print_report(final)


def print_report(final):
    print("\n" + "=" * 100)
    print("  CATBOOST vs EXISTING (XGBoost+LightGBM) ENSEMBLE — FULL-DATASET WALK-FORWARD")
    print("=" * 100)
    for fk, fv in final["per_fold"].items():
        print(f"\n--- {fk}  ({fv['test_start']} .. {fv['test_end']}, "
              f"train={fv['n_train']:,} test={fv['n_test']:,}) ---")
        for model in ("Existing_Ensemble", "CatBoost"):
            m = fv[model]
            print(f"  {model:<18} AUC={m['auc']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} F1={m['f1']:.4f} | "
                  f"PF={m['pf']:.3f} Exp={m['expectancy']:+.2f} WR={m['win_rate']*100:.1f}% "
                  f"DD={m['max_drawdown_pts']:.0f} N={m['trade_count']} | "
                  f"fit={m['fit_time_s']:.1f}s infer={m['inference_time_ms']:.1f}ms")

    print("\n" + "=" * 100)
    print("  SUMMARY ACROSS ALL FOLDS (mean / std / worst / best)")
    print("=" * 100)
    for model in ("Existing_Ensemble", "CatBoost"):
        print(f"\n{model}:")
        for mk in ("auc", "precision", "recall", "f1", "pf", "expectancy",
                   "win_rate", "max_drawdown_pts", "trade_count",
                   "fit_time_s", "inference_time_ms"):
            s = final["summary"][model][mk]
            if s["worst"] is None:
                print(f"  {mk:<20} no valid folds")
                continue
            print(f"  {mk:<20} mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"worst={s['worst']['value']:.4f} ({s['worst']['fold']})  "
                  f"best={s['best']['value']:.4f} ({s['best']['fold']})")

    print("\n" + "=" * 100)
    print("  STATISTICAL SIGNIFICANCE")
    print("=" * 100)
    bs = final["paired_bootstrap_net_pnl"]
    mn = final["mcnemar_paired_correctness"]
    print(f"Paired bootstrap (10,000 resamples) on per-fold net P&L "
          f"(CatBoost - Existing_Ensemble):")
    print(f"  mean diff = {bs['mean_diff']:+.2f} pts  "
          f"95% CI [{bs['ci_lo']:+.2f}, {bs['ci_hi']:+.2f}]  "
          f"p = {bs['p_value']:.4f}  (n={bs['n']} folds)")
    print(f"\nMcNemar's test (paired same-bar trade correctness, "
          f"bars where both models fired):")
    print(f"  n_pairs={mn['n_pairs']}  b01(Ens wrong/Cat right)={mn['b01']}  "
          f"b10(Ens right/Cat wrong)={mn['b10']}  "
          f"statistic={mn['statistic']:.4f}  p={mn['p_value']:.4f}")

    v = final["verdict"]
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    print(f"Folds where CatBoost net P&L > Existing_Ensemble: "
          f"{v['folds_where_catboost_net_pnl_higher']}/{v['folds_total']}")
    print(f"Statistically significant at alpha=0.05: "
          f"{v['statistically_significant_at_0.05']} "
          f"(bootstrap p={bs['p_value']:.4f}, McNemar p={mn['p_value']:.4f})")
    print(f"Consistent majority improvement across folds: "
          f"{v['consistent_majority_improvement']}")
    print(f"\n>>> RECOMMEND REPLACING PRODUCTION ENSEMBLE WITH CATBOOST: "
          f"{v['recommend_replacing_production_model']} <<<")
    if not v["recommend_replacing_production_model"]:
        print("    (Per instructions: do not recommend replacement unless "
              "CatBoost consistently outperforms across most folds WITH "
              "statistical significance. That bar is not met.)")
    print(f"\nTotal wall time: {final['total_wall_time_s']:.0f}s")
    print(f"Full results written to: {OUT_JSON}")


if __name__ == "__main__":
    main()
