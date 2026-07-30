"""
phase5_target_audit.py — Audit whether the prediction target is the bottleneck.

Trains a single XGBoost model per target formulation on the SAME features/folds
and measures which target has the highest OOS predictive signal (IC, R², AUC).

Targets tested:
  A. direction_15m           — current 3-class (CALL/PUT/SKIP)  → argmax accuracy
  B. return_15m              — regression on 15-min forward return         → R²
  C. abs_return_gt_friction  — binary: |return| > 0.12%                    → AUC
  D. trade_quality           — expected-value labels: sign(return) x (|ret|-friction)  → correlation with realized P&L
  E. realized_vol_15m        — regression on 15-bar realized vol           → R²

NO production models. NO ensembles. Just target-formulation signal audit.

Determines whether current target (A) is information-limiting relative to alternatives.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, gc, json
from pathlib import Path
from datetime import date, timedelta
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import build_frame, EMBARGO_DAYS, add_months
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score

FRICTION_PCT = 0.0012  # 0.12% per 15-min round-trip


def build_targets(feat):
    """Add all 5 target columns to feat. Uses only causal (leak-free) forward info.
    Assumes feat.index is sorted 5-min bar timestamps and 'close' is the spot."""
    df = feat.copy()
    px = df["close"]
    fwd_close = px.shift(-3)  # 15-min forward
    fwd_ret = fwd_close / px - 1
    # A: use existing "label" (already in leak-clean frame)
    df["tgt_A"] = df["label"]
    # B: continuous 15-min forward return
    df["tgt_B"] = fwd_ret
    # C: |return| > friction floor (binary)
    df["tgt_C"] = (np.abs(fwd_ret) > FRICTION_PCT).astype(int)
    # D: trade-quality score: sign(ret) * (|ret| - friction), clamped
    quality = np.sign(fwd_ret) * (np.abs(fwd_ret) - FRICTION_PCT)
    df["tgt_D"] = quality.where(np.abs(fwd_ret) > FRICTION_PCT, 0)
    # E: realized vol over next 3 bars (15-min window)
    logret = np.log(px / px.shift(1))
    df["tgt_E"] = logret.rolling(3).std().shift(-3)  # forward 15-min realized vol
    # keep only rows with all targets finite
    df = df.dropna(subset=["tgt_A", "tgt_B", "tgt_C", "tgt_D", "tgt_E"])
    return df


def _sample_weights(y_cat):
    """3-class sample weights (only used for target A)."""
    y = np.asarray(y_cat)
    skip_pct = (y == 2).mean()
    trade_w = skip_pct / max(1 - skip_pct, 1e-9)
    return np.where(y == 2, 1.0, trade_w)


def train_and_eval(feat, fcols, target_col, task, folds):
    """Walk-forward evaluate one target formulation. task in {'multi3','reg','bin'}."""
    all_pred, all_true = [], []
    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50: continue
        tr, te = feat[tr_mask], feat[te_mask]
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[fcols].values)
        Xte = sc.transform(te[fcols].values)
        ytr = tr[target_col].values
        yte = te[target_col].values

        if task == "multi3":
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                verbosity=0, n_jobs=-1)
            w = _sample_weights(ytr)
            model.fit(Xtr, ytr, sample_weight=w)
            p = model.predict_proba(Xte)
            all_pred.append(p); all_true.append(yte)
        elif task == "bin":
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                objective="binary:logistic", eval_metric="logloss",
                verbosity=0, n_jobs=-1)
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            all_pred.append(p); all_true.append(yte)
        else:  # regression
            model = xgb.XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.03,
                objective="reg:squarederror", verbosity=0, n_jobs=-1)
            model.fit(Xtr, ytr)
            p = model.predict(Xte)
            all_pred.append(p); all_true.append(yte)
        del model; gc.collect()
        print(f"    Fold {k} done (n_train={len(tr):,}, n_test={len(te):,})")

    if not all_pred: return None
    y = np.concatenate(all_true)
    p = np.concatenate(all_pred, axis=0)
    return y, p


def main():
    print("Phase 5 — Target audit (which target has strongest OOS signal?)")
    feat, fcols = build_frame()
    feat = build_targets(feat)
    print(f"After building targets: {len(feat):,} usable bars")

    folds = []; f0 = date(2024, 7, 1); f_end = date(2026, 5, 1)
    while f0 < f_end:
        f1 = min(add_months(f0, 3), f_end); folds.append((f0, f1)); f0 = f1

    results = {}

    print("\n--- Target A: direction_15m (current, multi3) ---")
    r = train_and_eval(feat, fcols, "tgt_A", "multi3", folds)
    if r is not None:
        y, p = r
        pred_class = p.argmax(axis=1)
        # accuracy (excluding SKIP-only trivial cases)
        acc = float((pred_class == y).mean())
        # directional accuracy on non-SKIP labels
        mask = y != 2
        dir_acc = float((pred_class[mask] == y[mask]).mean()) if mask.sum() else float("nan")
        results["A_direction_15m"] = {"n": int(len(y)),
                                       "argmax_acc": acc,
                                       "dir_acc_nonskip": dir_acc}
        print(f"  argmax_acc={acc:.4f}  dir_acc_nonskip={dir_acc:.4f}")

    print("\n--- Target B: return_15m (regression) ---")
    r = train_and_eval(feat, fcols, "tgt_B", "reg", folds)
    if r is not None:
        y, p = r
        r2 = r2_score(y, p)
        ic_p = stats.pearsonr(y, p)[0]
        ic_s = stats.spearmanr(y, p)[0]
        results["B_return_15m"] = {"n": int(len(y)), "r2": float(r2),
                                    "pearson_ic": float(ic_p),
                                    "spearman_ic": float(ic_s)}
        print(f"  R²={r2:.4f}  Pearson IC={ic_p:.4f}  Spearman IC={ic_s:.4f}")

    print("\n--- Target C: |return| > friction (binary) ---")
    r = train_and_eval(feat, fcols, "tgt_C", "bin", folds)
    if r is not None:
        y, p = r
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        results["C_return_gt_friction"] = {"n": int(len(y)), "auc": float(auc),
                                            "positive_rate": float(y.mean())}
        print(f"  AUC={auc:.4f}  positive_rate={y.mean():.4f}")

    print("\n--- Target D: trade_quality (regression on friction-adjusted EV) ---")
    r = train_and_eval(feat, fcols, "tgt_D", "reg", folds)
    if r is not None:
        y, p = r
        r2 = r2_score(y, p)
        ic_s = stats.spearmanr(y, p)[0]
        results["D_trade_quality"] = {"n": int(len(y)), "r2": float(r2),
                                       "spearman_ic": float(ic_s)}
        print(f"  R²={r2:.4f}  Spearman IC={ic_s:.4f}")

    print("\n--- Target E: realized_vol_15m (regression) ---")
    r = train_and_eval(feat, fcols, "tgt_E", "reg", folds)
    if r is not None:
        y, p = r
        r2 = r2_score(y, p)
        ic_s = stats.spearmanr(y, p)[0]
        results["E_realized_vol_15m"] = {"n": int(len(y)), "r2": float(r2),
                                          "spearman_ic": float(ic_s)}
        print(f"  R²={r2:.4f}  Spearman IC={ic_s:.4f}")

    with open(ROOT / "logs/phase5_target_audit.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 78)
    print("  Phase 5 — Target audit summary")
    print("=" * 78)
    print(json.dumps(results, indent=2))
    print("\n=== Interpretation guide ===")
    print("  Target A dir_acc_nonskip should baseline around 0.55-0.60 given weak features.")
    print("  If Target C AUC > 0.60 OR Target D IC > 0.10, the current 3-class target is")
    print("  substantially information-limiting relative to alternatives.")


if __name__ == "__main__":
    main()
