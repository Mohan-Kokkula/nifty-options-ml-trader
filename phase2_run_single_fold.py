"""
phase2_run_single_fold.py — Nested purged HPO for ONE outer fold, XGB only.

Called as a subprocess by phase2_xgb_hpo.py. Runs baseline eval (current
production XGB params) and Optuna HPO with K expanding-window inner folds
inside the outer training window (purged/embargoed), then retrains the
best trial on full outer_train and evaluates on outer_test.

Objective: mean(PF) + 1e-3 * mean(Sortino) over inner val folds.
Both are computed from ``simulate_trades`` on ``signals_from_probas`` at
the production thresholds — the actual production objective, not log-loss.

Outputs (under logs/phase2/fold_<n>/):
    baseline_trades.csv, baseline_metrics.json
    tuned_trades.csv,    tuned_metrics.json
    trials.parquet       (every trial: params, PF/Sortino/Sharpe/trades per inner)
    best_params.json
    best_model.xgb
    best_scaler.pkl
    summary.json         (baseline vs tuned + delta_pf)
    manifest.json        (source SHAs + config)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import (
    EMBARGO_DAYS, _proba3, _sample_weights, build_frame, signals_from_probas,
)
from backtest_options import build_iv_map, simulate_trades

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
from sklearn.preprocessing import StandardScaler


CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65
SORTINO_TIEBREAK = 1e-3
DEGENERATE_TRIAL_SCORE = -10.0

FIXED_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    verbosity=0,
    n_jobs=-1,
)

CURRENT_PARAMS = {
    **FIXED_PARAMS,
    "n_estimators": 700,
    "max_depth": 5,
    "learning_rate": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.5,
    "min_child_weight": 15,
    "gamma": 0.3,
    "reg_alpha": 1.5,
    "reg_lambda": 3.0,
    "early_stopping_rounds": 50,
}


# ---------------------------------------------------------------------------
_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase2_run_single_fold.py",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_manifest() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED if (ROOT / f).exists()}


# ---------------------------------------------------------------------------
def _fold_metrics(pnl: np.ndarray) -> dict:
    if len(pnl) == 0:
        return dict(n=0, pf=float("nan"), wr=float("nan"), net=0.0,
                    avg=0.0, dd=0.0, sharpe=float("nan"),
                    sortino=float("nan"))
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl <= 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf")
    eq = np.cumsum(pnl)
    dd = float(-(eq - np.maximum.accumulate(eq)).min())
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    sh = float(pnl.mean() / sd) if sd > 0 else float("nan")
    downside = pnl[pnl < 0]
    if downside.size and (downside ** 2).mean() > 0:
        sortino = float(pnl.mean() / np.sqrt((downside ** 2).mean()))
    else:
        sortino = float("nan")
    return dict(
        n=int(len(pnl)), pf=pf,
        wr=float((pnl > 0).mean()),
        net=float(pnl.sum()),
        avg=float(pnl.mean()),
        dd=dd, sharpe=sh, sortino=sortino,
    )


# _build_inner_folds was moved to brains/_hpo.build_inner_folds during
# Phase 3 so Phase 3's generic HPO driver could share it with Phase 2.
# We import under the original private name to keep the rest of this
# module unchanged.
from brains._hpo import build_inner_folds as _build_inner_folds  # noqa: E402


def _train_xgb(feat: pd.DataFrame, fcols: list[str], tr_mask: np.ndarray,
               params: dict, seed: int
               ) -> tuple[xgb.XGBClassifier | None, StandardScaler | None]:
    """Train XGB with an internal 5% early-stopping eval set."""
    sub = feat[tr_mask]
    if len(sub) < 5000:
        return None, None
    cut = int(len(sub) * 0.95)
    tr, ev = sub.iloc[:cut], sub.iloc[cut:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    Xev = sc.transform(ev[fcols].values)
    ytr, yev = tr["label"].values, ev["label"].values
    w = _sample_weights(tr)
    m = xgb.XGBClassifier(**params, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xev, yev)], verbose=False)
    return m, sc


def _predict_and_simulate(feat: pd.DataFrame, fcols: list[str],
                          val_mask: np.ndarray,
                          model: xgb.XGBClassifier, sc: StandardScaler,
                          iv: dict, exp: dict) -> pd.DataFrame:
    """Predict on val, generate signals, simulate trades."""
    test = feat[val_mask]
    if len(test) == 0:
        return pd.DataFrame(columns=["net_option"])
    Xte = sc.transform(test[fcols].values)
    p = _proba3(model, Xte)
    sig = signals_from_probas(p, CALL_THR, PUT_THR, SKIP_CEIL)
    return simulate_trades(test, sig, p, iv, exp)


# ---------------------------------------------------------------------------
def _make_objective(feat, fcols, outer_tr_mask, inner_folds, iv, exp,
                    seed: int):
    def objective(trial: "optuna.Trial") -> float:
        params = {
            **FIXED_PARAMS,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        pfs, sortinos, sharpes, counts = [], [], [], []
        t0 = time.time()
        for (train_end, val_start, val_end) in inner_folds:
            inner_tr_mask = outer_tr_mask & (feat.index.date < train_end)
            inner_val_mask = ((feat.index.date >= val_start)
                              & (feat.index.date < val_end))
            m, sc = _train_xgb(feat, fcols, inner_tr_mask, params, seed)
            if m is None:
                continue
            tdf = _predict_and_simulate(feat, fcols, inner_val_mask,
                                         m, sc, iv, exp)
            pnl = tdf["net_option"].values if len(tdf) else np.array([])
            met = _fold_metrics(pnl)
            pfs.append(met["pf"])
            sortinos.append(met["sortino"])
            sharpes.append(met["sharpe"])
            counts.append(met["n"])
            del m; gc.collect()
        runtime_s = time.time() - t0
        trial.set_user_attr("pfs", pfs)
        trial.set_user_attr("sortinos", sortinos)
        trial.set_user_attr("sharpes", sharpes)
        trial.set_user_attr("trade_counts", counts)
        trial.set_user_attr("runtime_s", runtime_s)
        finite_pfs = [x for x in pfs if np.isfinite(x)]
        finite_sortinos = [x for x in sortinos if np.isfinite(x)]
        if not finite_pfs:
            return DEGENERATE_TRIAL_SCORE
        mean_pf = float(np.mean(finite_pfs))
        mean_sortino = (float(np.mean(finite_sortinos))
                        if finite_sortinos else 0.0)
        return mean_pf + SORTINO_TIEBREAK * mean_sortino
    return objective


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fold_idx", type=int)
    ap.add_argument("test_start")
    ap.add_argument("test_end")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Optuna trials per outer fold (default 30).")
    ap.add_argument("--k-inner", type=int, default=3,
                    help="Number of inner CV folds (default 3).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-dir", default="logs/phase2")
    ap.add_argument("--verify-prereg", action="store_true")
    args = ap.parse_args()

    if args.verify_prereg:
        from verify_preregistration import verify
        verify()

    fold_root = ROOT / args.fold_dir / f"fold_{args.fold_idx}"
    fold_root.mkdir(parents=True, exist_ok=True)

    a_date = date.fromisoformat(args.test_start)
    b_date = date.fromisoformat(args.test_end)

    print(f"[Phase 2 / Fold {args.fold_idx}] build_frame ...")
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    cutoff = a_date - timedelta(days=EMBARGO_DAYS)
    outer_tr_mask = feat.index.date < cutoff
    outer_te_mask = ((feat.index.date >= a_date)
                     & (feat.index.date < b_date))

    if outer_tr_mask.sum() < 5000 or outer_te_mask.sum() < 50:
        print(f"[Fold {args.fold_idx}] insufficient data")
        return

    tr_max = feat.index[outer_tr_mask].max().date()
    te_min = feat.index[outer_te_mask].min().date()
    gap = (te_min - tr_max).days
    if gap < EMBARGO_DAYS:
        raise AssertionError(
            f"Purge violated: train_max={tr_max} test_min={te_min} "
            f"gap={gap} < embargo={EMBARGO_DAYS}")
    print(f"  outer train={outer_tr_mask.sum():,} bars up to {tr_max}, "
          f"test={outer_te_mask.sum():,} bars {te_min}..{b_date-timedelta(days=1)}, "
          f"gap={gap}d")

    outer_tr_dates = sorted(set(feat.index[outer_tr_mask].date))
    inner_folds = _build_inner_folds(outer_tr_dates, args.k_inner)
    print(f"  inner CV folds (K={args.k_inner}):")
    for i, (te_, vs, ve) in enumerate(inner_folds, 1):
        print(f"    inner {i}: train<{te_}  val {vs}..{ve}")

    # ---------- BASELINE ----------
    print(f"\n[Fold {args.fold_idx}] BASELINE (current production params) ...")
    t0 = time.time()
    b_m, b_sc = _train_xgb(feat, fcols, outer_tr_mask,
                            CURRENT_PARAMS, args.seed)
    if b_m is None:
        raise RuntimeError("baseline training failed (insufficient data?)")
    b_tdf = _predict_and_simulate(feat, fcols, outer_te_mask, b_m, b_sc, iv, exp)
    b_pnl = b_tdf["net_option"].values if len(b_tdf) else np.array([])
    b_met = _fold_metrics(b_pnl)
    b_met["elapsed_s"] = round(time.time() - t0, 1)
    b_met["config"] = "baseline_current_params"
    if len(b_tdf):
        b_tdf = b_tdf.copy(); b_tdf["fold"] = args.fold_idx
    b_tdf.to_csv(fold_root / "baseline_trades.csv", index=False)
    with open(fold_root / "baseline_metrics.json", "w") as fh:
        json.dump(b_met, fh, indent=2)
    print(f"  baseline: n={b_met['n']} PF={b_met['pf']:.3f} "
          f"Net=Rs.{b_met['net']:+,.0f} elapsed={b_met['elapsed_s']}s")
    del b_m, b_sc; gc.collect()

    # ---------- OPTUNA HPO ----------
    print(f"\n[Fold {args.fold_idx}] Optuna HPO (n_trials={args.n_trials}, "
          f"k_inner={args.k_inner}, seed={args.seed}) ...")
    objective = _make_objective(feat, fcols, outer_tr_mask, inner_folds,
                                 iv, exp, args.seed)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    hpo_t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    hpo_elapsed = time.time() - hpo_t0
    print(f"  HPO done in {hpo_elapsed:.1f}s.  best inner score "
          f"= {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")

    # Serialize trials
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "state": t.state.name, "value": t.value}
        row.update(t.params)
        for k in ("pfs", "sortinos", "sharpes", "trade_counts"):
            v = t.user_attrs.get(k)
            row[k] = json.dumps(v) if v is not None else None
        row["runtime_s"] = t.user_attrs.get("runtime_s")
        rows.append(row)
    trials_df = pd.DataFrame(rows)
    try:
        trials_df.to_parquet(fold_root / "trials.parquet", index=False)
    except Exception:
        trials_df.to_csv(fold_root / "trials.csv", index=False)

    with open(fold_root / "best_params.json", "w") as fh:
        json.dump(study.best_params, fh, indent=2)

    # ---------- TUNED: retrain full outer_train + evaluate outer_test ----------
    print(f"\n[Fold {args.fold_idx}] Retraining best params on full outer_train ...")
    best_params = {
        **FIXED_PARAMS,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        **study.best_params,
    }
    t0 = time.time()
    t_m, t_sc = _train_xgb(feat, fcols, outer_tr_mask, best_params, args.seed)
    t_tdf = _predict_and_simulate(feat, fcols, outer_te_mask, t_m, t_sc,
                                    iv, exp)
    t_pnl = t_tdf["net_option"].values if len(t_tdf) else np.array([])
    t_met = _fold_metrics(t_pnl)
    t_met["elapsed_s"] = round(time.time() - t0, 1)
    t_met["hpo_elapsed_s"] = round(hpo_elapsed, 1)
    t_met["config"] = "tuned_optuna"
    t_met["n_trials"] = args.n_trials
    t_met["k_inner"] = args.k_inner
    if len(t_tdf):
        t_tdf = t_tdf.copy(); t_tdf["fold"] = args.fold_idx
    t_tdf.to_csv(fold_root / "tuned_trades.csv", index=False)
    with open(fold_root / "tuned_metrics.json", "w") as fh:
        json.dump(t_met, fh, indent=2)

    t_m.save_model(str(fold_root / "best_model.xgb"))
    with open(fold_root / "best_scaler.pkl", "wb") as fh:
        pickle.dump(t_sc, fh)
    print(f"  tuned: n={t_met['n']} PF={t_met['pf']:.3f} "
          f"Net=Rs.{t_met['net']:+,.0f} eval_elapsed={t_met['elapsed_s']}s")

    delta_pf = t_met["pf"] - b_met["pf"]
    print(f"\n  [Fold {args.fold_idx}] delta_PF = {delta_pf:+.3f}  "
          f"(tuned - baseline)")

    # ---------- Fold summary + manifest ----------
    fold_summary = {
        "fold_idx": args.fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "baseline": b_met,
        "tuned": t_met,
        "delta_pf": delta_pf,
        "delta_net": t_met["net"] - b_met["net"],
        "best_params": study.best_params,
        "best_inner_score": float(study.best_value),
    }
    with open(fold_root / "summary.json", "w") as fh:
        json.dump(fold_summary, fh, indent=2)

    try:
        prereg = json.loads(
            (ROOT / "pre_registration"
                / "preregistration_active.json").read_text())
        prereg_v = prereg.get("protocol", {}).get("protocol_version")
        prereg_frozen = prereg.get("frozen_at_utc")
    except Exception:
        prereg_v, prereg_frozen = None, None
    manifest = {
        "fold_idx": args.fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "cutoff_train_before": str(cutoff),
        "embargo_days": EMBARGO_DAYS,
        "call_thr": CALL_THR, "put_thr": PUT_THR, "skip_ceil": SKIP_CEIL,
        "n_trials": args.n_trials,
        "k_inner": args.k_inner,
        "seed": args.seed,
        "inner_folds": [
            {"train_end": str(te_), "val_start": str(vs), "val_end": str(ve)}
            for (te_, vs, ve) in inner_folds
        ],
        "source_sha256": _source_manifest(),
        "preregistration_frozen_at": prereg_frozen,
        "preregistration_version": prereg_v,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(fold_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  wrote {fold_root}")


if __name__ == "__main__":
    main()
