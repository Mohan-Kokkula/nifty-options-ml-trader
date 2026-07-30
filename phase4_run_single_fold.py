"""
phase4_run_single_fold.py — Calibrate probabilities for ONE brain × ONE outer fold.

Pipeline:
  1. Verify Phase 3 outputs exist for this (brain, fold). Load
     ``predictions.parquet/csv`` → uncalibrated 3-class probs + y_true
     for the outer test window.
  2. Rebuild the SAME inner-CV folds via ``brains._hpo.build_inner_folds``.
  3. For each inner fold, train the brain on inner_train, predict on
     inner_val → concatenate to an OUT-OF-FOLD probability pool.
  4. Fit each calibrator (noop, platt, isotonic) on the OOF pool.
     Never touches outer test.
  5. Transform Phase-3 outer probs → re-run signals_from_probas and
     simulate_trades.
  6. Compute calibration metrics (top-1 ECE, class-conditional ECE,
     Brier, log-loss, reliability bins) and trading metrics (PF, WR,
     Net, MaxDD, Sharpe, Sortino, n_trades) for each config.
  7. Persist artifacts and manifest.

Artifacts under ``logs/phase4/{brain}/fold_{n}/``:
    manifest.json
    calibration_diagnostics.json     # ECE / Brier / log-loss / reliability for all configs
    reliability_data.json            # per-class reliability bins for plotting
    uncal_metrics.json               # Phase 3 outer metrics + calibration diagnostics
    {config}_calibrator.pkl          # platt, isotonic
    {config}_predictions.csv|parquet # calibrated per-bar probs
    {config}_trades.csv              # trades from calibrated re-simulation
    {config}_trade_pnl.csv           # 1-column trade P&L
    {config}_metrics.json            # trading + calibration metrics
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_options import build_iv_map, simulate_trades
from backtest_threshold_sweep import (
    EMBARGO_DAYS,
    _sample_weights,
    build_frame,
    signals_from_probas,
)
from brains import get as get_brain
from brains import list_brains
from brains._hpo import build_inner_folds
from calibrators import (class_conditional_ece, get as get_calibrator,
                          list_calibrators, multiclass_brier,
                          multiclass_log_loss, per_class_reliability_bins,
                          reliability_bins, top1_ece)

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65
PHASE4_VERSION = "1.0"

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase4_run_single_fold.py",
    "brains/__init__.py",
    "brains/_base.py",
    "brains/_hpo.py",
    "brains/xgb_adapter.py",
    "brains/lgb_adapter.py",
    "brains/cat_adapter.py",
    "brains/mlp_adapter.py",
    "calibrators/__init__.py",
    "calibrators/_base.py",
    "calibrators/_metrics.py",
    "calibrators/noop.py",
    "calibrators/platt.py",
    "calibrators/isotonic.py",
)


# ---------------------------------------------------------------------------
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in _MODULES_TRACKED:
        p = ROOT / f
        if p.exists():
            out[f] = _sha256(p)
    return out


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
    return dict(n=int(len(pnl)), pf=pf,
                wr=float((pnl > 0).mean()),
                net=float(pnl.sum()),
                avg=float(pnl.mean()),
                dd=dd, sharpe=sh, sortino=sortino)


# ---------------------------------------------------------------------------
def _load_phase3_predictions(brain_name: str, fold_idx: int
                              ) -> tuple[np.ndarray, np.ndarray,
                                          pd.DatetimeIndex]:
    """Load Phase-3 outer-test predictions for (brain, fold).

    Returns (p_uncal (n, 3), y_true (n,), timestamp_index).
    """
    fr = ROOT / f"logs/phase3/{brain_name}/fold_{fold_idx}"
    pq = fr / "predictions.parquet"
    csv = fr / "predictions.csv"
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except Exception:
            df = pd.read_csv(csv)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"Phase 3 predictions missing for {brain_name}/fold_{fold_idx}")
    p_uncal = df[["p_call", "p_put", "p_skip"]].values.astype(float)
    y_true = df["y_true"].values.astype(int)
    ts = pd.to_datetime(df["timestamp"])
    return p_uncal, y_true, pd.DatetimeIndex(ts)


def _train_inner_and_predict(brain, feat, fcols, outer_tr_mask,
                              inner_folds, seed: int
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Train ``brain`` on each inner-CV fold and return the concatenated
    (p_oof (N, 3), y_oof (N,)) OUT-OF-FOLD probabilities."""
    p_pieces, y_pieces = [], []
    for i, (train_end, val_start, val_end) in enumerate(inner_folds, 1):
        inner_tr_mask = outer_tr_mask & (feat.index.date < train_end)
        inner_val_mask = ((feat.index.date >= val_start)
                          & (feat.index.date < val_end))
        inner_tr = feat[inner_tr_mask]
        inner_val = feat[inner_val_mask]
        if len(inner_tr) < 5000 or len(inner_val) < 50:
            print(f"    inner fold {i}: insufficient data, skipping")
            continue
        cut = int(len(inner_tr) * 0.95)
        tr_early = inner_tr.iloc[:cut]
        ev_early = inner_tr.iloc[cut:]
        w = _sample_weights(tr_early)
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr_early[fcols].values)
        Xev = sc.transform(ev_early[fcols].values)
        Xval = sc.transform(inner_val[fcols].values)
        ytr = tr_early["label"].values
        yev = ev_early["label"].values
        yval = inner_val["label"].values
        t0 = time.time()
        m = brain.fit(Xtr, ytr, X_eval=Xev, y_eval=yev,
                       sample_weight=w, params=brain.default_params(),
                       seed=seed)
        p_val = brain.predict_proba_3class(m, Xval)
        p_pieces.append(p_val)
        y_pieces.append(yval)
        print(f"    inner fold {i}: n_val={len(yval)}, "
              f"train={time.time()-t0:.0f}s")
        del m; gc.collect()
    if not p_pieces:
        raise RuntimeError("no OOF predictions produced from inner CV")
    return np.concatenate(p_pieces, axis=0), np.concatenate(y_pieces, axis=0)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", choices=list_brains())
    ap.add_argument("fold_idx", type=int)
    ap.add_argument("test_start")
    ap.add_argument("test_end")
    ap.add_argument("--k-inner", type=int, default=3,
                    help="Number of purged inner CV folds for OOF (default 3).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-dir", default="logs/phase4")
    ap.add_argument("--verify-prereg", action="store_true")
    args = ap.parse_args()

    if args.verify_prereg:
        from verify_preregistration import verify
        verify()

    brain = get_brain(args.brain)
    fold_root = (ROOT / args.fold_dir / brain.name
                 / f"fold_{args.fold_idx}")
    fold_root.mkdir(parents=True, exist_ok=True)

    a_date = date.fromisoformat(args.test_start)
    b_date = date.fromisoformat(args.test_end)

    print(f"[Phase 4 / {brain.name} / Fold {args.fold_idx}] loading Phase 3 outputs ...")
    p_uncal_te, y_true_te, ts_te = _load_phase3_predictions(
        brain.name, args.fold_idx)
    print(f"  outer test: n={len(y_true_te)}")

    print(f"[Phase 4 / {brain.name} / Fold {args.fold_idx}] build_frame ...")
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    cutoff = a_date - timedelta(days=EMBARGO_DAYS)
    outer_tr_mask = feat.index.date < cutoff
    outer_te_mask = ((feat.index.date >= a_date)
                     & (feat.index.date < b_date))
    if outer_tr_mask.sum() < 5000 or outer_te_mask.sum() < 50:
        print(f"[Fold {args.fold_idx}] insufficient data")
        return

    # Purge assertion
    tr_max = feat.index[outer_tr_mask].max().date()
    te_min = feat.index[outer_te_mask].min().date()
    gap = (te_min - tr_max).days
    if gap < EMBARGO_DAYS:
        raise AssertionError(
            f"Purge violated: gap={gap} < embargo={EMBARGO_DAYS}")

    # Reindex outer test rows to line up with the outer-window feat frame,
    # since Phase 3 emitted predictions on outer_te_mask rows.
    outer_te = feat[outer_te_mask]
    if len(outer_te) != len(y_true_te):
        raise AssertionError(
            f"Phase 3 predictions length {len(y_true_te)} != outer_te rows "
            f"{len(outer_te)}; the frame or fold boundaries have drifted.")

    # --- Build inner CV folds and OOF pool
    outer_tr_dates = sorted(set(feat.index[outer_tr_mask].date))
    inner_folds = build_inner_folds(outer_tr_dates, args.k_inner)
    print(f"  inner CV K={args.k_inner}:")
    for i, (te_, vs, ve) in enumerate(inner_folds, 1):
        print(f"    inner {i}: train<{te_}  val {vs}..{ve}")

    print(f"[Phase 4] retraining {brain.name} on inner folds to build OOF pool ...")
    p_oof, y_oof = _train_inner_and_predict(
        brain, feat, fcols, outer_tr_mask, inner_folds, args.seed)
    print(f"  OOF pool: n={len(y_oof)}")

    # --- Diagnostic on Phase 3 outer probs (uncalibrated baseline)
    diag: dict[str, dict] = {}
    diag["uncal"] = {
        "top1_ece": top1_ece(y_true_te, p_uncal_te),
        "class_conditional_ece": class_conditional_ece(y_true_te, p_uncal_te),
        "brier": multiclass_brier(y_true_te, p_uncal_te),
        "log_loss": multiclass_log_loss(y_true_te, p_uncal_te),
    }
    reliability: dict[str, list] = {
        "uncal_top1": reliability_bins(y_true_te, p_uncal_te),
        "uncal_per_class": per_class_reliability_bins(y_true_te, p_uncal_te),
    }

    # --- Load Phase-3 outer trades for uncal baseline metrics (already computed)
    ph3 = ROOT / f"logs/phase3/{brain.name}/fold_{args.fold_idx}"
    try:
        ph3_metrics = json.loads((ph3 / "metrics.json").read_text())
    except Exception:
        ph3_metrics = {}

    uncal_metrics = {**ph3_metrics, "calibration": diag["uncal"]}
    with open(fold_root / "uncal_metrics.json", "w") as fh:
        json.dump(uncal_metrics, fh, indent=2, default=str)

    # --- Fit + apply each calibrator (except noop, which is trivial)
    results_summary: list[dict] = []
    # Baseline row for uncalibrated (from Phase 3)
    if ph3_metrics:
        results_summary.append({
            "config": "uncal", "n_trades": ph3_metrics.get("n"),
            "pf": ph3_metrics.get("pf"), "wr": ph3_metrics.get("wr"),
            "net": ph3_metrics.get("net"), "sharpe": ph3_metrics.get("sharpe"),
            "sortino": ph3_metrics.get("sortino"),
            "top1_ece": diag["uncal"]["top1_ece"],
            "brier": diag["uncal"]["brier"],
            "log_loss": diag["uncal"]["log_loss"],
        })

    for cfg in ("noop", "platt", "isotonic"):
        print(f"\n[{brain.name}/fold {args.fold_idx}] calibrator={cfg}")
        cal = get_calibrator(cfg)
        t0 = time.time()
        cal.fit(p_oof, y_oof, seed=args.seed)
        fit_elapsed = time.time() - t0

        # Transform outer test probs
        p_cal = cal.transform(p_uncal_te)
        # Re-simulate trades
        sig = signals_from_probas(p_cal, CALL_THR, PUT_THR, SKIP_CEIL)
        tdf = simulate_trades(outer_te, sig, p_cal, iv, exp)

        # Trading metrics
        pnl = (tdf["net_option"].values.astype(float)
               if len(tdf) and "net_option" in tdf.columns
               else np.array([]))
        trading = _fold_metrics(pnl)

        # Calibration metrics on outer test
        cal_diag = {
            "top1_ece": top1_ece(y_true_te, p_cal),
            "class_conditional_ece": class_conditional_ece(y_true_te, p_cal),
            "brier": multiclass_brier(y_true_te, p_cal),
            "log_loss": multiclass_log_loss(y_true_te, p_cal),
        }
        diag[cfg] = cal_diag
        reliability[f"{cfg}_top1"] = reliability_bins(y_true_te, p_cal)
        reliability[f"{cfg}_per_class"] = per_class_reliability_bins(
            y_true_te, p_cal)

        # Persist calibrator
        cal.save(fold_root, filename=f"{cfg}_calibrator.pkl")

        # Persist predictions
        pred_df = pd.DataFrame({
            "timestamp": ts_te,
            "y_true": y_true_te.astype(int),
            "p_call": p_cal[:, 0],
            "p_put": p_cal[:, 1],
            "p_skip": p_cal[:, 2],
            "signal": sig.astype(int),
        })
        try:
            pred_df.to_parquet(
                fold_root / f"{cfg}_predictions.parquet", index=False)
        except Exception:
            pred_df.to_csv(
                fold_root / f"{cfg}_predictions.csv", index=False)

        # Persist trades + trade_pnl
        if len(tdf):
            tdf = tdf.copy()
            tdf["fold"] = args.fold_idx
            tdf["brain"] = brain.name
            tdf["calibrator"] = cfg
        tdf.to_csv(fold_root / f"{cfg}_trades.csv", index=False)
        pd.DataFrame({"trade_pnl": pnl}).to_csv(
            fold_root / f"{cfg}_trade_pnl.csv", index=False)

        # Metrics json
        combined = {**trading,
                     "brain": brain.name, "fold_idx": args.fold_idx,
                     "calibrator": cfg,
                     "calibration": cal_diag,
                     "calibrator_fit_elapsed_s": round(fit_elapsed, 3)}
        with open(fold_root / f"{cfg}_metrics.json", "w") as fh:
            json.dump(combined, fh, indent=2, default=str)

        results_summary.append({
            "config": cfg, "n_trades": trading.get("n"),
            "pf": trading.get("pf"), "wr": trading.get("wr"),
            "net": trading.get("net"), "sharpe": trading.get("sharpe"),
            "sortino": trading.get("sortino"),
            "top1_ece": cal_diag["top1_ece"],
            "brier": cal_diag["brier"],
            "log_loss": cal_diag["log_loss"],
        })
        print(f"  {cfg}: n_trades={trading.get('n')} "
              f"PF={trading.get('pf'):.3f} Net=Rs.{trading.get('net'):+,.0f} "
              f"top1_ece={cal_diag['top1_ece']:.4f} "
              f"brier={cal_diag['brier']:.4f} "
              f"log_loss={cal_diag['log_loss']:.4f}")

    # --- Persist calibration diagnostics + reliability bundle
    with open(fold_root / "calibration_diagnostics.json", "w") as fh:
        json.dump(diag, fh, indent=2, default=str)
    with open(fold_root / "reliability_data.json", "w") as fh:
        json.dump(reliability, fh, indent=2, default=str)

    # --- Manifest
    try:
        prereg = json.loads(
            (ROOT / "pre_registration"
                / "preregistration_active.json").read_text())
        prereg_v = prereg.get("protocol", {}).get("protocol_version")
        prereg_frozen = prereg.get("frozen_at_utc")
    except Exception:
        prereg_v, prereg_frozen = None, None
    manifest = {
        "brain": brain.name,
        "phase4_version": PHASE4_VERSION,
        "fold_idx": args.fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "cutoff_train_before": str(cutoff),
        "embargo_days": EMBARGO_DAYS,
        "call_thr": CALL_THR, "put_thr": PUT_THR, "skip_ceil": SKIP_CEIL,
        "k_inner": args.k_inner,
        "seed": args.seed,
        "n_oof": int(len(y_oof)),
        "n_outer_test": int(len(y_true_te)),
        "source_sha256": _source_manifest(),
        "preregistration_frozen_at": prereg_frozen,
        "preregistration_version": prereg_v,
        "calibrators_evaluated": ["uncal", "noop", "platt", "isotonic"],
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(fold_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\n[{brain.name} Fold {args.fold_idx}] Phase 4 done.")
    print(f"  {'config':<10} {'n':>5} {'PF':>7} {'Net':>12} {'top1_ECE':>10} "
          f"{'Brier':>7} {'log_loss':>10}")
    for r in results_summary:
        pf = r.get("pf")
        pf_s = f"{pf:>7.3f}" if isinstance(pf, (int, float)) and pf is not None else "  n/a  "
        net = r.get("net") or 0.0
        print(f"  {r['config']:<10} {r.get('n_trades', 0):>5} {pf_s} "
              f"{net:>+12,.0f} {r.get('top1_ece', float('nan')):>10.4f} "
              f"{r.get('brier', float('nan')):>7.4f} "
              f"{r.get('log_loss', float('nan')):>10.4f}")


if __name__ == "__main__":
    main()
