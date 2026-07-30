"""
phase3_run_single_fold.py — Run ONE brain on ONE outer fold.

Symmetric with phase2_run_single_fold.py but generic over any brain in
the ``brains`` registry. Default behaviour is to train with the brain's
default parameters (no HPO). Passing ``--hpo`` triggers the nested
purged Optuna search via ``brains._hpo.run_hpo``.

Artifacts under ``logs/phase3/{brain}/fold_{n}/``:
    predictions.parquet (or predictions.csv)
    trades.csv
    trade_pnl.csv                        # single-column, for downstream analysis
    metrics.json
    manifest.json
    default_params.json  OR  best_params.json + trials.parquet(.csv)
    model.pkl                            # via BrainAdapter.save
    scaler.pkl                           # StandardScaler (needed for all brains)
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

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65
BRAIN_ADAPTER_VERSION = "1.0"

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase3_run_single_fold.py",
    "brains/__init__.py",
    "brains/_base.py",
    "brains/_hpo.py",
    "brains/xgb_adapter.py",
    "brains/lgb_adapter.py",
    "brains/cat_adapter.py",
    "brains/mlp_adapter.py",
)


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
    return dict(
        n=int(len(pnl)), pf=pf,
        wr=float((pnl > 0).mean()),
        net=float(pnl.sum()),
        avg=float(pnl.mean()),
        dd=dd, sharpe=sh, sortino=sortino,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", choices=list_brains())
    ap.add_argument("fold_idx", type=int)
    ap.add_argument("test_start")
    ap.add_argument("test_end")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hpo", action="store_true",
                    help="Optuna nested purged HPO (expensive).")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--fold-dir", default="logs/phase3")
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

    print(f"[Phase 3 / {brain.name} / Fold {args.fold_idx}] build_frame ...")
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
    print(f"  outer train={outer_tr_mask.sum():,}, "
          f"test={outer_te_mask.sum():,}, gap={gap}d")

    # -------- HPO or defaults --------
    hpo_elapsed = 0.0
    if args.hpo:
        from brains._hpo import run_hpo
        print(f"  running HPO n_trials={args.n_trials}, k_inner={args.k_inner}")
        best_params, trials_df, hpo_elapsed = run_hpo(
            brain, feat, fcols, outer_tr_mask, iv, exp,
            n_trials=args.n_trials, k_inner=args.k_inner, seed=args.seed,
        )
        try:
            trials_df.to_parquet(fold_root / "trials.parquet", index=False)
        except Exception:
            trials_df.to_csv(fold_root / "trials.csv", index=False)
        with open(fold_root / "best_params.json", "w") as fh:
            json.dump(best_params, fh, indent=2, default=str)
        params_used = {**brain.default_params(), **best_params}
        print(f"  HPO elapsed {hpo_elapsed:.1f}s; best_params={best_params}")
    else:
        params_used = brain.default_params()
        with open(fold_root / "default_params.json", "w") as fh:
            json.dump(params_used, fh, indent=2, default=str)

    # -------- Train final on full outer_train + evaluate outer_test --------
    outer_tr = feat[outer_tr_mask]
    outer_te = feat[outer_te_mask]

    cut = int(len(outer_tr) * 0.95)
    tr_early = outer_tr.iloc[:cut]
    ev_early = outer_tr.iloc[cut:]
    w = _sample_weights(tr_early)

    sc = StandardScaler()
    Xtr = sc.fit_transform(tr_early[fcols].values)
    Xev = sc.transform(ev_early[fcols].values)
    Xte = sc.transform(outer_te[fcols].values)
    ytr = tr_early["label"].values
    yev = ev_early["label"].values
    yte = outer_te["label"].values

    print(f"  training {brain.name} on outer_train ...")
    t0 = time.time()
    model = brain.fit(Xtr, ytr, X_eval=Xev, y_eval=yev,
                       sample_weight=w, params=params_used, seed=args.seed)
    train_elapsed = time.time() - t0

    p_te = brain.predict_proba_3class(model, Xte)
    sig = signals_from_probas(p_te, CALL_THR, PUT_THR, SKIP_CEIL)
    tdf = simulate_trades(outer_te, sig, p_te, iv, exp)

    # -------- Trades + trade_pnl --------
    if len(tdf):
        tdf = tdf.copy()
        tdf["fold"] = args.fold_idx
        tdf["brain"] = brain.name
    tdf.to_csv(fold_root / "trades.csv", index=False)
    pnl = (tdf["net_option"].values.astype(float)
           if len(tdf) and "net_option" in tdf.columns
           else np.array([]))
    pd.DataFrame({"trade_pnl": pnl}).to_csv(
        fold_root / "trade_pnl.csv", index=False)

    # -------- Predictions --------
    pred_df = pd.DataFrame({
        "timestamp": outer_te.index,
        "y_true": yte.astype(int),
        "p_call": p_te[:, 0],
        "p_put": p_te[:, 1],
        "p_skip": p_te[:, 2],
        "signal": sig.astype(int),
    })
    try:
        pred_df.to_parquet(fold_root / "predictions.parquet", index=False)
    except Exception:
        pred_df.to_csv(fold_root / "predictions.csv", index=False)

    # -------- Metrics --------
    metrics = _fold_metrics(pnl)
    metrics.update({
        "brain": brain.name,
        "fold_idx": args.fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "train_bars": int(outer_tr_mask.sum()),
        "test_bars": int(outer_te_mask.sum()),
        "hpo_enabled": bool(args.hpo),
        "n_trials": args.n_trials if args.hpo else 0,
        "k_inner": args.k_inner if args.hpo else 0,
        "hpo_elapsed_s": round(hpo_elapsed, 1),
        "train_elapsed_s": round(train_elapsed, 1),
        "seed": args.seed,
    })
    with open(fold_root / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    # -------- Model + scaler --------
    brain.save(model, fold_root)
    with open(fold_root / "scaler.pkl", "wb") as fh:
        pickle.dump(sc, fh)

    # -------- Manifest --------
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
        "brain_adapter_version": BRAIN_ADAPTER_VERSION,
        "fold_idx": args.fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "cutoff_train_before": str(cutoff),
        "embargo_days": EMBARGO_DAYS,
        "call_thr": CALL_THR, "put_thr": PUT_THR, "skip_ceil": SKIP_CEIL,
        "hpo_enabled": bool(args.hpo),
        "n_trials": args.n_trials if args.hpo else 0,
        "k_inner": args.k_inner if args.hpo else 0,
        "seed": args.seed,
        "source_sha256": _source_manifest(),
        "preregistration_frozen_at": prereg_frozen,
        "preregistration_version": prereg_v,
        "params_used": params_used,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(fold_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    del model; gc.collect()
    print(f"[{brain.name} Fold {args.fold_idx}] n_trades={metrics['n']} "
          f"PF={metrics['pf']:.3f} Net=Rs.{metrics['net']:+,.0f} "
          f"WR={metrics['wr']*100:.1f}% train={train_elapsed:.0f}s")


if __name__ == "__main__":
    main()
