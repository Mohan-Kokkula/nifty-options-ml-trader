"""
phase5_run_single_fold.py — Phase 5 ensemble worker for ONE outer fold.

This script is NOT executed as part of the infrastructure delivery.
It exists as a wired-up but untested-on-real-data harness ready to
consume the completed Phase 4 outputs once the overnight run finishes.

Pipeline (when eventually run):
  1. Discover participating brains for the fold via
     ``ensembles.discover_brains`` (registry ∩ Phase-3 outputs ∩ CLI).
  2. Load outer-test probabilities per brain from either raw Phase-3
     or calibrated Phase-4 (source selected by ``--input``).
  3. Generate OOF probabilities per brain by re-training via
     ``ensembles.generate_oof_predictions`` (cached under
     ``logs/phase5/_oof/{brain}/fold_{n}/``).
  4. For each requested ensemble in the registry, fit on OOF and
     transform outer-test predictions, then simulate trades.
  5. Persist the standardised artifact set + manifest.

The orchestrator is invoked via subprocess for isolation.

NOTE
----
This delivery ONLY compiles the script; it is never executed. Any
real-data execution requires the completed Phase 4 outputs and is out
of scope for the infrastructure phase.
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

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65
PHASE5_VERSION = "1.0"

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase5_run_single_fold.py",
    "brains/__init__.py",
    "brains/_base.py",
    "brains/_hpo.py",
    "calibrators/__init__.py",
    "calibrators/_base.py",
    "ensembles/__init__.py",
    "ensembles/_base.py",
    "ensembles/_validation.py",
    "ensembles/_selection.py",
    "ensembles/_diversity.py",
    "ensembles/_meta.py",
    "ensembles/_oof.py",
    "ensembles/mean_probability.py",
    "ensembles/median_probability.py",
    "ensembles/weighted_probability.py",
    "ensembles/performance_weighted.py",
    "ensembles/min_variance.py",
    "ensembles/stacking.py",
    "ensembles/confidence_weighted.py",
    "ensembles/uncertainty_weighted.py",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_manifest() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED
            if (ROOT / f).exists()}


def main() -> None:
    from ensembles import (
        discover_brains, get as get_ensemble, list_ensembles,
        load_brain_probs, ProbabilitySource,
    )
    from ensembles._oof import generate_oof_predictions
    from backtest_threshold_sweep import (
        EMBARGO_DAYS, build_frame, signals_from_probas,
    )
    from backtest_options import build_iv_map, simulate_trades

    ap = argparse.ArgumentParser()
    ap.add_argument("fold_idx", type=int)
    ap.add_argument("test_start")
    ap.add_argument("test_end")
    ap.add_argument("--brains", default="all",
                    help="'all' or comma-separated brain names.")
    ap.add_argument("--ensembles", default="all",
                    help="'all' or comma-separated ensemble names.")
    ap.add_argument("--input", default="raw",
                    choices=["raw", "calibrated_noop",
                              "calibrated_platt", "calibrated_isotonic"])
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-dir", default="logs/phase5")
    ap.add_argument("--verify-prereg", action="store_true")
    args = ap.parse_args()

    if args.verify_prereg:
        from verify_preregistration import verify
        verify()

    a_date = date.fromisoformat(args.test_start)
    b_date = date.fromisoformat(args.test_end)

    print(f"[Phase 5 / Fold {args.fold_idx}] discover brains ...")
    brains_used = discover_brains(ROOT, args.fold_idx, args.brains)
    print(f"  brains: {brains_used}")

    if args.ensembles == "all":
        ensembles_to_run = list_ensembles()
    else:
        ensembles_to_run = [e.strip() for e in args.ensembles.split(",")
                             if e.strip()]
        for e in ensembles_to_run:
            if e not in list_ensembles():
                raise SystemExit(f"unknown ensemble {e!r}")
    print(f"  ensembles: {ensembles_to_run}")

    # ---- Load outer test probs per brain
    outer_probs: dict[str, np.ndarray] = {}
    y_true = None
    timestamps = None
    data_hash: dict[str, str] = {}
    for brain in brains_used:
        probs, y, ts = load_brain_probs(ROOT, brain, args.fold_idx,
                                          source=args.input)
        outer_probs[brain] = probs
        if y_true is None:
            y_true = y
            timestamps = ts
        else:
            if len(y) != len(y_true) or not np.array_equal(y, y_true):
                raise SystemExit(
                    f"y_true mismatch between brains at outer-test load time")
        # Record which physical file the load ended up reading
        source_dir = (ROOT / ("logs/phase3" if args.input == "raw"
                              else "logs/phase4") / brain
                       / f"fold_{args.fold_idx}")
        for candidate in ("predictions.parquet", "predictions.csv",
                            f"{args.input.replace('calibrated_', '')}_predictions.parquet",
                            f"{args.input.replace('calibrated_', '')}_predictions.csv"):
            f = source_dir / candidate
            if f.exists():
                data_hash[f"{brain}:{candidate}"] = _sha256(f)
                break

    # ---- OOF generation per brain (rebuilds outer frame; expensive) ----
    from brains import get as get_brain
    print(f"[Phase 5 / Fold {args.fold_idx}] build_frame for OOF ...")
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    cutoff = a_date - timedelta(days=EMBARGO_DAYS)
    outer_tr_mask = feat.index.date < cutoff
    outer_te_mask = ((feat.index.date >= a_date)
                     & (feat.index.date < b_date))
    if outer_tr_mask.sum() < 5000 or outer_te_mask.sum() < 50:
        raise SystemExit(f"insufficient data for fold {args.fold_idx}")

    need_pnl = "min_variance" in ensembles_to_run
    oof_probs: dict[str, np.ndarray] = {}
    y_oof: np.ndarray | None = None
    pnl_by_brain: dict[str, dict[int, np.ndarray]] = {}
    for brain in brains_used:
        print(f"  OOF: {brain}")
        oof_cache = (ROOT / args.fold_dir / "_oof" / brain
                     / f"fold_{args.fold_idx}")
        oof_cache.mkdir(parents=True, exist_ok=True)
        b_adapter = get_brain(brain)
        result = generate_oof_predictions(
            b_adapter, feat, fcols, outer_tr_mask,
            k_inner=args.k_inner, seed=args.seed,
            collect_inner_pnl=need_pnl, iv=iv, exp=exp,
        )
        oof_probs[brain] = result.p_oof
        if y_oof is None:
            y_oof = result.y_oof
        if need_pnl and result.inner_pnl_by_fold is not None:
            pnl_by_brain[brain] = result.inner_pnl_by_fold
        # Persist the OOF cache
        pd.DataFrame(result.p_oof,
                       columns=["p_call", "p_put", "p_skip"]).to_csv(
            oof_cache / "p_oof.csv", index=False)
        pd.DataFrame({"y_oof": result.y_oof}).to_csv(
            oof_cache / "y_oof.csv", index=False)
        if need_pnl and result.inner_pnl_by_fold is not None:
            with open(oof_cache / "inner_pnl.json", "w") as fh:
                json.dump({int(k): v.tolist()
                            for k, v in result.inner_pnl_by_fold.items()},
                           fh)

    # ---- Fit + transform + simulate for each ensemble ----
    outer_te = feat[outer_te_mask]
    for ens_name in ensembles_to_run:
        print(f"[Phase 5] ensemble={ens_name}")
        ens_root = (ROOT / args.fold_dir / ens_name
                     / f"fold_{args.fold_idx}")
        ens_root.mkdir(parents=True, exist_ok=True)
        ens = get_ensemble(ens_name)

        t0 = time.time()
        fit_kwargs = {}
        if ens.requires_inner_pnl:
            fit_kwargs["brain_trade_pnl_by_fold"] = pnl_by_brain
        ens.fit(oof_probs, y_oof, seed=args.seed, **fit_kwargs)
        fit_elapsed = time.time() - t0

        t0 = time.time()
        p_test = ens.transform(outer_probs)
        transform_elapsed = time.time() - t0

        sig = signals_from_probas(p_test, CALL_THR, PUT_THR, SKIP_CEIL)
        t0 = time.time()
        tdf = simulate_trades(outer_te, sig, p_test, iv, exp)
        simulate_elapsed = time.time() - t0

        # Predictions + probabilities
        pred_df = pd.DataFrame({
            "timestamp": timestamps,
            "y_true": y_true.astype(int),
            "p_call": p_test[:, 0],
            "p_put": p_test[:, 1],
            "p_skip": p_test[:, 2],
            "signal": sig.astype(int),
        })
        try:
            pred_df.to_parquet(ens_root / "predictions.parquet", index=False)
        except Exception:
            pred_df.to_csv(ens_root / "predictions.csv", index=False)
        pd.DataFrame(p_test,
                       columns=["p_call", "p_put", "p_skip"]).to_csv(
            ens_root / "probabilities.csv", index=False)

        # Trades + P&L
        if len(tdf):
            tdf = tdf.copy()
            tdf["fold"] = args.fold_idx
            tdf["ensemble"] = ens_name
        tdf.to_csv(ens_root / "trades.csv", index=False)
        pnl_col = (tdf["net_option"].values.astype(np.float64)
                   if len(tdf) and "net_option" in tdf.columns
                   else np.array([], dtype=np.float64))
        pd.DataFrame({"trade_pnl": pnl_col}).to_csv(
            ens_root / "trade_pnl.csv", index=False)

        # Structural metrics (NO PF/Sharpe/Sortino — infra only)
        metrics = {
            "ensemble_type": ens_name,
            "fold_id": args.fold_idx,
            "n_trades": int(len(pnl_col)),
            "n_test_bars": int(len(y_true)),
            "n_bars_traded": int((sig != 2).sum()),
            "brains_used": list(brains_used),
            "input_source": args.input,
            "weights_summary": ens.weights_summary(),
            "fit_elapsed_s": round(fit_elapsed, 3),
            "transform_elapsed_s": round(transform_elapsed, 3),
            "simulate_elapsed_s": round(simulate_elapsed, 3),
            "n_oof_bars": int(len(y_oof)),
        }
        with open(ens_root / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)

        # Weights snapshot
        with open(ens_root / "weights.json", "w") as fh:
            json.dump(ens.weights_summary(), fh, indent=2, default=str)

        # Persist the fitted ensemble
        ens.save(ens_root)

        # Manifest
        try:
            prereg = json.loads(
                (ROOT / "pre_registration"
                    / "preregistration_active.json").read_text())
            prereg_v = prereg.get("protocol", {}).get("protocol_version")
            prereg_frozen = prereg.get("frozen_at_utc")
        except Exception:
            prereg_v, prereg_frozen = None, None
        manifest = {
            "protocol_version": prereg_v,
            "code_hash": _source_manifest(),
            "data_hash": data_hash,
            "random_seed": int(args.seed),
            "fold_id": int(args.fold_idx),
            "test_start": str(a_date),
            "test_end": str(b_date),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ensemble_type": ens_name,
            "participating_brains": list(brains_used),
            "input_source": args.input,
            "weights_summary": ens.weights_summary(),
            "n_test_bars": int(len(y_true)),
            "n_oof_bars": int(len(y_oof)),
            "k_inner": int(args.k_inner),
            "preregistration_frozen_at": prereg_frozen,
            "phase5_version": PHASE5_VERSION,
        }
        with open(ens_root / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)

        print(f"  {ens_name}: n_trades={metrics['n_trades']}  "
              f"fit={fit_elapsed:.2f}s transform={transform_elapsed:.2f}s")


if __name__ == "__main__":
    main()
