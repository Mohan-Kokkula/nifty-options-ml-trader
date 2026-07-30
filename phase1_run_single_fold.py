"""
phase1_run_single_fold.py — Train + evaluate ONE fold of avg4 in isolation.

Called as a subprocess by phase1_finish_avg4.py. Isolation gives fresh
Python memory per fold, defeating the OOM at Fold 7 that killed prior runs.

v2 upgrades:
  * Hard purge/embargo assertion at fold entry.
  * Opt-in --no-lookahead-weights that monkey-patches
    ``backtest_threshold_sweep._sample_weights`` for the duration of the
    fold. Source module is never mutated.
  * Rich per-fold artifact set under logs/phase1/fold_<n>/:
        trades.csv, predictions.parquet, metrics.json,
        manifest.json, leakage_check.json
  * Optional --verify-prereg drift gate.

Backward compatibility: legacy ``logs/phase1_fold_<n>.csv`` is still written.

CLI:
    python phase1_run_single_fold.py <fold_idx> <test_start> <test_end>
        [--no-lookahead-weights] [--verify-prereg] [--fold-dir DIR]
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import (
    EMBARGO_DAYS, _proba3, build_frame, signals_from_probas, train_fold,
)
from backtest_multibrain import train_rf_fold
from backtest_multibrain_v2 import train_nn_fold
from backtest_options import build_iv_map, simulate_trades

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_multibrain.py",
    "backtest_multibrain_v2.py",
    "backtest_options.py",
    "phase1_run_single_fold.py",
)


def _source_manifest() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED if (ROOT / f).exists()}


# ---------------------------------------------------------------------------
# Look-ahead-free sample-weight override for ablation runs
# ---------------------------------------------------------------------------
def _no_lookahead_sample_weights(sub: pd.DataFrame) -> np.ndarray:
    """Ablation replacement for backtest_threshold_sweep._sample_weights.

    Strips the trend-day (per-day close-open range) and expiry-day
    multipliers, both of which use end-of-day OHLC on intraday rows.
    Keeps only the class-imbalance weight.
    """
    y = sub["label"].values
    skip_pct = (y == 2).mean()
    trade_w = skip_pct / max(1 - skip_pct, 1e-9)
    return np.where(y == 2, 1.0, trade_w)


# ---------------------------------------------------------------------------
# Fold metrics
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
        n=int(len(pnl)),
        pf=pf,
        wr=float((pnl > 0).mean()),
        net=float(pnl.sum()),
        avg=float(pnl.mean()),
        dd=dd,
        sharpe=sh,
        sortino=sortino,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fold_idx", type=int)
    ap.add_argument("test_start")
    ap.add_argument("test_end")
    ap.add_argument("--no-lookahead-weights", action="store_true",
                    dest="ablate",
                    help="Ablate the trend-day + expiry sample-weight "
                         "multipliers (both consume end-of-day OHLC on "
                         "intraday training rows).")
    ap.add_argument("--fold-dir", default="logs/phase1",
                    help="Root directory for per-fold artifacts.")
    ap.add_argument("--verify-prereg", action="store_true",
                    help="Verify pre-registration manifest before doing "
                         "anything; abort if drift.")
    args = ap.parse_args()

    if args.verify_prereg:
        from verify_preregistration import verify
        verify()

    fold_idx = args.fold_idx
    a_date = date.fromisoformat(args.test_start)
    b_date = date.fromisoformat(args.test_end)
    fold_root = ROOT / args.fold_dir / f"fold_{fold_idx}"
    fold_root.mkdir(parents=True, exist_ok=True)

    print(f"[Fold {fold_idx}] build_frame...")
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    cutoff = a_date - timedelta(days=EMBARGO_DAYS)
    tr_mask = feat.index.date < cutoff
    te_mask = (feat.index.date >= a_date) & (feat.index.date < b_date)

    # Hard purge/embargo assertion.
    tr_max_date = feat.index[tr_mask].max().date() if tr_mask.sum() else None
    te_min_date = feat.index[te_mask].min().date() if te_mask.sum() else None
    if tr_max_date and te_min_date:
        gap_days = (te_min_date - tr_max_date).days
        if gap_days < EMBARGO_DAYS:
            raise AssertionError(
                f"Fold {fold_idx}: purge violated. train_max={tr_max_date} "
                f"test_min={te_min_date} gap={gap_days} < embargo={EMBARGO_DAYS}"
            )
    else:
        gap_days = None

    if tr_mask.sum() < 5000 or te_mask.sum() < 50:
        print(f"[Fold {fold_idx}] insufficient data "
              f"(train={tr_mask.sum()}, test={te_mask.sum()})")
        return

    # Optional monkey-patch: bypass look-ahead sample weighting for this run.
    import backtest_threshold_sweep as bts
    original_sw = None
    if args.ablate:
        original_sw = bts._sample_weights
        bts._sample_weights = _no_lookahead_sample_weights
        print(f"[Fold {fold_idx}] ablation: no-lookahead sample weights")

    t_start = time.time()
    try:
        print(f"[Fold {fold_idx}] training XGB+LGB...")
        models, sc_xl = train_fold(feat, fcols, tr_mask)
        test = feat[te_mask]
        Xte_xl = sc_xl.transform(test[fcols].values)
        p_xgb = _proba3(list(models.values())[0], Xte_xl)
        p_lgb = _proba3(list(models.values())[1], Xte_xl)
        del models; gc.collect()

        print(f"[Fold {fold_idx}] training RF...")
        rf, sc_rf = train_rf_fold(feat, fcols, tr_mask)
        Xte_rf = sc_rf.transform(test[fcols].values)
        p_rf = _proba3(rf, Xte_rf)
        del rf; gc.collect()

        print(f"[Fold {fold_idx}] training NN...")
        nn, sc_nn = train_nn_fold(feat, fcols, tr_mask)
        Xte_nn = sc_nn.transform(test[fcols].values)
        p_nn = _proba3(nn, Xte_nn)
        del nn; gc.collect()
    finally:
        if original_sw is not None:
            bts._sample_weights = original_sw

    elapsed = time.time() - t_start

    # Ensemble + trade simulation
    p_avg4 = (p_xgb + p_lgb + p_rf + p_nn) / 4
    sig = signals_from_probas(p_avg4, CALL_THR, PUT_THR, SKIP_CEIL)
    tdf = simulate_trades(test, sig, p_avg4, iv, exp)

    # Trades: new + legacy paths
    if len(tdf):
        tdf = tdf.copy()
        tdf["fold"] = fold_idx
    tdf.to_csv(fold_root / "trades.csv", index=False)
    tdf.to_csv(ROOT / f"logs/phase1_fold_{fold_idx}.csv", index=False)

    # Per-bar predictions
    pred_df = pd.DataFrame({
        "timestamp": test.index,
        "y_true": test["label"].values.astype(int),
        "p_call_xgb": p_xgb[:, 0], "p_put_xgb": p_xgb[:, 1], "p_skip_xgb": p_xgb[:, 2],
        "p_call_lgb": p_lgb[:, 0], "p_put_lgb": p_lgb[:, 1], "p_skip_lgb": p_lgb[:, 2],
        "p_call_rf": p_rf[:, 0], "p_put_rf": p_rf[:, 1], "p_skip_rf": p_rf[:, 2],
        "p_call_nn": p_nn[:, 0], "p_put_nn": p_nn[:, 1], "p_skip_nn": p_nn[:, 2],
        "p_call_avg4": p_avg4[:, 0], "p_put_avg4": p_avg4[:, 1], "p_skip_avg4": p_avg4[:, 2],
        "signal": sig.astype(int),
    })
    try:
        pred_df.to_parquet(fold_root / "predictions.parquet", index=False)
    except Exception:
        pred_df.to_csv(fold_root / "predictions.csv", index=False)

    # Metrics
    pnl = tdf["net_option"].values if len(tdf) and "net_option" in tdf.columns \
        else np.array([])
    metrics = _fold_metrics(pnl)
    metrics.update({
        "fold_idx": fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "train_bars": int(tr_mask.sum()),
        "test_bars": int(te_mask.sum()),
        "ablate_lookahead_weights": bool(args.ablate),
        "elapsed_seconds": round(elapsed, 1),
    })
    with open(fold_root / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Cache-consistency manifest
    try:
        prereg = json.loads(
            (ROOT / "pre_registration" / "preregistration_active.json").read_text()
        )
        prereg_frozen = prereg.get("frozen_at_utc")
        prereg_version = prereg.get("protocol", {}).get("protocol_version")
    except Exception:
        prereg_frozen, prereg_version = None, None
    manifest = {
        "fold_idx": fold_idx,
        "test_start": str(a_date),
        "test_end": str(b_date),
        "cutoff_train_before": str(cutoff),
        "embargo_days": EMBARGO_DAYS,
        "ablate_lookahead_weights": bool(args.ablate),
        "call_thr": CALL_THR,
        "put_thr": PUT_THR,
        "skip_ceil": SKIP_CEIL,
        "source_sha256": _source_manifest(),
        "preregistration_frozen_at": prereg_frozen,
        "preregistration_version": prereg_version,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(fold_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Leakage sanity summary
    leak = {
        "embargo_days_required": EMBARGO_DAYS,
        "train_max_date": str(tr_max_date) if tr_max_date else None,
        "test_min_date": str(te_min_date) if te_min_date else None,
        "gap_days": gap_days,
        "purge_satisfied": (
            gap_days is not None and gap_days >= EMBARGO_DAYS),
    }
    with open(fold_root / "leakage_check.json", "w") as fh:
        json.dump(leak, fh, indent=2)

    print(f"[Fold {fold_idx}] n={metrics['n']} PF={metrics['pf']:.3f} "
          f"Net=Rs.{metrics['net']:+,.0f} DD=Rs.{metrics['dd']:+,.0f} "
          f"WR={metrics['wr']*100:.1f}% elapsed={metrics['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
