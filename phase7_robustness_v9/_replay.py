"""Trade replay with configurable slippage / transaction-cost / execution
delay stress.

Phase 7 does not modify ``backtest_options.simulate_trades``. For slippage
and transaction-cost stress we recompute per-trade cost analytically from
the ``prem_entry`` / ``prem_exit`` columns using the same round-trip cost
formula that ``simulate_trades`` writes into every trade record. For
execution-delay stress the signals vector is shifted by ``k`` bars and
``simulate_trades`` is called unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Read-only Phase 0-6 imports
from backtest_options import (
    BROKERAGE_PER_ORDER, EXCH_TXN_RATE, GST_RATE,
    LOT_SIZE, SEBI_RATE, SPREAD_FLOOR_PTS, SPREAD_PCT,
    STAMP_BUY_RATE, STT_SELL_RATE, simulate_trades,
)
from threshold_opt import ThresholdCandidate, apply_thresholds
from threshold_opt._evaluate import _fold_metrics

from ._base import InvalidInputError


# ---------------------------------------------------------------------------
def spread_cost_component(prem_entry: float, qty: int = LOT_SIZE) -> float:
    """Round-trip spread cost — identical formula to ``round_trip_cost``."""
    spread_pts = max(SPREAD_FLOOR_PTS, SPREAD_PCT * float(prem_entry))
    return float(spread_pts * qty)


def non_spread_cost_component(prem_entry: float, prem_exit: float,
                                 qty: int = LOT_SIZE) -> float:
    """Round-trip cost minus the spread portion — matches ``round_trip_cost``.

    Includes brokerage, STT, exchange txn, SEBI, stamp duty and GST.
    """
    brokerage = BROKERAGE_PER_ORDER * 2
    stt = STT_SELL_RATE * float(prem_exit) * qty
    turnover = (float(prem_entry) + float(prem_exit)) * qty
    txn = EXCH_TXN_RATE * turnover
    sebi = SEBI_RATE * turnover
    stamp = STAMP_BUY_RATE * float(prem_entry) * qty
    gst = GST_RATE * (brokerage + txn + sebi)
    return float(brokerage + stt + txn + sebi + stamp + gst)


def apply_cost_stress(trades_df: pd.DataFrame,
                        *,
                        slippage_mult: float = 1.0,
                        tcost_mult: float = 1.0,
                        qty: int = LOT_SIZE) -> np.ndarray:
    """Return a per-trade ``net_option`` array under stressed cost model.

    ``slippage_mult`` scales the bid-ask spread component only.
    ``tcost_mult`` scales the brokerage/STT/fees/GST/stamp component only.
    All other trade mechanics are byte-identical to Phase 5.
    """
    if not len(trades_df):
        return np.array([], dtype=np.float64)
    needed = {"prem_entry", "prem_exit", "gross_option"}
    missing = needed - set(trades_df.columns)
    if missing:
        raise InvalidInputError(
            f"trades_df missing required columns: {missing}")
    pe = trades_df["prem_entry"].values.astype(np.float64)
    px = trades_df["prem_exit"].values.astype(np.float64)
    gross = trades_df["gross_option"].values.astype(np.float64)
    spread_cost = np.maximum(SPREAD_FLOOR_PTS, SPREAD_PCT * pe) * qty
    brokerage = BROKERAGE_PER_ORDER * 2
    stt = STT_SELL_RATE * px * qty
    turnover = (pe + px) * qty
    txn = EXCH_TXN_RATE * turnover
    sebi = SEBI_RATE * turnover
    stamp = STAMP_BUY_RATE * pe * qty
    gst = GST_RATE * (brokerage + txn + sebi)
    non_spread = brokerage + stt + txn + sebi + stamp + gst
    stressed_cost = slippage_mult * spread_cost + tcost_mult * non_spread
    return gross - stressed_cost


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoldReplay:
    fold: int
    trades: pd.DataFrame          # per-trade DataFrame from simulate_trades
    net_pnl: np.ndarray           # per-trade net_option (production cost)


def load_fold_data(target: str, folds=range(1, 9), *,
                     root: Path | None = None):
    """Return {fold: (probs, test_df, iv_map, expiries)}.

    Uses the frozen Phase 5 predictions and re-derives the feature frame /
    IV map via the unmodified Phase 5 helpers. Fails fast if any
    prediction file is missing or has a row-count mismatch.
    """
    from backtest_threshold_sweep import build_frame
    from backtest_options import build_iv_map
    from datetime import date

    r = root or Path(".")
    feat, _labels = build_frame()
    iv, exp = build_iv_map()

    # Phase 5 fold boundaries — identical calendar splits, read from
    # Phase 5 manifests (not hardcoded) to guarantee zero drift.
    import json as _json
    fold_windows = {}
    for f in folds:
        mp = (r / "logs" / "phase5" / target
              / f"fold_{f}" / "manifest.json")
        if not mp.exists():
            raise InvalidInputError(
                f"Phase 5 manifest missing: {mp}")
        m = _json.loads(mp.read_text())
        a = date.fromisoformat(m["test_start"])
        b = date.fromisoformat(m["test_end"])
        fold_windows[f] = (a, b)

    fold_data = {}
    for fold in folds:
        a, b = fold_windows[fold]
        mask = (feat.index.date >= a) & (feat.index.date < b)
        test_df = feat[mask]
        pred = pd.read_csv(
            r / "logs" / "phase5" / target / f"fold_{fold}"
              / "predictions.csv")
        if len(pred) != len(test_df):
            raise InvalidInputError(
                f"row-count mismatch for {target}/fold_{fold}: "
                f"predictions={len(pred)} test={len(test_df)}")
        probs = pred[["p_call", "p_put", "p_skip"]].values.astype(np.float64)
        fold_data[fold] = (probs, test_df, iv, exp)
    return fold_data


def simulate_candidate(cand: ThresholdCandidate,
                          fold_data: dict,
                          *,
                          exec_delay_bars: int = 0
                          ) -> dict[int, FoldReplay]:
    """Run the strategy for a candidate across pre-loaded fold data.

    ``exec_delay_bars`` shifts the signal vector forward by ``k`` bars
    before calling ``simulate_trades`` — bars 0..k-1 become SKIP.
    """
    if exec_delay_bars < 0:
        raise InvalidInputError("exec_delay_bars must be >= 0")
    out: dict[int, FoldReplay] = {}
    for fold, (probs, test_df, iv, exp) in fold_data.items():
        sig = apply_thresholds(probs, cand)
        if exec_delay_bars:
            new_sig = np.full_like(sig, 2)
            new_sig[exec_delay_bars:] = sig[:len(sig) - exec_delay_bars]
            sig = new_sig
        tdf = simulate_trades(test_df, sig, probs, iv, exp)
        if len(tdf) and "net_option" in tdf.columns:
            net = tdf["net_option"].values.astype(np.float64)
        else:
            net = np.array([], dtype=np.float64)
        out[fold] = FoldReplay(fold=fold, trades=tdf, net_pnl=net)
    return out


def pooled_metrics_from_replays(replays: dict[int, FoldReplay]) -> dict:
    """Return production-cost pooled metrics for a full replay set."""
    all_pnl = (np.concatenate([r.net_pnl for r in replays.values()])
                 if replays else np.array([], dtype=np.float64))
    return _fold_metrics(all_pnl)


def per_fold_metrics(replays: dict[int, FoldReplay]) -> dict[int, dict]:
    return {fold: _fold_metrics(r.net_pnl) for fold, r in replays.items()}


def pooled_metrics_from_stressed_pnl(pnl_by_fold: dict[int, np.ndarray]
                                        ) -> dict:
    """Same shape as ``pooled_metrics_from_replays`` but from raw arrays."""
    if not pnl_by_fold:
        return _fold_metrics(np.array([], dtype=np.float64))
    all_pnl = np.concatenate(list(pnl_by_fold.values()))
    return _fold_metrics(all_pnl)
