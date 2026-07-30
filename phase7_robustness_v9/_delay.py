"""Execution-delay stress: shift signals by k bars before replay.

The trade generation and pricing logic in ``backtest_options.simulate_trades``
is not modified; only the signals vector is shifted before it is passed in.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from threshold_opt import ThresholdCandidate

from ._base import EXEC_DELAYS_BARS
from ._replay import (
    per_fold_metrics, pooled_metrics_from_replays, simulate_candidate,
)


def run_delay_stress(cand: ThresholdCandidate,
                       fold_data: dict,
                       *,
                       delays: Iterable[int] = EXEC_DELAYS_BARS) -> dict:
    """Return {delay_bars: {pooled, per_fold}} for each delay level."""
    curve = {}
    per_fold_pnl_by_delay: dict[int, dict[int, np.ndarray]] = {}
    for d in delays:
        replays = simulate_candidate(cand, fold_data, exec_delay_bars=int(d))
        pooled = pooled_metrics_from_replays(replays)
        pf_by_fold = per_fold_metrics(replays)
        curve[int(d)] = {
            "delay_bars": int(d),
            "pooled_pf": pooled["pf"],
            "pooled_net": pooled["net"],
            "pooled_max_dd": pooled["dd"],
            "pooled_trade_count": pooled["n"],
            "pooled_wr": pooled["wr"],
            "per_fold_pf": {int(k): v["pf"] for k, v in pf_by_fold.items()},
        }
        per_fold_pnl_by_delay[int(d)] = {
            fold: r.net_pnl for fold, r in replays.items()}
    return {"curve": curve, "per_fold_pnl_by_delay": per_fold_pnl_by_delay}
