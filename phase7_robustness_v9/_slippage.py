"""R3 – slippage stress curves."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from threshold_opt._evaluate import _fold_metrics

from ._base import SLIPPAGE_MULTIPLIERS
from ._replay import apply_cost_stress


def _pool_stressed(trades_by_fold: Mapping[int, pd.DataFrame],
                     slippage_mult: float,
                     tcost_mult: float = 1.0
                     ) -> tuple[dict, dict[int, np.ndarray]]:
    """Return (pooled_metrics, per-fold stressed net_option arrays)."""
    per_fold: dict[int, np.ndarray] = {}
    for fold, tdf in trades_by_fold.items():
        per_fold[fold] = apply_cost_stress(
            tdf, slippage_mult=slippage_mult, tcost_mult=tcost_mult)
    all_pnl = (np.concatenate(list(per_fold.values()))
                 if per_fold else np.array([], dtype=np.float64))
    return _fold_metrics(all_pnl), per_fold


def run_slippage_curve(trades_by_fold: Mapping[int, pd.DataFrame],
                         multipliers=SLIPPAGE_MULTIPLIERS) -> dict:
    """Return {mult: metrics} + per-fold stressed pnl for the highest level."""
    curve = {}
    per_fold_by_mult: dict[float, dict[int, np.ndarray]] = {}
    for m in multipliers:
        metrics, per_fold = _pool_stressed(trades_by_fold, m)
        curve[float(m)] = {
            "slippage_multiplier": float(m),
            "pooled_pf": metrics["pf"],
            "pooled_net": metrics["net"],
            "pooled_max_dd": metrics["dd"],
            "pooled_trade_count": metrics["n"],
            "pooled_wr": metrics["wr"],
            "pooled_sharpe": metrics["sharpe"],
            "pooled_sortino": metrics["sortino"],
        }
        per_fold_by_mult[float(m)] = per_fold
    return {"curve": curve, "per_fold_by_mult": per_fold_by_mult}
