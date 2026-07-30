"""R4 – transaction-cost sensitivity + break-even solvers."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from threshold_opt._evaluate import _fold_metrics

from ._base import TCOST_MULTIPLIERS
from ._replay import apply_cost_stress


def _pool_stressed_net(trades_by_fold: Mapping[int, pd.DataFrame],
                          slippage_mult: float,
                          tcost_mult: float) -> float:
    parts = []
    for tdf in trades_by_fold.values():
        parts.append(apply_cost_stress(
            tdf, slippage_mult=slippage_mult, tcost_mult=tcost_mult))
    if not parts:
        return 0.0
    return float(np.concatenate(parts).sum())


def run_tcost_curve(trades_by_fold: Mapping[int, pd.DataFrame],
                      multipliers=TCOST_MULTIPLIERS) -> dict:
    curve = {}
    for m in multipliers:
        parts = []
        for tdf in trades_by_fold.values():
            parts.append(apply_cost_stress(
                tdf, slippage_mult=1.0, tcost_mult=m))
        pnl = (np.concatenate(parts) if parts
               else np.array([], dtype=np.float64))
        metrics = _fold_metrics(pnl)
        curve[float(m)] = {
            "tcost_multiplier": float(m),
            "pooled_pf": metrics["pf"],
            "pooled_net": metrics["net"],
            "pooled_max_dd": metrics["dd"],
            "pooled_trade_count": metrics["n"],
            "pooled_wr": metrics["wr"],
        }
    return curve


def _bisect_break_even(trades_by_fold: Mapping[int, pd.DataFrame],
                          axis: str,
                          *,
                          lo: float = 1.0,
                          hi: float = 20.0,
                          tol: float = 1e-3,
                          max_iter: int = 60) -> dict:
    """Return the multiplier at which pooled Net crosses zero.

    ``axis`` is ``"slippage"`` or ``"tcost"``.
    """
    def net_at(mult: float) -> float:
        if axis == "slippage":
            return _pool_stressed_net(trades_by_fold, mult, 1.0)
        if axis == "tcost":
            return _pool_stressed_net(trades_by_fold, 1.0, mult)
        raise ValueError(f"unknown axis: {axis}")

    n_lo = net_at(lo)
    n_hi = net_at(hi)

    # If Net is already <= 0 at 1x, break-even is at or below production.
    if n_lo <= 0:
        return {
            "axis": axis, "found": False,
            "reason": "pooled_net <= 0 at production baseline multiplier",
            "net_at_lo": n_lo, "net_at_hi": n_hi,
            "break_even_multiplier": float("nan"),
        }
    # If still positive at hi, extend upper bound geometrically once.
    if n_hi > 0:
        hi *= 5.0
        n_hi = net_at(hi)
        if n_hi > 0:
            return {
                "axis": axis, "found": False,
                "reason": f"pooled_net still positive at {hi}x",
                "net_at_lo": n_lo, "net_at_hi": n_hi,
                "break_even_multiplier": float("inf"),
            }

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        n_mid = net_at(mid)
        if abs(n_mid) < tol or (hi - lo) < tol:
            return {
                "axis": axis, "found": True,
                "break_even_multiplier": float(mid),
                "net_at_break_even": float(n_mid),
                "iterations": _ + 1,
                "net_at_lo": n_lo, "net_at_hi": n_hi,
            }
        if n_mid > 0:
            lo, n_lo = mid, n_mid
        else:
            hi, n_hi = mid, n_mid
    return {
        "axis": axis, "found": True,
        "break_even_multiplier": float(0.5 * (lo + hi)),
        "net_at_break_even": float(net_at(0.5 * (lo + hi))),
        "iterations": max_iter,
        "net_at_lo": n_lo, "net_at_hi": n_hi,
    }


def find_break_even_tcost(trades_by_fold: Mapping[int, pd.DataFrame]
                            ) -> dict:
    return _bisect_break_even(trades_by_fold, "tcost")


def find_break_even_slippage(trades_by_fold: Mapping[int, pd.DataFrame]
                               ) -> dict:
    return _bisect_break_even(trades_by_fold, "slippage")
