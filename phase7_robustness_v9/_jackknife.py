"""R2 – leave-one-fold-out jackknife.

For each fold i, drop it and pool the remaining folds. Report the pooled
metrics and ``ΔPF = pooled_PF_all - pooled_PF_minus_i``. Large |ΔPF|
means fold i disproportionately drives the aggregate.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from threshold_opt._evaluate import _fold_metrics


def _pf(pnl: np.ndarray) -> float:
    return float(_fold_metrics(pnl)["pf"])


def leave_one_out(pnl_by_fold: Mapping[int, np.ndarray]) -> dict:
    """Return per-fold LOO diagnostics."""
    folds = sorted(pnl_by_fold.keys())
    all_pnl = (np.concatenate([pnl_by_fold[f] for f in folds])
                 if folds else np.array([], dtype=np.float64))
    all_m = _fold_metrics(all_pnl)
    pf_all = all_m["pf"]

    per_drop = {}
    dpfs = []
    for drop in folds:
        kept = [f for f in folds if f != drop]
        arr = (np.concatenate([pnl_by_fold[f] for f in kept])
                if kept else np.array([], dtype=np.float64))
        m = _fold_metrics(arr)
        pf_minus = m["pf"]
        try:
            dpf = float(pf_all - pf_minus) if (
                np.isfinite(pf_all) and np.isfinite(pf_minus)) else float("nan")
        except Exception:
            dpf = float("nan")
        dpfs.append(dpf)
        per_drop[int(drop)] = {
            "kept_folds": kept,
            "pooled_pf": pf_minus,
            "pooled_net": m["net"],
            "pooled_max_dd": m["dd"],
            "pooled_trade_count": m["n"],
            "delta_pf_vs_all": dpf,
        }

    finite = [d for d in dpfs if np.isfinite(d)]
    influence = {
        "max_abs_delta_pf": float(max(abs(x) for x in finite)) if finite else float("nan"),
        "mean_abs_delta_pf": float(np.mean([abs(x) for x in finite])) if finite else float("nan"),
        "std_delta_pf": float(np.std(finite, ddof=0)) if len(finite) >= 2 else 0.0,
    }
    # Classify dependence
    if not finite:
        dependence = "UNKNOWN"
    elif influence["max_abs_delta_pf"] < 0.15:
        dependence = "LOW"
    elif influence["max_abs_delta_pf"] < 0.40:
        dependence = "MODERATE"
    else:
        dependence = "HIGH"

    return {
        "pooled_all_folds": {
            "pf": pf_all,
            "net": all_m["net"],
            "dd": all_m["dd"],
            "n": all_m["n"],
        },
        "per_dropped_fold": per_drop,
        "influence": influence,
        "single_fold_dependence": dependence,
    }
