"""R6 – stability diagnostics."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from threshold_opt._evaluate import _fold_metrics

from ._base import ROLLING_WINDOW


def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]


def fold_variance(pnl_by_fold: Mapping[int, np.ndarray]) -> dict:
    folds = sorted(pnl_by_fold.keys())
    per_fold_pf = []
    per_fold_net = []
    per_fold_dd = []
    for f in folds:
        m = _fold_metrics(pnl_by_fold[f])
        per_fold_pf.append(m["pf"])
        per_fold_net.append(m["net"])
        per_fold_dd.append(m["dd"])
    pfs = _finite(per_fold_pf)
    mean_pf = float(np.mean(pfs)) if pfs else float("nan")
    std_pf = float(np.std(pfs, ddof=0)) if len(pfs) >= 2 else 0.0
    cv_pf = (std_pf / abs(mean_pf)) if mean_pf and np.isfinite(mean_pf) else float("nan")
    return {
        "per_fold_pf": {int(f): pv for f, pv in zip(folds, per_fold_pf)},
        "per_fold_net": {int(f): nv for f, nv in zip(folds, per_fold_net)},
        "per_fold_max_dd": {int(f): dv for f, dv in zip(folds, per_fold_dd)},
        "mean_per_fold_pf": mean_pf,
        "std_per_fold_pf": std_pf,
        "cv_per_fold_pf": cv_pf,
        "n_folds_with_finite_pf": len(pfs),
    }


def rolling_report(pnl_by_fold: Mapping[int, np.ndarray],
                     w: int = ROLLING_WINDOW) -> list[dict]:
    folds = sorted(pnl_by_fold.keys())
    rows = []
    for i in range(0, len(folds) - w + 1):
        subset = folds[i:i + w]
        arr = np.concatenate([pnl_by_fold[f] for f in subset])
        m = _fold_metrics(arr)
        rows.append({
            "start_fold": subset[0],
            "end_fold": subset[-1],
            "pf": m["pf"],
            "net": m["net"],
            "max_dd": m["dd"],
            "n_trades": m["n"],
        })
    return rows


def flag_unstable_folds(pnl_by_fold: Mapping[int, np.ndarray],
                           z_thresh: float = 2.0) -> list[int]:
    folds = sorted(pnl_by_fold.keys())
    pfs = []
    for f in folds:
        pfs.append(_fold_metrics(pnl_by_fold[f])["pf"])
    finite = [(f, p) for f, p in zip(folds, pfs) if np.isfinite(p)]
    if len(finite) < 3:
        return []
    vals = np.array([p for _, p in finite], dtype=np.float64)
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0)) or 1e-9
    flagged = [int(f) for f, p in finite if abs((p - mu) / sd) > z_thresh]
    return flagged


def stability_report(pnl_by_fold: Mapping[int, np.ndarray]) -> dict:
    fv = fold_variance(pnl_by_fold)
    return {
        "fold_variance": fv,
        "rolling_metrics": rolling_report(pnl_by_fold),
        "unstable_folds_z_gt_2": flag_unstable_folds(pnl_by_fold),
    }
