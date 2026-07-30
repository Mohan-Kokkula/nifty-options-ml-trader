"""R5 – block bootstrap and paired bootstrap CIs (delegates to stat_utils)."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from stat_utils import block_bootstrap_ci
from stat_utils.metrics import profit_factor

from ._base import BOOTSTRAP_B, DEFAULT_SEED, InvalidInputError


def _pf_stat(pnl: np.ndarray) -> float:
    """Wraps stat_utils.profit_factor to give a plain-float statistic."""
    return float(profit_factor(pnl))


def _net_stat(pnl: np.ndarray) -> float:
    return float(pnl.sum()) if pnl.size else 0.0


def block_bootstrap_pf(pnl_by_fold: Mapping[int, np.ndarray],
                          *, seed: int = DEFAULT_SEED,
                          n_resamples: int = BOOTSTRAP_B,
                          ci_level: float = 0.9) -> dict:
    if not pnl_by_fold:
        raise InvalidInputError("empty pnl_by_fold")
    res = block_bootstrap_ci(
        pnl_by_fold, _pf_stat,
        n_resamples=n_resamples, ci_level=ci_level, seed=seed,
        stat_name="pooled_pf",
    )
    return {
        "stat_name": res.stat_name,
        "point_estimate": float(res.point_estimate),
        "lower": float(res.lower) if res.lower is not None else None,
        "upper": float(res.upper) if res.upper is not None else None,
        "ci_level": float(res.ci_level),
        "n_resamples": int(res.n_resamples),
        "n_valid_resamples": int(res.n_valid_resamples),
        "n_blocks": int(res.n_blocks),
        "block_unit": res.block_unit,
        "method": res.method,
        "seed": int(seed),
        "paired": False,
    }


def block_bootstrap_net(pnl_by_fold: Mapping[int, np.ndarray],
                          *, seed: int = DEFAULT_SEED,
                          n_resamples: int = BOOTSTRAP_B,
                          ci_level: float = 0.9) -> dict:
    if not pnl_by_fold:
        raise InvalidInputError("empty pnl_by_fold")
    res = block_bootstrap_ci(
        pnl_by_fold, _net_stat,
        n_resamples=n_resamples, ci_level=ci_level, seed=seed,
        stat_name="pooled_net",
    )
    return {
        "stat_name": res.stat_name,
        "point_estimate": float(res.point_estimate),
        "lower": float(res.lower) if res.lower is not None else None,
        "upper": float(res.upper) if res.upper is not None else None,
        "ci_level": float(res.ci_level),
        "n_resamples": int(res.n_resamples),
        "n_valid_resamples": int(res.n_valid_resamples),
        "n_blocks": int(res.n_blocks),
        "block_unit": res.block_unit,
        "method": res.method,
        "seed": int(seed),
        "paired": False,
    }


def paired_bootstrap_delta_pf(a_by_fold: Mapping[int, np.ndarray],
                                 b_by_fold: Mapping[int, np.ndarray],
                                 *, seed: int = DEFAULT_SEED,
                                 n_resamples: int = BOOTSTRAP_B,
                                 ci_level: float = 0.9) -> dict:
    """Paired-fold bootstrap CI on ΔPF = PF(a) - PF(b)."""
    common = sorted(set(a_by_fold.keys()) & set(b_by_fold.keys()))
    if not common:
        raise InvalidInputError("no common folds for paired bootstrap")
    stacked = {int(f): np.concatenate([a_by_fold[f], b_by_fold[f]])
               for f in common}
    lens = {int(f): (len(a_by_fold[f]), len(b_by_fold[f])) for f in common}

    def _safe_pf(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        return float(profit_factor(arr))

    def _dpf(concat: np.ndarray, fold: int) -> float:
        na, _nb = lens[fold]
        return _safe_pf(concat[:na]) - _safe_pf(concat[na:])

    # block_bootstrap_ci supports statistic(concat) — supply a wrapper that
    # ignores the fold label by carrying it via a closure over ordering.
    # To thread the fold id through, we resample folds ourselves.
    rng = np.random.default_rng(seed)
    folds = np.asarray(common, dtype=np.int64)
    n = len(folds)
    boot = np.empty(n_resamples, dtype=np.float64)
    valid = 0
    for i in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        vals = []
        for j in pick:
            f = int(folds[j])
            vals.append(_dpf(stacked[f], f))
        v = float(np.mean(vals))
        if np.isfinite(v):
            boot[valid] = v
            valid += 1
    boot = boot[:valid]

    # Point estimate uses full (unresampled) data — pooled PF diff.
    pooled_a = np.concatenate([a_by_fold[f] for f in common])
    pooled_b = np.concatenate([b_by_fold[f] for f in common])
    def _pf_or_nan(a):
        return float("nan") if a.size == 0 else float(profit_factor(a))
    point = _pf_or_nan(pooled_a) - _pf_or_nan(pooled_b)

    alpha = 1.0 - ci_level
    if valid < 2:
        lo = hi = None
    else:
        lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
        hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "stat_name": "delta_pf_a_minus_b",
        "point_estimate": point,
        "lower": lo,
        "upper": hi,
        "ci_level": float(ci_level),
        "n_resamples": int(n_resamples),
        "n_valid_resamples": int(valid),
        "n_blocks": int(n),
        "block_unit": "fold",
        "method": "percentile",
        "seed": int(seed),
        "paired": True,
        "common_folds": common,
    }
