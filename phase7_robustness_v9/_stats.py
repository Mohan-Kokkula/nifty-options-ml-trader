"""R7 – statistical testing on NEW comparisons only.

Phase 5/6 already ran DM/SPA/WRC on the winner-vs-baseline comparison.
Phase 7 uses the same primitives on comparisons introduced by the
robustness variants (delay-stress, slippage-stress, jackknife LOO).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from stat_utils import diebold_mariano, hansen_spa, white_reality_check


def _to_perf_arr(pnl_by_fold: Mapping[int, np.ndarray]) -> np.ndarray:
    """Reduce per-fold PnL streams to one performance number per fold (sum)."""
    folds = sorted(pnl_by_fold.keys())
    return np.array([float(pnl_by_fold[f].sum()) for f in folds],
                      dtype=np.float64)


def dm_winner_vs_baseline_by_fold(winner_by_fold: Mapping[int, np.ndarray],
                                     baseline_by_fold: Mapping[int, np.ndarray]
                                     ) -> dict:
    """Diebold-Mariano test on per-fold net PnL: is winner better than baseline?

    ``stat_utils.diebold_mariano(loss_a, loss_b, alternative='greater')`` tests
    ``H0: E[loss_a - loss_b] <= 0`` against ``H1: E[loss_a - loss_b] > 0``.
    The alternative direction is "loss_a is LARGER than loss_b" — i.e. model a
    performs WORSE than model b.

    We want the opposite direction: "winner has SMALLER loss than baseline"
    (winner performs BETTER). The mathematically correct call is therefore
    ``diebold_mariano(-baseline_perf, -winner_perf, alternative='greater')``,
    where ``loss_a = -baseline_perf`` and ``loss_b = -winner_perf``:

        H1: E[-baseline_perf - (-winner_perf)] > 0
          = E[winner_perf - baseline_perf] > 0

    Positive ``mean_loss_diff`` now means winner outperforms baseline (in
    per-fold net-PnL units), matching how Phase 6 reported the same test.
    """
    common = sorted(set(winner_by_fold.keys()) & set(baseline_by_fold.keys()))
    if len(common) < 3:
        return {"note": f"insufficient common folds ({len(common)}) for DM",
                "common_folds": common}
    winner_perf = np.array([float(winner_by_fold[f].sum()) for f in common],
                              dtype=np.float64)
    baseline_perf = np.array([float(baseline_by_fold[f].sum()) for f in common],
                                dtype=np.float64)
    # Argument order is (baseline_loss, winner_loss). Under
    # alternative='greater' this tests "baseline has more loss than winner"
    # = winner outperforms baseline.
    r = diebold_mariano(-baseline_perf, -winner_perf,
                          alternative="greater", lag=1)
    return {
        "common_folds": common,
        "statistic": float(r.statistic),
        "pvalue": float(r.pvalue),
        "alternative": r.alternative,
        "lag": int(r.lag),
        "n": int(r.n),
        "mean_loss_diff": float(r.mean_loss_diff),
        "se_loss_diff": float(r.se_loss_diff),
        "ci_lower": float(r.ci_lower) if r.ci_lower is not None else None,
        "ci_upper": float(r.ci_upper) if r.ci_upper is not None else None,
        "hypothesis": ("H1: E[winner_perf - baseline_perf] > 0 "
                          "(winner outperforms baseline in per-fold net PnL)"),
        "interpretation_positive_stat":
            "positive statistic → winner has lower loss than baseline (better)",
    }


def spa_and_wrc_over_variants(variant_pnl_by_fold: dict[str, dict[int, np.ndarray]],
                                 baseline_by_fold: Mapping[int, np.ndarray],
                                 *, seed: int = 42) -> dict:
    """Hansen SPA + White RC on {variant: per-fold PnL} vs baseline.

    Perf metric = per-fold total PnL (sum). Names preserved for reporting.
    """
    if not variant_pnl_by_fold:
        return {"note": "no variants supplied"}
    common = sorted(set.intersection(
        *[set(m.keys()) for m in variant_pnl_by_fold.values()],
        set(baseline_by_fold.keys())))
    if len(common) < 3:
        return {"note": f"insufficient common folds ({len(common)})"}
    bench = np.array([float(baseline_by_fold[f].sum()) for f in common],
                        dtype=np.float64)
    perf = {name: np.array(
                [float(m[f].sum()) for f in common], dtype=np.float64)
            for name, m in variant_pnl_by_fold.items()}
    spa = hansen_spa(perf, benchmark=bench, seed=seed)
    wrc = white_reality_check(perf, benchmark=bench, seed=seed)
    return {
        "common_folds": common,
        "n_variants": len(variant_pnl_by_fold),
        "hansen_spa": {
            "pvalue_lower": float(spa.pvalue_lower),
            "pvalue_consistent": float(spa.pvalue_consistent),
            "pvalue_upper": float(spa.pvalue_upper),
            "statistic": float(spa.statistic),
            "n_bootstrap": int(spa.n_bootstrap),
            "block_length": int(spa.block_length),
            "seed": int(seed),
        },
        "white_reality_check": {
            "pvalue": float(wrc.pvalue),
            "statistic": float(wrc.statistic),
            "n_bootstrap": int(wrc.n_bootstrap),
            "block_length": int(wrc.block_length),
            "seed": int(seed),
        },
    }
