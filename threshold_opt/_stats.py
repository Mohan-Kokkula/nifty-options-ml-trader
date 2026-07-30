"""Statistical comparison of winner vs baseline + top-K vs baseline.

All primitives come from ``stat_utils`` — Phase 6 introduces no new
statistical methods. Wrappers are provided so the orchestrator can call
one function per comparison and receive a JSON-safe result dict.
"""
from __future__ import annotations

import numpy as np


def _dpf(a: np.ndarray, b: np.ndarray) -> float:
    """Delta profit factor callable for paired bootstrap."""
    from stat_utils import profit_factor
    pfa = profit_factor(a) if len(a) else float("nan")
    pfb = profit_factor(b) if len(b) else float("nan")
    if not (np.isfinite(pfa) and np.isfinite(pfb)):
        return float("nan")
    return float(pfa - pfb)


def _folds_common(a_by_fold: dict, b_by_fold: dict) -> list[int]:
    return sorted(set(a_by_fold) & set(b_by_fold))


def _per_fold_pf_vector(pnl_by_fold: dict[int, np.ndarray],
                          folds: list[int]) -> np.ndarray:
    from stat_utils import profit_factor
    out = np.zeros(len(folds), dtype=np.float64)
    for i, k in enumerate(folds):
        pnl = pnl_by_fold[k]
        pf = profit_factor(pnl) if pnl.size else 0.0
        out[i] = pf if np.isfinite(pf) else 0.0
    return out


# ---------------------------------------------------------------------------
def compare_to_baseline(
    winner_pnl_by_fold: dict[int, np.ndarray],
    baseline_pnl_by_fold: dict[int, np.ndarray],
    seed: int = 42,
) -> dict:
    """Paired block-bootstrap 90% CI on ΔPF + Diebold-Mariano.

    DM is computed on the per-fold mean-P&L series (short but valid
    when the per-trade series are of different lengths between the two
    configurations).
    """
    from stat_utils import diebold_mariano, paired_block_bootstrap_ci

    common = _folds_common(winner_pnl_by_fold, baseline_pnl_by_fold)
    a = {k: winner_pnl_by_fold[k] for k in common}
    b = {k: baseline_pnl_by_fold[k] for k in common}

    ci = paired_block_bootstrap_ci(
        a, b, _dpf,
        n_resamples=10_000, ci_level=0.90, seed=int(seed),
        stat_name="dPF_winner_vs_baseline",
    )

    dm_result = None
    if len(common) >= 3:
        # Per-fold mean P&L differential
        a_means = np.array([winner_pnl_by_fold[k].mean()
                            if winner_pnl_by_fold[k].size else 0.0
                            for k in common])
        b_means = np.array([baseline_pnl_by_fold[k].mean()
                            if baseline_pnl_by_fold[k].size else 0.0
                            for k in common])
        try:
            dm = diebold_mariano(
                a_means, b_means, alternative="greater",
                lag=1, ci_level=0.90,
            )
            dm_result = dm.to_dict()
        except Exception as exc:
            dm_result = {"error": str(exc)}

    return {
        "paired_ci_90": ci.to_dict(),
        "diebold_mariano": dm_result,
        "common_folds": common,
    }


# ---------------------------------------------------------------------------
def top_k_comparison(
    top_k_pnl_by_fold: dict[str, dict[int, np.ndarray]],
    baseline_pnl_by_fold: dict[int, np.ndarray],
    seed: int = 42,
) -> dict:
    """Hansen SPA + White RC + Holm-Bonferroni over top-K candidates vs baseline.

    Per-fold Profit Factor is the performance measure. Fixed top-K
    ensures reproducibility (caller controls K by pre-filtering the
    input dict).
    """
    from stat_utils import (diebold_mariano, hansen_spa,
                              holm_bonferroni, white_reality_check)

    # Intersection of every candidate's folds with the baseline's
    if not top_k_pnl_by_fold:
        return {"note": "empty top-K set"}
    fold_sets = [set(v) for v in top_k_pnl_by_fold.values()]
    common = sorted(set.intersection(*fold_sets, set(baseline_pnl_by_fold)))
    if len(common) < 2:
        return {"note": f"insufficient common folds ({len(common)})",
                "common_folds": common}

    perf: dict[str, np.ndarray] = {}
    for name, pnl_by_fold in top_k_pnl_by_fold.items():
        perf[name] = _per_fold_pf_vector(pnl_by_fold, common)
    bench = _per_fold_pf_vector(baseline_pnl_by_fold, common)

    out: dict = {
        "k": len(top_k_pnl_by_fold),
        "common_folds": common,
    }
    try:
        spa = hansen_spa(perf, benchmark=bench,
                          n_bootstrap=5_000, seed=int(seed))
        out["hansen_spa"] = spa.to_dict()
    except Exception as exc:
        out["hansen_spa_error"] = str(exc)
    try:
        wrc = white_reality_check(perf, benchmark=bench,
                                    n_bootstrap=5_000, seed=int(seed))
        out["white_reality_check"] = wrc.to_dict()
    except Exception as exc:
        out["white_reality_check_error"] = str(exc)

    # Per-candidate DM vs baseline, then Holm
    dm_pvals: dict[str, float] = {}
    dm_stats: dict[str, dict] = {}
    for name, pnl_by_fold in top_k_pnl_by_fold.items():
        try:
            a_means = np.array([pnl_by_fold[k].mean()
                                if pnl_by_fold[k].size else 0.0
                                for k in common])
            b_means = np.array([baseline_pnl_by_fold[k].mean()
                                if baseline_pnl_by_fold[k].size else 0.0
                                for k in common])
            dm = diebold_mariano(a_means, b_means, alternative="greater",
                                  lag=1, ci_level=0.90)
            dm_pvals[name] = float(dm.pvalue)
            dm_stats[name] = dm.to_dict()
        except Exception as exc:
            dm_pvals[name] = 1.0
            dm_stats[name] = {"error": str(exc)}
    out["dm_pvalues"] = dm_pvals
    out["dm_statistics"] = dm_stats

    if dm_pvals:
        try:
            hb = holm_bonferroni(dm_pvals, alpha=0.10)
            out["holm_bonferroni"] = {k: v for k, v in hb.decisions.items()}
        except Exception as exc:
            out["holm_error"] = str(exc)

    return out
