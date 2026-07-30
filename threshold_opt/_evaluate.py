"""Per-candidate per-fold evaluation.

Replicates the mathematics of
``backtest_threshold_sweep.signals_from_probas`` locally so ``min_edge``
becomes a per-candidate variable rather than a module-level constant.
This is a re-implementation, NOT a modification of the original — the
original file is not touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._base import ThresholdCandidate


# ---------------------------------------------------------------------------
def apply_thresholds(probs: np.ndarray,
                       cand: ThresholdCandidate) -> np.ndarray:
    """Return signal array (0=CALL, 1=PUT, 2=SKIP) for each bar.

    Behaviour is identical to
    :func:`backtest_threshold_sweep.signals_from_probas` when called
    with ``(call_thr, put_thr, skip_ceil, min_edge=MIN_EDGE=0.05)`` —
    verified in the unit tests.
    """
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(
            f"probs must have shape (n, 3), got {probs.shape}")
    p_call = probs[:, 0]
    p_put = probs[:, 1]
    p_skip = probs[:, 2]
    sig = np.full(len(probs), 2, dtype=np.int8)
    sig[(p_call >= cand.call_thr)
        & (p_call - p_put >= cand.min_edge)
        & (p_skip < cand.skip_ceil)] = 0
    sig[(p_put >= cand.put_thr)
        & (p_put - p_call >= cand.min_edge)
        & (p_skip < cand.skip_ceil)] = 1
    return sig


# ---------------------------------------------------------------------------
def _fold_metrics(pnl: np.ndarray) -> dict:
    """Standard per-fold metrics for a trade P&L array."""
    if len(pnl) == 0:
        return dict(n=0, pf=float("nan"), wr=float("nan"), net=0.0,
                    avg=0.0, dd=0.0, sharpe=float("nan"),
                    sortino=float("nan"), expectancy=0.0,
                    avg_win=0.0, avg_loss=0.0)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    # Match stat_utils.profit_factor convention:
    #   * both zero → NaN (undefined)
    #   * losses=0, gains>0 → +inf
    #   * else → gains / losses
    if losses <= 0.0 and gains <= 0.0:
        pf = float("nan")
    elif losses <= 0.0:
        pf = float("inf")
    else:
        pf = float(gains / losses)
    eq = np.cumsum(pnl)
    dd = float(-(eq - np.maximum.accumulate(eq)).min())
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    sh = float(pnl.mean() / sd) if sd > 0 else float("nan")
    downside = pnl[pnl < 0]
    if downside.size and (downside ** 2).mean() > 0:
        sortino = float(pnl.mean() / np.sqrt((downside ** 2).mean()))
    else:
        sortino = float("nan")
    wr = float((pnl > 0).mean())
    wins = pnl[pnl > 0]
    loss = pnl[pnl <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(loss.mean()) if len(loss) else 0.0
    expectancy = wr * avg_win + (1.0 - wr) * avg_loss
    return dict(
        n=int(len(pnl)),
        pf=pf,
        wr=wr,
        net=float(pnl.sum()),
        avg=float(pnl.mean()),
        dd=dd,
        sharpe=sh,
        sortino=sortino,
        expectancy=float(expectancy),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateResult:
    """Full evaluation result for one candidate."""

    candidate: ThresholdCandidate
    per_fold: dict[int, dict]
    pooled: dict
    passes_min_trades: bool
    trade_pnl_by_fold: dict[int, np.ndarray]


def evaluate_candidate(
    cand: ThresholdCandidate,
    fold_data: dict[int, tuple[np.ndarray, Any, Any, Any]],
    min_trades: int = 50,
) -> CandidateResult:
    """Evaluate a candidate across pre-loaded fold data.

    Parameters
    ----------
    cand : ThresholdCandidate
    fold_data : Mapping[int, (probs, test_df, iv_map, exp_map)]
        Per-fold inputs. Loaded once and reused across many candidates
        so this function stays cheap in a grid loop.
    min_trades : int
        Minimum pooled trade count required for the result to pass the
        pre-registered filter. Rejected candidates are still returned
        (with ``passes_min_trades=False``) so the caller can retain the
        full search table.
    """
    from backtest_options import simulate_trades

    per_fold: dict[int, dict] = {}
    trade_pnl_by_fold: dict[int, np.ndarray] = {}
    for fold_idx, (probs, test_df, iv, exp) in fold_data.items():
        sig = apply_thresholds(probs, cand)
        tdf = simulate_trades(test_df, sig, probs, iv, exp)
        pnl = (tdf["net_option"].values.astype(np.float64)
               if len(tdf) and "net_option" in tdf.columns
               else np.array([], dtype=np.float64))
        per_fold[fold_idx] = _fold_metrics(pnl)
        trade_pnl_by_fold[fold_idx] = pnl

    all_pnl = (np.concatenate(list(trade_pnl_by_fold.values()))
               if trade_pnl_by_fold
               else np.array([], dtype=np.float64))
    pooled = _fold_metrics(all_pnl)
    passes = pooled["n"] >= int(min_trades)
    return CandidateResult(
        candidate=cand,
        per_fold=per_fold,
        pooled=pooled,
        passes_min_trades=bool(passes),
        trade_pnl_by_fold=trade_pnl_by_fold,
    )
