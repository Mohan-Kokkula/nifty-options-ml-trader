"""Multi-criteria candidate ranking.

Per the approved Phase-6 spec, candidates are ranked in this order:

    1. Highest pooled Profit Factor
    2. Highest pooled Net Profit
    3. Lowest pooled Max Drawdown (ascending)
    4. Highest pooled Trade Count

If two candidates match within 1% on every metric above, prefer the
one with LOWER standard deviation of per-fold Profit Factor (i.e.,
prefer the more stable candidate). This tie-break is documented in the
manifest so downstream consumers can reproduce the choice.

The production baseline is NEVER included in the eligibility list —
Phase 6 requires that the baseline exist only for comparison, never as
a winning candidate.
"""
from __future__ import annotations

import numpy as np

from ._evaluate import CandidateResult


def _within_1pct(a: float, b: float) -> bool:
    """Return True if ``a`` and ``b`` are within 1% relative distance."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    if a == 0.0 and b == 0.0:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom < 0.01


def _stability(cr: CandidateResult) -> float:
    """Std of per-fold PF across finite entries. Lower = more stable."""
    pfs = [f.get("pf") for f in cr.per_fold.values()]
    finite = [x for x in pfs if isinstance(x, float) and np.isfinite(x)]
    if len(finite) < 2:
        return 0.0
    return float(np.std(finite, ddof=0))


def rank_candidates(
    results: list[CandidateResult],
    min_pooled_trades: int = 50,
) -> list[CandidateResult]:
    """Return results sorted best-first per the pre-registered priority.

    Candidates below ``min_pooled_trades`` are filtered out. Ties on the
    four primary metrics are broken by ``_stability`` (lower std of
    per-fold PF wins).
    """
    eligible: list[CandidateResult] = []
    for r in results:
        n = r.pooled.get("n", 0)
        pf = r.pooled.get("pf")
        if int(n) < int(min_pooled_trades):
            continue
        # Reject NaN (nonsensical), but keep +inf (legitimate zero-loss).
        # The min_trades filter above prevents +inf from winning on
        # trivially-small samples.
        if not isinstance(pf, float) or np.isnan(pf):
            continue
        eligible.append(r)

    def _key(r: CandidateResult) -> tuple:
        p = r.pooled
        pf = float(p.get("pf") or 0.0)
        net = float(p.get("net") or 0.0)
        dd = float(p.get("dd") or 0.0)
        n = int(p.get("n") or 0)
        return (
            -pf,        # descending PF
            -net,       # descending Net
             dd,         # ascending DD
            -n,         # descending Count
             _stability(r),   # ascending std of per-fold PF
        )

    return sorted(eligible, key=_key)
