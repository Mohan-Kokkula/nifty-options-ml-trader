"""
structure_features.py — HH/HL price-structure feature (parallel output only).

Historical evidence (8-week replay, see project research log): of six candidate
structural features evaluated for incremental ML value over the existing
RSI/ADX/EMA-slope/VWAP-distance feature set, HH/HL structure was the only one
with a consistent (if modest) positive signal across mutual information,
conditional mutual information, permutation importance, and held-out AUC.

This module computes ONLY that one feature. It is intentionally NOT wired into
any trading decision, gate, or threshold — it is logged every cycle (same
"parallel output" pattern already used by core/ml_engine.py's probability
calibrator) so it accumulates real production data for the next model
retrain. Nothing here changes what the bot trades.

Causality note (important): a "swing high" in the usual technical-analysis
sense requires confirmation from bars AFTER it (closes[i] > closes[i-1] and
closes[i] > closes[i+1]), which is not computable live — bar i+1 doesn't
exist yet at decision time. This module instead uses a TRAILING-ONLY swing
definition (closes[i] compared only to the SWING_LAG bars before it), which
is deployable in real time and gives identical results whether computed
historically or live. Do not swap in a centered/confirmed definition without
re-deriving both the training-time and live-time code paths together.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_LOOKBACK = 20   # bars (5-min bars => 100 minutes), matches the historical analysis
DEFAULT_SWING_LAG = 2    # a bar must exceed the preceding N bars to count as a trailing swing point


def compute_hh_hl_structure(
    closes: list[float],
    lookback: int = DEFAULT_LOOKBACK,
    swing_lag: int = DEFAULT_SWING_LAG,
) -> dict:
    """
    Trailing-only (causal) higher-high/higher-low and lower-high/lower-low
    structure over the last `lookback` closes.

    Args:
        closes: chronological close prices, oldest first. Only the last
            `lookback` values are used; anything before that is ignored.
        lookback: number of trailing bars to evaluate (default 20, matches
            the historical replay this feature was validated against).
        swing_lag: a bar counts as a trailing swing high/low if it exceeds
            (or is below) every one of the preceding `swing_lag` bars.

    Returns:
        {
            "hh_hl_up": bool,     # swing highs AND swing lows both non-decreasing
            "hh_hl_down": bool,   # swing highs AND swing lows both non-increasing
            "n_swing_highs": int,
            "n_swing_lows": int,
            "n_bars_used": int,
            "insufficient_data": bool,  # True if fewer than lookback bars were available
        }

    Never raises — malformed input (too few bars, non-numeric values) returns
    insufficient_data=True with both flags False, matching this codebase's
    fail-open convention for advisory/diagnostic features.
    """
    empty = {
        "hh_hl_up": False, "hh_hl_down": False,
        "n_swing_highs": 0, "n_swing_lows": 0,
        "n_bars_used": 0, "insufficient_data": True,
    }
    try:
        if not closes or swing_lag < 1 or lookback < swing_lag + 2:
            return empty

        window = [float(c) for c in closes[-lookback:]]
        n = len(window)
        if n < swing_lag + 2:
            return empty

        swing_highs: list[float] = []
        swing_lows: list[float] = []
        for i in range(swing_lag, n):
            trailing = window[i - swing_lag:i]
            if window[i] > max(trailing):
                swing_highs.append(window[i])
            if window[i] < min(trailing):
                swing_lows.append(window[i])

        hh_hl_up = (
            len(swing_highs) >= 2 and len(swing_lows) >= 2
            and _non_decreasing(swing_highs) and _non_decreasing(swing_lows)
        )
        hh_hl_down = (
            len(swing_highs) >= 2 and len(swing_lows) >= 2
            and _non_increasing(swing_highs) and _non_increasing(swing_lows)
        )

        return {
            "hh_hl_up": bool(hh_hl_up),
            "hh_hl_down": bool(hh_hl_down),
            "n_swing_highs": len(swing_highs),
            "n_swing_lows": len(swing_lows),
            "n_bars_used": n,
            "insufficient_data": n < lookback,
        }
    except Exception:
        return empty


def _non_decreasing(vals: list[float]) -> bool:
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def _non_increasing(vals: list[float]) -> bool:
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
