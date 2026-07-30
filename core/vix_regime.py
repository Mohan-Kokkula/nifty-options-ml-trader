"""
vix_regime.py — Dynamic VIX Regime Engine
==========================================
Replaces the hard VIX > 22 cutoff with a tiered system that adjusts
SL/TP multipliers, confidence thresholds, and strike selection
based on the current VIX level AND direction.

Key insight: High VIX = wider ranges = bigger opportunity with adjusted risk.
VIX > 22 days average 208+ pts range vs 143 pts normal — skipping them
entirely leaves money on the table.

VIX also mean-reverts: when VIX > 25, next-day change averages -0.34,
meaning elevated VIX days often precede strong directional moves.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VIXRegime:
    """Parameters for the current VIX regime."""
    name: str               # CALM, NORMAL, ELEVATED, HIGH, VERY_HIGH, EXTREME
    vix_level: float        # Current VIX value
    vix_change: float       # VIX change from previous (positive = rising)
    vix_direction: str      # RISING, FALLING, STABLE

    # Trading adjustments
    tradeable: bool         # Whether to trade at all
    confidence_adj: int     # Add/subtract from confidence (e.g. -15)
    sl_multiplier: float    # Scale SL by this factor (e.g. 1.5 = 50% wider)
    tp_multiplier: float    # Scale TP by this factor
    min_ml_confidence: float  # ML must exceed this to generate signal
    preferred_strike: str   # ATM, OTM_1, OTM_2
    max_trades_per_day: int # Override daily trade limit
    trend_only: bool        # Only trade with confirmed trend (ADX > 25)


# VIX regime definitions
def classify_vix(vix: float, vix_prev: float = 0.0) -> VIXRegime:
    """
    Classify current VIX into a trading regime with adjusted parameters.

    Args:
        vix: Current India VIX level
        vix_prev: Previous day's VIX (for direction detection)

    Returns:
        VIXRegime with all trading adjustments
    """
    # VIX direction
    vix_change = vix - vix_prev if vix_prev > 0 else 0.0
    if abs(vix_change) < 0.5:
        vix_dir = "STABLE"
    elif vix_change > 0:
        vix_dir = "RISING"
    else:
        vix_dir = "FALLING"

    if vix < 12:
        regime = VIXRegime(
            name="CALM",
            vix_level=vix,
            vix_change=vix_change,
            vix_direction=vix_dir,
            tradeable=True,
            confidence_adj=+5,          # Slightly boost confidence
            sl_multiplier=0.85,         # Tighter SL (low vol = small moves)
            tp_multiplier=0.85,         # Proportionally smaller TP
            min_ml_confidence=0.30,     # Normal threshold
            preferred_strike="ATM",     # ATM premiums are cheap
            max_trades_per_day=5,
            trend_only=False,
        )
    elif vix < 16:
        regime = VIXRegime(
            name="NORMAL",
            vix_level=vix,
            vix_change=vix_change,
            vix_direction=vix_dir,
            tradeable=True,
            confidence_adj=0,
            sl_multiplier=1.0,
            tp_multiplier=1.0,
            min_ml_confidence=0.30,
            preferred_strike="ATM",
            max_trades_per_day=5,
            trend_only=False,
        )
    elif vix < 20:
        regime = VIXRegime(
            name="ELEVATED",
            vix_level=vix,
            vix_change=vix_change,
            vix_direction=vix_dir,
            tradeable=True,
            confidence_adj=-3,
            sl_multiplier=1.15,         # Slightly wider SL
            tp_multiplier=1.2,          # Bigger moves available
            min_ml_confidence=0.33,
            preferred_strike="ATM",
            max_trades_per_day=4,
            trend_only=False,
        )
    elif vix < 25:
        # HIGH VIX — this is where the old system blocked everything.
        # Now we trade with wider SL and only strong signals.
        # 2026-03-30: 600pt drop missed because thresholds too conservative.
        regime = VIXRegime(
            name="HIGH",
            vix_level=vix,
            vix_change=vix_change,
            vix_direction=vix_dir,
            tradeable=True,             # KEY CHANGE: still tradeable
            confidence_adj=-5,          # Reduced from -10 (was blocking valid PUT signals)
            sl_multiplier=1.5,          # 50% wider SL (absorb noise)
            tp_multiplier=1.6,          # Bigger targets (avg 208pt range)
            min_ml_confidence=0.35,     # Reasonable bar (was 0.40)
            preferred_strike="OTM_1",   # Cheaper premium = less at risk
            max_trades_per_day=3,       # Fewer trades
            trend_only=False,
        )
    elif vix < 30:
        # VERY HIGH — significant fear, but also massive opportunity.
        # VIX mean-reverts: next-day avg change is -0.34 at VIX > 25.
        # 2026-03-30 lesson: VIX=28.7, 600pt drop = massive PUT opportunity.
        # Don't block these — trade with wider SL and trend confirmation.
        regime = VIXRegime(
            name="VERY_HIGH",
            vix_level=vix,
            vix_change=vix_change,
            vix_direction=vix_dir,
            tradeable=True,             # Always tradeable
            confidence_adj=-10,         # Reduced from -15 (was blocking valid signals)
            sl_multiplier=1.8,          # Much wider SL (avg 238pt range)
            tp_multiplier=2.0,          # Large targets available
            min_ml_confidence=0.38,     # Lowered from 0.45 — strong trends are clear
            preferred_strike="OTM_1",   # OTM_1 not OTM_2 — need decent premium
            max_trades_per_day=3,       # Allow 3 (was 2)
            trend_only=False,           # Don't require ADX>25 — EMA trend is enough
        )
    else:
        # EXTREME (VIX > 30) — crash/panic territory.
        # Avg 322pt range. These are the biggest move days.
        # Trade both directions — VIX falling = recovery bounce, rising = trend continues.
        regime = VIXRegime(
            name="EXTREME",
            vix_level=vix,
            vix_change=vix_change,
            vix_direction=vix_dir,
            tradeable=True,             # Always tradeable (was falling-only)
            confidence_adj=-20,         # Was -25
            sl_multiplier=2.0,          # 2x SL for crash volatility
            tp_multiplier=2.5,          # Massive targets possible
            min_ml_confidence=0.42,     # Lowered from 0.50
            preferred_strike="OTM_1",
            max_trades_per_day=2,       # Max 2 trades
            trend_only=False,           # EMA trend filter is enough
        )

    return regime


def apply_vix_to_sl_tp(
    sl_pts: float,
    tp_pts: float,
    regime: VIXRegime,
    min_sl: float = 20.0,
    max_sl: float = 90.0,  # Raised from 60 for high-VIX regimes
    min_tp: float = 50.0,
    max_tp: float = 350.0,  # Raised from 200 for high-VIX regimes
) -> tuple:
    """
    Scale SL/TP by VIX regime multiplier, clamped to bounds.
    Returns (adjusted_sl, adjusted_tp).
    """
    adj_sl = max(min_sl, min(max_sl, sl_pts * regime.sl_multiplier))
    adj_tp = max(min_tp, min(max_tp, tp_pts * regime.tp_multiplier))

    # Ensure minimum R:R of 2:1
    if adj_tp < adj_sl * 2:
        adj_tp = adj_sl * 2

    return round(adj_sl, 1), round(adj_tp, 1)


def format_regime(regime: VIXRegime) -> str:
    """Format regime for logging."""
    return (
        f"VIX={regime.vix_level:.1f} ({regime.name}) "
        f"dir={regime.vix_direction} chg={regime.vix_change:+.1f} | "
        f"SL x{regime.sl_multiplier:.2f} TP x{regime.tp_multiplier:.2f} | "
        f"strike={regime.preferred_strike} "
        f"conf_adj={regime.confidence_adj:+d} "
        f"{'TREND-ONLY' if regime.trend_only else ''}"
    )
