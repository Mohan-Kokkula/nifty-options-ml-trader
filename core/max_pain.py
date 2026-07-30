"""
max_pain.py — Computes NIFTY Max-Pain strike from option chain OI.

Max-Pain theory:
    The strike where option WRITERS (sellers) profit MOST.
    Equivalently: the strike at which the total payout to BUYERS is MINIMUM.

    Markets often gravitate toward max-pain on expiry day (Tuesday for NIFTY).

USAGE:
    from core.max_pain import compute_max_pain, get_max_pain_bias

    mp = compute_max_pain(option_chain)
    # mp = 24000 (strike with maximum writer profit)

    bias = get_max_pain_bias(spot=24150, max_pain=24000)
    # bias = 'BEARISH'  (spot above MP → gravitates down)
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_max_pain(option_chain: list) -> Optional[int]:
    """
    Compute max-pain strike from option chain.

    option_chain: list of dicts, each with:
        {'strike': 24000, 'ce_oi': 12345, 'pe_oi': 6789}

    Returns: strike with maximum writer profit (= minimum buyer payout).
    """
    if not option_chain:
        return None

    # Get all unique strikes
    strikes = sorted({c.get("strike") for c in option_chain if c.get("strike")})
    if not strikes:
        return None

    # For each potential expiry close = strike S, compute total writer payout
    # Writer payout = OI × max(0, expiry_close - strike) for CE
    #               + OI × max(0, strike - expiry_close) for PE
    # We want to MAXIMIZE writer profit = MINIMIZE total payout to buyers

    min_payout = float("inf")
    max_pain_strike = None

    for s_close in strikes:
        total_payout = 0
        for c in option_chain:
            strike = c.get("strike", 0)
            ce_oi = c.get("ce_oi", 0) or 0
            pe_oi = c.get("pe_oi", 0) or 0
            if strike <= 0:
                continue
            # CE writer pays out if expiry > strike
            if s_close > strike:
                total_payout += ce_oi * (s_close - strike)
            # PE writer pays out if expiry < strike
            if s_close < strike:
                total_payout += pe_oi * (strike - s_close)
        if total_payout < min_payout:
            min_payout = total_payout
            max_pain_strike = s_close

    return max_pain_strike


def get_max_pain_bias(
    spot: float, max_pain: int, threshold_pct: float = 0.3,
) -> str:
    """
    Direction bias based on spot vs max-pain.

    Returns:
        'BEARISH' if spot >> max_pain (market likely to drop toward MP)
        'BULLISH' if spot << max_pain (market likely to rise toward MP)
        'NEUTRAL' if spot is near max_pain
    """
    if not max_pain or spot <= 0:
        return "NEUTRAL"
    diff_pct = (spot - max_pain) / max_pain * 100
    if diff_pct > threshold_pct:
        return "BEARISH"
    if diff_pct < -threshold_pct:
        return "BULLISH"
    return "NEUTRAL"


def fetch_option_chain_oi(kotak_client, expiry_str: str,
                          spot: float, strikes_each_side: int = 10) -> list:
    """
    Pull OI for a band of strikes around spot. Light wrapper around Kotak search.
    Returns list suitable for compute_max_pain().
    """
    try:
        # Use existing get_option_chain helper if available
        if hasattr(kotak_client, "get_option_chain"):
            chain = kotak_client.get_option_chain(
                underlying="NIFTY",
                expiry=expiry_str,
                strike_count=strikes_each_side,
            )
            out = []
            if isinstance(chain, dict) and "strikes" in chain:
                for s_data in chain["strikes"]:
                    out.append({
                        "strike": int(s_data.get("strike", 0)),
                        "ce_oi": int(s_data.get("ce_oi", 0) or 0),
                        "pe_oi": int(s_data.get("pe_oi", 0) or 0),
                    })
            return out
        return []
    except Exception as e:
        logger.debug(f"fetch_option_chain_oi failed: {e}")
        return []
