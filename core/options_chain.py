"""
options_chain.py — Fetch NIFTY option chain via Kotak Neo + compute Greeks.

Equivalent to Sensibull's option chain view, but built on Kotak Neo's
search_scrip + quotes APIs. No scraping, no ToS issues.

USAGE (called from claude_pilot before entry):
    from core.options_chain import get_atm_greeks

    g = get_atm_greeks(
        kotak_client=self.trader.client,
        spot=24050,
        dte_days=2,
        option_type='CE',
    )
    # {'strike': 24050, 'premium': 120, 'iv': 0.163, 'delta': 0.51, ...}

This adds the OPTIONS-AWARE signal that Tier 1 #1 needs:
    - Skip if IV too high (overpriced — even right direction loses to theta)
    - Skip if delta too low (deep OTM = won't move enough)
    - Adjust SL based on theta decay rate
"""
from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import Optional

from core.options_math import analyze_option, implied_volatility, compute_greeks

logger = logging.getLogger(__name__)

# Cache to avoid hitting Kotak repeatedly for the same strike in one cycle
_CACHE: dict = {}
_CACHE_TTL = 30   # seconds


def _round_to_strike(price: float, step: int = 50) -> int:
    return int(round(price / step) * step)


def get_atm_greeks(
    kotak_client,
    spot: float,
    dte_days: float,
    option_type: str = "CE",
    expiry_str: Optional[str] = None,
) -> Optional[dict]:
    """
    Pull ATM strike option premium from Kotak, compute IV + Greeks.
    Returns None on failure (caller should fail-open).
    """
    try:
        if not kotak_client or spot <= 0 or dte_days <= 0:
            return None

        atm_strike = _round_to_strike(spot, 50)
        cache_key = f"{atm_strike}_{option_type}_{int(time.time() // _CACHE_TTL)}"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        # Fetch premium via Kotak quote
        try:
            quote = kotak_client.get_option_quote(
                underlying="NIFTY",
                expiry=expiry_str,
                strike=atm_strike,
                option_type=option_type,
                quote_type="ltp",
            )
            premium = float(
                quote.get("ltp") or
                quote.get("data", {}).get("ltp") or
                0
            )
        except Exception as e:
            logger.debug(f"get_option_quote failed for ATM: {e}")
            return None

        if premium <= 0:
            return None

        # Compute IV + Greeks
        result = analyze_option(
            spot=spot,
            strike=atm_strike,
            dte_days=dte_days,
            premium=premium,
            opt_type=option_type,
        )
        result["strike"] = atm_strike
        result["premium"] = premium

        _CACHE[cache_key] = result
        return result
    except Exception as e:
        logger.debug(f"get_atm_greeks failed (fail-open): {e}")
        return None


def is_premium_overpriced(greeks: dict, vix_now: float) -> bool:
    """
    Return True if ATM IV is significantly above VIX → options overpriced.

    Rule: ATM IV > VIX + 4 percentage points → premium-decay risk
    (NSE VIX = ATM IV of front month, so they should be close.)
    """
    if not greeks or greeks.get("iv") is None or vix_now <= 0:
        return False
    iv_pct = greeks["iv"] * 100
    if iv_pct > vix_now + 4.0:
        return True
    return False


def is_delta_too_low(greeks: dict, min_delta: float = 0.35) -> bool:
    """
    Return True if option is too far OTM (delta too low) — won't move enough.
    For CE, delta=0.50 = ATM, lower = more OTM.
    For PE, delta=-0.50 = ATM, more negative = more ITM, less negative = more OTM.
    """
    if not greeks or greeks.get("delta") is None:
        return False
    abs_delta = abs(greeks["delta"])
    return abs_delta < min_delta


def theta_per_hour(greeks: dict) -> float:
    """Hourly theta cost — useful for estimating cost of holding."""
    if not greeks or greeks.get("theta_per_day") is None:
        return 0.0
    # NIFTY market: 6.25 trading hours/day
    return greeks["theta_per_day"] / 6.25
