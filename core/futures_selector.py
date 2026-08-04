"""
Futures Symbol Selector for Nifty Futures — order-placement resolver.

Companion to core/strike_selector.py (which resolves option CE/PE trading
symbols); this resolves the current-month NIFTY futures trading symbol for
order placement, not just market-data subscription.

Reuses the exact symbol-format logic already validated in production by
core/vwap_engine.py's _current_futures_symbol() (used for VWAP tick
subscription) and the same search_scrip lookup/fallback pattern already
proven working by core/tick_feed.py's _resolve_futures_token() — this
module adds the piece neither of those needed: returning a *validated
trading symbol string* (not just a token) suitable for
KotakNeoClient.place_order(symbol=...).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FuturesSelector:
    """
    Resolves and validates the current-month NIFTY futures trading symbol
    against the live broker instrument master before it's used to place
    an order. Never returns an unvalidated symbol — fails closed (None)
    if the broker doesn't confirm the contract exists.
    """

    def __init__(self, client, exchange: str = "NFO"):
        self.client = client
        self.exchange = exchange
        self._cached_symbol: Optional[str] = None
        self._cached_for_month: Optional[str] = None

    def resolve_trading_symbol(self) -> Optional[str]:
        """
        Returns the current-month NIFTY futures trading symbol (e.g.
        "NIFTY26AUGFUT"), validated against the live broker instrument
        master. Returns None if it can't be confirmed tradeable — callers
        must treat that as "do not place an order", not fall back to an
        unvalidated guess.
        """
        from core.vwap_engine import _current_futures_symbol
        wanted = _current_futures_symbol().upper()

        if self._cached_symbol and self._cached_for_month == wanted:
            return self._cached_symbol

        neo = getattr(self.client, "_neo", None)
        if neo is None:
            logger.warning("FuturesSelector: no underlying broker client available")
            return None

        try:
            results = neo.search_scrip(exchange_segment="nse_fo", symbol="nifty")
        except Exception as e:
            logger.warning(f"FuturesSelector: search_scrip failed: {e}")
            return None

        if not isinstance(results, list):
            logger.debug(
                "FuturesSelector: search_scrip returned non-list "
                f"(session not ready): type={type(results).__name__}"
            )
            return None
        if not results:
            logger.warning("FuturesSelector: search_scrip returned empty list for 'nifty'")
            return None

        # Pass 1: exact match on trading symbol (preferred contract month)
        for r in results:
            trd = str(r.get("pTrdSymbol", "") or r.get("trading_symbol", "")).upper()
            if trd == wanted:
                self._cached_symbol = trd
                self._cached_for_month = wanted
                logger.info(f"FuturesSelector: resolved trading symbol={trd}")
                return trd

        # Pass 2: any near-month NIFTY futures (ends with FUT, no option type)
        # — mirrors TickFeed._resolve_futures_token()'s fallback, so market-data
        # subscription and order placement can never silently disagree on
        # which contract is "current" during a same-day rollover edge case.
        for r in results:
            trd = str(r.get("pTrdSymbol", "")).upper()
            opt = str(r.get("pOptionType", "") or r.get("optionType", "")).strip()
            if trd.startswith("NIFTY") and trd.endswith("FUT") and not opt:
                logger.warning(
                    f"FuturesSelector: exact symbol {wanted} not found — "
                    f"using fallback {trd} instead"
                )
                self._cached_symbol = trd
                self._cached_for_month = wanted
                return trd

        logger.warning(
            f"FuturesSelector: NO tradeable NIFTY futures contract found "
            f"(wanted {wanted}). First 10 pTrdSymbols: "
            f"{[r.get('pTrdSymbol') for r in results[:10]]}"
        )
        return None

    def invalidate_cache(self) -> None:
        """Force re-resolution on next call — use on contract rollover day."""
        self._cached_symbol = None
        self._cached_for_month = None
