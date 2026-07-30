"""
Smart Strike Selector for Nifty Options.

Data sources (in priority order):
  1. Kotak Neo client.get_option_chain — delegates to NSE India API
  2. NSE India API — direct from exchange (richest data: OI, IV, volume)
  3. Calculated fallback — spot + ATM only
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

NIFTY_STRIKE_STEP = 50


def round_to_strike(price: float, step: float = NIFTY_STRIKE_STEP) -> int:
    return int(round(price / step) * step)


class StrikeSelector:
    """
    Fetches live data from Kotak Neo client (primary) + NSE (fallback).
    """

    def __init__(self, client, exchange: str = "NFO"):
        self.client = client
        self.exchange = exchange
        self._cached_expiry: Optional[str] = None
        self._cached_expiry_list: list[str] = []
        self._nse = None  # lazy init

    def _get_nse(self):
        """Lazy-init NSE fetcher."""
        if self._nse is None:
            try:
                from core.nse_fetcher import NSEFetcher
                self._nse = NSEFetcher(timeout=10)
                logger.info("NSE fetcher initialized")
            except Exception as e:
                logger.warning(f"NSE fetcher init failed: {e}")
        return self._nse

    def get_spot(self) -> float:
        """Fetch live Nifty spot price.

        Source priority:
          1. Native WebSocket (OHLCVBuilder.latest_ltp) — zero-latency, no API
          2. Kotak Neo REST quote — works when Kotak quotes API is healthy
          3. NSE option chain — has underlyingValue
          4. TradingView — last 5-min close
        """
        # ── Source 1: Native WebSocket tick buffer (fastest) ────────
        try:
            from core.ohlcv_builder import get_ohlcv_builder
            builder = get_ohlcv_builder()
            if builder is not None:
                ltp = float(builder.latest_ltp("NIFTY"))
                if ltp > 0:
                    logger.info(f"Nifty spot: {ltp:.2f} (WebSocket tick)")
                    return ltp
        except Exception as e:
            logger.debug(f"WebSocket spot fetch failed: {e}")

        # ── Source 2: Kotak Neo REST quote ──────────────────────────
        try:
            quote = self.client.get_quote("NIFTY", "NSE_INDEX")
            if isinstance(quote.get("data"), dict):
                ltp = float(quote["data"].get("ltp", 0))
            else:
                ltp = float(quote.get("ltp", 0))
            if ltp > 0:
                logger.info(f"Nifty spot: {ltp:.2f} (Kotak Neo)")
                return ltp
            else:
                logger.warning(f"Kotak Neo spot returned 0 (quote={quote})")
        except Exception as e:
            logger.warning(f"Kotak Neo spot failed: {e}")

        # Fallback 2: NSE option chain has underlyingValue
        nse = self._get_nse()
        if nse:
            try:
                chain = nse.get_option_chain("NIFTY", width=1)
                if chain.get("spot", 0) > 0:
                    logger.info(f"Nifty spot: {chain['spot']:.2f} (NSE)")
                    return chain["spot"]
            except Exception:
                pass

        # Fallback 3: TradingView — fetch latest 5-min close as spot
        try:
            from core.tv_fetcher import get_tv_fetcher
            tv = get_tv_fetcher()
            if tv:
                logger.info("Attempting TradingView spot fallback...")
                df = tv.get_ohlcv("NIFTY", "NSE", interval_minutes=5, n_bars=2)
                if df is not None and not df.empty:
                    ltp = float(df["close"].iloc[-1])
                    if ltp > 0:
                        logger.info(f"Nifty spot: {ltp:.2f} (TradingView)")
                        return ltp
                    else:
                        logger.warning("TV returned close=0")
                else:
                    logger.warning("TV returned empty DataFrame for NIFTY 5min")
            else:
                logger.warning("TV fetcher not available")
        except Exception as e:
            logger.warning(f"TV spot failed: {e}")

        raise ValueError("Cannot fetch Nifty spot from any source")

    def get_atm(self) -> int:
        spot = self.get_spot()
        return round_to_strike(spot)

    def get_next_expiry(self) -> str:
        """Get nearest expiry in DDMMMYY format (e.g. '24MAR26').
        Priority: 1) NIFTY_EXPIRY env var  2) Kotak Neo API  3) calculated fallback
        """
        import os

        # ── Priority 1: Manual override from settings.env ─────────
        env_expiry_is_stale = False
        env_expiry = os.getenv("NIFTY_EXPIRY", "").strip().upper()
        if env_expiry:
            from core.expiry_utils import _parse_expiry_str
            parsed = _parse_expiry_str(env_expiry)
            if parsed is None or parsed >= date.today():
                logger.info(f"Expiry: {env_expiry} (settings.env override)")
                self._cached_expiry = env_expiry
                return env_expiry
            env_expiry_is_stale = True
            logger.warning(
                f"NIFTY_EXPIRY={env_expiry} is in the past — ignoring stale "
                f"override, update settings.env! Falling back to broker/calculated expiry."
            )

        # ── Priority 2: Cached from previous call ─────────────────
        if self._cached_expiry:
            return self._cached_expiry

        # ── Priority 3: Kotak Neo search_scrip ────────────────────
        for exchange in ["NFO", "NSE_INDEX"]:
            try:
                dates = self.client.get_expiry_dates("NIFTY", exchange)
                if dates:
                    self._cached_expiry_list = dates
                    self._cached_expiry = dates[0]
                    logger.info(f"Expiry: {self._cached_expiry} (Kotak Neo/{exchange})")
                    return self._cached_expiry
            except Exception as e:
                logger.debug(f"Kotak Neo expiry failed ({exchange}): {e}")

        # ── Priority 4: Derive from NIFTY_EXPIRY env var ──────────
        # Skipped when Priority 1 already found this same env var stale —
        # it reads the identical value and would rediscover the same problem.
        if not env_expiry_is_stale:
            try:
                from core.expiry_utils import get_expiry_date
                exp_date = get_expiry_date()
                if exp_date is not None:
                    fallback = exp_date.strftime("%d%b%y").upper()
                    logger.info(f"Expiry from NIFTY_EXPIRY: {fallback}")
                    return fallback
            except Exception:
                pass
        # Last resort: next Tuesday (may be wrong if holiday-shifted)
        today = date.today()
        days_ahead = (1 - today.weekday()) % 7 or 7
        exp = today + timedelta(days=days_ahead)
        fallback = exp.strftime("%d%b%y").upper()
        logger.warning(
            f"Using calculated expiry: {fallback} — "
            f"set NIFTY_EXPIRY={fallback} in settings.env to fix"
        )
        return fallback

    def get_nse_expiry_format(self) -> str:
        """Get expiry in NSE format (e.g. '27-Mar-2026')."""
        nse = self._get_nse()
        if nse:
            try:
                dates = nse.get_expiry_dates("NIFTY")
                if dates:
                    return dates[0]
            except Exception:
                pass

        # Fallback: use NIFTY_EXPIRY env var
        try:
            from core.expiry_utils import get_expiry_date
            exp_date = get_expiry_date()
            if exp_date is not None:
                return exp_date.strftime("%d-%b-%Y")
        except Exception:
            pass
        # Last resort: next Tuesday
        today = date.today()
        days_ahead = (1 - today.weekday()) % 7 or 7
        exp = today + timedelta(days=days_ahead)
        return exp.strftime("%d-%b-%Y")

    def get_option_chain(self, width: int = 10) -> dict:
        """
        Fetch option chain. Tries Kotak Neo client first, falls back to NSE India.

        Returns standardized dict:
        {
            "spot": float,
            "atm": int,
            "expiry": str,
            "chain": [{strike, ce_premium, pe_premium, ce_oi, pe_oi, ...}],
            "pcr": float,
            "source": "KotakNeo" | "NSE",
        }
        """

        # ── Attempt 1: Kotak Neo (delegates to NSE internally) ──
        chain_result = self._try_broker_chain(width)
        if chain_result and chain_result.get("has_data"):
            return chain_result

        logger.info("Broker chain empty/zero → trying NSE India directly...")

        # ── Attempt 2: NSE India ──
        chain_result = self._try_nse_chain(width)
        if chain_result and chain_result.get("has_data"):
            return chain_result

        logger.warning("Both broker and NSE returned empty chains")

        # ── Fallback: spot-only ──
        try:
            spot = self.get_spot()
        except Exception:
            spot = 0
        atm = round_to_strike(spot) if spot > 0 else 0
        return {
            "spot": spot,
            "atm": atm,
            "expiry": self.get_next_expiry(),
            "strikes_count": 0,
            "chain": [],
            "pcr": 0,
            "has_data": False,
            "source": "FALLBACK",
        }

    def _try_broker_chain(self, width: int) -> dict:
        """Try fetching via Kotak Neo client (which uses NSE internally)."""
        try:
            expiry = self.get_next_expiry()
            chain_data = {}
            for exchange in ["NFO", "NSE_INDEX"]:
                try:
                    chain_data = self.client.get_option_chain(
                        underlying="NIFTY",
                        exchange=exchange,
                        expiry_date=expiry,
                        strike_count=width,
                    )
                    if chain_data.get("chain"):
                        break
                except Exception:
                    continue

            raw_chain = chain_data.get("chain", [])
            spot = float(chain_data.get("underlying_ltp", chain_data.get("spot", 0)))
            atm = int(chain_data.get("atm_strike", chain_data.get("atm", round_to_strike(spot))))

            parsed = []
            total_ce_oi, total_pe_oi = 0, 0
            has_real_data = False

            for entry in raw_chain:
                strike = int(entry.get("strike", 0))
                ce = entry.get("ce", {}) or {}
                pe = entry.get("pe", {}) or {}

                # Handle both nested format {ce: {ltp, oi}} and flat format {ce_ltp, ce_oi}
                ce_p  = float(ce.get("ltp", 0) or entry.get("ce_ltp", 0)
                              or entry.get("ce_premium", 0) or 0)
                pe_p  = float(pe.get("ltp", 0) or entry.get("pe_ltp", 0)
                              or entry.get("pe_premium", 0) or 0)
                ce_oi = int(ce.get("oi", entry.get("ce_oi", 0)) or 0)
                pe_oi = int(pe.get("oi", entry.get("pe_oi", 0)) or 0)

                if ce_p > 0 or pe_p > 0:
                    has_real_data = True
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi

                parsed.append({
                    "strike": strike,
                    "ce_symbol": ce.get("symbol", entry.get("ce_symbol", "")),
                    "ce_premium": ce_p,
                    "ce_label": ce.get("label", entry.get("ce_label", "")),
                    "ce_oi": ce_oi,
                    "pe_symbol": pe.get("symbol", entry.get("pe_symbol", "")),
                    "pe_premium": pe_p,
                    "pe_label": pe.get("label", entry.get("pe_label", "")),
                    "pe_oi": pe_oi,
                    "is_atm": strike == atm,
                    "moneyness_ce": "ATM" if strike == atm else (
                        "ITM" if strike < spot else "OTM"),
                    "moneyness_pe": "ATM" if strike == atm else (
                        "ITM" if strike > spot else "OTM"),
                })

            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            source = chain_data.get("source", "KotakNeo")
            logger.info(
                f"Broker chain ({source}): {len(parsed)} strikes | "
                f"spot={spot:.2f} | has_data={has_real_data}"
            )

            return {
                "spot": spot,
                "atm": atm,
                "expiry": expiry,
                "strikes_count": len(parsed),
                "chain": parsed,
                "pcr": round(pcr, 3),
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "has_data": has_real_data,
                "source": source,
            }

        except Exception as e:
            logger.debug(f"Broker chain failed: {e}")
            return None

    def _try_nse_chain(self, width: int) -> dict:
        """Try fetching from NSE India."""
        nse = self._get_nse()
        if not nse:
            return None

        try:
            result = nse.get_option_chain(
                symbol="NIFTY",
                width=width,
            )
            if result and result.get("has_data"):
                logger.info(
                    f"NSE chain: {result['strikes_count']} strikes | "
                    f"spot={result['spot']:.2f} | PCR={result.get('pcr', 0):.2f}"
                )
            return result
        except Exception as e:
            logger.debug(f"NSE chain failed: {e}")
            return None
