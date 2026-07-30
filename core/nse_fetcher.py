"""
nse_fetcher.py — Direct NSE Option Chain Fetcher
=================================================
Fetches option chain directly from NSE India website API.
Used as fallback when OpenAlgo returns empty/zero data.

NSE API: https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY&expiry=24-Mar-2026

NSE requires:
  1. First visit nseindia.com to get session cookies
  2. Use those cookies + proper headers for API calls
  3. Cookies expire after ~5 minutes, auto-refresh
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# NSE URLs
NSE_BASE = "https://www.nseindia.com"
NSE_OPTION_CHAIN_URL = NSE_BASE + "/api/option-chain-v3"

# Browser-like headers (NSE blocks non-browser requests)
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


class NSEFetcher:
    """
    Fetches live option chain data directly from NSE India.
    Handles session management (cookies), retries, and parsing.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._session: Optional[requests.Session] = None
        self._last_cookie_time: float = 0
        self._cookie_ttl: float = 240  # refresh cookies every 4 min
        self._cached_expiries: list[str] = []

    def _ensure_session(self):
        """Create session and fetch cookies from NSE option-chain page."""
        now = time.monotonic()
        if (self._session and
                now - self._last_cookie_time < self._cookie_ttl):
            return  # cookies still fresh

        logger.info("🌐 NSE: refreshing session cookies...")
        self._session = requests.Session()
        self._session.headers.update(NSE_HEADERS)

        try:
            # Step 1: Hit the option-chain HTML page (this sets proper cookies)
            # NSE requires a page visit before API calls — the homepage alone
            # returns 403 since late 2025.
            resp = self._session.get(
                NSE_BASE + "/option-chain",
                timeout=self.timeout,
                allow_redirects=True,
            )
            if resp.status_code == 200:
                self._last_cookie_time = now
                cookies = dict(self._session.cookies)
                logger.info(
                    f"🌐 NSE: session OK | cookies: {len(cookies)}"
                )
            else:
                # Fallback: try homepage
                resp = self._session.get(NSE_BASE, timeout=self.timeout)
                if resp.status_code == 200:
                    self._last_cookie_time = now
                    logger.info("🌐 NSE: session OK (homepage fallback)")
                else:
                    logger.warning(f"🌐 NSE returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"🌐 NSE session init failed: {e}")
            self._session = None

    def get_option_chain(self, symbol: str = "NIFTY",
                         expiry: str = "", width: int = 10) -> dict:
        """
        Fetch option chain from NSE.

        Args:
            symbol: "NIFTY" or "BANKNIFTY"
            expiry: Expiry date string like "24-Mar-2026" (empty = nearest)
            width: Number of strikes on each side of ATM

        Returns:
            Standardized dict matching our format:
            {
                "spot": float,
                "atm": int,
                "expiry": str,
                "chain": [{"strike", "ce_premium", "pe_premium",
                           "ce_oi", "pe_oi", "ce_iv", "pe_iv", ...}],
                "pcr": float,
                "total_ce_oi": int,
                "total_pe_oi": int,
                "source": "NSE",
            }
        """
        self._ensure_session()
        if not self._session:
            return self._empty_result("Session init failed")

        # Build URL params
        params = {
            "type": "Indices",
            "symbol": symbol.upper(),
        }
        if expiry:
            params["expiry"] = expiry

        try:
            resp = self._session.get(
                NSE_OPTION_CHAIN_URL,
                params=params,
                timeout=self.timeout,
            )

            if resp.status_code == 401 or resp.status_code == 403:
                # Session expired, retry once
                logger.info("🌐 NSE: cookies expired, refreshing...")
                self._last_cookie_time = 0
                self._ensure_session()
                if not self._session:
                    return self._empty_result("Session refresh failed")
                resp = self._session.get(
                    NSE_OPTION_CHAIN_URL,
                    params=params,
                    timeout=self.timeout,
                )

            if resp.status_code != 200:
                logger.warning(f"🌐 NSE API returned {resp.status_code}")
                return self._empty_result(f"HTTP {resp.status_code}")

            data = resp.json()
            return self._parse_response(data, width)

        except requests.exceptions.Timeout:
            logger.warning("🌐 NSE API timeout")
            return self._empty_result("Timeout")
        except Exception as e:
            logger.warning(f"🌐 NSE API error: {e}")
            return self._empty_result(str(e))

    def get_expiry_dates(self, symbol: str = "NIFTY") -> list[str]:
        """Fetch available expiry dates from NSE."""
        result = self.get_option_chain(symbol=symbol, width=1)
        return self._cached_expiries

    def get_nearest_expiry(self, symbol: str = "NIFTY") -> str:
        """Get nearest expiry date string for NSE API format."""
        dates = self.get_expiry_dates(symbol)
        if dates:
            return dates[0]

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

    def _parse_response(self, data: dict, width: int) -> dict:
        """Parse NSE option-chain-v3 response into our standard format."""

        records = data.get("records", {})
        filtered = data.get("filtered", {})

        # Get spot price
        spot = float(records.get("underlyingValue", 0))
        if spot <= 0:
            # Try from filtered data
            fdata = filtered.get("data", [])
            if fdata:
                for entry in fdata:
                    ce = entry.get("CE", {})
                    if ce and ce.get("underlyingValue", 0) > 0:
                        spot = float(ce["underlyingValue"])
                        break

        # Cache expiry dates
        self._cached_expiries = records.get("expiryDates", [])

        # ATM strike
        from core.strike_selector import round_to_strike
        atm = round_to_strike(spot) if spot > 0 else 0

        # Parse chain data — use "filtered" (single expiry) if available
        raw_data = filtered.get("data", []) or records.get("data", [])

        # Total OI from filtered
        total_ce_oi = 0
        total_pe_oi = 0
        if filtered.get("CE"):
            total_ce_oi = int(filtered["CE"].get("totOI", 0))
        if filtered.get("PE"):
            total_pe_oi = int(filtered["PE"].get("totOI", 0))

        # Parse individual strikes
        chain = []
        for entry in raw_data:
            strike = int(entry.get("strikePrice", 0))
            if strike == 0:
                continue

            # Filter to ATM ± width strikes
            if atm > 0 and abs(strike - atm) > width * 50:
                continue

            ce = entry.get("CE", {}) or {}
            pe = entry.get("PE", {}) or {}

            ce_premium = float(ce.get("lastPrice", 0) or 0)
            pe_premium = float(pe.get("lastPrice", 0) or 0)
            ce_oi = int(ce.get("openInterest", 0) or 0)
            pe_oi = int(pe.get("openInterest", 0) or 0)
            ce_iv = float(ce.get("impliedVolatility", 0) or 0)
            pe_iv = float(pe.get("impliedVolatility", 0) or 0)
            ce_vol = int(ce.get("totalTradedVolume", 0) or 0)
            pe_vol = int(pe.get("totalTradedVolume", 0) or 0)
            ce_oi_chg = int(ce.get("changeinOpenInterest", 0) or 0)
            pe_oi_chg = int(pe.get("changeinOpenInterest", 0) or 0)

            # Track totals if not available from filtered
            if total_ce_oi == 0:
                total_ce_oi += ce_oi
            if total_pe_oi == 0:
                total_pe_oi += pe_oi

            chain.append({
                "strike": strike,
                "ce_premium": ce_premium,
                "pe_premium": pe_premium,
                "ce_oi": ce_oi,
                "pe_oi": pe_oi,
                "ce_iv": ce_iv,
                "pe_iv": pe_iv,
                "ce_volume": ce_vol,
                "pe_volume": pe_vol,
                "ce_oi_change": ce_oi_chg,
                "pe_oi_change": pe_oi_chg,
                "ce_bid": float(ce.get("bidprice", 0) or 0),
                "ce_ask": float(ce.get("askPrice", 0) or 0),
                "pe_bid": float(pe.get("bidprice", 0) or 0),
                "pe_ask": float(pe.get("askPrice", 0) or 0),
                "is_atm": strike == atm,
                "moneyness_ce": "ATM" if strike == atm else (
                    "ITM" if strike < spot else "OTM"),
                "moneyness_pe": "ATM" if strike == atm else (
                    "ITM" if strike > spot else "OTM"),
            })

        # Sort by strike
        chain.sort(key=lambda x: x["strike"])

        # PCR
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

        # Find max OI strikes (support/resistance)
        max_ce_oi_strike = 0
        max_pe_oi_strike = 0
        max_ce_oi = 0
        max_pe_oi = 0
        for s in chain:
            if s["ce_oi"] > max_ce_oi:
                max_ce_oi = s["ce_oi"]
                max_ce_oi_strike = s["strike"]
            if s["pe_oi"] > max_pe_oi:
                max_pe_oi = s["pe_oi"]
                max_pe_oi_strike = s["strike"]

        # Expiry date from response
        expiry = ""
        if self._cached_expiries:
            expiry = self._cached_expiries[0]

        has_data = any(s["ce_premium"] > 0 or s["pe_premium"] > 0
                       for s in chain)

        logger.info(
            f"🌐 NSE chain: {len(chain)} strikes | "
            f"spot={spot:.2f} ATM={atm} | "
            f"CE_OI={total_ce_oi:,} PE_OI={total_pe_oi:,} PCR={pcr:.2f} | "
            f"Resistance={max_ce_oi_strike} Support={max_pe_oi_strike}"
        )

        return {
            "spot": spot,
            "atm": atm,
            "expiry": expiry,
            "strikes_count": len(chain),
            "chain": chain,
            "pcr": round(pcr, 3),
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "max_ce_oi_strike": max_ce_oi_strike,
            "max_pe_oi_strike": max_pe_oi_strike,
            "has_data": has_data,
            "source": "NSE",
            "timestamp": records.get("timestamp", ""),
        }

    def _empty_result(self, reason: str) -> dict:
        logger.warning(f"🌐 NSE: empty result — {reason}")
        return {
            "spot": 0,
            "atm": 0,
            "expiry": "",
            "strikes_count": 0,
            "chain": [],
            "pcr": 0,
            "total_ce_oi": 0,
            "total_pe_oi": 0,
            "has_data": False,
            "source": "NSE",
            "error": reason,
        }
