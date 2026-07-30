"""
market_intel.py -- Market Intelligence Layer
=============================================
Combines options flow, sentiment scoring, and event calendar
into a single pre-trade check that the pilot uses before every trade.

Merges the best of:
  - options_agent.py  -> Max Pain, PCR, OI support/resistance
  - sentiment_agent.py -> Weighted sentiment score (PCR + VIX + FII/DII + A/D)
  - news_agent.py     -> Economic calendar, risk keyword detection

Integration:
  The pilot calls intel.pre_trade_check() before confirming any ML signal.
  Returns: safe (bool), bias (BULLISH/BEARISH/NEUTRAL), score (0-100), context (str)

Data sources (all free, no API key needed):
  - NSE option chain API  -> PCR, OI, IV, Max Pain
  - NSE allIndices API    -> VIX, A/D ratio
  - NSE FII/DII API       -> institutional flow
  - Hardcoded calendar    -> RBI MPC, FOMC, Budget dates
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional

import requests

from core.nse_fetcher import NSEFetcher

logger = logging.getLogger(__name__)

_nse_fetcher: Optional[NSEFetcher] = None


def _get_nse_fetcher() -> NSEFetcher:
    global _nse_fetcher
    if _nse_fetcher is None:
        _nse_fetcher = NSEFetcher(timeout=10)
    return _nse_fetcher

# ======================================================================
# KOTAK NEO CLIENT INJECTION (primary OI data source)
# ======================================================================
# market_intel is a module (not a class), so we use a module-level ref.
# Call init_broker_client(client) once at startup to enable broker OI.

_openalgo_client = None          # type: Optional[Any]  # KotakNeoClient or OpenAlgoClient
_openalgo_expiry: str = ""       # cached nearest expiry
_kotak_client_backup = None      # always keep Kotak ref so we can fall back
_oa_retry_ts: float = 0.0        # last time we tried to (re)connect OpenAlgo
_OA_RETRY_INTERVAL = 300.0       # retry OpenAlgo every 5 min until it connects


def init_broker_client(client) -> None:
    """Inject Kotak Neo client so market_intel can fetch OI from broker."""
    global _openalgo_client, _kotak_client_backup
    _openalgo_client   = client
    _kotak_client_backup = client   # keep a permanent Kotak ref for fallback
    logger.info("Market intel: Kotak Neo client injected — broker OI enabled")


# Alias so claude_pilot.py (which calls init_openalgo_client) keeps working
init_openalgo_client = init_broker_client


def _try_connect_openalgo() -> bool:
    """
    Attempt to (re)connect to OpenAlgo REST API.
    Called from the OI fetch loop if OpenAlgo isn't the active client.

    OpenAlgo starts at 09:00 IST; the bot starts at 04:00.
    We don't even try before 09:10 IST — no point hammering a service
    that isn't up yet. After 09:10 we retry every 5 minutes until it connects.

    Returns True if OpenAlgo is now active.
    """
    global _openalgo_client, _oa_retry_ts
    import time as _time
    import os as _os
    from datetime import datetime as _dt

    # Only try after 09:10 IST — OpenAlgo starts at 09:00
    _now = _dt.now()
    if _now.hour < 9 or (_now.hour == 9 and _now.minute < 10):
        return False

    # Throttle — don't retry on every cycle, only every 5 min
    if _time.time() - _oa_retry_ts < _OA_RETRY_INTERVAL:
        return False
    _oa_retry_ts = _time.time()

    url = _os.getenv("OPENALGO_URL", "").strip()
    key = _os.getenv("OPENALGO_API_KEY", "").strip()
    if not url:
        return False

    try:
        from core.openalgo_client import OpenAlgoClient
        oa = OpenAlgoClient(base_url=url, api_key=key)
        if oa.ping():
            _openalgo_client = oa
            logger.info(
                f"✅ OpenAlgo connected at {url} — switching to OpenAlgo "
                f"as primary OI source (was Kotak direct)"
            )
            return True
        else:
            # ping() returned False — could be HTML (broker not logged in).
            # Try auto-login via scripts/openalgo_auto_login.py, then re-ping.
            logger.info(
                "OpenAlgo ping failed (HTML or unreachable) — "
                "attempting auto-login (scripts/openalgo_auto_login.py)..."
            )
            try:
                import importlib.util, sys as _sys
                _script = _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "scripts", "openalgo_auto_login.py"
                )
                _spec = importlib.util.spec_from_file_location(
                    "openalgo_auto_login", _script
                )
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _login_ok = _mod.auto_login(retries=1)
                if _login_ok:
                    # Re-ping after successful login
                    oa2 = OpenAlgoClient(base_url=url, api_key=key)
                    if oa2.ping():
                        _openalgo_client = oa2
                        logger.info(
                            f"✅ OpenAlgo connected (after auto-login) at {url}"
                        )
                        return True
                    logger.warning("OpenAlgo auto-login reported success but ping still fails")
                else:
                    logger.warning("OpenAlgo auto-login failed — will retry in 5 min")
            except Exception as _login_err:
                logger.debug(f"OpenAlgo auto-login attempt failed: {_login_err}")
    except Exception as _e:
        logger.debug(f"OpenAlgo retry failed: {_e}")
    return False

# ======================================================================
# ECONOMIC CALENDAR -- high-risk event dates
# ======================================================================

#  release_after = "HH:MM" — block lifts after this IST time (post-event window)
KNOWN_EVENTS = [
    {"date": "2026-04-09", "name": "RBI MPC Policy Decision",  "risk": "HIGH",   "release_after": "12:00"},
    {"date": "2026-06-06", "name": "RBI MPC Policy Decision",  "risk": "HIGH",   "release_after": "12:00"},
    {"date": "2026-08-06", "name": "RBI MPC Policy Decision",  "risk": "HIGH",   "release_after": "12:00"},
    {"date": "2026-10-01", "name": "RBI MPC Policy Decision",  "risk": "HIGH",   "release_after": "12:00"},
    {"date": "2026-12-03", "name": "RBI MPC Policy Decision",  "risk": "HIGH",   "release_after": "12:00"},
    {"date": "2026-02-01", "name": "Union Budget",             "risk": "HIGH",   "release_after": "14:30"},
    {"date": "2026-03-19", "name": "US FOMC Meeting",          "risk": "MEDIUM", "release_after": "09:30"},
    {"date": "2026-05-07", "name": "US FOMC Meeting",          "risk": "MEDIUM", "release_after": "09:30"},
    {"date": "2026-06-18", "name": "US FOMC Meeting",          "risk": "MEDIUM", "release_after": "09:30"},
    {"date": "2026-09-17", "name": "US FOMC Meeting",          "risk": "MEDIUM", "release_after": "09:30"},
    {"date": "2026-11-05", "name": "US FOMC Meeting",          "risk": "MEDIUM", "release_after": "09:30"},
    {"date": "2026-12-17", "name": "US FOMC Meeting",          "risk": "MEDIUM", "release_after": "09:30"},
]


def check_events_today() -> dict:
    """Check if today has any high-risk events.

    Honors per-event `release_after` window: if current IST time is past it,
    risk is downgraded HIGH→MEDIUM (informational, no block) so the bot can
    catch the post-event trend.
    """
    from datetime import datetime
    today_str = date.today().isoformat()
    now_hm = datetime.now().strftime("%H:%M")
    for evt in KNOWN_EVENTS:
        if evt["date"] != today_str:
            continue
        release = evt.get("release_after")
        risk = evt["risk"]
        released = bool(release and now_hm >= release)
        if released and risk == "HIGH":
            risk = "MEDIUM"   # downgrade — no longer blocks
        return {
            "event": True,
            "name": evt["name"] + (" (post-event window)" if released else ""),
            "risk": risk,
            "released": released,
        }
    return {"event": False, "name": "", "risk": "NONE", "released": False}


# ======================================================================
# NSE DATA FETCHER (shared session with proper cookies)
# ======================================================================

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

_nse_session: Optional[requests.Session] = None
_nse_cookie_time: float = 0


def _get_nse_session() -> requests.Session:
    """Get NSE session with fresh cookies."""
    global _nse_session, _nse_cookie_time
    now = time.monotonic()
    if _nse_session and (now - _nse_cookie_time) < 240:
        return _nse_session

    _nse_session = requests.Session()
    _nse_session.headers.update(NSE_HEADERS)
    try:
        resp = _nse_session.get(
            "https://www.nseindia.com/option-chain",
            timeout=10, allow_redirects=True,
        )
        if resp.status_code == 200:
            _nse_cookie_time = now
            logger.debug(f"NSE session OK | cookies={len(dict(_nse_session.cookies))}")
        else:
            # Fallback to homepage
            _nse_session.get("https://www.nseindia.com", timeout=10)
            _nse_cookie_time = now
    except Exception as e:
        logger.debug(f"NSE session init failed: {e}")
    return _nse_session


def _nse_get(url: str, timeout: int = 10) -> Optional[dict]:
    """GET from NSE API with session cookies."""
    try:
        session = _get_nse_session()
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        # Retry once on 401/403
        if resp.status_code in (401, 403):
            global _nse_cookie_time
            _nse_cookie_time = 0
            session = _get_nse_session()
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"NSE API failed ({url}): {e}")
    return None


# ======================================================================
# OPTIONS FLOW (PCR, Max Pain, OI levels)
# ======================================================================

@dataclass
class OptionsFlow:
    pcr: float = 1.0
    pcr_bias: str = "NEUTRAL"    # BULLISH / BEARISH / NEUTRAL
    max_pain: float = 0.0
    iv_atm: float = 15.0
    oi_support: float = 0.0      # highest PUT OI strike below spot (Max PE OI)
    oi_resistance: float = 0.0   # highest CALL OI strike above spot (Max CE OI)
    spot: float = 0.0
    available: bool = False
    # OI change detection — building vs unwinding
    total_ce_oi: float = 0.0
    total_pe_oi: float = 0.0
    ce_oi_change: float = 0.0    # change in total CE OI (positive = writing, bearish)
    pe_oi_change: float = 0.0    # change in total PE OI (positive = writing, bullish)
    oi_buildup: str = "NEUTRAL"  # CALL_WRITING / PUT_WRITING / LONG_BUILD / SHORT_BUILD / NEUTRAL
    # Top 3 OI walls for dynamic SL/TP
    oi_walls_above: list = None  # [(strike, ce_oi), ...] top 3 call OI walls above spot
    oi_walls_below: list = None  # [(strike, pe_oi), ...] top 3 put OI walls below spot
    # ATM OI tracking — for gamma squeeze detection
    atm_strike: float = 0.0
    atm_ce_oi: float = 0.0       # ATM call open interest
    atm_pe_oi: float = 0.0       # ATM put open interest
    atm_ce_oi_change: float = 0.0  # 5-min change in ATM CE OI
    atm_pe_oi_change: float = 0.0  # 5-min change in ATM PE OI
    atm_ce_unwind_pct: float = 0.0  # % change in ATM CE OI (negative = unwinding)
    atm_pe_unwind_pct: float = 0.0  # % change in ATM PE OI (negative = unwinding)
    # PCR momentum
    pcr_5m_ago: float = 1.0      # PCR from previous snapshot
    pcr_5m_momentum: float = 0.0 # current PCR - PCR 5 min ago

    def __post_init__(self):
        if self.oi_walls_above is None:
            self.oi_walls_above = []
        if self.oi_walls_below is None:
            self.oi_walls_below = []


# ── Snapshot history for 5-min deltas ─────────────────────────────
_prev_ce_oi: float = 0.0
_prev_pe_oi: float = 0.0
_prev_atm_ce_oi: float = 0.0
_prev_atm_pe_oi: float = 0.0
_prev_atm_strike: float = 0.0   # track ATM strike — OI change is invalid when strike rolls
_prev_pcr: float = 1.0


def _fetch_chain_openalgo() -> tuple:
    """
    Fetch option chain from OpenAlgo (Kotak Neo broker).
    Returns: (spot, strike_data, total_ce_oi, total_pe_oi) or (0, {}, 0, 0) on failure.

    OpenAlgo response format:
      {underlying_ltp, atm_strike, chain: [{strike, ce:{symbol,ltp,oi}, pe:{symbol,ltp,oi}}, ...]}
    """
    global _openalgo_expiry

    if not _openalgo_client:
        return 0, {}, 0, 0, "KotakNeo"

    try:
        # If currently using Kotak direct, try upgrading to OpenAlgo (every 5 min).
        # OpenAlgo starts at 09:00 — the 04:00 startup attempt always fails, so we
        # retry here until it connects. Once connected it stays as primary OI source.
        _is_openalgo_rest = hasattr(_openalgo_client, '_base')  # OpenAlgoClient has _base attr
        if not _is_openalgo_rest:
            if _try_connect_openalgo():
                _is_openalgo_rest = True  # just connected — use it this cycle too
        exchange_list = ["NSE", "NFO", "NSE_INDEX"] if _is_openalgo_rest else ["NFO", "NSE_INDEX"]
        # OpenAlgo's /api/v1/expiry only accepts NFO/BFO/MCX/CDS/CRYPTO for
        # instrumenttype=options — NSE/NSE_INDEX (the underlying's exchange)
        # returns HTTP 400.
        expiry_exchange = "NFO" if _is_openalgo_rest else "NSE_INDEX"

        # ── Expiry resolution ──────────────────────────────────────────────
        # Auto-reset if cached expiry has already passed — this handles the
        # case where NIFTY_EXPIRY=09JUN26 is still in .env on June 10 after
        # the contract expired. Without this check, the bot would keep trying
        # to fetch an expired chain all day.
        import os
        if _openalgo_expiry:
            try:
                from datetime import datetime as _dt
                _exp_dt = _dt.strptime(_openalgo_expiry, "%d%b%y")
                if _exp_dt.date() < _dt.now().date():
                    logger.warning(
                        f"NIFTY_EXPIRY={_openalgo_expiry} has EXPIRED — "
                        f"resetting for auto-detection. "
                        f"Update NIFTY_EXPIRY in .env to suppress this warning."
                    )
                    _openalgo_expiry = ""
            except Exception:
                pass

        # Get expiry if not set (fresh start or just reset above)
        if not _openalgo_expiry:
            try:
                dates = _openalgo_client.get_expiry_dates("NIFTY", expiry_exchange)
                if dates:
                    _openalgo_expiry = dates[0]  # nearest expiry
                    logger.info(f"OI fetch: auto-detected expiry: {_openalgo_expiry}")
            except Exception as e:
                logger.debug(f"Expiry auto-detect failed: {e}")
            # Fall back to settings.env if auto-detect unavailable
            if not _openalgo_expiry:
                _openalgo_expiry = os.getenv("NIFTY_EXPIRY", "")
                if _openalgo_expiry:
                    logger.info(f"OI fetch: using expiry from settings: {_openalgo_expiry}")

        # Fetch chain — 20 strikes each side for good OI wall coverage
        logger.info(f"OI fetch: expiry='{_openalgo_expiry}' exchanges={exchange_list}")
        chain_data = None
        for exchange in exchange_list:
            try:
                chain_data = _openalgo_client.get_option_chain(
                    underlying="NIFTY",
                    exchange=exchange,
                    expiry_date=_openalgo_expiry,
                    strike_count=20,
                )
                if chain_data and chain_data.get("chain"):
                    break
            except Exception:
                continue

        if not chain_data or not chain_data.get("chain"):
            # OpenAlgo REST is reachable (ping succeeded) but its optionchain
            # endpoint returned nothing usable — e.g. OpenAlgo's instrument
            # master needs refreshing server-side. Fall back to the KotakNeo
            # direct client we keep a permanent ref to, rather than degrading
            # all the way to the NSE public API (unreliable from VPS IPs).
            if _is_openalgo_rest and _kotak_client_backup is not None:
                logger.warning(
                    "OpenAlgo optionchain returned no usable chain — "
                    "falling back to KotakNeo direct for this cycle"
                )
                for exchange in ("NFO", "NSE_INDEX"):
                    try:
                        chain_data = _kotak_client_backup.get_option_chain(
                            underlying="NIFTY",
                            exchange=exchange,
                            expiry_date="",
                            strike_count=20,
                        )
                        if chain_data and chain_data.get("chain"):
                            break
                    except Exception:
                        continue

        if not chain_data or not chain_data.get("chain"):
            logger.warning(
                "Broker option chain: empty response "
                "(Kotak Neo direct fetch returned no usable chain)"
            )
            return 0, {}, 0, 0, "KotakNeo"

        spot = float(chain_data.get("underlying_ltp", 0))
        raw_chain = chain_data.get("chain", [])

        if spot == 0 or not raw_chain:
            return 0, {}, 0, 0, "KotakNeo"

        # Use pre-computed totals if the client already summed them (OpenAlgo REST does)
        total_ce_oi = float(chain_data.get("total_ce_oi", 0))
        total_pe_oi = float(chain_data.get("total_pe_oi", 0))
        strike_data = {}

        # Handle BOTH formats automatically:
        #   - OpenAlgo nested:  {strike, ce:{oi,ltp,iv}, pe:{oi,ltp,iv}}
        #   - Kotak Neo flat:   {strike, ce_oi, ce_ltp, ce_iv, pe_oi, pe_ltp, pe_iv}
        for entry in raw_chain:
            strike = float(entry.get("strike", 0))
            if strike == 0:
                continue

            # Nested format first
            ce = entry.get("ce", {}) or {}
            pe = entry.get("pe", {}) or {}
            c_oi = float(ce.get("oi", 0) or 0)
            p_oi = float(pe.get("oi", 0) or 0)
            c_iv = float(ce.get("iv", 0) or 0)
            p_iv = float(pe.get("iv", 0) or 0)

            # Flat-key fallback (Kotak Neo direct output)
            if c_oi == 0 and "ce_oi" in entry:
                c_oi = float(entry.get("ce_oi", 0) or 0)
                c_iv = float(entry.get("ce_iv", 0) or 0)
            if p_oi == 0 and "pe_oi" in entry:
                p_oi = float(entry.get("pe_oi", 0) or 0)
                p_iv = float(entry.get("pe_iv", 0) or 0)

            # Accumulate only if client didn't pre-compute totals
            if total_ce_oi == 0:
                total_ce_oi += c_oi
            if total_pe_oi == 0:
                total_pe_oi += p_oi

            strike_data[strike] = {
                "call_oi": c_oi, "put_oi": p_oi,
                "call_iv": c_iv, "put_iv": p_iv,
            }

        src_tag = chain_data.get("source") or ("OpenAlgo" if _is_openalgo_rest else "KotakNeo")
        logger.info(
            f"{src_tag} OI: {len(strike_data)} strikes | spot={spot:.2f} | "
            f"CE_OI={total_ce_oi:,.0f} PE_OI={total_pe_oi:,.0f}"
        )
        return spot, strike_data, total_ce_oi, total_pe_oi, src_tag

    except Exception as e:
        logger.warning(f"Broker option chain fetch failed: {e}")
        return 0, {}, 0, 0, "KotakNeo"


def _fetch_chain_nse() -> tuple:
    """
    Fetch option chain from NSE India (fallback) using NSEFetcher.
    Returns: (spot, strike_data, total_ce_oi, total_pe_oi) or (0, {}, 0, 0) on failure.
    """
    try:
        fetcher = _get_nse_fetcher()
        result = fetcher.get_option_chain(symbol="NIFTY", width=20)
        if not result.get("has_data") or result.get("total_ce_oi", 0) == 0:
            return 0, {}, 0, 0

        spot = float(result.get("spot", 0))
        total_ce_oi = float(result.get("total_ce_oi", 0))
        total_pe_oi = float(result.get("total_pe_oi", 0))

        strike_data = {}
        for s in result.get("chain", []):
            strike = float(s.get("strike", 0))
            if strike == 0:
                continue
            strike_data[strike] = {
                "call_oi": float(s.get("ce_oi", 0)),
                "put_oi": float(s.get("pe_oi", 0)),
                "call_iv": float(s.get("ce_iv", 0)),
                "put_iv": float(s.get("pe_iv", 0)),
            }

        return spot, strike_data, total_ce_oi, total_pe_oi

    except Exception as e:
        logger.warning(f"NSE chain fetch failed: {e}")
        return 0, {}, 0, 0


def fetch_options_flow() -> OptionsFlow:
    """
    Fetch option chain and compute PCR, Max Pain, OI levels, OI change.
    Priority: OpenAlgo (Kotak Neo) → NSE India fallback.
    """
    global _prev_ce_oi, _prev_pe_oi, _prev_atm_ce_oi, _prev_atm_pe_oi, _prev_pcr, _prev_atm_strike

    # ── Try broker chain first (KotakNeo direct or OpenAlgo REST) ─
    spot, strike_data, total_ce_oi, total_pe_oi, source = _fetch_chain_openalgo()

    # ── Fallback to NSE if broker returned no OI ──────────────────
    if total_ce_oi == 0:
        spot, strike_data, total_ce_oi, total_pe_oi = _fetch_chain_nse()
        source = "NSE"

    if total_ce_oi == 0 or not strike_data:
        logger.warning("OI fetch FAILED from both Kotak Neo and NSE — no OI data")
        return OptionsFlow()

    logger.info(f"OI source: {source} | spot={spot:.2f} | strikes={len(strike_data)}")

    # PCR
    pcr = total_pe_oi / total_ce_oi
    if pcr > 1.3:
        pcr_bias = "BULLISH"
    elif pcr < 0.7:
        pcr_bias = "BEARISH"
    else:
        pcr_bias = "NEUTRAL"

    # OI support / resistance (single strongest wall)
    below = {s: d for s, d in strike_data.items() if s <= spot}
    above = {s: d for s, d in strike_data.items() if s > spot}
    support = max(below, key=lambda s: below[s]["put_oi"]) if below else 0
    resistance = max(above, key=lambda s: above[s]["call_oi"]) if above else 0

    # Top 3 OI walls above spot (call OI = resistance) and below spot (put OI = support)
    walls_above = sorted(
        [(s, d["call_oi"]) for s, d in above.items() if d["call_oi"] > 0],
        key=lambda x: x[1], reverse=True
    )[:3]
    walls_below = sorted(
        [(s, d["put_oi"]) for s, d in below.items() if d["put_oi"] > 0],
        key=lambda x: x[1], reverse=True
    )[:3]

    # OI change detection (vs previous snapshot)
    ce_oi_change = total_ce_oi - _prev_ce_oi if _prev_ce_oi > 0 else 0.0
    pe_oi_change = total_pe_oi - _prev_pe_oi if _prev_pe_oi > 0 else 0.0
    _prev_ce_oi = total_ce_oi
    _prev_pe_oi = total_pe_oi

    # Classify OI buildup pattern
    # CE OI building = call writers adding → bearish (resistance strengthening)
    # PE OI building = put writers adding → bullish (support strengthening)
    oi_buildup = "NEUTRAL"
    if ce_oi_change > 0 and pe_oi_change > 0:
        # Both building — check which is dominant
        if ce_oi_change > pe_oi_change * 1.5:
            oi_buildup = "CALL_WRITING"    # bearish: heavy call writing
        elif pe_oi_change > ce_oi_change * 1.5:
            oi_buildup = "PUT_WRITING"     # bullish: heavy put writing
        else:
            oi_buildup = "NEUTRAL"         # balanced
    elif ce_oi_change > 0 and pe_oi_change <= 0:
        oi_buildup = "SHORT_BUILD"         # bearish: CE writing + PE unwinding
    elif pe_oi_change > 0 and ce_oi_change <= 0:
        oi_buildup = "LONG_BUILD"          # bullish: PE writing + CE unwinding
    elif ce_oi_change < 0 and pe_oi_change < 0:
        oi_buildup = "SHORT_COVERING" if abs(ce_oi_change) > abs(pe_oi_change) else "LONG_UNWINDING"

    logger.info(
        f"OI change: CE={ce_oi_change:+,.0f} PE={pe_oi_change:+,.0f} → {oi_buildup} | "
        f"Walls above: {[(s, f'{oi:,.0f}') for s, oi in walls_above[:2]]} "
        f"Walls below: {[(s, f'{oi:,.0f}') for s, oi in walls_below[:2]]}"
    )

    # Max Pain
    max_pain = _calc_max_pain(strike_data)

    # ATM strike + IV
    atm_strike = min(strike_data.keys(), key=lambda s: abs(s - spot))
    atm = strike_data.get(atm_strike, {})
    iv = (atm.get("call_iv", 0) + atm.get("put_iv", 0)) / 2

    # ── ATM OI tracking (for gamma squeeze detection) ─────────────
    atm_ce_oi = atm.get("call_oi", 0)
    atm_pe_oi = atm.get("put_oi", 0)

    # 5-min ATM OI change
    # BUG FIX: when spot moves ~50pts the ATM strike rolls (e.g. 23350→23400).
    # Comparing the new ATM's OI against the old ATM's OI produces
    # meaningless +78% / +117% "changes" (different strikes!).
    # Zero out the change whenever the ATM strike has changed.
    atm_strike_rolled = (_prev_atm_strike > 0 and _prev_atm_strike != atm_strike)
    if atm_strike_rolled:
        logger.debug(
            f"ATM strike rolled {_prev_atm_strike:.0f} → {atm_strike:.0f} — "
            f"OI change zeroed to avoid false signal"
        )
    atm_ce_oi_change = (
        atm_ce_oi - _prev_atm_ce_oi
        if (_prev_atm_ce_oi > 0 and not atm_strike_rolled) else 0.0
    )
    atm_pe_oi_change = (
        atm_pe_oi - _prev_atm_pe_oi
        if (_prev_atm_pe_oi > 0 and not atm_strike_rolled) else 0.0
    )

    # ATM unwind rate (% change — negative = unwinding/short-covering)
    atm_ce_unwind_pct = (atm_ce_oi_change / _prev_atm_ce_oi * 100) if (_prev_atm_ce_oi > 0 and not atm_strike_rolled) else 0.0
    atm_pe_unwind_pct = (atm_pe_oi_change / _prev_atm_pe_oi * 100) if (_prev_atm_pe_oi > 0 and not atm_strike_rolled) else 0.0

    _prev_atm_ce_oi = atm_ce_oi
    _prev_atm_pe_oi = atm_pe_oi
    _prev_atm_strike = atm_strike

    # ── PCR 5-min momentum ────────────────────────────────────────
    pcr_5m_momentum = round(pcr - _prev_pcr, 4) if _prev_pcr > 0 else 0.0
    pcr_5m_ago = _prev_pcr
    _prev_pcr = pcr

    logger.info(
        f"ATM OI ({atm_strike:.0f}): CE={atm_ce_oi:,.0f}({atm_ce_oi_change:+,.0f} "
        f"{atm_ce_unwind_pct:+.1f}%) PE={atm_pe_oi:,.0f}({atm_pe_oi_change:+,.0f} "
        f"{atm_pe_unwind_pct:+.1f}%) | PCR_mom={pcr_5m_momentum:+.4f}"
    )

    return OptionsFlow(
        pcr=round(pcr, 3), pcr_bias=pcr_bias,
        max_pain=round(max_pain), iv_atm=round(iv, 1),
        oi_support=round(support), oi_resistance=round(resistance),
        spot=spot, available=True,
        total_ce_oi=total_ce_oi, total_pe_oi=total_pe_oi,
        ce_oi_change=ce_oi_change, pe_oi_change=pe_oi_change,
        oi_buildup=oi_buildup,
        oi_walls_above=walls_above, oi_walls_below=walls_below,
        atm_strike=atm_strike, atm_ce_oi=atm_ce_oi, atm_pe_oi=atm_pe_oi,
        atm_ce_oi_change=atm_ce_oi_change, atm_pe_oi_change=atm_pe_oi_change,
        atm_ce_unwind_pct=round(atm_ce_unwind_pct, 2),
        atm_pe_unwind_pct=round(atm_pe_unwind_pct, 2),
        pcr_5m_ago=round(pcr_5m_ago, 3), pcr_5m_momentum=pcr_5m_momentum,
    )


def _calc_max_pain(strike_data: dict) -> float:
    """Max pain = strike where total option buyer loss is maximum."""
    strikes = sorted(strike_data.keys())
    if not strikes:
        return 0.0
    min_loss = float("inf")
    pain = strikes[0]
    for test in strikes:
        loss = 0
        for s, d in strike_data.items():
            if test > s:
                loss += (test - s) * d["call_oi"]
            if test < s:
                loss += (s - test) * d["put_oi"]
        if loss < min_loss:
            min_loss = loss
            pain = test
    return float(pain)


# ======================================================================
# SENTIMENT SCORE (weighted: PCR + VIX + FII/DII + A/D ratio)
# ======================================================================

@dataclass
class SentimentScore:
    score: float = 50.0       # 0-100
    mood: str = "NEUTRAL"     # BEARISH / NEUTRAL / BULLISH / EUPHORIA
    bias: str = "NEUTRAL"     # BUY / SELL / NEUTRAL / AVOID
    safe: bool = True
    vix: float = 15.0
    fii_net: float = 0.0
    ad_ratio: float = 1.0
    reason: str = ""


def fetch_sentiment(pcr: float = 1.0) -> SentimentScore:
    """Compute weighted sentiment from multiple factors."""

    # VIX
    vix = _fetch_vix()

    # FII/DII
    fii_net = _fetch_fii_net()

    # Advance/Decline
    ad_ratio = _fetch_ad_ratio()

    # PCR score (0-100): PCR 1.3+ = 100, PCR 0.5 = 0
    pcr_score = min(100, max(0, (pcr - 0.5) / 0.8 * 100))

    # VIX score
    if vix <= 12:
        vix_score = 60      # complacent
    elif vix <= 20:
        vix_score = 80      # ideal
    elif vix <= 25:
        vix_score = 40      # elevated
    else:
        vix_score = 10      # panic

    # FII score: +2000Cr = 90, 0 = 50, -2000Cr = 10
    fii_score = min(90, max(10, 50 + fii_net / 40))

    # A/D score: 2.0 = 90, 1.0 = 50, 0.5 = 20
    ad_score = min(90, max(10, 50 + (ad_ratio - 1.0) * 40))

    # Weighted total
    total = round(
        pcr_score * 0.35 + vix_score * 0.30 +
        fii_score * 0.20 + ad_score * 0.15, 1
    )

    # Mood and bias
    if total >= 80:
        mood, bias, safe = "EUPHORIA", "AVOID", False
        reason = f"Overextended (score={total}) - skip entries"
    elif total >= 60:
        mood, bias, safe = "BULLISH", "BUY", True
        reason = f"Bullish sentiment (score={total})"
    elif total >= 35:
        mood, bias, safe = "NEUTRAL", "NEUTRAL", True
        reason = f"Neutral sentiment (score={total})"
    else:
        mood, bias, safe = "BEARISH", "SELL", True
        reason = f"Bearish sentiment (score={total})"

    if vix > 30:
        safe = False
        reason = f"VIX={vix:.1f} > 30 - extreme panic"

    return SentimentScore(
        score=total, mood=mood, bias=bias, safe=safe,
        vix=vix, fii_net=fii_net, ad_ratio=ad_ratio, reason=reason,
    )


def _fetch_vix() -> float:
    """Fetch India VIX from NSE allIndices API."""
    data = _nse_get("https://www.nseindia.com/api/allIndices")
    if data:
        for item in data.get("data", []):
            if "VIX" in item.get("index", "").upper():
                v = float(item.get("last", 0))
                if v > 0:
                    return v
    return 15.0


def _fetch_fii_net() -> float:
    """Fetch FII net buy/sell in crores."""
    data = _nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
    if data:
        for row in (data if isinstance(data, list) else []):
            if "FII" in str(row.get("category", "")).upper():
                buy = float(str(row.get("buyValue", "0")).replace(",", "") or 0)
                sell = float(str(row.get("sellValue", "0")).replace(",", "") or 0)
                return round(buy - sell, 2)
    return 0.0


def _fetch_ad_ratio() -> float:
    """Fetch Advance/Decline ratio from NSE."""
    data = _nse_get("https://www.nseindia.com/api/allIndices")
    if data:
        for item in data.get("data", []):
            if item.get("index") == "NIFTY 50":
                adv = int(item.get("advances", 1))
                dec = int(item.get("declines", 1))
                return round(adv / max(dec, 1), 3)
    return 1.0


# ======================================================================
# PRE-TRADE CHECK -- single function the pilot calls
# ======================================================================

@dataclass
class MarketIntel:
    safe: bool = True
    bias: str = "NEUTRAL"          # BULLISH / BEARISH / NEUTRAL
    sentiment_score: float = 50.0
    pcr: float = 1.0
    max_pain: float = 0.0
    iv: float = 15.0
    vix: float = 15.0
    oi_support: float = 0.0        # Max PE OI strike below spot
    oi_resistance: float = 0.0     # Max CE OI strike above spot
    fii_net: float = 0.0
    event_today: str = ""
    context_str: str = ""          # one-line summary for Claude prompt
    # OI change data
    oi_buildup: str = "NEUTRAL"    # CALL_WRITING / PUT_WRITING / LONG_BUILD / SHORT_BUILD / etc.
    ce_oi_change: float = 0.0
    pe_oi_change: float = 0.0
    oi_walls_above: list = None    # [(strike, ce_oi), ...] resistance walls
    oi_walls_below: list = None    # [(strike, pe_oi), ...] support walls
    # ATM OI + Gamma squeeze data
    atm_strike: float = 0.0
    atm_ce_oi: float = 0.0
    atm_pe_oi: float = 0.0
    atm_ce_oi_change: float = 0.0
    atm_pe_oi_change: float = 0.0
    atm_ce_unwind_pct: float = 0.0  # negative = CE writers unwinding (squeeze fuel)
    atm_pe_unwind_pct: float = 0.0  # negative = PE writers unwinding
    # PCR momentum
    pcr_5m_momentum: float = 0.0   # current_PCR - PCR_5m_ago
    # Phase 3 fix (2026-06-09): distinguish real PCR from the OptionsFlow default of 1.0.
    # False = OI fetch failed; the pcr field holds the default sentinel value (1.0),
    # not a measured Put/Call ratio.  True = broker or NSE chain returned real OI data.
    pcr_available: bool = False

    def __post_init__(self):
        if self.oi_walls_above is None:
            self.oi_walls_above = []
        if self.oi_walls_below is None:
            self.oi_walls_below = []


# Cache
_cached_intel: Optional[MarketIntel] = None
_cached_intel_time: float = 0
INTEL_CACHE_TTL = 300  # 5 minutes


def pre_trade_check(force_refresh: bool = False) -> MarketIntel:
    """
    Single function the pilot calls before every trade decision.
    Fetches options flow + sentiment + events, returns unified verdict.
    Cached for 5 minutes.
    """
    global _cached_intel, _cached_intel_time
    now = time.monotonic()
    if not force_refresh and _cached_intel and (now - _cached_intel_time) < INTEL_CACHE_TTL:
        return _cached_intel

    logger.info("Market intel: fetching options flow + sentiment + events...")

    # 1. Events
    evt = check_events_today()

    # 2. Options flow
    options = fetch_options_flow()

    # 3. Sentiment (pass PCR from options if available)
    sentiment = fetch_sentiment(pcr=options.pcr if options.available else 1.0)

    # 4. Combine
    safe = True
    reasons = []

    # Event block
    if evt["risk"] == "HIGH":
        safe = False
        reasons.append(f"HIGH-RISK EVENT: {evt['name']}")
    elif evt["risk"] == "MEDIUM":
        reasons.append(f"Event: {evt['name']} (trade with caution)")

    # Sentiment block
    if not sentiment.safe:
        safe = False
        reasons.append(sentiment.reason)

    # VIX — dynamic regime (no hard block until VIX > 30)
    if sentiment.vix > 30:
        safe = False
        reasons.append(f"VIX={sentiment.vix:.1f} EXTREME PANIC")
    elif sentiment.vix > 25:
        reasons.append(f"VIX={sentiment.vix:.1f} VERY HIGH - trend-only, wider SL")
    elif sentiment.vix > 22:
        reasons.append(f"VIX={sentiment.vix:.1f} HIGH - OTM preferred, wider SL")
    elif sentiment.vix > 18:
        reasons.append(f"VIX={sentiment.vix:.1f} elevated")

    # BUG FIX: Kotak option chain API doesn't include IV — chain gives OI+LTP only.
    # _compute_options_flow() ends up with iv_atm=0. Patch it here from the
    # IVEngine singleton (which computes IV via Black-Scholes and runs earlier
    # in the pilot cycle, so its cache is always fresh).
    if options.iv_atm == 0 and options.available:
        try:
            from core.iv_engine import get_iv_engine
            _iv_eng = get_iv_engine()
            if _iv_eng is not None and _iv_eng._cache is not None:
                _cached_iv = _iv_eng._cache.atm_iv
                if _cached_iv > 0:
                    options.iv_atm = _cached_iv
        except Exception:
            pass

    # IV spike block
    if options.iv_atm > 25:
        reasons.append(f"IV={options.iv_atm:.0f}% HIGH - expect big move")

    # Determine bias
    if options.available:
        bias = options.pcr_bias
    else:
        bias = sentiment.bias if sentiment.bias != "AVOID" else "NEUTRAL"

    # Build context string for Claude
    parts = []
    if options.available:
        parts.append(f"PCR={options.pcr:.2f}({options.pcr_bias})")
        parts.append(f"MaxPain={options.max_pain:.0f}")
        parts.append(f"IV={options.iv_atm:.0f}%")
        parts.append(f"Support={options.oi_support:.0f}")
        parts.append(f"Resistance={options.oi_resistance:.0f}")
        parts.append(f"OI_Buildup={options.oi_buildup}")
        if options.oi_walls_above:
            parts.append(f"Wall_Above={options.oi_walls_above[0][0]:.0f}")
        if options.oi_walls_below:
            parts.append(f"Wall_Below={options.oi_walls_below[0][0]:.0f}")
    parts.append(f"VIX={sentiment.vix:.1f}")
    if sentiment.fii_net != 0:
        parts.append(f"FII={sentiment.fii_net:+.0f}Cr")
    parts.append(f"Sentiment={sentiment.score:.0f}/100({sentiment.mood})")
    if evt["event"]:
        parts.append(f"EVENT={evt['name']}")
    if reasons:
        parts.append(f"FLAGS={';'.join(reasons)}")

    context_str = " | ".join(parts)

    intel = MarketIntel(
        safe=safe,
        bias=bias,
        sentiment_score=sentiment.score,
        pcr=options.pcr if options.available else 1.0,
        pcr_available=options.available,
        max_pain=options.max_pain,
        iv=options.iv_atm,
        vix=sentiment.vix,
        oi_support=options.oi_support,
        oi_resistance=options.oi_resistance,
        fii_net=sentiment.fii_net,
        event_today=evt["name"],
        context_str=context_str,
        oi_buildup=options.oi_buildup,
        ce_oi_change=options.ce_oi_change,
        pe_oi_change=options.pe_oi_change,
        oi_walls_above=options.oi_walls_above,
        oi_walls_below=options.oi_walls_below,
        atm_strike=options.atm_strike,
        atm_ce_oi=options.atm_ce_oi,
        atm_pe_oi=options.atm_pe_oi,
        atm_ce_oi_change=options.atm_ce_oi_change,
        atm_pe_oi_change=options.atm_pe_oi_change,
        atm_ce_unwind_pct=options.atm_ce_unwind_pct,
        atm_pe_unwind_pct=options.atm_pe_unwind_pct,
        pcr_5m_momentum=options.pcr_5m_momentum,
    )

    _cached_intel = intel
    _cached_intel_time = now

    logger.info(f"Market intel: {context_str}")
    logger.info(f"Market intel: safe={safe} bias={bias} score={sentiment.score:.0f}")

    return intel


def is_signal_aligned(signal_direction: str, intel: Optional[MarketIntel] = None) -> tuple:
    """
    Check if a trade direction aligns with OI data + market sentiment.
    Returns (aligned: bool, reason: str)

    Hard blocks (returns False):
      - Extreme PCR: > 1.8 blocks PUT (too many puts = contrarian bullish)
      - Extreme PCR: < 0.5 blocks CALL (too few puts = complacent, bearish)
      - OI buildup directly contradicts signal

    Soft warnings (returns True with reason):
      - Moderate PCR misalignment (0.7-1.3 range)
    """
    if intel is None:
        intel = pre_trade_check()

    # --- Hard gate: Extreme PCR ---
    # PCR > 1.8 = extreme put writing = strong support = contrarian BULLISH
    # Taking a PUT against this wall of support is dangerous
    if signal_direction == "PUT" and intel.pcr > 1.8:
        return False, (
            f"HARD BLOCK: PCR={intel.pcr:.2f} (extreme put writing = "
            f"strong support) — PUT against massive OI support"
        )

    # PCR < 0.5 = extreme call writing = heavy resistance = contrarian BEARISH
    # Taking a CALL against this wall of resistance is dangerous
    if signal_direction == "CALL" and intel.pcr < 0.5:
        return False, (
            f"HARD BLOCK: PCR={intel.pcr:.2f} (extreme call writing = "
            f"heavy resistance) — CALL against massive OI resistance"
        )

    # --- Hard gate: OI buildup contradicts signal ---
    # CALL_WRITING or SHORT_BUILD = bearish OI activity → blocks CALL
    if signal_direction == "CALL" and intel.oi_buildup in ("CALL_WRITING", "SHORT_BUILD"):
        return False, (
            f"HARD BLOCK: OI={intel.oi_buildup} (CE OI building "
            f"+{intel.ce_oi_change:,.0f}) — institutions writing calls, blocks CALL"
        )

    # PUT_WRITING or LONG_BUILD = bullish OI activity → blocks PUT
    if signal_direction == "PUT" and intel.oi_buildup in ("PUT_WRITING", "LONG_BUILD"):
        return False, (
            f"HARD BLOCK: OI={intel.oi_buildup} (PE OI building "
            f"+{intel.pe_oi_change:,.0f}) — institutions writing puts, blocks PUT"
        )

    # --- Soft warning: Moderate PCR misalignment ---
    if intel.bias == "NEUTRAL":
        return True, "Market neutral - no conflict"

    if signal_direction == "CALL" and intel.bias == "BEARISH":
        return True, f"WARNING: PCR={intel.pcr:.2f} bearish - proceed with caution"

    if signal_direction == "PUT" and intel.bias == "BULLISH":
        return True, f"WARNING: PCR={intel.pcr:.2f} bullish - proceed with caution"

    return True, f"Market {intel.bias} + OI {intel.oi_buildup} aligns with {signal_direction}"


# ======================================================================
# ASYNC NON-BLOCKING OI FETCH
# ======================================================================
# The position monitor runs every 5 seconds. Blocking it for 2-3s to
# fetch the NSE option chain would stall SL/TP/trail checks.
# This wrapper runs the fetch in a background thread and caches the result.

import asyncio
import concurrent.futures

_async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="oi_fetch")
_async_intel: Optional[MarketIntel] = None
_async_fetch_lock = threading.Lock()
_async_last_fetch: float = 0.0
ASYNC_FETCH_INTERVAL = 300  # 5 minutes


def fetch_intel_async() -> Optional[MarketIntel]:
    """
    Non-blocking OI fetch. Returns cached MarketIntel immediately.
    Triggers a background refresh if cache is stale (>5 min).
    Safe to call from the position monitor's 5-second loop.
    """
    global _async_intel, _async_last_fetch

    now = time.monotonic()

    # Return cache if fresh
    if _async_intel and (now - _async_last_fetch) < ASYNC_FETCH_INTERVAL:
        return _async_intel

    # Submit background fetch (non-blocking)
    with _async_fetch_lock:
        if (now - _async_last_fetch) >= ASYNC_FETCH_INTERVAL:
            _async_last_fetch = now  # Prevent duplicate submits
            _async_executor.submit(_do_async_fetch)

    return _async_intel  # Return stale cache while refresh runs


def _do_async_fetch():
    """Runs in background thread — fetches fresh intel and updates cache."""
    global _async_intel
    try:
        intel = pre_trade_check(force_refresh=True)
        _async_intel = intel
        logger.debug(f"Async OI fetch complete: PCR={intel.pcr:.2f} buildup={intel.oi_buildup}")
    except Exception as e:
        logger.debug(f"Async OI fetch failed: {e}")
