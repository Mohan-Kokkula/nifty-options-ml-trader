"""
openalgo_client.py — HTTP REST client for OpenAlgo broker bridge.

OpenAlgo is an open-source broker middleware that exposes a clean REST API
over Kotak Neo (and other brokers). It handles session management, TOTP login,
and provides option chain data in a standard format.

Setup:
    1. Deploy OpenAlgo on Hostinger: https://github.com/marketcalls/openalgo
    2. Set OPENALGO_URL and OPENALGO_API_KEY in config/settings.env
    3. Bot auto-detects OpenAlgo at startup and uses it for option chain + OI

API used (OpenAlgo's REST API is POST-only with the API key in the JSON
body — a GET to any /api/v1/* path falls through to the SPA and returns
index.html, not an error):
    POST /api/v1/funds        {"apikey": ...}
    POST /api/v1/expiry       {"apikey": ..., "symbol": "NIFTY", "exchange": "NFO", "instrumenttype": "options"}
    POST /api/v1/optionchain  {"apikey": ..., "underlying": "NIFTY", "exchange": "NFO", "expiry_date": "16JUN26", "strike_count": 20}
    POST /api/v1/placeorder   (future: order routing)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime as _dt
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT   = 8    # seconds
_RETRY_MAX = 2

# ── 2026-06-12 PHASE-1 IMPL: intraday OI snapshot archive ────────────────
# Every successful chain fetch is appended to data/oi_archive/oi_YYYY-MM-DD.csv
# (one row per strike per snapshot). Purely passive research capture — reads
# nothing, gates nothing, and any failure is swallowed after a log.
_ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "oi_archive",
)
_ARCHIVE_COLS = ("snapshot_ts,expiry,spot,atm_strike,strike,"
                 "ce_oi,ce_ltp,ce_iv,pe_oi,pe_ltp,pe_iv,pcr_total\n")

# Reliability/observability state for the archive hook only (read via
# get_archive_health() for monitoring; never read by any trading path).
_archive_health = {
    "last_success_ts": None,       # str "%Y-%m-%d %H:%M:%S", last row actually written
    "last_error": None,
    "last_error_ts": None,
    "consecutive_failures": 0,
    "total_snapshots_written": 0,
    "total_rows_written": 0,
    "duplicate_snapshots_skipped": 0,
}
_FAILURE_LOG_EVERY = 10   # rate-limit repeated-failure warnings to avoid log spam
_last_archived_ts = None  # dedup: skip re-writing an identical (ts) snapshot


def get_archive_health() -> dict:
    """Read-only snapshot of the OI archive hook's own health. Never
    consulted by any trading/signal-generation code path — monitoring only."""
    return dict(_archive_health)


def _archive_snapshot(result: dict, expiry: str) -> None:
    global _last_archived_ts
    try:
        if not result or not result.get("chain"):
            return
        now = _dt.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        # Dedup: get_option_chain() can be called more than once for the same
        # real-world snapshot (different callers within the same second).
        # Re-archiving identical rows was silently inflating the file with
        # exact duplicates; skip if this exact timestamp was just archived.
        if ts == _last_archived_ts:
            _archive_health["duplicate_snapshots_skipped"] += 1
            return

        os.makedirs(_ARCHIVE_DIR, exist_ok=True)
        path = os.path.join(_ARCHIVE_DIR, f"oi_{now:%Y-%m-%d}.csv")
        new_file = not os.path.exists(path)
        spot = result.get("underlying_ltp", 0)
        atm = result.get("atm_strike", 0)
        pcr = result.get("pcr", 0)
        n_rows = 0
        with open(path, "a", encoding="utf-8", newline="") as fh:
            if new_file:
                fh.write(_ARCHIVE_COLS)
            for row in result["chain"]:
                ce, pe = row.get("ce", {}), row.get("pe", {})
                fh.write(
                    f"{ts},{expiry},{spot},{atm},{row.get('strike', 0)},"
                    f"{ce.get('oi', 0)},{ce.get('ltp', 0)},{ce.get('iv', 0)},"
                    f"{pe.get('oi', 0)},{pe.get('ltp', 0)},{pe.get('iv', 0)},"
                    f"{pcr}\n"
                )
                n_rows += 1
        _last_archived_ts = ts
        _archive_health["last_success_ts"] = ts
        _archive_health["consecutive_failures"] = 0
        _archive_health["total_snapshots_written"] += 1
        _archive_health["total_rows_written"] += n_rows
    except Exception as _e:                       # never disturb trading
        _archive_health["last_error"] = str(_e)
        _archive_health["last_error_ts"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        _archive_health["consecutive_failures"] += 1
        if _archive_health["consecutive_failures"] % _FAILURE_LOG_EVERY == 1:
            logger.warning(
                f"OI archive write failed ({_archive_health['consecutive_failures']} "
                f"consecutive): {_e}"
            )
        else:
            logger.debug(f"OI archive write failed (ignored): {_e}")


class OpenAlgoClient:
    """
    Thin HTTP client for OpenAlgo REST API.
    Compatible with the same interface KotakNeoClient exposes so
    market_intel.py works with either client.
    """

    def __init__(self, base_url: str, api_key: str):
        self._base    = base_url.rstrip("/")
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
        })
        logger.info(f"OpenAlgoClient: connected to {self._base}")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if OpenAlgo's REST API responds with {"status": "success"}.

        OpenAlgo's API is POST-only with the API key in the JSON body.
        A GET (or header-based auth) doesn't match any backend route and
        falls through to the SPA's catch-all, returning index.html with
        HTTP 200 — that's NOT a working API regardless of login state.
        """
        try:
            resp = self._session.post(
                f"{self._base}/api/v1/funds",
                json={"apikey": self._api_key},
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                return False
            ct = resp.headers.get("Content-Type", "")
            if "application/json" not in ct:
                logger.warning(
                    "OpenAlgo /api/v1/funds returned non-JSON (web UI) — "
                    "API not initialized. Log in to OpenAlgo at "
                    f"{self._base} first, then restart the bot."
                )
                return False
            return resp.json().get("status") == "success"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Option chain — returns OpenAlgo standard format
    # (same dict shape market_intel._fetch_chain_openalgo() already parses)
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NSE",
        expiry_date: str = "",        # "26MAY26" or "26-MAY-2026" — we normalise
        strike_count: int = 20,
    ) -> dict:
        """
        Fetch option chain from OpenAlgo.

        Returns dict:
          {
            "underlying_ltp": float,
            "atm_strike":     int,
            "chain": [
              {"strike": float,
               "ce": {"oi": int, "ltp": float, "iv": float},
               "pe": {"oi": int, "ltp": float, "iv": float}},
              ...
            ]
          }
        or empty dict on failure.
        """
        expiry_fmt = self._normalise_expiry(expiry_date)

        body = {
            "apikey":       self._api_key,
            "underlying":   underlying.upper(),
            "exchange":     exchange.upper(),
            "expiry_date":  expiry_fmt,
            "strike_count": strike_count,
        }
        logger.debug(
            "OpenAlgo optionchain request: "
            f"{ {k: v for k, v in body.items() if k != 'apikey'} }"
        )

        for attempt in range(_RETRY_MAX):
            resp = None
            try:
                resp = self._session.post(
                    f"{self._base}/api/v1/optionchain",
                    json=body,
                    timeout=_TIMEOUT,
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"OpenAlgo optionchain: HTTP {resp.status_code} | "
                        f"body={resp.text[:200]!r} | attempt {attempt+1}"
                    )
                    time.sleep(1)
                    continue

                if not resp.text or not resp.text.strip():
                    logger.warning(
                        f"OpenAlgo optionchain: HTTP 200 but EMPTY body | "
                        f"body={ {k: v for k, v in body.items() if k != 'apikey'} }"
                    )
                    time.sleep(1)
                    continue

                data = resp.json()

                # OpenAlgo returns {status, data: [...]} or direct chain list
                if isinstance(data, dict):
                    if data.get("status") == "error":
                        logger.warning(f"OpenAlgo optionchain error: {data.get('message')}")
                        return {}
                    chain_raw = data.get("data", data.get("chain", []))
                else:
                    chain_raw = data  # raw list

                if not chain_raw:
                    logger.warning("OpenAlgo optionchain: empty chain returned")
                    return {}

                result = self._normalise_chain(chain_raw, underlying)

                # OpenAlgo returns the real spot price and ATM strike at the
                # top level of the response (computed server-side from the
                # broker quote) — prefer these over our chain-based estimates.
                if isinstance(data, dict):
                    ltp = data.get("underlying_ltp")
                    if ltp:
                        result["underlying_ltp"] = float(ltp)
                    atm = data.get("atm_strike")
                    if atm:
                        result["atm_strike"] = int(atm)

                # 2026-06-12 PHASE-1 IMPL: archive every snapshot for future
                # intraday-OI research. Passive — never touches decisions,
                # never raises.
                _archive_snapshot(result, expiry_fmt)

                return result

            except (requests.RequestException, ValueError) as e:
                # Log the raw body so we can see exactly what OpenAlgo returned
                try:
                    raw = resp.text[:400] if resp is not None else "<no response>"
                    logger.warning(
                        f"OpenAlgo optionchain failed ({type(e).__name__}: {e}) | "
                        f"HTTP {resp.status_code if resp is not None else '?'} | "
                        f"body={raw!r}"
                    )
                except Exception:
                    logger.warning(f"OpenAlgo optionchain request failed: {e}")
                time.sleep(1)

        return {}

    def get_expiry_dates(self, symbol: str = "NIFTY", exchange: str = "NFO") -> list:
        """Return list of expiry date strings (DD-MMM-YY) from OpenAlgo.

        OpenAlgo's /api/v1/expiry only accepts exchange in
        {NFO, BFO, MCX, CDS, CRYPTO} for instrumenttype=options — passing
        the underlying's exchange (NSE/NSE_INDEX) returns a 400.
        """
        try:
            resp = self._session.post(
                f"{self._base}/api/v1/expiry",
                json={
                    "apikey":         self._api_key,
                    "symbol":         symbol,
                    "exchange":       exchange,
                    "instrumenttype": "options",
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                dates = data.get("data", data) if isinstance(data, dict) else data
                return dates if isinstance(dates, list) else []
        except Exception as e:
            logger.debug(f"OpenAlgo expiry fetch failed: {e}")
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_expiry(expiry: str) -> str:
        """
        Convert any expiry format to 'DDMMMYY' (e.g. '16JUN26', no
        separators) — the format POST /api/v1/optionchain's 'expiry_date'
        field requires.

        NOTE: this is DIFFERENT from /api/v1/expiry's response format,
        which is 'DD-MMM-YY' (e.g. '16-JUN-26'). OpenAlgo's
        get_available_strikes() builds its DB lookup key as
        expiry_date[:2] + '-' + expiry_date[2:5] + '-' + expiry_date[5:];
        feeding it 'DD-MMM-YY' (dashes already present) produces a garbled
        key ('16--JU-N-26') that matches nothing, causing a spurious
        "No strikes found ... update master contract" 404.

        Input could be: '16JUN26', '16-JUN-26', '16-JUN-2026',
        '2026-06-16', etc.
        """
        if not expiry:
            return ""
        expiry = expiry.strip().upper()

        from datetime import datetime
        for fmt in ("%d%b%y", "%d-%b-%y", "%d-%b-%Y", "%d%b%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(expiry, fmt)
                return dt.strftime("%d%b%y").upper()
            except ValueError:
                continue

        return expiry  # return as-is if can't parse

    @staticmethod
    def _normalise_chain(chain_raw: list, underlying: str) -> dict:
        """
        Normalise OpenAlgo chain response into our standard format:
          {underlying_ltp, atm_strike, chain: [{strike, ce:{oi,ltp,iv}, pe:{oi,ltp,iv}}]}

        OpenAlgo can return data in slightly different shapes depending on version.
        We handle all variants.
        """
        chain_out   = []
        spot        = 0.0
        total_ce_oi = 0.0

        for entry in chain_raw:
            # Strike price
            strike = float(
                entry.get("strikePrice") or
                entry.get("strike_price") or
                entry.get("strike") or 0
            )
            if strike <= 0:
                continue

            # CE side — try nested and flat keys
            ce_raw = entry.get("CE") or entry.get("ce") or {}
            pe_raw = entry.get("PE") or entry.get("pe") or {}

            def _safe(d, *keys):
                for k in keys:
                    v = d.get(k)
                    if v not in (None, "", "0", 0):
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return 0.0

            ce = {
                "oi":  _safe(ce_raw, "openInterest",        "oi",  "OI"),
                "ltp": _safe(ce_raw, "lastPrice",            "ltp", "LTP"),
                "iv":  _safe(ce_raw, "impliedVolatility",    "iv",  "IV"),
            }
            pe = {
                "oi":  _safe(pe_raw, "openInterest",        "oi",  "OI"),
                "ltp": _safe(pe_raw, "lastPrice",            "ltp", "LTP"),
                "iv":  _safe(pe_raw, "impliedVolatility",    "iv",  "IV"),
            }

            # Underlying spot (present on first few records usually)
            if spot == 0:
                spot = float(
                    entry.get("underlyingValue") or
                    entry.get("underlying_ltp") or
                    entry.get("underlying") or 0
                )

            total_ce_oi += ce["oi"]
            chain_out.append({"strike": strike, "ce": ce, "pe": pe})

        # Estimate ATM from spot
        atm = 0
        if spot > 0 and chain_out:
            atm = int(round(spot / 50) * 50)
        elif chain_out:
            # midpoint of chain
            strikes = [e["strike"] for e in chain_out]
            atm = int(strikes[len(strikes) // 2])

        total_pe_oi = sum(e["pe"]["oi"] for e in chain_out)
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

        logger.info(
            f"OpenAlgo chain: {len(chain_out)} strikes | spot={spot:.0f} | "
            f"CE_OI={total_ce_oi:,.0f} PE_OI={total_pe_oi:,.0f} PCR={pcr:.2f}"
        )

        return {
            "underlying_ltp": spot,
            "atm_strike":     atm,
            "chain":          chain_out,
            "pcr":            pcr,
            "total_ce_oi":    total_ce_oi,
            "total_pe_oi":    total_pe_oi,
            "source":         "OpenAlgo",
        }


def create_openalgo_client() -> Optional[OpenAlgoClient]:
    """
    Create OpenAlgoClient from environment variables.
    Returns None if OPENALGO_URL is not set.
    """
    url = os.getenv("OPENALGO_URL", "").strip()
    key = os.getenv("OPENALGO_API_KEY", "").strip()

    if not url:
        return None

    client = OpenAlgoClient(base_url=url, api_key=key)
    if client.ping():
        logger.info(f"OpenAlgo: reachable at {url} — using as primary OI source")
        return client
    else:
        logger.warning(f"OpenAlgo: NOT reachable at {url} — falling back to Kotak direct")
        return None
