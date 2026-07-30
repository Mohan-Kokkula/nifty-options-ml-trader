"""
oi_proxy.py — Lightweight OI/option chain proxy server.

Runs as a separate process alongside the main bot.
Fetches NSE option chain every 60s and serves it locally
so the bot doesn't hit NSE directly (avoiding IP blocks).

Usage:
    python scripts/oi_proxy.py &

Serves two endpoint families:
  1. Native proxy format (original):
       http://localhost:5050/optionchain?symbol=NIFTY

  2. OpenAlgo-compatible REST (NEW — lets openalgo_client.py use this proxy):
       http://localhost:5050/api/v1/funds          → health check (always 200)
       http://localhost:5050/api/v1/optionchain    → NSE chain in OA format
       http://localhost:5050/api/v1/expiry         → current expiry list

Set in config/settings.env:
    OPENALGO_URL=http://localhost:5050
    OPENALGO_API_KEY=  (leave blank — not used by proxy)

Then start proxy on Hostinger BEFORE the bot:
    nohup python scripts/oi_proxy.py >> logs/oi_proxy.log 2>&1 &
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("oi_proxy")

PORT      = 5050
REFRESH   = 60   # seconds between NSE fetches
_cache: dict = {}
_lock         = threading.Lock()

NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/option-chain",
    "X-Requested-With":"XMLHttpRequest",
}


def _fetch_nse(symbol: str = "NIFTY") -> dict:
    """Fetch NSE option chain and return parsed data."""
    try:
        # Step 1: get cookies
        req0 = Request("https://www.nseindia.com/option-chain", headers=NSE_HEADERS)
        with urlopen(req0, timeout=10) as r:
            cookies = r.getheader("Set-Cookie") or ""

        headers = {**NSE_HEADERS, "Cookie": cookies}

        # Step 2: fetch option chain
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        records = data.get("records", {})
        oc_data = records.get("data", [])
        spot    = float(records.get("underlyingValue", 0))

        chain = []
        for item in oc_data:
            strike = float(item.get("strikePrice", 0))
            if strike <= 0:
                continue
            ce = item.get("CE", {}) or {}
            pe = item.get("PE", {}) or {}
            chain.append({
                "strike": strike,
                "ce": {
                    "oi":  float(ce.get("openInterest", 0) or 0),
                    "ltp": float(ce.get("lastPrice", 0) or 0),
                    "iv":  float(ce.get("impliedVolatility", 0) or 0),
                },
                "pe": {
                    "oi":  float(pe.get("openInterest", 0) or 0),
                    "ltp": float(pe.get("lastPrice", 0) or 0),
                    "iv":  float(pe.get("impliedVolatility", 0) or 0),
                },
            })

        if not chain:
            return {}

        total_ce = sum(e["ce"]["oi"] for e in chain)
        total_pe = sum(e["pe"]["oi"] for e in chain)
        pcr = total_pe / total_ce if total_ce > 0 else 1.0

        result = {
            "underlying_ltp": spot,
            "atm_strike":     int(round(spot / 50) * 50),
            "chain":          chain,
            "pcr":            round(pcr, 4),
            "total_ce_oi":    total_ce,
            "total_pe_oi":    total_pe,
            "source":         "NSE_proxy",
        }
        log.info(f"NSE fetch OK: spot={spot} strikes={len(chain)} PCR={pcr:.2f}")
        return result

    except Exception as e:
        log.warning(f"NSE fetch failed: {e}")
        return {}


def _refresh_loop():
    """Background thread: refresh cache every REFRESH seconds."""
    while True:
        data = _fetch_nse("NIFTY")
        if data:
            with _lock:
                _cache["NIFTY"] = data
                _cache["ts"]    = time.time()
        time.sleep(REFRESH)


def _to_openalgo_format(cache_data: dict) -> list:
    """
    Convert our normalized chain format to OpenAlgo/NSE-compatible format.
    This is the format openalgo_client._normalise_chain() knows how to parse.

    Each item returned:
      {
        "strikePrice": 23500,
        "underlyingValue": 23750.5,   ← spot embedded so _normalise_chain() picks it up
        "CE": {"openInterest": ..., "lastPrice": ..., "impliedVolatility": ...},
        "PE": {"openInterest": ..., "lastPrice": ..., "impliedVolatility": ...},
      }
    """
    spot  = cache_data.get("underlying_ltp", 0)
    chain = cache_data.get("chain", [])
    result = []
    for item in chain:
        ce = item.get("ce", {})
        pe = item.get("pe", {})
        result.append({
            "strikePrice":    item["strike"],
            "underlyingValue": spot,       # embedded so normalise_chain extracts spot
            "CE": {
                "openInterest":      ce.get("oi", 0),
                "lastPrice":         ce.get("ltp", 0),
                "impliedVolatility": ce.get("iv", 0),
            },
            "PE": {
                "openInterest":      pe.get("oi", 0),
                "lastPrice":         pe.get("ltp", 0),
                "impliedVolatility": pe.get("iv", 0),
            },
        })
    return result


def _get_expiry_str() -> str:
    """
    Read NIFTY_EXPIRY from environment and convert to OpenAlgo format (26-MAY-2026).
    Falls back to empty string if not set.
    """
    expiry = os.environ.get("NIFTY_EXPIRY", "").strip().upper()
    if not expiry:
        return ""
    from datetime import datetime
    for fmt in ("%d%b%y", "%d%b%Y"):
        try:
            dt = datetime.strptime(expiry, fmt)
            return dt.strftime("%d-%b-%Y").upper()
        except ValueError:
            continue
    return expiry


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request noise

    def _send_json(self, body_dict: dict, status: int = 200):
        body = json.dumps(body_dict).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_503(self):
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b'{"error":"no data yet"}')

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)
        symbol = (params.get("symbol", ["NIFTY"])[0]).upper()

        # ── Route: /api/v1/funds (OpenAlgo health check — always 200) ──────────
        if path == "/api/v1/funds":
            self._send_json({"status": "ok", "data": {"balance": 0}})
            return

        # ── Route: /api/v1/expiry (return current expiry from env) ─────────────
        if path == "/api/v1/expiry":
            expiry = _get_expiry_str()
            dates  = [expiry] if expiry else []
            self._send_json({"status": "ok", "data": dates})
            return

        # ── Route: /api/v1/optionchain (OpenAlgo-compatible format) ────────────
        if path == "/api/v1/optionchain":
            with _lock:
                cache_data = _cache.get(symbol, {})
            if not cache_data:
                self._send_503()
                return
            oa_chain = _to_openalgo_format(cache_data)
            self._send_json({
                "status": "ok",
                "data":   oa_chain,           # openalgo_client reads "data" key
            })
            return

        # ── Route: /optionchain (original native proxy format) ──────────────────
        if path == "/optionchain" or path == "/":
            with _lock:
                data = _cache.get(symbol, {})
            if data:
                self._send_json(data)
            else:
                self._send_503()
            return

        # Unknown path
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'{"error":"not found"}')


if __name__ == "__main__":
    log.info(f"OI Proxy starting on port {PORT} — fetching NSE every {REFRESH}s")
    log.info(f"Endpoints:")
    log.info(f"  Native   : http://localhost:{PORT}/optionchain?symbol=NIFTY")
    log.info(f"  OpenAlgo : http://localhost:{PORT}/api/v1/optionchain?symbol=NIFTY")
    log.info(f"  Health   : http://localhost:{PORT}/api/v1/funds")
    log.info(f"  Expiry   : http://localhost:{PORT}/api/v1/expiry")
    expiry = _get_expiry_str()
    if expiry:
        log.info(f"  Current expiry from env: {expiry}")
    else:
        log.warning("  NIFTY_EXPIRY not set in env — /api/v1/expiry will return []")

    # Initial fetch before serving
    data = _fetch_nse("NIFTY")
    if data:
        _cache["NIFTY"] = data
        _cache["ts"]    = time.time()
        log.info("Initial NSE fetch OK")
    else:
        log.warning("Initial NSE fetch failed — will retry in background")

    # Start background refresh
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    log.info(f"Ready — set OPENALGO_URL=http://localhost:{PORT} in settings.env")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("OI Proxy stopped")
