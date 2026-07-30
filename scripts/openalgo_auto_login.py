#!/usr/bin/env python3
"""
scripts/openalgo_auto_login.py
==============================
Auto-login to OpenAlgo and authenticate the Kotak Neo broker session.

Confirmed login flow (source: openalgo/blueprints/brlogin.py + auth.py):
  Step 1  GET  /auth/csrf-token              → receive CSRF token
  Step 2  POST /auth/login (multipart)       → admin session (username + password)
  Step 3  GET  fresh CSRF token              → needed for subsequent POST
  Step 4  POST /kotak/callback (form-data)   → submit mobile + mpin + totp
                                               Flask calls Kotak API server-side
  Step 5  POST /api/v1/funds {"apikey": ...} → verify {"status": "success"}

NOTE: The React JS bundle shows /kotak/auth and /auth/broker/kotak — these are
NOT Flask routes. The real route is /kotak/callback (blueprints/brlogin.py).

Usage:
    python scripts/openalgo_auto_login.py          # run once
    python scripts/openalgo_auto_login.py --check  # check only, no login
    python scripts/openalgo_auto_login.py --retries 3

Called by cron at 09:00 IST Mon-Fri (see scripts/crontab.openclaw).
Also called by _try_connect_openalgo() in core/market_intel.py when ping
returns HTML (broker not yet authenticated).

Required in config/settings.env:
    OPENALGO_URL=http://187.127.150.248:5000
    OPENALGO_API_KEY=...
    OPENALGO_ADMIN_USER=MohanKokkula
    OPENALGO_ADMIN_PASSWORD=...
    OPENALGO_MOBILE=9480893593       # 10-digit, no +91 prefix
    OPENALGO_MPIN=192017             # 6-digit trading PIN
    OPENALGO_TOTP_SECRET=...         # base32 TOTP secret for OpenAlgo/Kotak
    TELEGRAM_BOT_TOKEN=...           # optional - for alerts
    TELEGRAM_CHAT_ID=...             # optional
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Load settings.env before anything else ──────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / "config" / "settings.env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass  # fall back to pre-exported env vars

import requests
import pyotp

# ── Config from env ──────────────────────────────────────────────────────────
OPENALGO_URL   = os.getenv("OPENALGO_URL",           "").strip().rstrip("/")
API_KEY        = os.getenv("OPENALGO_API_KEY",        "").strip()
ADMIN_USER     = os.getenv("OPENALGO_ADMIN_USER",     "").strip()
ADMIN_PASS     = os.getenv("OPENALGO_ADMIN_PASSWORD", "").strip()
MOBILE         = os.getenv("OPENALGO_MOBILE",         "").strip()
MPIN           = os.getenv("OPENALGO_MPIN",           "").strip()
TOTP_SECRET    = os.getenv("OPENALGO_TOTP_SECRET",    "").strip()
TG_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN",      "").strip()
TG_CHAT        = os.getenv("TELEGRAM_CHAT_ID",        "").strip()

_TIMEOUT = 20   # seconds per HTTP request (Kotak API call can take ~5s)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openalgo_login")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _telegram(msg: str) -> None:
    """Send a Telegram message (non-fatal — never raises)."""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def check_connected() -> bool:
    """
    Return True if OpenAlgo API is live (POST /api/v1/funds returns
    {"status": "success"} JSON). False means either unreachable or broker
    session not established yet.

    OpenAlgo's API is POST-only with the API key in the JSON body — a GET
    with an x-api-key header doesn't match any backend route and falls
    through to the SPA, returning index.html with HTTP 200.
    """
    if not OPENALGO_URL or not API_KEY:
        return False
    try:
        r = requests.post(
            f"{OPENALGO_URL}/api/v1/funds",
            json={"apikey": API_KEY},
            timeout=8,
        )
        if r.status_code != 200:
            return False
        ct = r.headers.get("Content-Type", "")
        if "application/json" not in ct:
            return False
        return r.json().get("status") == "success"
    except Exception:
        return False


def _get_csrf(sess: requests.Session) -> str:
    """Fetch a fresh CSRF token from OpenAlgo."""
    r = sess.get(f"{OPENALGO_URL}/auth/csrf-token", timeout=_TIMEOUT)
    r.raise_for_status()
    token = r.json().get("csrf_token", "")
    if not token:
        raise ValueError("Empty CSRF token in response")
    return token


# ── Main login flow ───────────────────────────────────────────────────────────

def auto_login(retries: int = 2) -> bool:
    """
    Run the full OpenAlgo + Kotak broker login flow.
    Returns True on success, False on all-attempts failure.
    """
    # ── Pre-flight checks ──────────────────────────────────────────────────
    missing = []
    if not OPENALGO_URL:      missing.append("OPENALGO_URL")
    if not ADMIN_USER:        missing.append("OPENALGO_ADMIN_USER")
    if not ADMIN_PASS:        missing.append("OPENALGO_ADMIN_PASSWORD")
    if not MOBILE:            missing.append("OPENALGO_MOBILE")
    if not MPIN:              missing.append("OPENALGO_MPIN")
    if not TOTP_SECRET:       missing.append("OPENALGO_TOTP_SECRET")
    if missing:
        log.error(f"Missing required settings.env keys: {', '.join(missing)}")
        return False

    # ── Already connected? ─────────────────────────────────────────────────
    if check_connected():
        log.info("OpenAlgo already connected — /api/v1/funds returns JSON. Nothing to do.")
        return True

    log.info(f"OpenAlgo at {OPENALGO_URL} — starting auto-login (Kotak broker)...")

    for attempt in range(1, retries + 1):
        log.info(f"--- Attempt {attempt}/{retries} ---")
        try:
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": "OpenClaw-AutoLogin/1.0",
                "Origin":     OPENALGO_URL,
                "Referer":    f"{OPENALGO_URL}/broker/kotak/totp",
                "Accept":     "application/json, text/plain, */*",
            })

            # ── Step 1: CSRF token ─────────────────────────────────────────
            log.info("Step 1 → GET /auth/csrf-token")
            csrf1 = _get_csrf(sess)
            log.info(f"  token={csrf1[:20]}... OK")

            # ── Step 2: Admin login ────────────────────────────────────────
            log.info(f"Step 2 → POST /auth/login (user={ADMIN_USER})")
            r = sess.post(
                f"{OPENALGO_URL}/auth/login",
                files={
                    "username":   (None, ADMIN_USER),
                    "password":   (None, ADMIN_PASS),
                    "csrf_token": (None, csrf1),
                },
                timeout=_TIMEOUT,
            )
            log.info(f"  HTTP {r.status_code}")
            try:
                resp = r.json()
                log.info(f"  {resp}")
                if resp.get("status") == "error":
                    log.error(f"  Admin login rejected: {resp.get('message','')}")
                    break   # wrong credentials — no point retrying
            except Exception:
                pass
            if not sess.cookies.get("session"):
                log.warning("  No session cookie — credentials may be incorrect")
            else:
                log.info("  Session cookie set OK")

            # ── Step 3: Fresh CSRF for the broker POST ─────────────────────
            log.info("Step 3 → GET fresh CSRF token for broker auth")
            csrf2 = _get_csrf(sess)
            log.info(f"  token={csrf2[:20]}... OK")

            # ── Step 4: POST /kotak/callback with mobile + mpin + totp ────
            # Source: openalgo/blueprints/brlogin.py  /<broker>/callback POST
            # Flask reads: request.form.get("mobile"), "totp", "mpin"
            # Then calls auth_api.authenticate_broker(mobile, totp, mpin) server-side
            totp_code = pyotp.TOTP(TOTP_SECRET).now()
            log.info(f"Step 4 → POST /kotak/callback  mobile={MOBILE}  mpin={MPIN}  totp={totp_code}")
            r = sess.post(
                f"{OPENALGO_URL}/kotak/callback",
                headers={"X-CSRFToken": csrf2},
                data={
                    "mobile": MOBILE,
                    "totp":   totp_code,
                    "mpin":   MPIN,
                },
                timeout=_TIMEOUT,
            )
            log.info(f"  HTTP {r.status_code} | CT={r.headers.get('Content-Type','')[:40]}")
            try:
                resp = r.json()
                log.info(f"  Response: {resp}")
                if isinstance(resp, dict) and resp.get("status") == "error":
                    log.warning(f"  Broker auth error: {resp.get('message','')}")
            except Exception:
                # Might return a redirect/HTML on success — that's OK
                if r.status_code in (200, 302):
                    log.info(f"  Non-JSON response ({len(r.text)}B) — checking funds endpoint...")
                else:
                    log.warning(f"  Unexpected response: {r.text[:200]}")

            # ── Step 5: Verify ────────────────────────────────────────────
            log.info("Step 5 → Verifying /api/v1/funds ...")
            time.sleep(3)   # give OpenAlgo time to store the broker token
            if check_connected():
                log.info("SUCCESS — OpenAlgo API returning JSON. Kotak broker is live.")
                _telegram(
                    "<b>OpenAlgo auto-login SUCCESS</b>\n"
                    f"Kotak broker connected at {OPENALGO_URL}\n"
                    "Real OI/PCR data available from next bot cycle."
                )
                return True
            else:
                log.warning("  /api/v1/funds still returns HTML after login attempt")

        except requests.exceptions.ConnectionError:
            log.warning("  Connection error — OpenAlgo may not be running yet")
        except requests.exceptions.Timeout:
            log.warning(f"  Request timed out after {_TIMEOUT}s")
        except Exception as exc:
            log.error(f"  Unexpected error: {exc}")

        if attempt < retries:
            log.info(f"  Waiting 15s before retry {attempt + 1}...")
            time.sleep(15)

    log.error("FAILED — All auto-login attempts exhausted")
    _telegram(
        "<b>OpenAlgo auto-login FAILED</b>\n"
        f"Manual login required: {OPENALGO_URL}\n"
        "OI/PCR data will use fallback (pcr=1.0) until fixed."
    )
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAlgo auto-login for Kotak broker")
    parser.add_argument(
        "--check", action="store_true",
        help="Only check connection status, do not attempt login"
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Number of login attempts (default: 2)"
    )
    args = parser.parse_args()

    if args.check:
        connected = check_connected()
        log.info(f"OpenAlgo status: {'CONNECTED' if connected else 'NOT CONNECTED'}")
        sys.exit(0 if connected else 1)

    success = auto_login(retries=args.retries)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
