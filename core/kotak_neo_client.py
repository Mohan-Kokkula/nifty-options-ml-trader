"""
Kotak Neo API client — direct replacement for OpenAlgoClient.

Same public interface as OpenAlgoClient so all callers (trader.py,
strike_selector.py, market_intel.py) work without changes beyond the import.

Login flow (automated):
  - TOTP mode: uses pyotp to generate TOTP from KOTAK_TOTP_SECRET,
    then validates with KOTAK_MPIN.
  - Access-token mode: pass KOTAK_ACCESS_TOKEN to skip login entirely
    (useful for scheduled/server runs where token is pre-generated).

Symbol resolution for place_option_order:
  OpenAlgo had an /optionsorder endpoint that resolved ATM/OTM offsets
  automatically. Kotak Neo requires an explicit trading symbol, so we:
    1. Fetch Nifty spot via quotes() or NSE fallback
    2. Calculate the strike from the offset (e.g. OTM1 = ATM+50 for CE)
    3. Resolve the exact pTrdSymbol via search_scrip()
    4. Place the order with that resolved symbol

Option chain / expiry data:
  Delegated to the existing NSEFetcher (already proven reliable) rather
  than making many search_scrip calls.  If NSE is unavailable the
  caller's own fallback logic handles it.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional

import requests

from security import WithdrawalGuard, RateLimiter
import core.sdk_timeout_patch as _sdk_timeout_patch
from core.sdk_timeout_patch import patch_sdk_timeouts


class BrokerCallTimeout(Exception):
    """Raised when a broker order call (place/modify/cancel/close) times out.

    Deliberately a distinct type from a confirmed broker rejection: on a
    timeout the outcome at the broker is UNKNOWN (the order may have been
    received and processed even though the response never arrived). Callers
    MUST treat this differently from a normal failure -- verify actual
    broker state via a separate query (position book / order book) before
    assuming success or failure, and must never blindly resubmit the same
    order on this exception alone (duplicate-order risk)."""
    pass

logger = logging.getLogger(__name__)

# C4: the vendored SDK has no native timeout anywhere — apply once here,
# before any thread can call into it.
patch_sdk_timeouts()

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = [0.3, 0.8, 1.5]

# Circuit breaker: after this many consecutive call failures, is_healthy()
# reports unhealthy so callers can pause opening NEW positions (mirrors the
# MAX_CONSECUTIVE_FAILURES pattern already used in core/tv_fetcher.py).
MAX_CONSECUTIVE_FAILURES = 5

# Session health check interval (seconds) — check every 30 min during market hours
_HEALTH_CHECK_INTERVAL = 30 * 60

NIFTY_STRIKE_STEP = 50

# ATM=0 steps, OTM moves away from ATM, ITM moves into the money
OFFSET_STEPS: dict[str, int] = {
    "ATM": 0,
    "OTM1": 1, "OTM2": 2, "OTM3": 3, "OTM4": 4,
    "ITM1": -1, "ITM2": -2, "ITM3": -3,
}


def _round_to_strike(price: float, step: int = NIFTY_STRIKE_STEP) -> int:
    return int(round(price / step) * step)


def _filter_underlying(results: list, underlying: str) -> list:
    """
    Filter Kotak search_scrip results down to the desired underlying.

    Kotak's search_scrip(symbol='nifty') returns BOTH NIFTY and FINNIFTY
    contracts. Worse, the pSymbolName field can contain values like:
      'NIFTY', 'NIFTY 50', 'Nifty', 'NIFTY-I', 'NIFTYBEES', etc.

    Strategy:
      1. If underlying='NIFTY':
         - Accept any pSymbolName starting with 'NIFTY'
         - Reject 'FINNIFTY' and 'NIFTYBEES' explicitly
      2. Other underlyings: prefer exact match, else 'startswith' match.
    """
    want = (underlying or "").upper().strip()
    if not want:
        return results or []

    out = []
    for r in (results or []):
        sym = str(r.get("pSymbolName", "")).upper().strip()
        if not sym:
            continue
        if want == "NIFTY":
            if "FINNIFTY" in sym or "NIFTYBEES" in sym or "MIDCPNIFTY" in sym:
                continue
            if sym.startswith("NIFTY"):
                out.append(r)
        else:
            if sym == want or sym.startswith(want):
                out.append(r)
    return out


def _tx_code(action: str) -> str:
    """Map 'BUY'/'SELL' (case-insensitive) → Kotak Neo's 'B'/'S'.
    Kotak SDK rejects 'BUY'/'SELL' with ApiValueError; it accepts 'B', 'Buy', 'S', 'Sell'.
    """
    a = str(action or "").strip().upper()
    if a in ("B", "BUY"):  return "B"
    if a in ("S", "SELL"): return "S"
    return a  # pass-through so SDK validation surfaces the real input


class KotakNeoClient:
    """
    Secure wrapper around Kotak Neo API.
    Drop-in replacement for OpenAlgoClient — identical public interface.

    Every order request is guarded by WithdrawalGuard and RateLimiter.
    Auto-retries transient failures; does NOT retry order endpoints on
    timeout (duplicate-order risk).
    """

    def __init__(
        self,
        consumer_key: str,
        rate_limiter: RateLimiter,
        environment: str = "prod",
        neo_fin_key: Optional[str] = None,
        mobile: str = "",
        password: str = "",
        totp_secret: str = "",
        ucc: str = "",
        mpin: str = "",
    ):
        """
        Auto-login flow (Neo SDK v2 TOTP flow):
          1. Construct NeoAPI(consumer_key, environment)
          2. totp_login(mobile_number, ucc, totp=<6-digit>) → view token
          3. totp_validate(mpin=<6-digit>)                  → fully authenticated

        The TOTP secret is the base32 string Kotak issues when you set up
        Authenticator-app 2FA. We compute the 6-digit code on the fly so no
        manual phone interaction is needed.

        Note: `password` kwarg is accepted for backwards compatibility and is
        used as `mpin` if `mpin` itself is not provided (most users had set
        KOTAK_PASSWORD before we discovered Kotak's TOTP flow needs MPIN).
        """
        try:
            from neo_api_client import NeoAPI
        except ImportError as e:
            raise ImportError(
                "neo-api-client not installed. Run: pip install neo-api-client"
            ) from e

        self.rate_limiter   = rate_limiter
        self._consumer_key  = consumer_key
        self._environment   = environment
        self._neo_fin_key   = neo_fin_key
        self._mobile        = mobile
        self._password      = password
        self._totp_secret   = totp_secret
        self._ucc           = ucc
        # MPIN: prefer explicit mpin, else reuse password field (back-compat)
        self._mpin          = mpin or password

        self._consecutive_failures: int = 0
        self._unhealthy_alerted: bool = False   # edge-trigger for is_healthy() alert
        self._session_valid: bool = False
        self._session_expired_at: Optional[datetime] = None
        self._notifier = None                     # set via set_notifier()
        self._post_login_callbacks: list = []     # fired after every successful login
        # 2026-04-29 FIX: must be RLock — _do_login() and ensure_logged_in()
        # both acquire this. Plain Lock() deadlocks the same thread on nested
        # acquisition (the daily login scheduler hung at 09:05 because of this).
        self._login_lock = threading.RLock()      # serialize relogins (reentrant!)

        # Build SDK client — login is deferred until ensure_logged_in() is
        # called by the pilot just before market open. This avoids burning
        # a session token overnight (Kotak tokens expire after ~24h, and
        # logging in at 09:10 gives us a fresh one for the whole session).
        # 2026-04-28 FIX: neo_fin_key MUST be 'neotradeapi' per Kotak v2 spec.
        # If env var is empty string (not None), force the default.
        effective_nfk = (neo_fin_key or "").strip() or "neotradeapi"
        self._neo_fin_key = effective_nfk
        self._neo = NeoAPI(
            consumer_key=consumer_key,
            environment=environment,
            neo_fin_key=effective_nfk,
        )

        # 2026-04-29: REVERTED to eager login by default.
        # Lazy login + scheduler thread caused deadlocks and hard-to-debug
        # hangs. Eager login is the simple, proven-working approach.
        # Set KOTAK_LAZY_LOGIN=true to defer to first market-hour use.
        import os
        lazy = os.getenv("KOTAK_LAZY_LOGIN", "false").lower() == "true"
        if not lazy:
            logger.info("Kotak: eager login (default since 2026-04-29 RLock fix)")
            self._do_login()
            self._start_health_check()
        elif self._is_market_session_window():
            logger.info("Kotak: lazy mode but market open → logging in now")
            self._do_login()
            self._start_health_check()
        else:
            logger.info(
                "Kotak: lazy mode + market closed → deferring login until 09:10 IST"
            )

    @staticmethod
    def _is_market_session_window() -> bool:
        """Are we within (or close to) market hours? 09:10-15:30 IST Mon-Fri."""
        now = datetime.now()
        if now.weekday() >= 5:           # Sat/Sun
            return False
        h, m = now.hour, now.minute
        # 09:10 .. 15:30 covers pre-open warmup + active trading
        if (h == 9 and m >= 10) or (10 <= h <= 14) or (h == 15 and m <= 30):
            return True
        return False

    def _start_health_check(self):
        """Start background session monitor (idempotent)."""
        if getattr(self, "_health_thread", None) is not None:
            return
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="KotakSessionMonitor",
        )
        self._health_thread.start()

    def ensure_logged_in(self) -> bool:
        """
        Public entry point — pilot calls this at 09:10 IST and before any
        live API call. Idempotent: no-op if session already valid.
        """
        if self._session_valid:
            return True
        with self._login_lock:
            if self._session_valid:   # double-check after lock
                return True
            try:
                self._do_login()
                self._start_health_check()
                return True
            except Exception as e:
                logger.error(f"Kotak ensure_logged_in failed: {e}")
                return False

    # ------------------------------------------------------------------
    # Login / TOTP
    # ------------------------------------------------------------------

    def _generate_totp(self) -> str:
        """Generate the current 6-digit TOTP from the configured secret."""
        try:
            import pyotp
        except ImportError as e:
            raise ImportError(
                "pyotp not installed. Run: pip install pyotp==2.9.0"
            ) from e
        if not self._totp_secret:
            raise ValueError("KOTAK_TOTP_SECRET is empty — cannot generate TOTP")
        # Strip whitespace and remove any spaces a user may have pasted
        secret_clean = self._totp_secret.strip().replace(" ", "")
        return pyotp.TOTP(secret_clean).now()

    @staticmethod
    def _is_error_response(resp) -> tuple[bool, str]:
        """Inspect a Kotak SDK response for an error envelope. Returns (is_error, message)."""
        if isinstance(resp, dict):
            for key in ("Error Message", "errMsg", "error", "Message"):
                msg = resp.get(key)
                if msg:
                    return True, str(msg)
            # Some endpoints flag status:'error'
            if str(resp.get("status", "")).lower() == "error":
                return True, str(resp.get("statusDescription", "")
                                  or resp.get("message", "")
                                  or "Kotak Neo error")
        return False, ""

    def _do_login(self) -> None:
        """
        Perform full login + TOTP 2FA. Sets self._session_valid on success.
        Raises RuntimeError on any auth failure (caller decides whether to
        retry or alert the user).
        """
        with self._login_lock:
            if self._session_valid:
                return
            missing = [n for n, v in (
                ("KOTAK_MOBILE",       self._mobile),
                ("KOTAK_UCC",          self._ucc),
                ("KOTAK_MPIN",         self._mpin),
                ("KOTAK_TOTP_SECRET",  self._totp_secret),
            ) if not v]
            if missing:
                raise RuntimeError(
                    "Kotak Neo auto-login requires: " + ", ".join(missing) +
                    ". UCC is the Unique Client Code shown under Profile in "
                    "the Kotak Neo app. MPIN is the 6-digit PIN you set for "
                    "app login. TOTP_SECRET is the base32 string from the "
                    "Authenticator-app QR setup screen."
                )

            # Step 1 — totp_login: mobile + ucc + generated TOTP → view token
            try:
                totp_code = self._generate_totp()
                logger.debug(f"Kotak totp_login: submitting TOTP (len={len(totp_code)})")
                login_resp = self._neo.totp_login(
                    mobile_number=self._mobile,
                    ucc=self._ucc,
                    totp=totp_code,
                )
            except Exception as e:
                raise RuntimeError(f"Kotak totp_login() raised: {e}") from e

            is_err, msg = self._is_error_response(login_resp)
            if is_err:
                raise RuntimeError(
                    f"Kotak totp_login rejected: {msg}. "
                    f"Check KOTAK_MOBILE, KOTAK_UCC, KOTAK_TOTP_SECRET (base32, "
                    f"no spaces), and that system clock is NTP-synced."
                )

            # Step 2 — totp_validate with MPIN → full session
            try:
                tfa_resp = self._neo.totp_validate(mpin=self._mpin)
            except Exception as e:
                raise RuntimeError(f"Kotak totp_validate() raised: {e}") from e

            is_err, msg = self._is_error_response(tfa_resp)
            if is_err:
                raise RuntimeError(
                    f"Kotak totp_validate rejected: {msg}. Check KOTAK_MPIN "
                    f"(6-digit PIN you use to unlock the Kotak Neo app)."
                )

            self._session_valid       = True
            self._session_expired_at  = None
            self._consecutive_failures = 0
            self._unhealthy_alerted   = False
            logger.info(
                f"✅ Kotak Neo: logged in ({self._environment}) "
                f"as {self._mobile[-4:]}****  — session active"
            )

            # Fire post-login callbacks (e.g. TickFeed.retry_futures_subscription).
            # Run each in a daemon thread so we don't block the login flow.
            for _cb in self._post_login_callbacks:
                try:
                    threading.Thread(
                        target=_cb,
                        daemon=True,
                        name="PostLoginCallback",
                    ).start()
                except Exception as _e:
                    logger.debug(f"Post-login callback launch failed: {_e}")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def set_notifier(self, notifier) -> None:
        """Wire in NotificationManager so session alerts go to WhatsApp/email."""
        self._notifier = notifier

    def register_post_login_callback(self, callback) -> None:
        """
        Register a callable to fire immediately after every successful login.

        The callback is invoked in a short-lived daemon thread so it does not
        block the login flow.  Use this to wire components that need an
        authenticated session before they can initialise (e.g. TickFeed needs
        a valid session to resolve the futures instrument token via search_scrip).

        Safe to call multiple times — callbacks accumulate and are all fired.
        """
        self._post_login_callbacks.append(callback)
        logger.debug(
            f"KotakNeoClient: post-login callback registered "
            f"({callback.__qualname__ if hasattr(callback, '__qualname__') else callback})"
        )

    @property
    def is_session_valid(self) -> bool:
        return self._session_valid

    def _relogin(self) -> None:
        """
        Called when session expiry is detected in an API response.
        Attempts automatic re-login using stored credentials + TOTP.
        Only alerts (and pauses trading) if re-login ITSELF fails.
        """
        self._session_valid = False
        self._session_expired_at = datetime.now()
        logger.warning("Kotak Neo: session expired — attempting auto-relogin...")

        try:
            # Rebuild the SDK client to drop any stale internal state, then
            # run the full login + TOTP flow again.
            try:
                from neo_api_client import NeoAPI
            except ImportError:
                raise
            self._neo = NeoAPI(
                consumer_key=self._consumer_key,
                environment=self._environment,
                neo_fin_key=self._neo_fin_key,
            )
            self._do_login()
            logger.info("✅ Kotak Neo: auto-relogin successful — trading resumes")
            return
        except Exception as e:
            logger.critical(f"Kotak Neo auto-relogin FAILED: {e}")

        # Re-login failed — notify and keep trading paused
        msg = (
            "🔴 KOTAK NEO AUTO-RELOGIN FAILED\n"
            "Trading is PAUSED — no new orders will be placed.\n\n"
            "Likely causes:\n"
            "  • KOTAK_PASSWORD changed\n"
            "  • KOTAK_TOTP_SECRET wrong or expired\n"
            "  • Account locked after too many attempts\n"
            "  • System clock out of sync (TOTP is time-sensitive)\n\n"
            "Check logs and restart after fixing credentials."
        )
        logger.critical(msg)
        if self._notifier:
            try:
                self._notifier.notify(
                    subject="Kotak Neo Relogin Failed — Action Required",
                    message=msg,
                )
            except Exception as e:
                logger.debug(f"Relogin-failure notification failed: {e}")

    def _check_health(self) -> bool:
        """
        Proactive session health check using limits() API.
        Returns True if session is alive, False if expired.

        2026-05-07: Hardened detection — checks status code AND multiple
        error fields, not just message text. Kotak's exact error format
        is {'stCode': 100008, 'errMsg': 'unauthorized', 'stat': 'Not_Ok'}.
        """
        try:
            resp = self._neo.limits(segment="FO")
            if isinstance(resp, dict):
                # Status-code check (most reliable)
                st_code = resp.get("stCode") or resp.get("statusCode")
                if st_code in (401, 403, 100008, "100008"):
                    logger.warning(f"Health check: stCode={st_code} → session expired")
                    return False
                # Status flag check
                if str(resp.get("stat", "")).lower() in ("not_ok", "fail", "error"):
                    err_msg = str(resp.get("errMsg", "") or resp.get("Error Message", ""))
                    if err_msg:
                        logger.warning(f"Health check: stat=Not_Ok msg={err_msg[:80]}")
                        return False
                # Fallback: keyword search
                msg = " ".join(str(resp.get(k, "") or "") for k in (
                    "Error Message", "errMsg", "error", "message"
                )).lower()
                if any(k in msg for k in ("2fa", "session", "login", "token", "unauthori", "expir")):
                    return False
                # Positive signal: real account data present
                if any(k in resp for k in ("Category", "EntityId", "BoardLotLimit", "data")):
                    return True
            return True
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("unauthori", "token", "session", "login", "401", "403", "100008")):
                return False
            # Network error — don't mark session invalid, just warn
            logger.debug(f"Health check network error (session may still be ok): {e}")
            return True

    def is_healthy(self) -> bool:
        """
        Cheap, local circuit-breaker check (no API call) — False once
        _call_with_retry has recorded MAX_CONSECUTIVE_FAILURES in a row.
        Callers should use this to pause opening NEW positions during a
        sustained broker outage; it resets automatically the moment a call
        succeeds (see _call_with_retry). Existing open positions should
        keep being monitored/closed regardless of this flag.
        """
        return self._consecutive_failures < MAX_CONSECUTIVE_FAILURES

    def _health_check_loop(self) -> None:
        """
        Background thread — checks session health every 30 minutes.
        Only runs during market hours (9:00–15:35 IST) to avoid
        false alarms from NSE maintenance windows.
        """
        time.sleep(60)   # wait 1 min after startup before first check
        while True:
            try:
                now = datetime.now()
                in_market_hours = (
                    (now.hour == 9 and now.minute >= 0) or
                    (10 <= now.hour <= 14) or
                    (now.hour == 15 and now.minute <= 35)
                )

                if in_market_hours and self._session_valid:
                    alive = self._check_health()
                    if not alive:
                        logger.error("KotakSessionMonitor: health check FAILED — session expired")
                        self._relogin()
                    else:
                        logger.debug("KotakSessionMonitor: session OK")

                # 2026-05-07 BUG FIX: midnight reset was BACKWARDS.
                # Old code set session_valid=True at midnight even though no
                # login happened, fooling the bot into thinking it was logged
                # in with a stale (expired) token. Correct behavior: FORCE
                # relogin flag (session_valid=False) so next API call triggers
                # ensure_logged_in() and gets a fresh token.
                if now.hour == 0 and now.minute < 2:
                    if self._session_valid:
                        logger.info(
                            "KotakSessionMonitor: midnight — invalidating session "
                            "(token will refresh on next call / 09:10 scheduler)"
                        )
                        self._session_valid = False

            except Exception as e:
                logger.debug(f"KotakSessionMonitor loop error: {e}")

            time.sleep(_HEALTH_CHECK_INTERVAL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expiry_oa_to_neo(self, expiry_oa: str) -> str:
        """Convert any common expiry format ('24MAR26', '24-MAR-26',
        '24MAR2026', '24-MAR-2026') to Kotak Neo 'DDMMMYYYY' (e.g.
        '24MAR2026'). Accepts the same format set as
        core.expiry_utils._parse_expiry_str() for consistency."""
        s = expiry_oa.strip().upper()
        for fmt in ("%d%b%y", "%d-%b-%y", "%d%b%Y", "%d-%b-%Y"):
            try:
                return datetime.strptime(s, fmt).strftime("%d%b%Y").upper()
            except ValueError:
                continue
        # Unrecognized format — return as-is (unchanged pre-existing behavior)
        return s

    def _offset_to_strike(self, spot: float, offset: str, option_type: str) -> int:
        """
        Calculate the exact strike for a given ATM/OTM/ITM offset.
        CE: OTM moves UP from ATM; PE: OTM moves DOWN from ATM.
        """
        atm = _round_to_strike(spot)
        steps = OFFSET_STEPS.get(offset.upper(), 0)
        direction = 1 if option_type.upper() == "CE" else -1
        return atm + direction * steps * NIFTY_STRIKE_STEP

    def _resolve_nifty_token(self) -> tuple[str, str]:
        """
        Resolve the Nifty 50 index instrument identifier for Kotak Neo quotes().

        In Kotak Neo API v2, the `instrument_token` for indices is the literal
        neo-symbol string (e.g. "Nifty 50"), NOT the numeric NSE token 26000.
        We short-circuit to the known-good pair ("Nifty 50", "nse_cm") and
        fall back to search_scrip() only if that somehow fails a live probe.
        """
        # Fast-path: verify the canonical Neo symbol works by hitting quotes().
        try:
            probe = self._neo.quotes(
                instrument_tokens=[
                    {"instrument_token": "Nifty 50", "exchange_segment": "nse_cm"}
                ],
                quote_type="ltp",
            )
            if isinstance(probe, list) and probe and float(probe[0].get("ltp") or 0) > 0:
                logger.info("Nifty 50 resolved: neo-symbol='Nifty 50' on nse_cm")
                return "Nifty 50", "nse_cm"
        except Exception as e:
            logger.debug(f"Nifty probe quotes() failed: {e}")

        # Fallback: search_scrip discovery
        search_terms = ["nifty 50", "nifty50", "nifty"]
        segments = ["nse_cm", "nse_idx"]
        twofa_seen = False
        for seg in segments:
            for term in search_terms:
                try:
                    results = self._neo.search_scrip(
                        exchange_segment=seg, symbol=term
                    )
                    # Detect the Kotak 2FA-required error envelope explicitly
                    # (it returns a dict, not a list, with "Error Message").
                    if isinstance(results, dict):
                        err = str(results.get("Error Message", "")
                                  or results.get("error", "")).lower()
                        if "2fa" in err or "two factor" in err or "complete the 2fa" in err:
                            twofa_seen = True
                            logger.debug(f"search_scrip {seg}/{term} → 2FA error: {results}")
                            continue
                        logger.debug(f"search_scrip {seg}/{term} returned dict (unexpected): {results}")
                        continue
                    if isinstance(results, list):
                        for r in results:
                            name = str(r.get("pTrdSymbol", "") or r.get("pSymbolName", "")).upper()
                            token = str(r.get("pSymbol", "") or r.get("instrument_token", ""))
                            if token and ("NIFTY 50" in name or name == "NIFTY"):
                                logger.info(
                                    f"Nifty 50 token resolved: {token} "
                                    f"({name}) on {seg}"
                                )
                                return token, seg
                except Exception as e:
                    logger.debug(f"search_scrip {seg}/{term} failed: {e}")
        # If every call came back with the 2FA error, the user's token is
        # NOT 2FA-authenticated. Surface this loudly — silent fallback to
        # token 26000 + NSE makes the real cause invisible.
        if twofa_seen:
            self._session_valid = False
            msg = (
                "🔴 KOTAK NEO TOKEN IS NOT 2FA-AUTHENTICATED\n"
                "All API calls are returning: "
                "'Complete the 2fa process before accessing this application'.\n\n"
                "To fix:\n"
                "  1. Open Kotak Neo App on your phone\n"
                "  2. Invest tab → Trade API → Generate / Refresh session\n"
                "  3. Complete OTP from SMS\n"
                "  4. Copy the new token\n"
                "  5. Update KOTAK_CONSUMER_KEY in config/settings.env\n"
                "  6. Restart the bot (Ctrl-C → python main.py)"
            )
            logger.error(msg)
            if self._notifier:
                try:
                    self._notifier.send(msg)
                except Exception:
                    pass
            raise RuntimeError("Kotak Neo token requires 2FA — see logs for steps")
        # Last-resort: return the known-good neo-symbol anyway and let caller
        # surface any downstream error. We deliberately do NOT fall back to
        # numeric "26000" — Kotak's quotes() endpoint rejects that as an
        # "Invalid neosymbol value" for index queries in v2.
        logger.warning("search_scrip could not confirm Nifty 50 — returning 'Nifty 50'/nse_cm anyway")
        return "Nifty 50", "nse_cm"

    def _get_spot_price(self) -> float:
        """Fetch Nifty 50 spot price directly from Kotak Neo.
        Token is resolved dynamically via search_scrip (not hardcoded).
        Falls back to NSE only if Kotak Neo API is unavailable.
        """
        # Use cached token if already resolved.
        # NOTE: default for _nifty_token must be None (NOT a tuple) — the old
        # default `(None, None)` is a non-empty tuple, which is truthy, so
        # `if not token:` evaluated False and the resolver never ran on the
        # first call. The API was then called with token=(None,None), which
        # silently returned ltp=0 and triggered the "spot returned 0" warning
        # at every startup. Bug fix 2026-04-20.
        token        = getattr(self, "_nifty_token",        None)
        exchange_seg = getattr(self, "_nifty_exchange_seg", None)
        if not token or not exchange_seg:
            token, exchange_seg = self._resolve_nifty_token()
            self._nifty_token        = token
            self._nifty_exchange_seg = exchange_seg

        # Primary — Kotak Neo quotes with dynamically resolved token
        try:
            resp = self._neo.quotes(
                instrument_tokens=[
                    {"instrument_token": token, "exchange_segment": exchange_seg}
                ],
                quote_type="ltp",
            )
            # Response shapes seen in the wild:
            #   [{"ltp": "24393.9500", "exchange_token": "Nifty 50", ...}]
            #   {"data": [ ... ]}
            #   {"fault": {...}}  ← error, skip
            if isinstance(resp, dict) and "fault" in resp:
                logger.debug(f"Kotak quotes fault: {resp['fault']}")
                data = []
            else:
                data = resp.get("data", resp) if isinstance(resp, dict) else (resp or [])
            if isinstance(data, list) and data:
                raw_ltp = (
                    data[0].get("ltp")
                    or data[0].get("last_traded_price")
                    or 0
                )
                ltp = float(raw_ltp) if raw_ltp not in ("", None) else 0.0
                if ltp > 0:
                    logger.debug(f"Nifty spot: {ltp:.2f} (Kotak Neo token={token})")
                    return ltp
        except Exception as e:
            logger.debug(f"Kotak Neo spot failed: {e}")

        # Fallback — NSE (only if Kotak Neo is down)
        try:
            from core.nse_fetcher import NSEFetcher
            spot = float(NSEFetcher(timeout=10).get_option_chain("NIFTY", width=1).get("spot", 0))
            if spot > 0:
                logger.debug(f"Nifty spot: {spot:.2f} (NSE fallback)")
                return spot
        except Exception as e:
            logger.debug(f"NSE spot fallback failed: {e}")

        raise ValueError("Cannot fetch Nifty spot from Kotak Neo or NSE")

    def _resolve_option_symbol(
        self,
        underlying: str,
        expiry_neo: str,
        strike: int,
        option_type: str,
    ) -> str:
        """
        Use search_scrip to find the exact Kotak Neo trading symbol.
        Falls back to constructing the symbol in standard NSE format.
        """
        try:
            results = self._neo.search_scrip(
                exchange_segment="nse_fo",
                symbol=underlying.lower(),
                expiry=expiry_neo,
                option_type=option_type.upper(),
                strike_price=str(strike),
            )
            if isinstance(results, list) and results:
                # CRITICAL: search_scrip(symbol="nifty") returns BOTH NIFTY and FINNIFTY.
                # Filter to underlying using the tolerant helper (matches NIFTY,
                # 'NIFTY 50', 'NIFTY-I' etc. but rejects FINNIFTY/NIFTYBEES).
                filtered = _filter_underlying(results, underlying)
                chosen = filtered[0] if filtered else results[0]
                if not filtered:
                    logger.warning(
                        f"search_scrip: no '{underlying.upper()}' match, got "
                        f"{[r.get('pSymbolName') for r in results[:5]]} — using first"
                    )
                trd_sym = chosen.get("pTrdSymbol") or chosen.get("trading_symbol")
                if trd_sym:
                    logger.debug(f"Resolved symbol via search_scrip: {trd_sym}")
                    return trd_sym
        except Exception as e:
            logger.debug(f"search_scrip failed: {e}")

        # Fallback: construct symbol in standard NSE F&O format
        # expiry_neo is DDMMMYYYY e.g. "24MAR2026" → "24MAR26" for symbol
        try:
            dt = datetime.strptime(expiry_neo, "%d%b%Y")
            exp_sym = dt.strftime("%d%b%y").upper()
        except ValueError:
            exp_sym = expiry_neo[:7].upper()
        symbol = f"{underlying.upper()}{exp_sym}{strike}{option_type.upper()}"
        logger.debug(f"Constructed symbol (fallback): {symbol}")
        return symbol

    def _normalize_order_response(self, raw: dict, symbol: str = "", underlying_ltp: float = 0) -> dict:
        """Normalize Kotak Neo order response to OpenAlgo-compatible format."""
        if not isinstance(raw, dict):
            return {"status": "failed", "orderid": "", "symbol": symbol, "underlying_ltp": underlying_ltp}

        # Kotak Neo success: {"stat": "Ok", "nOrdNo": "..."}
        if raw.get("stat") == "Ok" or raw.get("nOrdNo"):
            return {
                "status": "success",
                "orderid": raw.get("nOrdNo", raw.get("order_id", "")),
                "symbol": symbol,
                "underlying_ltp": underlying_ltp,
                "raw": raw,
            }

        # Error response
        err_msg = raw.get("errMsg") or raw.get("Error") or str(raw)
        logger.error(f"Kotak Neo order failed: {err_msg}")
        return {
            "status": "failed",
            "orderid": "",
            "symbol": symbol,
            "underlying_ltp": underlying_ltp,
            "error": err_msg,
            "raw": raw,
        }

    def _rate_gate(self, endpoint: str, payload: dict) -> None:
        """Apply WithdrawalGuard + RateLimiter before any request."""
        if not self._session_valid:
            raise PermissionError(
                "BLOCKED: Kotak Neo session expired — update token and restart bot. "
                "See logs for instructions."
            )
        if not WithdrawalGuard.check(endpoint, payload):
            raise PermissionError(
                f"BLOCKED: Withdrawal/transfer attempt on endpoint {endpoint}"
            )
        self.rate_limiter.wait_and_acquire()

    def _check_session_error(self, resp) -> bool:
        """Return True if response indicates session expiry / unauthorized."""
        if not isinstance(resp, dict):
            return False
        # 1. Top-level error text fields
        msg = str(
            resp.get("Error Message", "") or resp.get("errMsg", "") or
            resp.get("message", "") or ""
        ).lower()
        if any(k in msg for k in ("2fa", "session", "login", "unauthori", "token expir")):
            return True
        # 2. Nested fault block — quotes() returns {"fault": {"code": "100008", ...}}
        fault = resp.get("fault") or {}
        if isinstance(fault, dict):
            code = str(fault.get("code", ""))
            if code in ("401", "403", "100008"):
                return True
            fault_msg = str(
                fault.get("description", "") or fault.get("message", "") or ""
            ).lower()
            if any(k in fault_msg for k in ("unauthori", "session", "2fa", "token")):
                return True
        # 3. Top-level status codes
        st_code = str(resp.get("stCode", "") or resp.get("statusCode", ""))
        if st_code in ("401", "403", "100008"):
            return True
        return False

    def _alert_broker_timeout(self, call_name: str, attempt: int, exc: Exception) -> None:
        """Structured log + Telegram alert for a broker call timeout. Called
        from every timeout site so no caller can forget to alert (fix #1 of
        the operational-safety patch: every timeout must generate a
        structured log entry and a Telegram alert)."""
        logger.critical(
            "BROKER_CALL_TIMEOUT",
            extra={
                "event": "broker_call_timeout",
                "call": call_name,
                "attempt": attempt,
                "connect_timeout_sec": getattr(_sdk_timeout_patch, "CONNECT_TIMEOUT_SEC", None),
                "read_timeout_sec": getattr(_sdk_timeout_patch, "READ_TIMEOUT_SEC", None),
                "error": str(exc),
            },
        )
        logger.critical(
            f"🚨 BROKER CALL TIMEOUT: {call_name} (attempt {attempt}) — outcome "
            f"UNKNOWN, did not receive a response within the configured "
            f"timeout. Caller must verify actual broker state before "
            f"assuming success or failure. Error: {exc}"
        )
        if self._notifier:
            try:
                self._notifier.notify(
                    subject="🚨 Broker Call Timeout",
                    message=(
                        f"Broker call TIMED OUT: {call_name}\n"
                        f"Outcome is UNKNOWN — order may or may not have been "
                        f"received by the broker.\n"
                        f"Attempt: {attempt}\nError: {exc}"
                    ),
                )
            except Exception as _e:
                logger.debug(f"Timeout notification failed: {_e}")

    def _call_with_retry(self, fn, *args, is_order: bool = False, **kwargs):
        """Call a NeoAPI method with retry on transient errors."""
        last_err = None
        call_name = getattr(fn, "__name__", str(fn))
        retries = 2 if is_order else MAX_RETRIES
        for attempt in range(retries):
            try:
                resp = fn(*args, **kwargs)
                if self._check_session_error(resp):
                    self._relogin()
                    resp = fn(*args, **kwargs)
                self._consecutive_failures = 0
                self._unhealthy_alerted = False
                return resp
            except Exception as e:
                if is_order and isinstance(e, requests.exceptions.Timeout):
                    # Order calls only (fix #1 scope): deterministic, bounded
                    # failure -- outcome at the broker is UNKNOWN. Never
                    # retried blindly (duplicate-order risk); the caller
                    # receives a distinctly-typed exception so it cannot be
                    # mistaken for a confirmed rejection. Non-order (read)
                    # calls, and non-timeout exceptions on order calls, fall
                    # through unchanged to the existing logic below.
                    self._consecutive_failures += 1
                    self._alert_broker_timeout(call_name, attempt + 1, e)
                    raise BrokerCallTimeout(f"{call_name} timed out: {e}") from e
                self._consecutive_failures += 1
                last_err = e
                if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES and not self._unhealthy_alerted:
                    self._unhealthy_alerted = True
                    logger.critical(
                        f"Kotak Neo: {self._consecutive_failures} consecutive call "
                        f"failures — broker unhealthy, new-position entry will pause "
                        f"until a call succeeds"
                    )
                    if self._notifier:
                        try:
                            self._notifier.notify(
                                subject="Kotak Neo — Broker Unhealthy",
                                message=(
                                    f"🔴 {self._consecutive_failures} consecutive Kotak Neo "
                                    f"call failures.\nNew trade entry is paused until a call "
                                    f"succeeds. Existing positions are still monitored."
                                ),
                            )
                        except Exception as _e:
                            logger.debug(f"Unhealthy-broker notification failed: {_e}")
                if is_order and attempt == 0:
                    # For order endpoints: log but don't retry (duplicate risk)
                    logger.error(f"Kotak Neo order call failed: {e} — not retrying")
                    raise
                if attempt < retries - 1:
                    wait = RETRY_BACKOFF[attempt]
                    logger.warning(f"Kotak Neo call failed (attempt {attempt+1}): {e} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Kotak Neo call failed after {retries} attempts: {e}")
                    raise
        raise RuntimeError(f"Kotak Neo call failed: {last_err}")

    # ------------------------------------------------------------------
    # Order Management
    # ------------------------------------------------------------------

    def place_option_order(
        self,
        underlying: str,
        exchange: str,
        expiry_date: str,
        offset: str,
        option_type: str,
        action: str,
        quantity: int,
        price_type: str = "MARKET",
        product: str = "MIS",
        price: float = 0.0,
        trigger_price: float = 0.0,
        strategy: str = "OpenClawNifty",
    ) -> dict:
        """
        Place an option order by resolving ATM/OTM offset to a real symbol.
        This replicates OpenAlgo's /optionsorder auto-resolution locally.

        Args:
            underlying: "NIFTY"
            exchange: ignored (always uses nse_fo)
            expiry_date: "24MAR26" (OpenAlgo format — auto-converted)
            offset: "ATM", "OTM1", "OTM2", "OTM3", "ITM1", "ITM2"
            option_type: "CE" or "PE"
            action: "BUY" or "SELL"
            quantity: number of units (e.g. 75 for 1 Nifty lot)
        """
        self._rate_gate("/optionsorder", {"action": action})

        # 1. Spot price → strike
        spot = self._get_spot_price()
        strike = self._offset_to_strike(spot, offset, option_type)

        # 2. Resolve exact Kotak Neo trading symbol
        expiry_neo = self._expiry_oa_to_neo(expiry_date)
        trading_symbol = self._resolve_option_symbol(underlying, expiry_neo, strike, option_type)

        logger.info(
            f"Option order: {action} {trading_symbol} x{quantity} "
            f"@ {price_type} (offset={offset}, spot={spot:.0f}, strike={strike})"
        )

        # 3. Place order
        # ENTRY orders use IOC (Immediate Or Cancel) for LIMIT type:
        # If not filled instantly at our price, the order cancels itself.
        # This prevents orphaned hanging orders when market gaps past our limit,
        # keeping the position monitor clean and avoiding phantom fills.
        order_type = "MKT" if price_type.upper() in ("MARKET", "MKT") else "L"
        validity   = "DAY" if order_type == "MKT" else "IOC"

        order_req = {
            "exchange_segment": "nse_fo",
            "product": product,
            "price": str(price) if price else "0",
            "order_type": order_type,
            "quantity": str(quantity),
            "validity": validity,
            "trading_symbol": trading_symbol,
            "transaction_type": _tx_code(action),
            "trigger_price": str(trigger_price) if trigger_price else "0",
        }
        logger.info(f"📤 KotakNeo place_order REQUEST: {order_req}")

        raw = self._call_with_retry(self._neo.place_order, is_order=True, **order_req)

        logger.info(f"📥 KotakNeo place_order RESPONSE: {raw}")

        result = self._normalize_order_response(raw, symbol=trading_symbol, underlying_ltp=spot)

        # IOC post-check: verify the order actually filled (not cancelled by exchange)
        if validity == "IOC" and result.get("status") == "success":
            order_id = result.get("orderid", "")
            if order_id:
                try:
                    time.sleep(0.5)    # brief settle — IOC resolves in milliseconds
                    report = self._call_with_retry(self._neo.order_report)
                    orders = report if isinstance(report, list) else \
                             (report.get("data", []) if isinstance(report, dict) else [])
                    for o in orders:
                        if str(o.get("nOrdNo", "")) == str(order_id):
                            status = str(o.get("ordSt", "")).upper()
                            if status in ("CANCELLED", "REJECTED", "CANCEL"):
                                logger.warning(
                                    f"IOC entry CANCELLED by exchange "
                                    f"(market moved) — order_id={order_id}"
                                )
                                return {
                                    "status": "ABORTED",
                                    "reason": "IOC_NOT_FILLED",
                                    "orderid": order_id,
                                }
                            break
                except Exception as _e:
                    logger.debug(f"IOC post-check failed (non-fatal): {_e}")

        return result

    def place_order(
        self,
        symbol: str,
        exchange: str,
        action: str,
        quantity: int,
        price_type: str = "MARKET",
        product: str = "MIS",
        price: float = 0.0,
        trigger_price: float = 0.0,
        disclosed_quantity: int = 0,
        strategy: str = "OpenClawNifty",
    ) -> dict:
        """Place a regular order directly by symbol."""
        self._rate_gate("/placeorder", {"action": action, "symbol": symbol})

        exchange_segment = "nse_fo" if exchange.upper() in ("NFO", "NSE_FO", "NSE_INDEX") else "nse_cm"
        order_type = "MKT" if price_type.upper() in ("MARKET", "MKT") else "L"
        # LIMIT entry orders → IOC to prevent orphaned hanging orders.
        # EXIT / close orders always use DAY (we must exit no matter what).
        is_exit = action.upper() == "SELL"
        validity = "DAY" if (order_type == "MKT" or is_exit) else "IOC"

        logger.info(f"Placing order: {action} {quantity}x {symbol} @ {price_type} ({validity})")
        order_req = {
            "exchange_segment": exchange_segment,
            "product": product,
            "price": str(price) if price else "0",
            "order_type": order_type,
            "quantity": str(quantity),
            "validity": validity,
            "trading_symbol": symbol,
            "transaction_type": _tx_code(action),
            "trigger_price": str(trigger_price) if trigger_price else "0",
            "disclosed_quantity": str(disclosed_quantity),
        }
        logger.info(f"📤 KotakNeo place_order REQUEST: {order_req}")

        raw = self._call_with_retry(self._neo.place_order, is_order=True, **order_req)

        logger.info(f"📥 KotakNeo place_order RESPONSE: {raw}")
        return self._normalize_order_response(raw, symbol=symbol)

    def place_smart_order(
        self,
        symbol: str,
        exchange: str,
        action: str,
        quantity: int,
        position_size: int,
        price_type: str = "MARKET",
        product: str = "MIS",
        price: float = 0.0,
        trigger_price: float = 0.0,
        strategy: str = "OpenClawNifty",
    ) -> dict:
        """Smart order — delegates to place_order (Kotak Neo has no direct equivalent)."""
        return self.place_order(
            symbol=symbol,
            exchange=exchange,
            action=action,
            quantity=quantity,
            price_type=price_type,
            product=product,
            price=price,
            trigger_price=trigger_price,
            strategy=strategy,
        )

    def modify_order(
        self,
        order_id: str,
        symbol: str,
        exchange: str,
        action: str,
        quantity: int,
        price_type: str = "LIMIT",
        product: str = "MIS",
        price: float = 0.0,
        trigger_price: float = 0.0,
        strategy: str = "OpenClawNifty",
    ) -> dict:
        """Modify an existing order."""
        self._rate_gate("/modifyorder", {"orderid": order_id})
        logger.info(f"Modifying order {order_id}")
        order_type = "L" if price_type.upper() in ("LIMIT", "L") else "MKT"
        raw = self._call_with_retry(
            self._neo.modify_order,
            order_id=order_id,
            price=str(price),
            order_type=order_type,
            quantity=str(quantity),
            validity="DAY",
            trigger_price=str(trigger_price) if trigger_price else "0",
            is_order=True,
        )
        return raw if isinstance(raw, dict) else {"status": "success", "raw": raw}

    def cancel_order(self, order_id: str, strategy: str = "OpenClawNifty") -> dict:
        """Cancel a specific order."""
        self._rate_gate("/cancelorder", {"orderid": order_id})
        logger.info(f"Cancelling order {order_id}")
        raw = self._call_with_retry(
            self._neo.cancel_order, order_id=order_id, is_order=True
        )
        return raw if isinstance(raw, dict) else {"status": "success"}

    def close_position(
        self,
        symbol: str,
        exchange: str,
        product: str = "MIS",
        strategy: str = "OpenClawNifty",
    ) -> dict:
        """
        Square off an open position by placing an opposing order.
        Reads the net quantity from the position book.
        """
        self._rate_gate("/closeposition", {"symbol": symbol})
        logger.info(f"Closing position: {symbol}")

        # Determine net qty from positions
        try:
            pos_resp = self._neo.positions()
            positions = []
            if isinstance(pos_resp, dict):
                positions = pos_resp.get("data", []) or pos_resp.get("Data", [])
            elif isinstance(pos_resp, list):
                positions = pos_resp

            net_qty = 0
            matched_symbol = None
            for pos in positions:
                trd = pos.get("trdSym", pos.get("trading_symbol", ""))
                # Match on symbol name (strip -FO suffix if present)
                base_trd = trd.replace("-FO", "").replace("-fo", "")
                base_sym = symbol.replace("-FO", "").replace("-fo", "")
                if base_trd == base_sym or trd == symbol:
                    buy_qty = int(pos.get("flBuyQty", pos.get("buyQty", 0)) or 0)
                    sell_qty = int(pos.get("flSellQty", pos.get("sellQty", 0)) or 0)
                    net_qty = buy_qty - sell_qty
                    matched_symbol = trd
                    break

            if net_qty == 0:
                logger.warning(f"close_position: no open qty found for {symbol}")
                return {"status": "success", "message": "No open position to close"}

            action = "SELL" if net_qty > 0 else "BUY"
            qty = abs(net_qty)
            sym = matched_symbol or symbol

        except Exception as e:
            logger.warning(f"close_position: could not read positions ({e}) — aborting close")
            return {"status": "error", "message": str(e)}

        exchange_segment = "nse_fo" if exchange.upper() in ("NFO", "NSE_FO", "NSE_INDEX") else "nse_cm"
        order_req = {
            "exchange_segment": exchange_segment,
            "product": product,
            "price": "0",
            "order_type": "MKT",
            "quantity": str(qty),
            "validity": "DAY",
            "trading_symbol": sym,
            "transaction_type": _tx_code(action),
        }
        logger.info(f"📤 KotakNeo place_order REQUEST (close_position): {order_req}")

        raw = self._call_with_retry(self._neo.place_order, is_order=True, **order_req)

        logger.info(f"📥 KotakNeo place_order RESPONSE (close_position): {raw}")
        return self._normalize_order_response(raw, symbol=sym)

    def cancel_all_orders(self, strategy: str = "OpenClawNifty") -> dict:
        """Cancel all open orders."""
        self._rate_gate("/cancelallorder", {})
        logger.info("Cancelling all open orders")

        try:
            book_resp = self._neo.order_report()
            orders = []
            if isinstance(book_resp, dict):
                orders = book_resp.get("data", []) or book_resp.get("Data", [])
            elif isinstance(book_resp, list):
                orders = book_resp

            cancelled = []
            errors = []
            open_statuses = {"open", "pending", "trigger pending", "modify pending"}
            for order in orders:
                status = str(order.get("ordSt", order.get("status", ""))).lower()
                if status in open_statuses:
                    oid = order.get("nOrdNo", order.get("order_id", ""))
                    if oid:
                        try:
                            # Independent-verification fix #2 (2026-07-21):
                            # previously called self._neo.cancel_order()
                            # directly, bypassing _call_with_retry entirely
                            # -- no BrokerCallTimeout typing, no structured
                            # timeout log, no dedicated alert, on the exact
                            # call path emergency_stop() depends on. Now
                            # routed through the same is_order=True timeout
                            # protection as place_order/modify_order/
                            # cancel_order, so a timeout here is bounded,
                            # never blindly retried, and alerted.
                            self._call_with_retry(
                                self._neo.cancel_order, order_id=oid, is_order=True
                            )
                            cancelled.append(oid)
                        except Exception as e:
                            errors.append(f"{oid}: {e}")

            logger.info(f"Cancelled {len(cancelled)} orders; errors: {errors}")
            status = "success" if not errors else "error"
            return {"status": status, "cancelled": cancelled, "errors": errors}

        except Exception as e:
            logger.error(f"cancel_all_orders failed: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Portfolio & Market Data
    # ------------------------------------------------------------------

    def get_positions(self) -> dict:
        """Fetch current positions."""
        self._rate_gate("/positionbook", {})
        return self._call_with_retry(self._neo.positions)

    def get_open_nifty_positions(self) -> list:
        """
        Return a list of open NIFTY option positions (net_qty != 0).

        Each entry is a normalised dict:
          {
            "symbol":    str,   # trading symbol, e.g. "NIFTY2660923200PE"
            "direction": str,   # "CALL" (CE) or "PUT" (PE)
            "net_qty":   int,   # positive = long, negative = short
            "ltp":       float, # last traded price of the option
          }

        Returns [] on any error or when session is not authenticated.
        Used exclusively by position-reconciliation on startup.
        """
        try:
            resp = self._call_with_retry(self._neo.positions)
            rows = []
            if isinstance(resp, dict):
                rows = resp.get("data", []) or resp.get("Data", []) or []
            elif isinstance(resp, list):
                rows = resp

            result = []
            for row in rows:
                symbol = str(row.get("trdSym", row.get("trading_symbol", "")) or "")
                # Only NIFTY options (not BANKNIFTY, MIDCAP etc.)
                if not (symbol.upper().startswith("NIFTY") and
                        (symbol.upper().endswith("CE") or symbol.upper().endswith("PE"))):
                    continue

                buy_qty  = int(row.get("flBuyQty",  row.get("buyQty",  0)) or 0)
                sell_qty = int(row.get("flSellQty", row.get("sellQty", 0)) or 0)
                net_qty  = buy_qty - sell_qty
                if net_qty == 0:
                    continue

                direction = "CALL" if symbol.upper().endswith("CE") else "PUT"
                ltp = float(row.get("ltp", row.get("LTP", 0)) or 0)

                result.append({
                    "symbol":    symbol,
                    "direction": direction,
                    "net_qty":   net_qty,
                    "ltp":       ltp,
                })

            return result
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).debug(
                f"get_open_nifty_positions failed (session may not be ready): {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Exchange-Resident Stop-Loss (Phase 2)
    # ------------------------------------------------------------------

    def place_exchange_sl(
        self,
        symbol: str,
        qty: int,
        trigger_price: float,
        limit_price: float,
        product: str = "MIS",
    ) -> str:
        """
        Place a SL (stop-loss LIMIT) DAY SELL order at the exchange.

        This is the exchange-resident floor protection that fires even if the
        VPS crashes. It is placed immediately after a position is opened and
        cancelled before the software monitor closes the position.

        Kotak's RMS rejects SL-M (stop-loss MARKET) orders on index options
        outright ("STOP LOSS MARKET ORDER NOT ALLOWED IN OPTIONS, block type:
        ALL") — so this uses order_type "SL" (stop-loss limit) instead, which
        requires both a trigger and a limit price.

        Parameters
        ----------
        symbol        : trading symbol (e.g. "NIFTY2660923200PE")
        qty           : quantity to sell (same as position qty)
        trigger_price : OPTION PREMIUM (₹) at which the order arms
                        (the order is on the option contract, so the price
                        must be in premium terms, not spot-index points)
        limit_price   : OPTION PREMIUM (₹) limit once armed — set slightly
                        below trigger_price so the SELL fills on touch
        product       : product code matching the open position (default "MIS")

        Returns
        -------
        order_id (str) on success, "" on failure.
        Failure is logged at WARNING but never raises — entry is never blocked
        by SL placement failure.
        """
        try:
            self._rate_gate("/exchange_sl", {})
            exchange_seg = "nse_fo"
            order_req = {
                "exchange_segment": exchange_seg,
                "product":          product,
                "price":            str(round(limit_price, 2)),
                "order_type":       "SL",
                "quantity":         str(qty),
                "validity":         "DAY",
                "trading_symbol":   symbol,
                "transaction_type": "S",          # SELL
                "trigger_price":    str(round(trigger_price, 2)),
                "tag":              "OPENCLAW_SL",
            }
            logger.info(
                f"PlaceExchangeSL: {symbol} qty={qty} "
                f"trigger=Rs.{trigger_price:.2f} limit=Rs.{limit_price:.2f}"
            )
            raw = self._call_with_retry(
                self._neo.place_order, is_order=True, **order_req
            )
            result = self._normalize_order_response(raw, symbol=symbol)
            order_id = result.get("orderid", "") or result.get("nOrdNo", "")
            if order_id:
                logger.info(
                    f"ExchangeSL placed: order_id={order_id} "
                    f"trigger=Rs.{trigger_price:.2f} limit=Rs.{limit_price:.2f} "
                    f"symbol={symbol}"
                )
                return str(order_id)
            logger.warning(
                f"ExchangeSL: order placed but no order_id returned — "
                f"raw={raw}"
            )
            return ""
        except Exception as e:
            logger.warning(
                f"ExchangeSL: place failed ({e}) — software-only protection active"
            )
            return ""

    def cancel_exchange_sl(self, order_id: str) -> bool:
        """
        Cancel the exchange SL-M order before software close fires.

        Called immediately before `close_position()` to prevent a double-sell
        (software close + exchange SL both triggering).

        Returns True on success, False on failure (logged; never raises).
        """
        if not order_id:
            return True   # nothing to cancel
        try:
            self._rate_gate("/cancel_exchange_sl", {})
            # Independent-verification fix #3 (2026-07-21): is_order=True so
            # a timeout here raises BrokerCallTimeout (bounded, alerted,
            # structured-logged) instead of silently falling through to the
            # normal retries=MAX_RETRIES backoff loop -- this call sits
            # directly inside the fix #2 close-then-cancel-SL chain, and an
            # unknown-outcome cancel must never be blindly retried.
            self._call_with_retry(
                self._neo.cancel_order, order_id=str(order_id), is_order=True
            )
            logger.info(f"ExchangeSL cancelled: order_id={order_id}")
            return True
        except Exception as e:
            logger.warning(
                f"ExchangeSL: cancel failed for {order_id} ({e}) — "
                f"exchange SL may double-fire; position may go net-short; "
                f"check broker order book immediately"
            )
            return False

    def update_exchange_sl(
        self,
        order_id: str,
        new_trigger: float,
        new_limit: float,
        qty: int,
        symbol: str,
    ) -> bool:
        """
        Modify the trigger/limit price of an existing exchange SL order.

        Used when the software trailing stop advances the SL level. Updating
        the exchange SL keeps the floor protection in sync with the software SL.

        new_trigger / new_limit are OPTION PREMIUM (₹) values, matching the
        order_type "SL" (stop-loss limit) placed by place_exchange_sl().

        Returns True on success, False on failure (non-fatal — software SL
        continues to trail; exchange SL retains previous trigger).
        """
        if not order_id:
            return False
        try:
            self._rate_gate("/modify_exchange_sl", {})
            # Independent-verification fix #3 (2026-07-21): is_order=True,
            # same reasoning as cancel_exchange_sl above.
            self._call_with_retry(
                self._neo.modify_order,
                order_id=str(order_id),
                price=str(round(new_limit, 2)),
                order_type="SL",
                quantity=str(qty),
                validity="DAY",
                trigger_price=str(round(new_trigger, 2)),
                trading_symbol=symbol,
                is_order=True,
            )
            logger.info(
                f"ExchangeSL updated: order_id={order_id} "
                f"new_trigger=Rs.{new_trigger:.2f} new_limit=Rs.{new_limit:.2f}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"ExchangeSL: update failed for {order_id} ({e}) — "
                f"exchange SL retains old trigger; software SL is active"
            )
            return False

    def get_open_exchange_sl_orders(self) -> list[dict]:
        """
        Return all open SL-M SELL orders tagged 'OPENCLAW_SL'.

        Used by position reconciliation on startup to detect/rebuild missing
        exchange SL orders after a VPS crash or restart.

        Returns list of {order_id, symbol, trigger_price, qty}.
        """
        try:
            self._rate_gate("/orderbook", {})
            raw = self._call_with_retry(self._neo.order_report)
            orders = []
            if isinstance(raw, dict):
                orders = raw.get("data", []) or raw.get("Data", []) or []
            elif isinstance(raw, list):
                orders = raw

            result = []
            for o in orders:
                # Filter: SL-M SELL OPENCLAW_SL tag, open/pending status
                tag    = str(o.get("tag", "") or o.get("tags", "") or "")
                otype  = str(o.get("ordTyp", o.get("order_type", ""))).upper()
                txtype = str(o.get("trnsTp", o.get("transaction_type", ""))).upper()
                status = str(o.get("ordSt", o.get("status", ""))).lower()

                if (
                    "OPENCLAW_SL" in tag
                    and "SL" in otype
                    and txtype in ("S", "SELL")
                    and status in ("open", "pending", "trigger pending")
                ):
                    sym = str(o.get("trdSym", o.get("trading_symbol", "")))
                    trg = float(o.get("trgPrc", o.get("trigger_price", 0)) or 0)
                    qty = int(o.get("qty", o.get("quantity", 0)) or 0)
                    oid = str(o.get("nOrdNo", o.get("order_id", "")))
                    result.append({
                        "order_id": oid, "symbol": sym,
                        "trigger_price": trg, "qty": qty,
                    })
            return result
        except Exception as e:
            logger.debug(f"get_open_exchange_sl_orders failed: {e}")
            return []

    def get_orderbook(self) -> dict:
        """Fetch order book."""
        self._rate_gate("/orderbook", {})
        return self._call_with_retry(self._neo.order_report)

    def get_tradebook(self) -> dict:
        """Fetch trade book."""
        self._rate_gate("/tradebook", {})
        return self._call_with_retry(self._neo.trade_report)

    def get_holdings(self) -> dict:
        """Fetch holdings."""
        self._rate_gate("/holdings", {})
        return self._call_with_retry(self._neo.holdings)

    def get_funds(self) -> dict:
        """Fetch available funds/margins (FO segment)."""
        self._rate_gate("/funds", {})
        return self._call_with_retry(self._neo.limits, segment="FO", exchange="NSE", product="ALL")

    def get_quote(self, symbol: str, exchange: str) -> dict:
        """
        Get live quote for a symbol.
        For Nifty index → uses 'Nifty 50' token.
        For option symbols → fetches instrument token via search_scrip first.
        """
        self._rate_gate("/quotes", {"symbol": symbol})

        # Index quote — use same dynamically resolved token as _get_spot_price
        if symbol.upper() in ("NIFTY", "NIFTY50", "NIFTY 50"):
            try:
                ltp = self._get_spot_price()
                if ltp > 0:
                    return {"data": {"ltp": ltp}, "status": "success"}
            except Exception as e:
                logger.debug(f"get_quote Nifty failed: {e}")
            return {"data": {"ltp": 0}, "status": "error"}

        # Option/equity quote — resolve token first
        try:
            results = self._call_with_retry(
                self._neo.search_scrip,
                exchange_segment="nse_fo",
                symbol=symbol.lower().replace("-fo", ""),
            )
            if isinstance(results, list) and results:
                token = str(results[0].get("pSymbol", ""))
                if token:
                    resp = self._call_with_retry(
                        self._neo.quotes,
                        instrument_tokens=[{"instrument_token": token, "exchange_segment": "nse_fo"}],
                        quote_type="ltp",
                    )
                    data = resp.get("data", resp) if isinstance(resp, dict) else resp
                    if isinstance(data, list) and data:
                        ltp = float(data[0].get("ltp") or data[0].get("last_traded_price") or 0)
                        return {"data": {"ltp": ltp}, "status": "success"}
        except Exception as e:
            logger.debug(f"get_quote option {symbol} failed: {e}")

        return {"data": {"ltp": 0}, "status": "error"}

    def get_market_depth(self, symbol: str, exchange: str) -> dict:
        """Get Level-5 market depth."""
        self._rate_gate("/depth", {"symbol": symbol})
        try:
            results = self._neo.search_scrip(exchange_segment="nse_fo", symbol=symbol.lower())
            if isinstance(results, list) and results:
                token = str(results[0].get("pSymbol", ""))
                if token:
                    return self._call_with_retry(
                        self._neo.quotes,
                        instrument_tokens=[{"instrument_token": token, "exchange_segment": "nse_fo"}],
                        quote_type="depth",
                    )
        except Exception as e:
            logger.debug(f"get_market_depth {symbol} failed: {e}")
        return {}

    def get_history(
        self,
        symbol: str,
        exchange: str,
        interval: str = "5m",
        start_date: str = "",
        end_date: str = "",
    ) -> dict:
        """Historical OHLCV — not natively supported by Kotak Neo SDK; returns empty."""
        logger.debug("get_history: not supported by Kotak Neo SDK; use tv_fetcher instead")
        return {"data": [], "status": "not_supported"}

    def search_symbol(self, symbol: str, exchange: str) -> dict:
        """Search for a symbol."""
        self._rate_gate("/search", {"symbol": symbol})
        exchange_segment = "nse_fo" if exchange.upper() in ("NFO", "NSE_FO", "NSE_INDEX") else "nse_cm"
        return self._call_with_retry(
            self._neo.search_scrip,
            exchange_segment=exchange_segment,
            symbol=symbol.lower(),
        )

    # ------------------------------------------------------------------
    # Options Data — Expiry, Chain, Symbol
    # ------------------------------------------------------------------

    def get_expiry_dates(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
    ) -> list[str]:
        """
        Return available expiry dates in OpenAlgo format: ['24MAR26', '31MAR26', ...].
        Uses search_scrip to find all contracts and extracts unique expiries.
        """
        self._rate_gate("/expiry", {})
        try:
            results = self._call_with_retry(
                self._neo.search_scrip,
                exchange_segment="nse_fo",
                symbol=underlying.lower(),
            )
            if not isinstance(results, list) or not results:
                return []

            seen: set[str] = set()
            dates: list[str] = []
            today = datetime.now()
            today_ts = today.timestamp()

            # Kotak SDK uses different field names across versions.
            # Also try parsing from the trading symbol itself (most reliable).
            EXPIRY_TS_FIELDS = (
                "lExpiryDate", "pExpiryDate", "ExpiryDate",
                "expiry_date", "expiry", "pExpiry",
            )
            EXPIRY_STR_FIELDS = (
                "pExpiry", "expiry", "expiryDate", "maturityDate",
            )
            import re as _re
            # NSE option symbol pattern: NIFTY + YY + M{1,2} + DD + strike + CE/PE
            _SYM_EXPIRY = _re.compile(
                r'^NIFTY(\d{2})(1[0-2]|[1-9])(\d{2})\d{4,6}(?:CE|PE)$'
            )
            _MONTHS_SHORT = [
                "", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
            ]

            def _try_from_symbol(sym: str) -> str:
                m = _SYM_EXPIRY.match(sym.upper())
                if m:
                    yy, mon, dd = m.group(1), int(m.group(2)), m.group(3)
                    if 1 <= mon <= 12:
                        return f"{dd}{_MONTHS_SHORT[mon]}{yy}"
                return ""

            for scrip in results:
                oa_fmt = ""

                # Try timestamp fields
                if not oa_fmt:
                    for fld in EXPIRY_TS_FIELDS:
                        v = scrip.get(fld)
                        if v and isinstance(v, (int, float)) and v > today_ts:
                            try:
                                oa_fmt = datetime.fromtimestamp(v).strftime("%d%b%y").upper()
                                break
                            except Exception:
                                pass

                # Try string date fields (e.g. "09-JUN-2026", "09JUN2026")
                if not oa_fmt:
                    for fld in EXPIRY_STR_FIELDS:
                        v = str(scrip.get(fld, "") or "").strip()
                        if v:
                            try:
                                for fmt in ("%d-%b-%Y", "%d%b%Y", "%d/%m/%Y",
                                            "%Y-%m-%d", "%d-%m-%Y"):
                                    try:
                                        dt_p = datetime.strptime(v, fmt)
                                        if dt_p > today:
                                            oa_fmt = dt_p.strftime("%d%b%y").upper()
                                            break
                                    except ValueError:
                                        pass
                                if oa_fmt:
                                    break
                            except Exception:
                                pass

                # Parse expiry from trading symbol (most reliable, works for any SDK version)
                if not oa_fmt:
                    for sym_field in ("pTrdSymbol", "trading_symbol", "symbol"):
                        sym = str(scrip.get(sym_field, "") or "")
                        if sym:
                            oa_fmt = _try_from_symbol(sym)
                            if oa_fmt:
                                break

                if oa_fmt and oa_fmt not in seen:
                    # Validate it's a future date
                    try:
                        dt_check = datetime.strptime(oa_fmt, "%d%b%y")
                        if dt_check > today:
                            seen.add(oa_fmt)
                            dates.append((dt_check, oa_fmt))
                    except Exception:
                        pass

            dates.sort(key=lambda x: x[0])
            result = [d[1] for d in dates]
            logger.info(f"Expiry dates for {underlying}: {result[:5]}")
            return result

        except Exception as e:
            logger.warning(f"get_expiry_dates failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Native Kotak option-data helpers
    # ------------------------------------------------------------------
    #
    # Kotak Neo SDK (neo-api-client v2) does NOT expose a REST
    # `option_chain` endpoint. The previous implementation wrapped NSE
    # India's public API and labelled it "KotakNeo→NSE" — that source is
    # blocked from cloud VPS IPs and is the reason every chain fetch was
    # returning empty in production.
    #
    # The native path is two-step:
    #   1. search_scrip(symbol=underlying, expiry=expiry_neo, ...)
    #         → returns instrument_token + pTrdSymbol per contract
    #   2. quotes(instrument_tokens=[{token, segment}, ...], quote_type=...)
    #         → returns LTP / bid / ask / OI / IV per token
    #
    # Both methods are authenticated calls over the broker session — they
    # cannot be IP-blocked and don't depend on NSE's public site.

    def _empty_quote(self) -> dict:
        return {
            "ltp": 0.0, "bid": 0.0, "ask": 0.0,
            "oi": 0.0, "iv": 0.0, "symbol": "", "token": "",
        }

    def _empty_chain(self) -> dict:
        return {
            "spot": 0, "underlying_ltp": 0,
            "atm": 0, "atm_strike": 0,
            "expiry": "", "chain": [],
            "pcr": 1.0, "total_ce_oi": 0, "total_pe_oi": 0,
            "source": "EMPTY",
        }

    def _parse_option_quote(self, resp, symbol: str = "", token: str = "") -> dict:
        """Normalize a Kotak Neo quotes() response into a flat dict.

        Field-name aliases cover every wire format the SDK has used
        across versions (ltp / last_traded_price / ltP, bPrc / bid / ...).
        """
        if not isinstance(resp, dict) and not isinstance(resp, list):
            return {**self._empty_quote(), "symbol": symbol, "token": token}
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        if isinstance(data, list):
            rec = data[0] if data else {}
        elif isinstance(data, dict):
            rec = data
        else:
            rec = {}

        def _f(*keys) -> float:
            for k in keys:
                v = rec.get(k)
                if v in (None, "", "0", 0):
                    continue
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return 0.0

        return {
            "symbol": symbol or rec.get("trading_symbol") or rec.get("pTrdSymbol", ""),
            "token":  token  or str(rec.get("instrument_token") or rec.get("tk", "")),
            "ltp":    _f("ltp", "last_traded_price", "ltP", "LTP"),
            "bid":    _f("bPrc", "bid", "best_bid", "bid_price", "bp"),
            "ask":    _f("aPrc", "ask", "best_ask", "ask_price", "sp"),
            "oi":     _f("open_int", "oi", "openInterest", "openIntr", "OI"),
            "iv":     _f("iv", "implied_volatility", "impliedVolatility", "IV"),
        }

    def get_option_quote(
        self,
        underlying: str,
        expiry: str,           # OpenAlgo "28APR26" or Neo "28APR2026"
        strike: int,
        option_type: str,      # "CE" or "PE"
        quote_type: str = "depth",
    ) -> dict:
        """
        Native Kotak Neo single-strike option quote.

        Used by `_smart_execute` for IOC price discovery — replaces the
        broken approx-symbol → search_scrip(symbol=full_string) path.

        Returns a flat dict (always populated, even on failure):
            {symbol, token, ltp, bid, ask, oi, iv}
        """
        self._rate_gate("/option_quote", {})
        expiry_neo = self._expiry_oa_to_neo(expiry)
        try:
            results = self._call_with_retry(
                self._neo.search_scrip,
                exchange_segment="nse_fo",
                symbol=underlying.lower(),
                expiry=expiry_neo,
                option_type=option_type.upper(),
                strike_price=str(strike),
            )
            if not isinstance(results, list) or not results:
                logger.debug(
                    f"get_option_quote: search_scrip empty for "
                    f"{underlying} {expiry_neo} {strike}{option_type}"
                )
                return self._empty_quote()
            # Filter using tolerant helper (NIFTY/'NIFTY 50'/'NIFTY-I' OK, FINNIFTY out)
            filtered = _filter_underlying(results, underlying)
            scrip = filtered[0] if filtered else results[0]
            if not filtered:
                logger.warning(
                    f"get_option_quote: no '{underlying.upper()}' match, got "
                    f"{[r.get('pSymbolName') for r in results[:5]]}"
                )
            token   = str(scrip.get("pSymbol", ""))
            trd_sym = scrip.get("pTrdSymbol", "")
            if not token:
                return self._empty_quote()

            resp = self._call_with_retry(
                self._neo.quotes,
                instrument_tokens=[
                    {"instrument_token": token, "exchange_segment": "nse_fo"}
                ],
                quote_type=quote_type,
            )
            return self._parse_option_quote(resp, symbol=trd_sym, token=token)
        except Exception as e:
            logger.debug(
                f"get_option_quote {underlying} {expiry_neo} "
                f"{strike}{option_type} failed: {e}"
            )
            return self._empty_quote()

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
        expiry_date: str = "",
        strike_count: int = 10,
    ) -> dict:
        """
        Native Kotak Neo option chain — assembled in two steps:

          1. search_scrip(symbol=underlying, expiry=expiry_neo)
             → returns every CE+PE contract for that expiry. We filter
               to ±`strike_count` strikes around current ATM.

          2. quotes(instrument_tokens=[...], quote_type="depth")
             → bulk LTP / bid / ask / OI / IV. Chunked at 50 tokens per
               request (Kotak's documented batch ceiling).

        Returns the same dict shape the rest of the codebase consumes:
            {spot, atm_strike, expiry, chain: [{strike, ce_*, pe_*}, ...],
             pcr, total_ce_oi, total_pe_oi, source}
        """
        self._rate_gate("/optionchain", {})

        expiry_neo = (
            self._expiry_oa_to_neo(expiry_date)
            if expiry_date else self._get_nearest_expiry()
        )
        if not expiry_neo:
            logger.warning("get_option_chain: no expiry resolvable")
            return self._empty_chain()

        try:
            # Step 1 — enumerate every contract for this expiry
            all_scrips = self._call_with_retry(
                self._neo.search_scrip,
                exchange_segment="nse_fo",
                symbol=underlying.lower(),
                expiry=expiry_neo,
            )
            if not isinstance(all_scrips, list) or not all_scrips:
                logger.warning(
                    f"Kotak chain: search_scrip empty for "
                    f"{underlying} {expiry_neo}"
                )
                return self._empty_chain()

            # CRITICAL filter: search_scrip(symbol="nifty") also returns FINNIFTY.
            # Use tolerant helper — accepts 'NIFTY', 'NIFTY 50', 'NIFTY-I' etc.
            raw_count = len(all_scrips)
            all_scrips = _filter_underlying(all_scrips, underlying)
            if not all_scrips:
                logger.warning(
                    f"Kotak chain: no '{underlying.upper()}' contracts for "
                    f"expiry {expiry_neo} (raw count was {raw_count})"
                )
                return self._empty_chain()
            logger.debug(
                f"Kotak chain: {underlying} filter {raw_count}→{len(all_scrips)} contracts"
            )

            spot = self._get_spot_price()
            if spot <= 0:
                logger.warning("Kotak chain: spot unavailable, cannot pick ATM band")
                return self._empty_chain()
            atm_strike = int(round(spot / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP)
            min_strike = atm_strike - strike_count * NIFTY_STRIKE_STEP
            max_strike = atm_strike + strike_count * NIFTY_STRIKE_STEP

            # Group ATM-band contracts by strike → {strike: {"CE": (tok,sym), "PE": (tok,sym)}}
            # Kotak SDK has historically used several field names for strike + opt-type.
            # Try in order of likelihood so a field rename does not silently break us.
            STRIKE_FIELDS = ("dStrikePrice;", "dStrikePrice", "strikePrice",
                             "strike_price", "dStrike", "strike")
            OPT_FIELDS    = ("pOptionType", "optionType", "option_type",
                             "pOptType", "optType")

            def _extract_strike(s: dict) -> int:
                for f in STRIKE_FIELDS:
                    v = s.get(f)
                    if v in (None, "", 0, "0"):
                        continue
                    try:
                        sk = float(v)
                    except (TypeError, ValueError):
                        continue
                    # Paise-scaled values (e.g. 2370000 for ₹23,700) → normalise
                    if sk > 1_000_000:
                        sk = sk / 100.0
                    return int(round(sk))
                return 0

            def _extract_opt(s: dict) -> str:
                for f in OPT_FIELDS:
                    v = (s.get(f) or "").strip().upper() if isinstance(s.get(f), str) else ""
                    if v in ("CE", "PE", "CALL", "PUT", "C", "P"):
                        # normalise CALL→CE, PUT→PE, C→CE, P→PE
                        if v in ("CALL", "C"): return "CE"
                        if v in ("PUT",  "P"): return "PE"
                        return v
                return ""

            strike_map: dict = {}
            for scrip in all_scrips:
                try:
                    strike = _extract_strike(scrip)
                    if strike <= 0:
                        continue
                    if strike < min_strike or strike > max_strike:
                        continue
                    opt = _extract_opt(scrip)
                    if opt not in ("CE", "PE"):
                        continue
                    token   = str(scrip.get("pSymbol", "") or scrip.get("instrument_token", ""))
                    trd_sym = scrip.get("pTrdSymbol", "") or scrip.get("trading_symbol", "")
                    if not token:
                        continue
                    strike_map.setdefault(strike, {})[opt] = (token, trd_sym)
                except (TypeError, ValueError):
                    continue

            if not strike_map:
                # Diagnostic dump — show what fields Kotak actually returned so
                # we can spot scaling issues (paise vs rupees), wrong expiry,
                # or renamed fields in the SDK.
                all_strikes_raw = []
                for s in all_scrips:
                    sk = _extract_strike(s)
                    if sk > 0:
                        all_strikes_raw.append(sk)
                sample_strikes = sorted(set(all_strikes_raw))[:20]
                # Show keys of first scrip so we can spot field renames
                first_keys = list(all_scrips[0].keys()) if all_scrips else []
                first_sample = {k: all_scrips[0].get(k) for k in first_keys[:25]} if all_scrips else {}
                logger.warning(
                    f"Kotak chain: no strikes found in band "
                    f"[{min_strike}, {max_strike}] (ATM={atm_strike}). "
                    f"Sample strikes parsed: {sample_strikes} "
                    f"(total={len(all_scrips)} scrips, expiry={expiry_neo})"
                )
                logger.warning(
                    f"Kotak chain DIAG: first-scrip keys={first_keys[:25]} | "
                    f"first-scrip sample={first_sample}"
                )
                return self._empty_chain()

            # Step 2 — batch-quote all tokens (chunk at 50)
            instruments: list = []
            token_to_meta: dict = {}
            for strike, types in strike_map.items():
                for opt, (token, trd_sym) in types.items():
                    instruments.append(
                        {"instrument_token": token, "exchange_segment": "nse_fo"}
                    )
                    token_to_meta[token] = (strike, opt, trd_sym)

            all_quotes: dict = {}
            CHUNK = 50
            for i in range(0, len(instruments), CHUNK):
                chunk = instruments[i:i + CHUNK]
                try:
                    resp = self._call_with_retry(
                        self._neo.quotes,
                        instrument_tokens=chunk,
                        # No quote_type = full quote: returns ltp + open_int + ohlc + depth
                        # quote_type="depth" only returns depth (no ltp, no OI)
                        # quote_type="ltp"   only returns ltp (no OI)
                    )
                    # Detect session-error fault BEFORE processing data
                    if isinstance(resp, dict) and self._check_session_error(resp):
                        logger.warning(
                            f"Kotak chain: quotes() returned session error "
                            f"(fault={resp.get('fault')}) — triggering re-login"
                        )
                        self._relogin()
                        # Retry this chunk once after re-login
                        resp = self._neo.quotes(instrument_tokens=chunk)

                    data = resp.get("data", resp) if isinstance(resp, dict) else resp
                    if isinstance(data, list):
                        for rec in data:
                            # Store under every possible token key variant so
                            # the lookup below never misses due to field renames.
                            # "exchange_token" is the key the full-quote response uses.
                            for tk_field in ("exchange_token", "instrument_token",
                                             "tk", "pSymbol", "token", "instrumentToken"):
                                tk = str(rec.get(tk_field) or "")
                                if tk:
                                    all_quotes[tk] = rec
                    elif isinstance(data, dict) and data.get("fault"):
                        logger.warning(
                            f"Kotak chain quotes chunk {i}-{i+CHUNK}: "
                            f"fault={data['fault']}"
                        )
                except Exception as e:
                    logger.debug(f"Kotak chain quote chunk {i}-{i+CHUNK} failed: {e}")

            if not all_quotes:
                logger.warning(
                    f"Kotak chain: quotes() returned 0 records for "
                    f"{len(instruments)} tokens — session may have expired"
                )

            # Step 3 — assemble OpenAlgo-shaped chain
            chain: list = []
            total_ce_oi = 0.0
            total_pe_oi = 0.0
            for strike in sorted(strike_map.keys()):
                entry = {"strike": strike}
                for opt in ("CE", "PE"):
                    meta = strike_map[strike].get(opt)
                    if not meta:
                        continue
                    tok, trd_sym = meta
                    rec = all_quotes.get(tok)
                    if not rec:
                        continue
                    parsed = self._parse_option_quote(
                        {"data": [rec]}, symbol=trd_sym, token=tok
                    )
                    pfx = "ce" if opt == "CE" else "pe"
                    entry[f"{pfx}_premium"] = parsed["ltp"]
                    entry[f"{pfx}_ltp"]     = parsed["ltp"]
                    entry[f"{pfx}_bid"]     = parsed["bid"]
                    entry[f"{pfx}_ask"]     = parsed["ask"]
                    entry[f"{pfx}_oi"]      = parsed["oi"]
                    entry[f"{pfx}_iv"]      = parsed["iv"]
                    entry[f"{pfx}_symbol"]  = parsed["symbol"]
                    entry[f"{pfx}_token"]   = parsed["token"]
                    if opt == "CE":
                        total_ce_oi += parsed["oi"]
                    else:
                        total_pe_oi += parsed["oi"]
                chain.append(entry)

            pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 1.0
            return {
                "spot":           spot,
                "underlying_ltp": spot,
                "atm":            atm_strike,
                "atm_strike":     atm_strike,
                "expiry":         expiry_date or expiry_neo,
                "chain":          chain,
                "pcr":            pcr,
                "total_ce_oi":    total_ce_oi,
                "total_pe_oi":    total_pe_oi,
                "source":         "KotakNeo",
            }
        except Exception as e:
            logger.warning(f"Kotak option chain failed: {e}")
            return self._empty_chain()

    def _get_nearest_expiry(self) -> str:
        """Return nearest expiry in Neo (DDMMMYYYY) format, or '' on failure."""
        try:
            from core.expiry_utils import get_expiry_date
            oa = get_expiry_date()   # returns a date object, e.g. date(2026, 5, 26)
            if oa is None:
                logger.warning("_get_nearest_expiry: NIFTY_EXPIRY not set in settings.env")
                return ""
            # get_expiry_date() returns a date object — convert to "26MAY26" string first
            oa_str = oa.strftime("%d%b%y").upper()   # → "26MAY26"
            return self._expiry_oa_to_neo(oa_str)    # → "26MAY2026"
        except Exception as e:
            logger.warning(f"_get_nearest_expiry failed: {e}")
            return ""

    def get_option_symbol(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
        expiry_date: str = "",
        offset: str = "ATM",
        option_type: str = "CE",
    ) -> dict:
        """Resolve the trading symbol for a given offset."""
        spot = self._get_spot_price()
        strike = self._offset_to_strike(spot, offset, option_type)
        expiry_neo = self._expiry_oa_to_neo(expiry_date)
        symbol = self._resolve_option_symbol(underlying, expiry_neo, strike, option_type)
        return {
            "symbol": symbol,
            "exchange": "NSE",
            "exchange_segment": "nse_fo",
            "strike": strike,
            "option_type": option_type,
            "expiry": expiry_date,
        }
