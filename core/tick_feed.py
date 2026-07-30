"""
tick_feed.py — Kotak Neo WebSocket Tick Ingestion Worker
=========================================================
Subscribes to Kotak Neo's live WebSocket feed and pushes raw ticks
into a thread-safe queue consumed by OHLCVBuilder.

Architecture:
    Kotak Neo WebSocket
          │  on_message callback
          ▼
    TickFeed._on_tick()
          │  puts normalized tick dict
          ▼
    multiprocessing.Queue / queue.Queue
          │  consumed by
          ▼
    OHLCVBuilder  (see ohlcv_builder.py)

Subscribed instruments (two feeds):
  1. Nifty 50 Index  (token=26000, isIndex=True)
     → provides spot OHLCV for EMA / RSI / price structure checks
  2. Nifty Current-Month Futures (token resolved via search_scrip)
     → provides volume data needed for accurate VWAP

Tick normalization:
  Each tick pushed to queue is a dict:
  {
    "symbol":    "NIFTY" | "NIFTY_FUT",
    "ltp":       float,       # last traded price
    "volume":    float,       # traded volume this tick (0 for index)
    "oi":        float,       # open interest (futures only)
    "timestamp": datetime,    # UTC-normalized to IST
  }

Usage:
    feed = TickFeed(neo_client=client, tick_queue=q)
    feed.start()
    feed.stop()
"""

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Reconnect delay on WebSocket failure
_RECONNECT_DELAY = 5.0
# How often to re-check for market open while idle outside market hours
_MARKET_CLOSED_POLL = 300.0
# Nifty 50 index token (NSE standard — never changes)
# In Kotak Neo v2, index subscriptions use the neo-symbol string, not the
# numeric NSE token. "26000" silently fails; "Nifty 50" is the correct value.
_NIFTY_INDEX_TOKEN = "Nifty 50"


class TickFeed:
    """
    WebSocket tick subscriber for Kotak Neo.
    Runs in a background thread; pushes ticks to a shared queue.
    """

    def __init__(
        self,
        neo_client,                         # KotakNeoClient instance
        tick_queue: queue.Queue,            # shared queue → OHLCVBuilder
        futures_token: Optional[str] = None,# resolved at startup if None
        spot_callback=None,                 # optional callable(ltp: float) for each Nifty spot tick
    ):
        self._client        = neo_client
        self._neo           = neo_client._neo   # underlying NeoAPI object
        self._queue         = tick_queue
        self._futures_token = futures_token
        self._spot_callback = spot_callback     # e.g. pilot.update_tick_spot
        self._running       = False
        self._thread: Optional[threading.Thread] = None
        self._connected     = False
        self._lock          = threading.Lock()
        # Set by _on_close / _on_error — loop blocks on this instead of
        # re-subscribing every time the non-blocking subscribe() returns.
        self._disconnect_evt = threading.Event()

        # Stats
        self.ticks_received = 0
        self.ticks_dropped  = 0

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def start(self):
        """Start the WebSocket worker in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="TickFeed",
        )
        self._thread.start()
        logger.info("TickFeed: started")

    def stop(self):
        """Signal clean shutdown."""
        self._running = False
        self._disconnect_evt.set()   # wake the wait loop so it can exit
        try:
            self._neo.un_subscribe_feeds()
        except Exception:
            pass
        logger.info(
            f"TickFeed: stopped | "
            f"ticks={self.ticks_received} dropped={self.ticks_dropped}"
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ──────────────────────────────────────────────────────────
    # Background loop
    # ──────────────────────────────────────────────────────────

    def _run_loop(self):
        """Outer retry loop — reconnects on WebSocket failure."""
        while self._running:
            if not self._client._is_market_session_window():
                # Outside market hours — Kotak drops the WS every couple of
                # minutes here, which would otherwise spam ERROR-level
                # reconnect logs all night. Disconnect once and idle until
                # the next market session window.
                if self._connected:
                    try:
                        self._neo.un_subscribe_feeds()
                    except Exception:
                        pass
                    self._connected = False
                    logger.info(
                        "TickFeed: market closed — disconnecting until next "
                        f"session (re-checking every {int(_MARKET_CLOSED_POLL)}s)"
                    )
                self._disconnect_evt.clear()
                self._disconnect_evt.wait(timeout=_MARKET_CLOSED_POLL)
                continue
            try:
                self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"TickFeed connection error: {e}")
                self._connected = False
            if self._running:
                logger.warning(
                    f"TickFeed: reconnecting in {_RECONNECT_DELAY}s..."
                )
                time.sleep(_RECONNECT_DELAY)

    def _connect_and_subscribe(self):
        """Establish WebSocket connection and subscribe to instruments."""
        # Resolve futures token if not yet known
        if not self._futures_token:
            self._futures_token = self._resolve_futures_token()

        index_tokens, fut_tokens = self._build_subscription_tokens()
        logger.info(
            f"TickFeed: subscribing → index={[t['instrument_token'] for t in index_tokens]} "
            f"futures={[t['instrument_token'] for t in fut_tokens]}"
        )

        # Wire callbacks into NeoAPI
        self._neo.on_message = self._on_message
        self._neo.on_error   = self._on_error
        self._neo.on_close   = self._on_close
        self._neo.on_open    = self._on_open

        # SINGLE subscribe() call — Kotak Neo SDK opens a NEW WebSocket for
        # every subscribe() call. Two concurrent calls (futures + index) create
        # two WebSocket connections; Kotak drops one every ~2 min causing an
        # infinite reconnect storm. Fix: subscribe ONCE with the most useful
        # tokens available.
        #
        # Priority:
        #   1. Futures available → subscribe futures only (isIndex=False).
        #      Futures ticks carry full OHLCV + volume/OI for VWAP. NIFTY
        #      spot/index bars continue to come from TradingView (unchanged).
        #   2. No futures token → subscribe index only (isIndex=True). Gives
        #      live spot ticks via spot_callback but no volume data.
        self._disconnect_evt.clear()
        if fut_tokens:
            # Futures-only subscription: one connection, full volume data
            logger.info(
                "TickFeed: single WS subscribe — futures only "
                f"(token={fut_tokens[0]['instrument_token']})"
            )
            self._neo.subscribe(
                instrument_tokens=fut_tokens,
                isIndex=False,
                isDepth=False,
            )
        else:
            # Fallback: index-only subscription when futures token unavailable
            logger.info("TickFeed: single WS subscribe — index only (no futures token)")
            self._neo.subscribe(
                instrument_tokens=index_tokens,
                isIndex=True,
                isDepth=False,
            )
        # Block this thread until the WS actually drops (or we're stopped).
        while self._running and not self._disconnect_evt.is_set():
            self._disconnect_evt.wait(timeout=5.0)

    def _build_subscription_tokens(self) -> tuple[list, list]:
        """Build instrument-token lists for NeoAPI.subscribe().

        Returns (index_tokens, futures_tokens) — they must be subscribed
        separately because Kotak Neo's WebSocket uses a different frame
        format for indices (isIndex=True) vs. tradeable instruments.
        """
        index_tokens = [
            {"instrument_token": _NIFTY_INDEX_TOKEN, "exchange_segment": "nse_cm"}
        ]
        fut_tokens: list = []
        if self._futures_token and self._futures_token != _NIFTY_INDEX_TOKEN:
            fut_tokens.append(
                {"instrument_token": self._futures_token, "exchange_segment": "nse_fo"}
            )
        return index_tokens, fut_tokens

    def _resolve_futures_token(self) -> Optional[str]:
        """
        Look up current-month Nifty futures instrument token via search_scrip.

        BUG FIX 2026-05-25: Kotak's search_scrip is a PREFIX SEARCH on the
        underlying name — NOT a lookup by full trading symbol.
        Passing symbol="nifty26mayfut" returns nothing (or a pre-auth dict).
        Correct approach: search symbol="nifty", then filter for the FUT contract.
        """
        try:
            from core.vwap_engine import _current_futures_symbol
            fut_symbol = _current_futures_symbol()   # e.g. "NIFTY26MAYFUT"
            fut_upper  = fut_symbol.upper()

            results = self._neo.search_scrip(
                exchange_segment="nse_fo",
                symbol="nifty",           # underlying prefix — NOT the full symbol
            )

            if not isinstance(results, list):
                # Kotak returns a dict when session is not yet authenticated
                # e.g. {"Error Message": "Complete the 2fa process..."}.
                # Suppress noise — retry_futures_subscription() fires post-login.
                logger.debug(
                    "TickFeed: search_scrip returned non-list "
                    f"(session not ready): type={type(results).__name__}"
                )
                return None

            if not results:
                logger.warning(
                    "TickFeed: search_scrip returned empty list for 'nifty'"
                )
                return None

            # Pass 1: exact match on trading symbol
            for r in results:
                trd = str(
                    r.get("pTrdSymbol", "") or r.get("trading_symbol", "")
                ).upper()
                if trd == fut_upper:
                    token = str(
                        r.get("pSymbol", "") or r.get("instrument_token", "")
                    )
                    if token:
                        logger.info(
                            f"TickFeed: resolved futures token={token} ({fut_symbol})"
                        )
                        return token

            # Pass 2: any near-month NIFTY futures (ends with FUT, no option type)
            for r in results:
                trd = str(r.get("pTrdSymbol", "")).upper()
                opt = str(
                    r.get("pOptionType", "") or r.get("optionType", "")
                ).strip()
                if trd.startswith("NIFTY") and trd.endswith("FUT") and not opt:
                    token = str(
                        r.get("pSymbol", "") or r.get("instrument_token", "")
                    )
                    if token:
                        logger.info(
                            f"TickFeed: resolved futures token={token} ({trd}) "
                            f"[fallback — wanted {fut_symbol}]"
                        )
                        return token

            logger.warning(
                f"TickFeed: futures token NOT found for {fut_symbol}. "
                f"First 10 pTrdSymbols: "
                f"{[r.get('pTrdSymbol') for r in results[:10]]}"
            )

        except Exception as e:
            logger.warning(f"TickFeed: futures token lookup failed: {e}")
        return None

    def retry_futures_subscription(self) -> bool:
        """
        Called after Kotak session is established (post-login callback).

        At bot startup (~04:00) the Kotak session is not yet authenticated
        (KOTAK_LAZY_LOGIN=true), so both the index AND futures subscriptions
        either silently fail or get rejected with "Please complete Login Flow".
        This method fires on the post-login callback and re-subscribes BOTH
        instruments so that OHLCVBuilder receives real ticks from 09:05 onwards.

        Safe to call multiple times — no-op if already subscribed.
        Returns True if at least one subscription succeeded.
        """
        any_success = False

        # ── Step 1: Re-subscribe Nifty 50 index ──────────────────────────
        # The initial subscribe() at 04:00 is rejected by Kotak because the
        # session isn't authenticated yet. The WebSocket stays connected but
        # no ticks flow. Re-subscribing here (post-login) fixes that.
        if not self._connected:
            logger.debug("TickFeed: WebSocket not connected — skipping re-subscribe")
        else:
            try:
                index_tokens = [
                    {"instrument_token": _NIFTY_INDEX_TOKEN,
                     "exchange_segment": "nse_cm"}
                ]
                self._neo.subscribe(
                    instrument_tokens=index_tokens,
                    isIndex=True,
                    isDepth=False,
                )
                logger.info(
                    "TickFeed: ✓ Nifty 50 index re-subscribed post-login "
                    f"(token={_NIFTY_INDEX_TOKEN}) — spot ticks now flowing"
                )
                any_success = True
            except Exception as e:
                logger.warning(
                    f"TickFeed: index post-login re-subscription failed: {e}"
                )

        # ── Step 2: Resolve and subscribe futures ─────────────────────────
        if self._futures_token:
            logger.debug("TickFeed: futures already subscribed — skip retry")
            any_success = True
        else:
            token = self._resolve_futures_token()
            if not token:
                logger.warning(
                    "TickFeed: post-login retry — futures token still unresolvable"
                )
            else:
                self._futures_token = token
                fut_tokens = [
                    {"instrument_token": token, "exchange_segment": "nse_fo"}
                ]
                try:
                    self._neo.subscribe(
                        instrument_tokens=fut_tokens,
                        isIndex=False,
                        isDepth=False,
                    )
                    logger.info(
                        f"TickFeed: ✓ NIFTY_FUT subscription added post-login "
                        f"(token={token}) — volume/OI ticks now flowing"
                    )
                    any_success = True
                except Exception as e:
                    logger.warning(
                        f"TickFeed: futures post-login subscription failed: {e}"
                    )

        return any_success

    # ──────────────────────────────────────────────────────────
    # WebSocket callbacks
    # ──────────────────────────────────────────────────────────

    # Kotak Neo SDK invokes these as `cb(ws_instance)` / `cb(ws_instance, err)`
    # / `cb(ws_instance, msg)`. Use *args so we tolerate either calling
    # convention across SDK versions (some pass the ws handle, some don't).
    def _on_open(self, *args):
        self._connected = True
        self._disconnect_evt.clear()
        logger.info("TickFeed: WebSocket connected ✓")

    def _on_close(self, *args):
        self._connected = False
        self._disconnect_evt.set()
        logger.warning("TickFeed: WebSocket closed")

    def _on_error(self, *args):
        self._connected = False
        self._disconnect_evt.set()
        err = args[-1] if args else ""
        logger.error(f"TickFeed: WebSocket error: {err}")

    def _on_message(self, *args):
        # Last positional arg is the actual message payload
        message = args[-1] if args else None
        """
        Normalize raw Kotak Neo tick message and push to queue.
        Message format varies by SDK version — we handle both dict and list.
        """
        try:
            ticks = self._normalize_message(message)
            for tick in ticks:
                try:
                    self._queue.put_nowait(tick)
                    self.ticks_received += 1
                except queue.Full:
                    self.ticks_dropped += 1
                    logger.debug("TickFeed: queue full — tick dropped")

                # Push Nifty spot price to position monitor (1s high-freq path)
                if self._spot_callback and tick.get("symbol") == "NIFTY":
                    try:
                        self._spot_callback(tick["ltp"])
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"TickFeed._on_message error: {e}")

    def _normalize_message(self, message) -> list:
        """
        Convert raw Kotak Neo WebSocket payload to list of normalized tick dicts.

        Kotak Neo SDK sends ticks as dicts or lists depending on version.
        Known field names (vary across SDK versions):
          ltp:  ltp / last_traded_price / ltP
          vol:  volume / vol / trdVol
          oi:   oi / openInterest / openIntr
          token:instrument_token / tk / pSymbol
        """
        if message is None:
            return []

        # Handle list of ticks
        if isinstance(message, list):
            result = []
            for m in message:
                t = self._parse_single_tick(m)
                if t:
                    result.append(t)
            return result

        # Single tick dict
        if isinstance(message, dict):
            t = self._parse_single_tick(message)
            return [t] if t else []

        return []

    def _parse_single_tick(self, m: dict) -> Optional[dict]:
        """Parse a single tick dict from Kotak Neo WebSocket."""
        if not isinstance(m, dict):
            return None

        # LTP — try multiple field names
        ltp = (
            m.get("ltp") or m.get("last_traded_price") or
            m.get("ltP") or m.get("LTP") or 0
        )
        try:
            ltp = float(ltp)
        except (TypeError, ValueError):
            return None
        if ltp <= 0:
            return None

        # Volume
        vol = (
            m.get("volume") or m.get("vol") or m.get("trdVol") or
            m.get("totalBuyQuantity") or 0
        )
        try:
            vol = float(vol)
        except (TypeError, ValueError):
            vol = 0.0

        # Open Interest
        oi = m.get("oi") or m.get("openInterest") or m.get("openIntr") or 0
        try:
            oi = float(oi)
        except (TypeError, ValueError):
            oi = 0.0

        # Instrument token → determine symbol
        token = str(
            m.get("instrument_token") or m.get("tk") or
            m.get("pSymbol") or ""
        )
        symbol = (
            "NIFTY_FUT" if (
                self._futures_token and token == self._futures_token
            ) else "NIFTY"
        )

        # Timestamp
        ts = m.get("exchange_timestamp") or m.get("exch_timestamp") or m.get("ts")
        if ts:
            try:
                timestamp = datetime.fromisoformat(str(ts))
            except Exception:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        return {
            "symbol":    symbol,
            "ltp":       ltp,
            "volume":    vol,
            "oi":        oi,
            "timestamp": timestamp,
        }
