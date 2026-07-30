"""
tv_fetcher.py — TradingView Historical Data Fetcher
====================================================
Fetches OHLCV bars directly from TradingView using your premium account.
Replaces OpenAlgo's /history endpoint for ML feature generation.

Install:  pip install tradingview-datafeed

Credentials from .env:
    TV_USERNAME  — TradingView login email
    TV_PASSWORD  — TradingView password

Session management:
    - Health check before each fetch (ping with a small request)
    - Auto-reconnect on stale/dead session
    - Retry with backoff (up to 3 attempts)
    - Session refresh every 30 minutes to prevent staleness

Standalone test:
    python -m core.tv_fetcher
"""

import logging
import os
import time
import threading
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_tv_instance = None
_tv_lock = threading.Lock()
_tv_last_success: float = 0.0      # timestamp of last successful fetch
_tv_session_created: float = 0.0    # timestamp when session was created
_tv_consecutive_failures: int = 0

# ── In-memory OHLCV cache ────────────────────────────────────────────
# When TradingView is temporarily unreachable, serve the last successful
# response if it's < TV_CACHE_MAX_AGE_SECS old. This prevents the bot
# from skipping a full analysis cycle due to a 1-2 min TV outage.
# Key: (symbol, exchange, interval_minutes) → {"df": DataFrame, "ts": float}
_tv_cache: dict = {}
TV_CACHE_MAX_AGE_SECS = int(os.getenv("TV_CACHE_MAX_AGE_SECS", "600"))  # 10 min

# ── Data provenance registry ─────────────────────────────────────────
# Records WHICH source served the most recent fetch for each
# (symbol, exchange, interval) and how old the data is. Consumers (e.g. the
# pilot's entry gate) read this to refuse trading on stale / delayed data.
# Sources: "TV" (live) | "TV_CACHE" (<10min stale) | "KOTAK_WS" (real-time)
#          | "YFINANCE" (~15min lag) | "NONE" (all sources failed)
_last_fetch_meta: dict = {}   # cache_key -> {"source","fetch_ts","bar_ts"}


def _record_meta(cache_key, source: str, df=None) -> None:
    """Record provenance + freshness for the most recent fetch of a feed.

    NOTE on bar_ts: pandas `Timestamp(naive).timestamp()` interprets a tz-NAIVE
    timestamp as UTC. TradingView returns IST-naive bars, so a raw conversion
    would be off by the +5:30 IST offset on any server. We localize naive
    timestamps to Asia/Kolkata (the NSE exchange tz) before converting, so
    bar_ts is a correct POSIX epoch for TV/cache feeds. bar_age is used only
    as a DIAGNOSTIC (logged), never as a hard trading gate — see
    ClaudePilot._entry_data_is_fresh() — because source timezones differ
    (Kotak-WS bars use server-local time) and absolute bar-age is unreliable.
    """
    bar_ts = None
    try:
        if df is not None and len(df) > 0:
            ts = pd.Timestamp(df.index[-1])
            if ts.tzinfo is None:
                ts = ts.tz_localize("Asia/Kolkata")
            bar_ts = float(ts.timestamp())
    except Exception:
        bar_ts = None
    _last_fetch_meta[cache_key] = {
        "source": source,
        "fetch_ts": time.time(),
        "bar_ts": bar_ts,
    }


def get_data_freshness(symbol: str = "NIFTY", exchange: str = "NSE",
                       interval_minutes=5) -> dict:
    """
    Return freshness metadata for the last fetch of a feed:
      {source, fetch_age_sec, bar_age_sec}

    fetch_age_sec = seconds since we last called the source
    bar_age_sec   = seconds since the timestamp of the latest bar returned
    Returns source="NONE" with huge ages if the feed was never fetched.
    """
    key = (symbol.upper(), exchange.upper(), interval_minutes)
    meta = _last_fetch_meta.get(key)
    if not meta:
        return {"source": "NONE", "fetch_age_sec": 1e9, "bar_age_sec": 1e9}
    now = time.time()
    return {
        "source": meta["source"],
        "fetch_age_sec": now - meta["fetch_ts"],
        "bar_age_sec": (now - meta["bar_ts"]) if meta["bar_ts"] else 1e9,
    }

# ── yFinance fallback ─────────────────────────────────────────────────
# When TV is down AND cache is stale, try Yahoo Finance as a secondary
# data source. yf returns slightly delayed data (~5-15 min lag for free
# tier) but is good enough to keep the ML pipeline running.
#
# Symbol mapping: TV symbol → Yahoo Finance symbol
_YF_SYMBOL_MAP = {
    ("NIFTY",     "NSE"): "^NSEI",       # Nifty 50 index
    ("NIFTY",     "NFO"): "^NSEI",       # futures proxy → use spot
    ("NIFTY1!",   "NSE"): "^NSEI",       # continuous futures → use spot
    ("INDIAVIX",  "NSE"): "^INDIAVIX",   # India VIX
    ("INDIA VIX", "NSE"): "^INDIAVIX",
    ("INDIA_VIX", "NSE"): "^INDIAVIX",
}
# Interval mapping: TV minutes → yfinance interval string
_YF_INTERVAL_MAP = {
    3: "5m",    # yfinance has no 3m — use 5m as nearest
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    "D": "1d",
    "Dm": "1d",
}
# Period (lookback) per interval — enough to cover n_bars + weekends/gaps
_YF_PERIOD_MAP = {
    "5m":  "5d",
    "15m": "5d",
    "30m": "7d",
    "1h":  "7d",
    "1d":  "3mo",
}


def _yf_fetch(symbol: str, exchange: str,
              interval_minutes, n_bars: int) -> "pd.DataFrame":
    """
    Fetch OHLCV from Yahoo Finance as a fallback when TradingView is down.
    Returns a standardized DataFrame (open/high/low/close/volume, DatetimeIndex)
    or an empty DataFrame on failure.
    """
    try:
        import yfinance as yf

        yf_sym = _YF_SYMBOL_MAP.get((symbol.upper(), exchange.upper()))
        if not yf_sym:
            logger.debug(f"yFinance: no symbol mapping for {symbol}:{exchange}")
            return pd.DataFrame()

        yf_interval = _YF_INTERVAL_MAP.get(interval_minutes, "5m")
        period      = _YF_PERIOD_MAP.get(yf_interval, "5d")

        ticker = yf.Ticker(yf_sym)
        raw = ticker.history(period=period, interval=yf_interval,
                             auto_adjust=True, prepost=False)

        if raw is None or raw.empty:
            logger.warning(f"yFinance: empty response for {yf_sym} @{yf_interval}")
            return pd.DataFrame()

        # Standardize → lowercase columns, keep OHLCV only
        raw.columns = raw.columns.str.lower()
        cols = [c for c in ["open", "high", "low", "close", "volume"]
                if c in raw.columns]
        df = raw[cols].astype(float).dropna()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df.tail(n_bars)

        logger.info(
            "yFinance FALLBACK: %d bars %s @%s | close=%.2f",
            len(df), yf_sym, yf_interval,
            float(df["close"].iloc[-1]) if len(df) else 0,
        )
        return df

    except Exception as e:
        logger.warning(f"yFinance fallback failed for {symbol}@{interval_minutes}m: {e}")
        return pd.DataFrame()

# Session expires after 25 minutes — force reconnect to prevent stale data.
# Lowering this just adds noise and latency; 25 min matches observed session stability.
# Override with TV_SESSION_MAX_AGE_SECS env var (e.g. for free accounts try 600).
SESSION_MAX_AGE_SECS = int(os.getenv("TV_SESSION_MAX_AGE_SECS", "1500"))  # 25 min default
# After 2 consecutive failures, force reconnect (don't waste cycles)
MAX_CONSECUTIVE_FAILURES = 2
# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = [0.5, 1.0, 2.0]  # seconds between retries


def _create_tv_session():
    """Create a fresh TradingView session."""
    global _tv_instance, _tv_session_created, _tv_consecutive_failures
    try:
        from tvDatafeed import TvDatafeed
    except ImportError:
        raise ImportError("Run: pip install tradingview-datafeed")

    # Suppress the noisy signin error from tvDatafeed library
    import logging as _logging
    _logging.getLogger("tvDatafeed.main").setLevel(_logging.CRITICAL)

    username = os.getenv("TV_USERNAME", "")
    password = os.getenv("TV_PASSWORD", "")

    # Close existing session if any
    if _tv_instance is not None:
        try:
            if hasattr(_tv_instance, 'ws') and _tv_instance.ws:
                _tv_instance.ws.close()
        except Exception:
            pass
        _tv_instance = None

    if username and password:
        logger.info("TV: logging in as %s", username)
        _tv_instance = TvDatafeed(username=username, password=password)
    else:
        logger.warning("TV: no credentials — anonymous session")
        _tv_instance = TvDatafeed()

    _tv_session_created = time.time()
    _tv_consecutive_failures = 0
    return _tv_instance


def _is_session_healthy() -> bool:
    """Check if the current TV session is usable."""
    global _tv_instance, _tv_session_created, _tv_consecutive_failures

    if _tv_instance is None:
        return False

    # Session too old — force refresh
    age = time.time() - _tv_session_created
    if age > SESSION_MAX_AGE_SECS:
        logger.debug("TV: session age %.0f min ≥ %.0f min limit — refreshing", age / 60, SESSION_MAX_AGE_SECS / 60)
        return False

    # No successful fetch in last 5 min — session likely stale
    if _tv_last_success > 0 and (time.time() - _tv_last_success) > 5 * 60:
        logger.warning("TV: no successful fetch in 5 min — reconnecting")
        return False

    # Too many consecutive failures — session likely dead
    if _tv_consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.warning(
            "TV: %d consecutive failures — session dead, reconnecting",
            _tv_consecutive_failures
        )
        return False

    # Check websocket connection if available
    try:
        ws = getattr(_tv_instance, 'ws', None)
        if ws is not None:
            # tvDatafeed uses websocket-client; check if connected
            if hasattr(ws, 'connected') and not ws.connected:
                logger.warning("TV: websocket disconnected — reconnecting")
                return False
            if hasattr(ws, 'sock') and ws.sock is None:
                logger.warning("TV: websocket socket is None — reconnecting")
                return False
    except Exception:
        pass

    return True


def _get_tv():
    """Get a healthy TV session, reconnecting if needed."""
    global _tv_instance
    with _tv_lock:
        if not _is_session_healthy():
            _create_tv_session()
        return _tv_instance


def _force_reconnect():
    """Force a new session (called after fetch failure)."""
    global _tv_instance
    with _tv_lock:
        logger.info("TV: forcing reconnection...")
        _create_tv_session()


class TVFetcher:
    """Wraps tvDatafeed to deliver a clean OHLCV DataFrame with auto-reconnect."""

    _INTERVAL_MAP = {
        1:   "in_1_minute",
        3:   "in_3_minute",
        5:   "in_5_minute",
        10:  "in_10_minute",
        15:  "in_15_minute",
        30:  "in_30_minute",
        45:  "in_45_minute",
        60:  "in_1_hour",
        120: "in_2_hour",
        240: "in_4_hour",
        "D": "in_daily",
        "W": "in_weekly",
    }

    def __init__(self):
        self._ready: Optional[bool] = None

    def get_ohlcv(self, symbol="NIFTY", exchange="NSE",
                  interval_minutes=5, n_bars=100) -> pd.DataFrame:
        global _tv_last_success, _tv_consecutive_failures, _tv_cache

        try:
            from tvDatafeed import Interval
        except ImportError:
            raise ImportError("Run: pip install tradingview-datafeed")

        key = self._INTERVAL_MAP.get(interval_minutes)
        if key is None:
            raise ValueError(f"Unsupported interval: {interval_minutes}")

        cache_key = (symbol.upper(), exchange.upper(), interval_minutes)
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                tv = _get_tv()
                if tv is None:
                    raise ConnectionError("TV session is None after init")

                raw = tv.get_hist(
                    symbol=symbol.upper(), exchange=exchange.upper(),
                    interval=getattr(Interval, key), n_bars=n_bars + 10,
                )

                if raw is None or raw.empty:
                    raise ValueError("Empty response from TradingView")

                raw.columns = raw.columns.str.strip().str.lower()
                cols = [c for c in ["open", "high", "low", "close", "volume"]
                        if c in raw.columns]
                df = raw[cols].astype(float).dropna()
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df = df.tail(n_bars)

                # Success — reset failure counter and populate cache
                _tv_consecutive_failures = 0
                _tv_last_success = time.time()
                _tv_cache[cache_key] = {"df": df.copy(), "ts": time.time()}
                _record_meta(cache_key, "TV", df)

                logger.info("TV: %d bars %s:%s @%sm | close=%.2f",
                            len(df), exchange, symbol, interval_minutes,
                            df["close"].iloc[-1] if len(df) else 0)
                return df

            except Exception as e:
                last_error = e
                _tv_consecutive_failures += 1

                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    logger.warning(
                        "TV fetch failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, e, wait
                    )
                    time.sleep(wait)
                    # Force reconnect on retry
                    _force_reconnect()
                else:
                    logger.error(
                        "TV fetch FAILED after %d attempts: %s (symbol=%s exchange=%s interval=%s)",
                        MAX_RETRIES, last_error, symbol, exchange, interval_minutes
                    )

        # ── Fallback tier 1: in-memory cache (< 10 min) ──────────────────
        cached = _tv_cache.get(cache_key)
        if cached is not None:
            age = time.time() - cached["ts"]
            if age <= TV_CACHE_MAX_AGE_SECS:
                logger.warning(
                    "TV CACHE FALLBACK: serving stale data for %s:%s @%sm "
                    "(age=%.0fs < %ds) — live fetch failed",
                    exchange, symbol, interval_minutes, age, TV_CACHE_MAX_AGE_SECS
                )
                _record_meta(cache_key, "TV_CACHE", cached["df"])
                return cached["df"]
            else:
                logger.warning(
                    "TV cache too stale (age=%.0fs > %ds) for %s:%s @%sm",
                    age, TV_CACHE_MAX_AGE_SECS, exchange, symbol, interval_minutes
                )

        # ── Fallback tier 2: Kotak WebSocket OHLCVBuilder (real-time) ───────
        # OHLCVBuilder accumulates live Kotak ticks → real OHLCV bars, zero lag.
        # Only available after 09:05 IST when the post-login callback re-subscribes
        # both the Nifty index and futures feeds.
        if symbol.upper() in ("NIFTY", "NIFTY1!"):
            try:
                from core.ohlcv_builder import get_ohlcv_builder
                _ob = get_ohlcv_builder()
                if _ob is not None:
                    _iv = interval_minutes if isinstance(interval_minutes, int) else 5
                    df_kotak = _ob.get_ohlcv(
                        symbol="NIFTY",
                        interval_minutes=_iv,
                        n_bars=n_bars,
                    )
                    if not df_kotak.empty and len(df_kotak) >= 3:
                        logger.warning(
                            "KOTAK OHLCV FALLBACK: %d bars @%sm from WebSocket "
                            "ticks (real-time, zero lag) — TV was down",
                            len(df_kotak), _iv
                        )
                        # Populate cache so next cycle uses fresh Kotak data
                        _tv_cache[cache_key] = {
                            "df": df_kotak.copy(), "ts": time.time()
                        }
                        _record_meta(cache_key, "KOTAK_WS", df_kotak)
                        return df_kotak
            except Exception as _e:
                logger.debug(f"OHLCVBuilder fallback skipped: {_e}")

        # ── Fallback tier 3: yFinance (Yahoo Finance, ~15 min lag) ────────
        # Last resort when both TV and Kotak WebSocket are unavailable.
        logger.warning(
            "TV + Kotak down for %s:%s @%sm — trying yFinance (delayed data)",
            exchange, symbol, interval_minutes
        )
        df_yf = _yf_fetch(symbol, exchange, interval_minutes, n_bars)
        if not df_yf.empty:
            _tv_cache[cache_key] = {"df": df_yf.copy(), "ts": time.time()}
            _record_meta(cache_key, "YFINANCE", df_yf)
            return df_yf

        # ── All sources exhausted ─────────────────────────────────────────
        logger.error(
            "ALL DATA SOURCES FAILED for %s:%s @%sm "
            "(TV + cache + Kotak WebSocket + yFinance) — returning empty",
            exchange, symbol, interval_minutes
        )
        _record_meta(cache_key, "NONE", None)
        return pd.DataFrame()

    def get_nifty_5min(self,  n_bars=100): return self.get_ohlcv("NIFTY", "NSE", 5,  n_bars)
    def get_nifty_15min(self, n_bars=60):  return self.get_ohlcv("NIFTY", "NSE", 15, n_bars)
    def get_nifty_30min(self, n_bars=40):  return self.get_ohlcv("NIFTY", "NSE", 30, n_bars)
    def get_nifty_60min(self, n_bars=20):  return self.get_ohlcv("NIFTY", "NSE", 60, n_bars)

    def is_available(self):
        if self._ready is None:
            self._ready = not self.get_nifty_5min(n_bars=5).empty
        return self._ready


_fetcher_instance: Optional[TVFetcher] = None


def get_tv_fetcher() -> "NativeOHLCVProvider":
    """
    Return the primary OHLCV provider.

    Returns NativeOHLCVProvider (live WebSocket → TradingView fallback)
    so every existing call-site automatically gets native-first data
    without any changes.  The return type is a superset of TVFetcher —
    all the same methods exist.
    """
    return get_native_provider()


# ──────────────────────────────────────────────────────────────────────
# NativeOHLCVProvider
# ──────────────────────────────────────────────────────────────────────

class NativeOHLCVProvider:
    """
    Primary OHLCV source: live WebSocket ticks via OHLCVBuilder.
    Falls back to TVFetcher when native data is unavailable
    (e.g. pre-session warmup, builder not started, insufficient bars).

    Exposes the same interface as TVFetcher so callers are unaffected.
    """

    def __init__(self):
        self._tv = TVFetcher()

    def _builder(self):
        """Lazily import to avoid circular imports."""
        from core.ohlcv_builder import get_ohlcv_builder
        return get_ohlcv_builder()

    def get_ohlcv(self, symbol: str = "NIFTY", exchange: str = "NSE",
                  interval_minutes: int = 5, n_bars: int = 100) -> pd.DataFrame:
        """
        Return OHLCV DataFrame.
        1st choice: live native data from OHLCVBuilder (WebSocket)
        Fallback:   TradingView historical data
        """
        builder = self._builder()
        if builder is not None and builder.is_ready(symbol.upper(), interval_minutes):
            df = builder.get_ohlcv(symbol.upper(), interval_minutes, n_bars)
            if not df.empty:
                logger.debug(
                    "NativeOHLCV: %s @%sm — %d bars (native WebSocket)",
                    symbol, interval_minutes, len(df)
                )
                return df

        # Fallback to TradingView (DEBUG only — this fires for every call when
        # builder isn't ready: e.g. daily bars, 3m bars, and first ~30 min
        # after startup. Not an error — just the normal warm-up path.)
        logger.debug(
            "NativeOHLCV: %s @%sm — TV fallback "
            "(builder=%s ready=%s)",
            symbol, interval_minutes,
            "None" if builder is None else "present",
            False if builder is None else builder.is_ready(symbol.upper(), interval_minutes),
        )
        return self._tv.get_ohlcv(symbol, exchange, interval_minutes, n_bars)

    # Convenience wrappers — match TVFetcher API
    def get_nifty_5min(self,  n_bars: int = 100): return self.get_ohlcv("NIFTY", "NSE", 5,  n_bars)
    def get_nifty_15min(self, n_bars: int = 60):  return self.get_ohlcv("NIFTY", "NSE", 15, n_bars)
    def get_nifty_30min(self, n_bars: int = 40):  return self.get_ohlcv("NIFTY", "NSE", 30, n_bars)
    def get_nifty_60min(self, n_bars: int = 20):  return self.get_ohlcv("NIFTY", "NSE", 60, n_bars)

    def is_available(self) -> bool:
        builder = self._builder()
        if builder is not None and builder.is_ready("NIFTY", 5):
            return True
        return self._tv.is_available()

    def data_source(self) -> str:
        """Return human-readable label of active data source."""
        builder = self._builder()
        if builder is not None and builder.is_ready("NIFTY", 5):
            return f"native-ws ({builder.tick_count('NIFTY')} ticks)"
        return "tradingview"


_native_provider_instance: Optional[NativeOHLCVProvider] = None


def get_native_provider() -> NativeOHLCVProvider:
    """Return the singleton NativeOHLCVProvider (primary OHLCV source)."""
    global _native_provider_instance
    if _native_provider_instance is None:
        _native_provider_instance = NativeOHLCVProvider()
    return _native_provider_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    print("Testing TradingView fetcher...")
    df = TVFetcher().get_nifty_5min(n_bars=10)
    print(df.tail(5).to_string() if not df.empty else
          "No data — check TV_USERNAME / TV_PASSWORD in .env")
