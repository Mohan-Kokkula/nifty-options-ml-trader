"""
ohlcv_builder.py — Live OHLCV Builder from WebSocket Ticks
===========================================================
Consumes raw ticks from TickFeed's queue and builds rolling OHLCV
DataFrames via pandas resample — completely native, no TradingView.

Architecture:
    TickFeed (WebSocket)
          │  tick dicts → queue
          ▼
    OHLCVBuilder._consume_loop()
          │  appends to _tick_buf (deque)
          │  rebuilds on each tick
          ▼
    self._df_5m   — 5-minute OHLCV (up to 200 bars)
    self._df_15m  — 15-minute OHLCV (up to 100 bars)
          │
          ├─ (read by VWAPEngine, ClaudePilot, MLEngine)
          │    get_ohlcv(symbol, interval_minutes) → pd.DataFrame
          │
          └─ EOD Saver (15:30 IST daily)
               data/nifty_5m_YYYY-MM-DD.csv
               data/nifty_fut_5m_YYYY-MM-DD.csv
               data/nifty_15m_YYYY-MM-DD.csv
               └─ accumulated → train_model.py retrains ML model

Tick dict format (from TickFeed):
    {
      "symbol":    "NIFTY" | "NIFTY_FUT",
      "ltp":       float,
      "volume":    float,
      "oi":        float,
      "timestamp": datetime,
    }

OHLCV columns returned:
    open, high, low, close, volume  (standard — same as TVFetcher)

Usage:
    q = queue.Queue(maxsize=10_000)
    builder = OHLCVBuilder(tick_queue=q)
    builder.start()
    df = builder.get_ohlcv("NIFTY", 5)    # → pd.DataFrame
    df = builder.get_ohlcv("NIFTY_FUT", 15)
    builder.stop()
"""

import logging
import queue
import threading
import time
from collections import deque
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Maximum ticks to keep in memory (covers 1 full trading day at ~1 tick/sec)
_MAX_TICKS    = 30_000
# Minimum bars required before we consider data "ready"
_MIN_BARS_5M  = 3
# How often to log stats (seconds)
_LOG_INTERVAL = 300.0
# EOD save time (IST): 15:30
_EOD_HOUR     = 15
_EOD_MINUTE   = 30
# Where to save daily CSVs
_DATA_DIR     = Path("data")


class OHLCVBuilder:
    """
    Consumes ticks from a queue and maintains live OHLCV DataFrames
    for NIFTY (spot) and NIFTY_FUT (futures) at 5m and 15m intervals.

    Runs two background daemon threads:
      1. _consume_loop  — drains tick queue, rebuilds OHLCV cache
      2. _eod_loop      — saves today's bars to CSV at 15:30 IST
    """

    def __init__(self, tick_queue: queue.Queue):
        self._queue   = tick_queue
        self._running = False
        self._thread:     Optional[threading.Thread] = None
        self._eod_thread: Optional[threading.Thread] = None
        self._lock    = threading.Lock()

        # Per-symbol tick buffers: symbol → deque of tick dicts
        self._ticks: Dict[str, deque] = {
            "NIFTY":     deque(maxlen=_MAX_TICKS),
            "NIFTY_FUT": deque(maxlen=_MAX_TICKS),
        }

        # Cached DataFrames: (symbol, interval_minutes) → pd.DataFrame
        self._cache: Dict[tuple, pd.DataFrame] = {}

        # EOD save tracking
        self._eod_saved_date: Optional[date] = None

        # Stats
        self.ticks_processed = 0
        self._last_log_time  = time.time()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def start(self):
        """Start the background consumer + EOD saver threads."""
        if self._running:
            return
        self._running = True

        self._thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="OHLCVBuilder",
        )
        self._thread.start()

        self._eod_thread = threading.Thread(
            target=self._eod_loop,
            daemon=True,
            name="OHLCVBuilder-EOD",
        )
        self._eod_thread.start()

        logger.info("OHLCVBuilder: started — consuming ticks + EOD saver active")

    def stop(self):
        """Signal clean shutdown — save data before stopping."""
        # Save whatever we have before shutting down
        self._save_eod_data(reason="shutdown")
        self._running = False
        logger.info(
            f"OHLCVBuilder: stopped | "
            f"ticks_processed={self.ticks_processed}"
        )

    def get_ohlcv(self, symbol: str = "NIFTY",
                  interval_minutes: int = 5,
                  n_bars: int = 100) -> pd.DataFrame:
        """
        Return a live OHLCV DataFrame built from WebSocket ticks.
        Columns: open, high, low, close, volume
        Returns empty DataFrame if not enough data yet.
        """
        key = (symbol.upper(), interval_minutes)
        with self._lock:
            df = self._cache.get(key, pd.DataFrame())

        if df.empty:
            return df

        return df.tail(n_bars).copy()

    def is_ready(self, symbol: str = "NIFTY",
                 interval_minutes: int = 5) -> bool:
        """True once we have at least MIN_BARS_5M complete bars."""
        df = self.get_ohlcv(symbol, interval_minutes, n_bars=_MIN_BARS_5M + 1)
        return len(df) >= _MIN_BARS_5M

    def latest_ltp(self, symbol: str = "NIFTY") -> float:
        """Return most recent LTP from tick buffer (fast, no resample)."""
        with self._lock:
            buf = self._ticks.get(symbol.upper())
        if buf:
            return buf[-1]["ltp"]
        return 0.0

    def tick_count(self, symbol: str = "NIFTY") -> int:
        with self._lock:
            return len(self._ticks.get(symbol.upper(), []))

    def get_daily_bar(self, symbol: str = "NIFTY") -> pd.DataFrame:
        """Return today's single daily OHLCV bar (updates live)."""
        return self.get_ohlcv(symbol.upper(), 1440, n_bars=1)

    # ──────────────────────────────────────────────────────────
    # Background consumer
    # ──────────────────────────────────────────────────────────

    def _consume_loop(self):
        """Drain the tick queue and rebuild OHLCV on each new tick."""
        while self._running:
            try:
                tick = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            symbol = tick.get("symbol", "").upper()
            if symbol not in self._ticks:
                continue

            with self._lock:
                self._ticks[symbol].append(tick)

            self._rebuild(symbol)
            self.ticks_processed += 1

            # Periodic stats log
            now = time.time()
            if now - self._last_log_time >= _LOG_INTERVAL:
                self._log_stats()
                self._last_log_time = now

    # ──────────────────────────────────────────────────────────
    # EOD saver loop
    # ──────────────────────────────────────────────────────────

    def _eod_loop(self):
        """
        Checks every 60 seconds if it's 15:30 IST.
        Saves today's complete OHLCV bars to CSV once per day.
        """
        while self._running:
            time.sleep(60)
            try:
                now = datetime.now()
                today = now.date()

                is_eod_time = (
                    now.hour == _EOD_HOUR and
                    now.minute >= _EOD_MINUTE
                )
                already_saved = (self._eod_saved_date == today)

                if is_eod_time and not already_saved:
                    self._save_eod_data(reason="eod")
                    self._eod_saved_date = today

                # Reset for next day
                if now.hour < _EOD_HOUR:
                    self._eod_saved_date = None

            except Exception as e:
                logger.error(f"OHLCVBuilder EOD loop error: {e}")

    def _save_eod_data(self, reason: str = "eod"):
        """
        Save today's OHLCV bars to CSV files in data/ directory.

        Files created:
            data/nifty_5m_YYYY-MM-DD.csv
            data/nifty_15m_YYYY-MM-DD.csv
            data/nifty_fut_5m_YYYY-MM-DD.csv

        These accumulate daily and are used by train_model.py
        to retrain the ML model on real market data.
        """
        today_str = date.today().isoformat()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        saved = []

        # Master accumulated files (what train_model_v8.py reads)
        # Each day's bars are APPENDED so the file grows over time
        # After 1 month → ~1,500 rows in nifty_5min.csv → enough to retrain ML
        master_map = [
            ("NIFTY",     5,    "nifty_5min.csv"),
            ("NIFTY",     15,   "nifty_15min.csv"),
            ("NIFTY",     30,   "nifty_30min.csv"),
            ("NIFTY",     60,   "nifty_60min.csv"),
            ("NIFTY",     1440, "nifty_day.csv"),
            ("NIFTY_FUT", 5,    "nifty_fut_5min.csv"),
        ]

        for symbol, interval, filename in master_map:
            try:
                df = self.get_ohlcv(symbol, interval, n_bars=200)
                if df.empty:
                    logger.debug(
                        f"OHLCVBuilder EOD: no {symbol} {interval}m data to save"
                    )
                    continue

                master_path = _DATA_DIR / filename

                if master_path.exists():
                    # Load existing, merge, deduplicate by timestamp
                    existing = pd.read_csv(
                        master_path, index_col=0, parse_dates=True
                    )
                    combined = pd.concat([existing, df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    combined.to_csv(master_path)
                    saved.append(f"{filename} ({len(combined)} rows total)")
                else:
                    # First day — create fresh
                    df.to_csv(master_path)
                    saved.append(f"{filename} ({len(df)} rows — first day)")

                # Also keep a daily backup (for reference / debugging)
                backup_name = filename.replace(".csv", f"_{today_str}.csv")
                df.to_csv(_DATA_DIR / backup_name)

            except Exception as e:
                logger.warning(
                    f"OHLCVBuilder EOD: failed to save {filename}: {e}"
                )

        if saved:
            logger.info(
                f"OHLCVBuilder EOD ({reason}): saved — "
                + " | ".join(saved)
            )
        else:
            logger.warning(
                f"OHLCVBuilder EOD ({reason}): nothing saved "
                f"(no data collected yet)"
            )

        # Fetch and append today's India VIX from NSE
        try:
            from core.vix_fetcher import update_vix_csv
            update_vix_csv()
        except Exception as e:
            logger.warning(f"OHLCVBuilder EOD: VIX fetch failed: {e}")

    # ──────────────────────────────────────────────────────────
    # OHLCV resample
    # ──────────────────────────────────────────────────────────

    def _rebuild(self, symbol: str):
        """
        Resample tick buffer into 5m and 15m OHLCV DataFrames.
        Called after every new tick — uses pandas resample for correctness.
        """
        with self._lock:
            ticks = list(self._ticks[symbol])

        if not ticks:
            return

        df_raw = pd.DataFrame(ticks)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.set_index("timestamp").sort_index()
        df_raw = df_raw[["ltp", "volume"]].rename(columns={"ltp": "price"})

        # Clip to today's session only (9:15 IST onwards)
        today = df_raw.index[-1].date()
        session_start = pd.Timestamp(
            year=today.year, month=today.month, day=today.day,
            hour=9, minute=15, tz=df_raw.index.tz
        )
        df_raw = df_raw[df_raw.index >= session_start]

        if df_raw.empty:
            return

        for interval_min in (5, 15, 30, 60):
            rule = f"{interval_min}min"
            try:
                ohlcv = df_raw["price"].resample(
                    rule,
                    closed="left",
                    label="left",
                    origin=session_start,
                ).ohlc()

                ohlcv["volume"] = df_raw["volume"].resample(
                    rule,
                    closed="left",
                    label="left",
                    origin=session_start,
                ).sum()

                ohlcv = ohlcv.dropna(subset=["open"])
                # Drop the currently-forming (last) bar — it's incomplete
                if len(ohlcv) >= 1:
                    ohlcv = ohlcv.iloc[:-1]

                ohlcv = ohlcv.astype(float)

                key = (symbol, interval_min)
                with self._lock:
                    self._cache[key] = ohlcv

            except Exception as e:
                logger.debug(
                    f"OHLCVBuilder resample error ({symbol} {interval_min}m): {e}"
                )

        # Daily bar — full session OHLC from all ticks (saved at EOD only)
        try:
            daily = pd.DataFrame({
                "open":   [float(df_raw["price"].iloc[0])],
                "high":   [float(df_raw["price"].max())],
                "low":    [float(df_raw["price"].min())],
                "close":  [float(df_raw["price"].iloc[-1])],
                "volume": [float(df_raw["volume"].sum())],
            }, index=[session_start])
            key = (symbol, 1440)   # 1440 minutes = 1 day
            with self._lock:
                self._cache[key] = daily
        except Exception as e:
            logger.debug(f"OHLCVBuilder daily bar error ({symbol}): {e}")

    def _log_stats(self):
        nifty_ticks = len(self._ticks.get("NIFTY", []))
        fut_ticks   = len(self._ticks.get("NIFTY_FUT", []))
        df5  = self._cache.get(("NIFTY", 5),  pd.DataFrame())
        df15 = self._cache.get(("NIFTY", 15), pd.DataFrame())
        logger.info(
            f"OHLCVBuilder stats | "
            f"processed={self.ticks_processed} | "
            f"NIFTY ticks={nifty_ticks} | FUT ticks={fut_ticks} | "
            f"5m bars={len(df5)} | 15m bars={len(df15)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Singleton accessor  (wired from main.py after startup)
# ──────────────────────────────────────────────────────────────────────

_builder_instance: Optional[OHLCVBuilder] = None


def set_ohlcv_builder(builder: OHLCVBuilder):
    """Register the global OHLCVBuilder (called once from main.py)."""
    global _builder_instance
    _builder_instance = builder


def get_ohlcv_builder() -> Optional[OHLCVBuilder]:
    return _builder_instance
