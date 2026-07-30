"""
vwap_engine.py — Session-Anchored VWAP from Nifty Current Month Futures
========================================================================
WHY futures and not spot:
  - Nifty Spot (NSE Index) has NO volume data — VWAP on it is meaningless.
  - Nifty Futures have real exchange-traded volume → true institutional
    fair-value anchor.

Calculation:
  VWAP = Σ(Typical_Price × Volume) / Σ(Volume)
  typical_price = (high + low + close) / 3
  Resets every trading day at 09:15 IST (session-anchored, not rolling).

Data sources (in priority order):
  1. Native WebSocket — live NIFTY_FUT ticks via OHLCVBuilder (zero external deps)
  2. TradingView — "NIFTY1!" continuous front-month futures (5m bars)
  3. Kotak Neo historical — current-month futures symbol (e.g. NIFTY25APRFUT)
  4. Fallback: 0.0 (callers must check and handle gracefully)

Usage:
  engine = VWAPEngine(tv_fetcher=get_tv_fetcher(), kotak_client=client)
  vwap   = engine.get_current_vwap()          # float (0.0 = unavailable)
  df     = engine.get_vwap_dataframe()        # DataFrame with 'vwap' column
  feats  = engine.get_vwap_features(spot)     # dict for ML pipeline
"""

import logging
from datetime import datetime, date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TradingView continuous front-month Nifty futures symbol
_TV_FUT_SYMBOL   = "NIFTY1!"
_TV_FUT_EXCHANGE = "NSE"

# Cache TTL — refresh at most once per minute
_CACHE_TTL_SEC = 60


class VWAPEngine:
    """
    Session-anchored VWAP built from Nifty Current Month Futures OHLCV.

    Thread-safe for read access; write (refresh) should happen from a
    single scheduler thread (the main pilot loop).
    """

    def __init__(self, tv_fetcher=None, kotak_client=None):
        self._tv      = tv_fetcher
        self._client  = kotak_client
        self._cache_df: Optional[pd.DataFrame] = None
        self._cache_ts: Optional[datetime]     = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_vwap(self) -> float:
        """
        Returns the current session VWAP as a float.
        Returns 0.0 when data is unavailable — callers must check.
        """
        df = self._get_futures_ohlcv()
        if df is None or df.empty:
            logger.warning("VWAPEngine: no futures data — returning 0.0")
            return 0.0
        series = self._compute_vwap_series(df)
        if series is None or series.empty:
            return 0.0
        val = float(series.iloc[-1])
        return val if not np.isnan(val) else 0.0

    def get_vwap_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Returns the full 5m futures DataFrame with a 'vwap' column appended.
        Useful for structure filtering and band calculation.
        """
        df = self._get_futures_ohlcv()
        if df is None or df.empty:
            return None
        df = df.copy()
        df["vwap"] = self._compute_vwap_series(df)
        return df

    def get_vwap_features(self, spot: float) -> dict:
        """
        Returns a flat feature dict suitable for ML pipeline injection.
        Includes VWAP level, distance from spot, and band width.
        """
        df = self.get_vwap_dataframe()
        if df is None or df.empty or "vwap" not in df.columns:
            return _zero_vwap_features()

        try:
            vwap_now    = float(df["vwap"].iloc[-1])
            vwap_prev   = float(df["vwap"].iloc[-2]) if len(df) >= 2 else vwap_now
            vwap_trend  = vwap_now - vwap_prev           # rising or falling

            # VWAP bands (±1 std dev of typical-price deviations)
            typical = (df["high"].astype(float) +
                       df["low"].astype(float) +
                       df["close"].astype(float)) / 3.0
            vol = df["volume"].astype(float) if "volume" in df.columns \
                else pd.Series(1.0, index=df.index)

            # Today's bars only
            today = date.today()
            mask  = df.index.date == today
            if mask.sum() < 2:
                mask = pd.Series(True, index=df.index)   # fallback: all bars

            tp_today  = typical[mask]
            vol_today = vol[mask]
            vwap_td   = df["vwap"][mask]

            variance = ((tp_today - vwap_td) ** 2 * vol_today).sum() / vol_today.sum()
            std_dev   = float(np.sqrt(max(variance, 0.0)))
            band_width = std_dev * 2          # upper - lower band

            if np.isnan(vwap_now):
                return _zero_vwap_features()

            return {
                "vwap":             round(vwap_now, 2),
                "vwap_dist":        round(spot - vwap_now, 2),   # + = above
                "vwap_dist_pct":    round((spot - vwap_now) / vwap_now * 100, 3),
                "vwap_trend":       round(vwap_trend, 2),
                "vwap_std":         round(std_dev, 2),
                "vwap_band_width":  round(band_width, 2),
                "above_vwap":       int(spot > vwap_now),
                "vwap_avail":       1,
            }
        except Exception as e:
            logger.error(f"VWAPEngine.get_vwap_features failed: {e}")
            return _zero_vwap_features()

    # ------------------------------------------------------------------
    # Internal: data fetching
    # ------------------------------------------------------------------

    def _get_futures_ohlcv(self) -> Optional[pd.DataFrame]:
        """
        Fetch 5m OHLCV for Nifty Futures, with 60-second caching.
        Source priority:
          1. Native WebSocket (OHLCVBuilder — NIFTY_FUT ticks, volume > 0)
          2. TradingView NIFTY1! (external, session required)
          3. Kotak Neo historical API
        """
        now = datetime.now()
        if (self._cache_df is not None and
                self._cache_ts is not None and
                (now - self._cache_ts).total_seconds() < _CACHE_TTL_SEC):
            return self._cache_df

        df = self._fetch_from_native_ws()
        if df is not None and not df.empty:
            self._cache_df = df
            self._cache_ts = now
            return df

        df = self._fetch_from_tradingview()
        if df is not None and not df.empty:
            self._cache_df = df
            self._cache_ts = now
            return df

        df = self._fetch_from_kotak()
        if df is not None and not df.empty:
            self._cache_df = df
            self._cache_ts = now
            return df

        logger.warning("VWAPEngine: all data sources exhausted — no futures bars")
        return None

    def _fetch_from_native_ws(self) -> Optional[pd.DataFrame]:
        """Fetch NIFTY_FUT 5m OHLCV from live WebSocket OHLCVBuilder."""
        try:
            from core.ohlcv_builder import get_ohlcv_builder
            builder = get_ohlcv_builder()
            if builder is None or not builder.is_ready("NIFTY_FUT", 5):
                return None
            df = builder.get_ohlcv("NIFTY_FUT", 5, n_bars=100)
            if df.empty:
                return None
            # Futures ticks carry volume; validate it
            if "volume" not in df.columns or df["volume"].sum() <= 0:
                logger.debug("VWAPEngine: native NIFTY_FUT has zero volume — skipping")
                return None
            logger.debug(
                f"VWAPEngine: {len(df)} futures bars from native WebSocket "
                f"(vol={df['volume'].sum():,.0f})"
            )
            return df
        except Exception as e:
            logger.debug(f"VWAPEngine native WS fetch failed: {e}")
            return None

    def _fetch_from_tradingview(self) -> Optional[pd.DataFrame]:
        """Fetch Nifty futures 5m bars from TradingView using NIFTY1! symbol."""
        if self._tv is None:
            return None
        try:
            df = self._tv.get_ohlcv(
                symbol=_TV_FUT_SYMBOL,
                exchange=_TV_FUT_EXCHANGE,
                interval_minutes=5,
                n_bars=100,
            )
            if df is None or df.empty:
                return None

            if "volume" not in df.columns:
                logger.warning("VWAPEngine: TradingView futures has no volume column")
                return None

            total_vol = df["volume"].astype(float).sum()
            if total_vol <= 0:
                logger.warning(
                    "VWAPEngine: TradingView futures volume is zero — "
                    "NIFTY1! may not carry volume on your TradingView plan"
                )
                return None

            logger.debug(
                f"VWAPEngine: {len(df)} futures bars from TradingView "
                f"(vol={total_vol:,.0f})"
            )
            return df
        except Exception as e:
            logger.debug(f"VWAPEngine TradingView fetch failed: {e}")
            return None

    def _fetch_from_kotak(self) -> Optional[pd.DataFrame]:
        """Fetch 5m bars from Kotak Neo historical for current-month futures."""
        if self._client is None:
            return None
        try:
            symbol = _current_futures_symbol()
            raw    = self._client.get_history(symbol, "NFO", interval="5m")
            if not isinstance(raw, dict) or not raw.get("data"):
                return None

            df = pd.DataFrame(raw["data"])
            # Normalise column names
            rename = {"o": "open", "h": "high", "l": "low",
                      "c": "close", "v": "volume", "t": "time"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            if "time" in df.columns:
                df.index = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
                df = df.drop(columns=["time"], errors="ignore")

            if "volume" not in df.columns:
                return None

            logger.debug(
                f"VWAPEngine: {len(df)} futures bars from Kotak Neo ({symbol})"
            )
            return df
        except Exception as e:
            logger.debug(f"VWAPEngine Kotak fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal: VWAP computation
    # ------------------------------------------------------------------

    def _compute_vwap_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Compute session-anchored VWAP.
        Resets cumulative sums at start of each trading day (09:15 IST).
        Returns a pd.Series aligned with df.index.
        """
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)

            if "volume" not in df.columns:
                df["volume"] = 1.0   # equal-weight fallback
            df["volume"] = df["volume"].astype(float).clip(lower=0.01)

            df["typical"] = (
                df["high"].astype(float) +
                df["low"].astype(float) +
                df["close"].astype(float)
            ) / 3.0
            df["tpv"] = df["typical"] * df["volume"]

            # Session anchor: group by calendar date, cumsum within each day
            df["_date"] = df.index.normalize()     # truncate to midnight
            df["cum_tpv"] = df.groupby("_date")["tpv"].cumsum()
            df["cum_vol"] = df.groupby("_date")["volume"].cumsum()
            vwap = df["cum_tpv"] / df["cum_vol"].replace(0, float("nan"))

            return vwap
        except Exception as e:
            logger.error(f"VWAPEngine._compute_vwap_series error: {e}")
            return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _current_futures_symbol() -> str:
    """
    Build the current-month Nifty Futures symbol.
    Format: NIFTY<YY><MON>FUT   e.g. NIFTY25APRFUT

    Rolls to the next calendar month after the 25th (approximate expiry
    boundary — actual roll date depends on last Thursday of month).
    """
    today = date.today()
    if today.day > 25:
        import calendar
        yr  = today.year + (1 if today.month == 12 else 0)
        mon = 1 if today.month == 12 else today.month + 1
        roll = today.replace(year=yr, month=mon, day=1)
    else:
        roll = today
    return f"NIFTY{roll.strftime('%y%b').upper()}FUT"   # e.g. NIFTY25APRFUT


def _zero_vwap_features() -> dict:
    return {
        "vwap":             0.0,
        "vwap_dist":        0.0,
        "vwap_dist_pct":    0.0,
        "vwap_trend":       0.0,
        "vwap_std":         0.0,
        "vwap_band_width":  0.0,
        "above_vwap":       0,
        "vwap_avail":       0,
    }
