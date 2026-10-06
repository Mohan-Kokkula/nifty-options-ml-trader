"""
macd_engine.py — MACD Momentum Signal Engine (Strategy 2)
=========================================================
Primary signal: MACD histogram crossover on 5m
Filters:        ADX trend strength (>15), VIX regime
Standalone:     No PSAR confirmation — backtest-proven best without it

Signal logic:
  CALL: MACD histogram crosses above zero on 5m + ADX>15 + VIX filter
  PUT:  MACD histogram crosses below zero on 5m + ADX>15 + VIX filter
  SKIP: no crossover, ADX too low, or time filter
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine

logger = logging.getLogger(__name__)


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9) -> tuple:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_adx(df: pd.DataFrame, period: int = 14):
    """Compute ADX for trend strength."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


class MACDEngine(StrategyEngine):
    """MACD histogram crossover strategy — standalone with ADX + VIX.

    Uses MACD(8,17,9) on 5m as primary signal — tuned for NIFTY intraday.
    Standard MACD(12,26,9) is too slow for 5m bars.
    """

    BASE_SL = 60
    BASE_TP = 120
    MACD_FAST = 8
    MACD_SLOW = 17
    MACD_SIGNAL = 9
    CROSSOVER_BARS = 2  # histogram must have crossed within last N bars

    def __init__(self):
        super().__init__("MACD")
        self._last_vix_mult = 1.0

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {"strategy": "MACD", "vix": round(vix, 2)}

        # Time filter
        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        if len(df5) < 30 or len(df15) < 10:
            return self._skip("insufficient_data", indicators)

        # ── Primary: MACD histogram crossover on 5m ──
        macd_line, signal_line, histogram = _compute_macd(
            df5["close"], self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL
        )

        # Current and previous histogram values
        hist_cur = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        hist_prev2 = float(histogram.iloc[-3]) if len(histogram) > 2 else hist_prev

        indicators["macd_line"] = round(float(macd_line.iloc[-1]), 2)
        indicators["signal_line"] = round(float(signal_line.iloc[-1]), 2)
        indicators["histogram"] = round(hist_cur, 2)
        indicators["histogram_prev"] = round(hist_prev, 2)

        # Detect crossover: histogram changed sign within last CROSSOVER_BARS
        cross_bull = False
        cross_bear = False

        # Check last N bars for zero-cross
        hist_vals = histogram.iloc[-(self.CROSSOVER_BARS + 1):].values
        for i in range(1, len(hist_vals)):
            if hist_vals[i] > 0 and hist_vals[i - 1] <= 0:
                cross_bull = True
            if hist_vals[i] < 0 and hist_vals[i - 1] >= 0:
                cross_bear = True

        indicators["cross_bull"] = cross_bull
        indicators["cross_bear"] = cross_bear

        if not cross_bull and not cross_bear:
            return self._skip("no_macd_crossover", indicators)

        # ── ADX trend strength ──
        adx_series = _compute_adx(df5)
        adx_val = float(adx_series.iloc[-2]) if len(adx_series) > 1 and \
            not np.isnan(adx_series.iloc[-2]) else 25.0
        indicators["adx"] = round(adx_val, 1)

        if adx_val < 15:
            return self._skip(f"adx_too_low ({adx_val:.0f})", indicators)

        # ── VIX multiplier ──
        vix_mult = self._vix_multiplier(vix)
        self._last_vix_mult = vix_mult

        # ── Generate signal (standalone — no PSAR confirmation) ──
        if cross_bull:
            confidence = self._calc_confidence(hist_cur, adx_val, vix)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        if cross_bear:
            confidence = self._calc_confidence(abs(hist_cur), adx_val, vix)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        return self._skip("no_signal", indicators)

    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0

    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0, atr: float = 0.0) -> tuple:
        vix_mult = self._vix_multiplier(vix)

        if atr > 0:
            sl = max(self.SL_ATR_MULT * atr * vix_mult, self.BASE_SL)
            tp = max(self.TP_ATR_MULT * atr * vix_mult, self.BASE_TP)
        else:
            sl = self.BASE_SL * vix_mult
            tp = self.BASE_TP * vix_mult

        if max_loss_budget > 0 and lot_size > 0:
            max_sl = (max_loss_budget / lot_size) * 0.995
            if max_sl < sl:
                ratio = max_sl / sl
                sl = max_sl
                tp = tp * ratio

        return round(sl, 1), round(tp, 1)

    def _calc_confidence(self, hist_magnitude: float, adx: float,
                         vix: float) -> float:
        """Confidence from MACD histogram strength + ADX + VIX."""
        base = 0.50

        if hist_magnitude > 5:
            base += 0.10
        elif hist_magnitude > 2:
            base += 0.05

        if adx > 30:
            base += 0.08
        elif adx > 20:
            base += 0.04

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
