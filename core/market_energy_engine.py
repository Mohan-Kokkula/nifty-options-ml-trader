"""
market_energy_engine.py — Market Energy Score Engine (Strategy 11)
=================================================================
Multi-dimensional scoring system with 7 independent evidence layers.

Score components (max 8 pts):
  1. Volume expansion     +1  (volume > 1.5x 20-bar avg)
  2. Force Index          +1  (price_change * volume, directional)
  3. ATR expansion        +1  (ATR > its 20-bar SMA)
  4. RSI momentum         +1  (RSI>50 rising = bull, <50 falling = bear)
  5. ADX trending         +1  (ADX > 20)
  6. Structure breakout   +2  (close breaks 20-bar swing high/low)
  7. VWAP position        +1  (above daily VWAP = bull, below = bear)

Signal: Score >= 6 AND bull_score > bear_score (or vice versa).
SL/TP: Fixed 35/70 pts (2:1 R:R).
Filters: Max 1 per direction per day, max 2 total per day.
Cooldown: 5 bars after SL.
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine

logger = logging.getLogger(__name__)


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_vwap_daily(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"] if "volume" in df.columns else pd.Series(1000000, index=df.index)
    tp_vol = typical * vol
    vwap = pd.Series(np.nan, index=df.index)
    dates = df.index.date
    for d in sorted(set(dates)):
        mask = dates == d
        cv = vol[mask].cumsum()
        ctv = tp_vol[mask].cumsum()
        vwap[mask] = ctv / cv.replace(0, np.nan)
    return vwap


class MarketEnergyEngine(StrategyEngine):
    """Multi-dimensional Market Energy Score engine."""

    BASE_SL = 35
    BASE_TP = 70
    MIN_SCORE = 6
    SWING_LOOKBACK = 20
    COOLDOWN_BARS = 5
    MAX_PER_DIR_DAY = 1
    MAX_PER_DAY = 2
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20

    def __init__(self):
        super().__init__("MktEnergy")
        self._last_sl_bar = {"CALL": -999, "PUT": -999}
        self._daily_dir_count = {"CALL": 0, "PUT": 0}
        self._daily_total = 0
        self._last_date = None
        self._bar_counter = 0

    def reset_daily(self):
        super().reset_daily()
        self._daily_dir_count = {"CALL": 0, "PUT": 0}
        self._daily_total = 0

    def on_exit(self, direction: str, exit_reason: str):
        if exit_reason == "SL":
            self._last_sl_bar[direction] = self._bar_counter

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        self._bar_counter += 1
        indicators = {"strategy": "MktEnergy", "vix": round(vix, 2)}

        if len(df5) < 60:
            return self._skip("insufficient_data", indicators)

        if vix > 28:
            return self._skip("vix_too_high", indicators)

        if current_hm >= 1500:
            return self._skip("eod_cutoff", indicators)

        spot = float(df5["close"].iloc[-1])
        indicators["spot"] = round(spot, 2)

        today = df5.index[-1].date()
        if self._last_date != today:
            self._daily_dir_count = {"CALL": 0, "PUT": 0}
            self._daily_total = 0
            self._last_date = today

        bull_score, bear_score, component_detail = self._compute_energy(df5)
        indicators["bull_score"] = bull_score
        indicators["bear_score"] = bear_score
        indicators.update(component_detail)

        direction = None
        score = 0
        if bull_score >= self.MIN_SCORE and bull_score > bear_score:
            direction = "CALL"
            score = bull_score
        elif bear_score >= self.MIN_SCORE and bear_score > bull_score:
            direction = "PUT"
            score = bear_score

        if direction is None:
            return self._skip("low_energy", indicators)

        indicators["direction"] = direction
        indicators["score"] = score

        bars_since_sl = self._bar_counter - self._last_sl_bar[direction]
        if bars_since_sl < self.COOLDOWN_BARS:
            return self._skip(f"sl_cooldown ({bars_since_sl}/{self.COOLDOWN_BARS})", indicators)

        if self._daily_dir_count[direction] >= self.MAX_PER_DIR_DAY:
            return self._skip(f"max_dir_day ({direction})", indicators)

        if self._daily_total >= self.MAX_PER_DAY:
            return self._skip("max_daily_total", indicators)

        self._daily_dir_count[direction] += 1
        self._daily_total += 1

        confidence = self._calc_confidence(score, vix)
        indicators["confidence"] = round(confidence, 3)

        if direction == "CALL":
            p_call = confidence
            p_put = (1.0 - confidence) * 0.15
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators
        else:
            p_put = confidence
            p_call = (1.0 - confidence) * 0.15
            p_skip = 1.0 - p_put - p_call
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

    def _compute_energy(self, df5: pd.DataFrame) -> tuple:
        bull = 0
        bear = 0
        detail = {}

        close = float(df5["close"].iloc[-1])
        close_prev = float(df5["close"].iloc[-2])
        high_cur = float(df5["high"].iloc[-1])
        low_cur = float(df5["low"].iloc[-1])

        vol = df5["volume"] if "volume" in df5.columns else pd.Series(1000000, index=df5.index)
        vol_cur = float(vol.iloc[-1])
        vol_avg = float(vol.tail(20).mean())

        # 1. Volume expansion
        vol_expand = vol_cur > 1.5 * vol_avg if vol_avg > 0 else False
        if vol_expand:
            bull += 1
            bear += 1
        detail["vol_expand"] = vol_expand

        # 2. Force Index
        force = (close - close_prev) * vol_cur
        if force > 0:
            bull += 1
        elif force < 0:
            bear += 1
        detail["force_index"] = round(force, 0)

        # 3. ATR expansion
        atr_series = _compute_atr(df5)
        atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0
        atr_avg = float(atr_series.tail(20).mean()) if not atr_series.tail(20).isna().all() else 0
        atr_expand = atr_val > atr_avg if atr_avg > 0 else False
        if atr_expand:
            bull += 1
            bear += 1
        detail["atr"] = round(atr_val, 2)
        detail["atr_expand"] = atr_expand

        # 4. RSI momentum
        rsi = _compute_rsi(df5["close"])
        rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
        rsi_prev = float(rsi.iloc[-2]) if not np.isnan(rsi.iloc[-2]) else 50
        if rsi_val > 50 and rsi_val > rsi_prev:
            bull += 1
        elif rsi_val < 50 and rsi_val < rsi_prev:
            bear += 1
        detail["rsi"] = round(rsi_val, 1)

        # 5. ADX trending
        adx = _compute_adx(df5)
        adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 15
        if adx_val > 20:
            bull += 1
            bear += 1
        detail["adx"] = round(adx_val, 1)

        # 6. Structure breakout (+2)
        swing_high = float(df5["high"].iloc[-self.SWING_LOOKBACK - 1:-1].max())
        swing_low = float(df5["low"].iloc[-self.SWING_LOOKBACK - 1:-1].min())
        if close > swing_high:
            bull += 2
        if close < swing_low:
            bear += 2
        detail["swing_high"] = round(swing_high, 1)
        detail["swing_low"] = round(swing_low, 1)

        # 7. VWAP position
        vwap = _compute_vwap_daily(df5)
        vwap_val = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else close
        if close > vwap_val:
            bull += 1
        elif close < vwap_val:
            bear += 1
        detail["vwap"] = round(vwap_val, 1)

        return bull, bear, detail

    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0, atr: float = 0.0) -> tuple:
        vix_mult = self._vix_multiplier(vix)
        sl = self.BASE_SL * vix_mult
        tp = self.BASE_TP * vix_mult

        if max_loss_budget > 0 and lot_size > 0:
            max_sl = (max_loss_budget / lot_size) * 0.995
            if max_sl < sl:
                ratio = max_sl / sl
                sl = max_sl
                tp = tp * ratio

        return round(sl, 1), round(tp, 1)

    def _calc_confidence(self, score: int, vix: float) -> float:
        base = 0.50

        if score >= 8:
            base += 0.12
        elif score >= 7:
            base += 0.07
        elif score >= 6:
            base += 0.04

        if vix >= 22:
            base -= 0.03

        return min(0.82, max(0.40, round(base, 3)))
