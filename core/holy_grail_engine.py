"""
holy_grail_engine.py — Raschke Holy Grail Engine (Strategy 8)
=============================================================
ADX > 35 (strong trend) + EMA20 pullback + Supertrend Sensitivity
consensus as directional confirmation.

Backtest: PF 2.30, WR 60.7%, +Rs.1,23,360 over 69 trading days

Signal logic:
  CALL: ADX > 35 (strong uptrend)
        + Price pulled back to EMA20 within last 5 bars
        + Bullish candle (close > open) bouncing off EMA20
        + EMA20 rising (current > 5 bars ago)
        + Supertrend consensus >= 70% bullish
  PUT:  ADX > 35 (strong downtrend)
        + Price pulled back to EMA20 within last 5 bars
        + Bearish candle (close < open) rejecting from EMA20
        + EMA20 falling + ST consensus >= 70% bearish
  SKIP: ADX < 35, no EMA touch, or consensus weak
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine
from core.supertrend_engine import compute_supertrend

logger = logging.getLogger(__name__)


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


class HolyGrailEngine(StrategyEngine):
    """ADX>35 + EMA20 pullback + Supertrend Sensitivity consensus."""

    BASE_SL = 60
    BASE_TP = 120
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0
    ADX_THRESHOLD = 35
    EMA_PERIOD = 20
    LOOKBACK_BARS = 5  # how far back to check for EMA20 touch
    CONSENSUS_THRESHOLD = 0.70

    LEN_START = 5
    LEN_STEP = 1
    MULT_START = 1.0
    MULT_STEP = 0.1
    GRID_SIZE = 10

    def __init__(self):
        super().__init__("HolyGrail")
        self._st_cache_date = None
        self._st_directions = None

    def _compute_st_grid(self, df5: pd.DataFrame):
        today = df5.index[-1].date()
        if self._st_cache_date == today and self._st_directions is not None:
            return
        dirs = {}
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                atr_len = self.LEN_START + i * self.LEN_STEP
                mult = round(self.MULT_START + j * self.MULT_STEP, 1)
                dirs[(atr_len, mult)] = compute_supertrend(df5, atr_len, mult).values
        self._st_directions = dirs
        self._st_cache_date = today

    def _get_consensus(self, bar_idx: int) -> tuple:
        bull = sum(1 for d in self._st_directions.values() if d[bar_idx] == 1)
        total = len(self._st_directions)
        bull_pct = bull / total
        return bull_pct, 1.0 - bull_pct

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {"strategy": "HolyGrail", "vix": round(vix, 2)}

        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        if len(df5) < 50:
            return self._skip("insufficient_data", indicators)

        # VIX filter
        if vix > 28:
            return self._skip("vix_too_high", indicators)
        vix_mult = self._vix_multiplier(vix)

        # ADX
        adx_series = _compute_adx(df5)
        adx_val = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else 0
        indicators["adx"] = round(adx_val, 1)

        if adx_val < self.ADX_THRESHOLD:
            return self._skip(f"adx_low ({adx_val:.0f})", indicators)

        # EMA20
        ema20 = df5["close"].ewm(span=self.EMA_PERIOD, adjust=False).mean()
        ema_cur = float(ema20.iloc[-1])
        ema_prev = float(ema20.iloc[-self.LOOKBACK_BARS - 1]) if len(ema20) > self.LOOKBACK_BARS else ema_cur
        spot = float(df5["close"].iloc[-1])
        indicators["ema20"] = round(ema_cur, 2)
        indicators["spot"] = round(spot, 2)

        # EMA20 touch check: price touched EMA within last LOOKBACK_BARS
        touched = False
        for j in range(1, self.LOOKBACK_BARS + 1):
            idx = len(df5) - 1 - j
            if idx < 0:
                break
            bar_low = float(df5["low"].iloc[idx])
            bar_high = float(df5["high"].iloc[idx])
            ema_at = float(ema20.iloc[idx])
            if bar_low <= ema_at <= bar_high:
                touched = True
                break
        indicators["ema_touched"] = touched

        if not touched:
            return self._skip("no_ema_pullback", indicators)

        # Current bar direction
        bar_body = float(df5["close"].iloc[-1]) - float(df5["open"].iloc[-1])
        indicators["bar_body"] = round(bar_body, 2)

        # Supertrend consensus
        self._compute_st_grid(df5)
        bar_idx = len(df5) - 1
        bull_pct, bear_pct = self._get_consensus(bar_idx)
        indicators["st_bull_pct"] = round(bull_pct, 2)
        indicators["st_bear_pct"] = round(bear_pct, 2)

        # EMA20 direction
        ema_rising = ema_cur > ema_prev
        ema_falling = ema_cur < ema_prev
        indicators["ema_rising"] = ema_rising

        # CALL: above EMA, bullish bar, EMA rising, ST consensus
        if spot > ema_cur and bar_body > 0 and ema_rising:
            if bull_pct < self.CONSENSUS_THRESHOLD:
                return self._skip(f"st_weak_bull ({bull_pct:.0%})", indicators)
            confidence = self._calc_confidence(adx_val, bull_pct, vix)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        # PUT: below EMA, bearish bar, EMA falling, ST consensus
        if spot < ema_cur and bar_body < 0 and ema_falling:
            if bear_pct < self.CONSENSUS_THRESHOLD:
                return self._skip(f"st_weak_bear ({bear_pct:.0%})", indicators)
            confidence = self._calc_confidence(adx_val, bear_pct, vix)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        return self._skip("no_hg_setup", indicators)

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

    def _calc_confidence(self, adx: float, consensus_pct: float,
                         vix: float) -> float:
        base = 0.55

        if adx > 50:
            base += 0.10
        elif adx > 40:
            base += 0.06

        if consensus_pct >= 0.90:
            base += 0.12
        elif consensus_pct >= 0.80:
            base += 0.08
        elif consensus_pct >= 0.70:
            base += 0.05

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
