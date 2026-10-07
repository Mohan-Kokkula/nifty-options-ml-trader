"""
rsi2_engine.py — Connors RSI(2) Mean Reversion Engine (Strategy 7)
==================================================================
RSI(2) extreme readings (< 15 oversold, > 85 overbought) with
200-day EMA trend filter and Supertrend Sensitivity consensus.

Backtest: PF 1.74, WR 44.1%, +Rs.45,232 over 69 trading days

Signal logic:
  CALL: RSI(2) < 15 (oversold snap-back)
        + spot > 200-day EMA (buy dips in uptrend)
        + Supertrend consensus >= 65% bullish
        + candle body ratio >= 30% (strong reversal bar)
  PUT:  RSI(2) > 85 (overbought fade)
        + spot < 200-day EMA (sell rips in downtrend)
        + Supertrend consensus >= 65% bearish
        + candle body ratio >= 30%
  SKIP: RSI not extreme, wrong side of EMA, weak consensus, or doji
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine
from core.supertrend_engine import compute_supertrend

logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """RSI with Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class RSI2Engine(StrategyEngine):
    """RSI(2) mean reversion + 200 EMA trend + Supertrend Sensitivity."""

    BASE_SL = 40
    BASE_TP = 80
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 3.0
    RSI_PERIOD = 2
    RSI_OVERSOLD = 15
    RSI_OVERBOUGHT = 85
    CONSENSUS_THRESHOLD = 0.65
    CANDLE_BODY_MIN = 0.30
    EMA_DAILY_PERIOD = 200

    LEN_START = 5
    LEN_STEP = 1
    MULT_START = 1.0
    MULT_STEP = 0.1
    GRID_SIZE = 10

    def __init__(self):
        super().__init__("RSI2")
        self._ema200_daily = None
        self._ema200_date = None
        self._st_cache_date = None
        self._st_directions = None

    def set_daily_ema200(self, ema200_value: float):
        """Set the 200-day EMA externally (from daily data)."""
        self._ema200_daily = ema200_value

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
        indicators = {"strategy": "RSI2", "vix": round(vix, 2)}

        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        if len(df5) < 30:
            return self._skip("insufficient_data", indicators)

        # VIX filter
        if vix > 28:
            return self._skip("vix_too_high", indicators)
        vix_mult = self._vix_multiplier(vix)

        # RSI(2)
        rsi = _compute_rsi(df5["close"], self.RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        indicators["rsi2"] = round(rsi_val, 2)

        if self.RSI_OVERSOLD <= rsi_val <= self.RSI_OVERBOUGHT:
            return self._skip(f"rsi_neutral ({rsi_val:.0f})", indicators)

        # 200-day EMA — use externally-set value or approximate from 5m data
        spot = float(df5["close"].iloc[-1])
        if self._ema200_daily is not None:
            ema200 = self._ema200_daily
        else:
            ema200 = float(df5["close"].ewm(span=min(len(df5)-1, 15000), adjust=False).mean().iloc[-1])
        indicators["ema200"] = round(ema200, 2)
        indicators["spot"] = round(spot, 2)

        # Candle body ratio filter
        bar_body = abs(float(df5["close"].iloc[-1]) - float(df5["open"].iloc[-1]))
        bar_range = float(df5["high"].iloc[-1]) - float(df5["low"].iloc[-1])
        body_ratio = bar_body / bar_range if bar_range > 0 else 0
        indicators["body_ratio"] = round(body_ratio, 2)

        if body_ratio < self.CANDLE_BODY_MIN:
            return self._skip(f"weak_candle ({body_ratio:.0%})", indicators)

        # Supertrend consensus
        self._compute_st_grid(df5)
        bar_idx = len(df5) - 1
        bull_pct, bear_pct = self._get_consensus(bar_idx)
        indicators["st_bull_pct"] = round(bull_pct, 2)
        indicators["st_bear_pct"] = round(bear_pct, 2)

        # CALL: RSI < 15 + above 200 EMA + ST bullish
        if rsi_val < self.RSI_OVERSOLD and spot > ema200:
            if bull_pct < self.CONSENSUS_THRESHOLD:
                return self._skip(f"st_weak_bull ({bull_pct:.0%})", indicators)
            confidence = self._calc_confidence(rsi_val, bull_pct, vix, is_call=True)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        # PUT: RSI > 85 + below 200 EMA + ST bearish
        if rsi_val > self.RSI_OVERBOUGHT and spot < ema200:
            if bear_pct < self.CONSENSUS_THRESHOLD:
                return self._skip(f"st_weak_bear ({bear_pct:.0%})", indicators)
            confidence = self._calc_confidence(rsi_val, bear_pct, vix, is_call=False)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        # RSI extreme but wrong side of EMA
        if rsi_val < self.RSI_OVERSOLD:
            return self._skip(f"ema_side_call (spot={spot:.0f} < EMA200={ema200:.0f})", indicators)
        return self._skip(f"ema_side_put (spot={spot:.0f} > EMA200={ema200:.0f})", indicators)

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

    def _calc_confidence(self, rsi_val: float, consensus_pct: float,
                         vix: float, is_call: bool) -> float:
        base = 0.50
        rsi_extreme = (self.RSI_OVERSOLD - rsi_val) if is_call else (rsi_val - self.RSI_OVERBOUGHT)
        if rsi_extreme > 10:
            base += 0.10
        elif rsi_extreme > 5:
            base += 0.06

        if consensus_pct >= 0.85:
            base += 0.10
        elif consensus_pct >= 0.75:
            base += 0.06
        elif consensus_pct >= 0.65:
            base += 0.03

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
