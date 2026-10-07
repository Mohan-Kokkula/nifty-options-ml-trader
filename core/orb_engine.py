"""
orb_engine.py — Opening Range Breakout Engine (Strategy 6)
==========================================================
Crabel-style ORB: break of first 30-min range on NR4 compression days,
confirmed by Supertrend Parameter Sensitivity consensus (100 combos).

Backtest: PF 3.20, WR 66.7%, MaxDD -100pts over 69 trading days

Signal logic:
  CALL: Price breaks above 30-min OR high
        + NR4 (prior day was narrowest-range in 4 days)
        + Supertrend consensus >= 70% bullish
  PUT:  Price breaks below 30-min OR low
        + NR4 + Supertrend consensus >= 70% bearish
  SKIP: no breakout, no NR4, consensus weak, or time filter
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine
from core.supertrend_engine import compute_supertrend

logger = logging.getLogger(__name__)


class ORBEngine(StrategyEngine):
    """Opening Range Breakout + NR4 + Supertrend Sensitivity consensus."""

    BASE_SL = 50
    BASE_TP = 100
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0
    CONSENSUS_THRESHOLD = 0.70
    OR_BARS = 6  # 6 x 5min = 30 min opening range
    OR_MIN_RANGE = 10  # minimum OR width in points

    LEN_START = 5
    LEN_STEP = 1
    MULT_START = 1.0
    MULT_STEP = 0.1
    GRID_SIZE = 10

    def __init__(self):
        super().__init__("ORB")
        self._or_high = None
        self._or_low = None
        self._or_date = None
        self._breakout_done = False
        self._st_cache_date = None
        self._st_directions = None

    def _compute_st_grid(self, df5: pd.DataFrame):
        """Compute 100-combo Supertrend grid, cached per day."""
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
        """Get (bull_pct, bear_pct) at a specific bar index."""
        bull = sum(1 for d in self._st_directions.values() if d[bar_idx] == 1)
        total = len(self._st_directions)
        bull_pct = bull / total
        return bull_pct, 1.0 - bull_pct

    def _is_nr4(self, df5: pd.DataFrame) -> bool:
        """Check if previous day was NR4 (narrowest range in last 4 days)."""
        dates = sorted(set(df5.index.date))
        if len(dates) < 5:
            return False
        day_ranges = {}
        for d in dates[-5:]:
            day_bars = df5[df5.index.date == d]
            day_ranges[d] = float(day_bars["high"].max()) - float(day_bars["low"].min())
        last_4 = [day_ranges[d] for d in dates[-5:-1]]
        prev_day = dates[-2]
        return day_ranges[prev_day] == min(last_4)

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {"strategy": "ORB", "vix": round(vix, 2)}

        if len(df5) < 50:
            return self._skip("insufficient_data", indicators)

        today = df5.index[-1].date()

        # Reset per-day state
        if self._or_date != today:
            self._or_date = today
            self._breakout_done = False
            today_bars = df5[df5.index.date == today]
            if len(today_bars) >= self.OR_BARS:
                or_data = today_bars.iloc[:self.OR_BARS]
                self._or_high = float(or_data["high"].max())
                self._or_low = float(or_data["low"].min())
            else:
                self._or_high = None
                self._or_low = None

        if self._or_high is None:
            return self._skip("or_not_formed", indicators)

        or_width = self._or_high - self._or_low
        indicators["or_high"] = round(self._or_high, 2)
        indicators["or_low"] = round(self._or_low, 2)
        indicators["or_width"] = round(or_width, 2)

        if or_width < self.OR_MIN_RANGE:
            return self._skip(f"or_too_narrow ({or_width:.0f})", indicators)

        if self._breakout_done:
            return self._skip("breakout_already_taken", indicators)

        # Only trade after OR is formed (09:45+) and before 15:00
        if current_hm < 945:
            return self._skip("or_forming", indicators)
        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        # NR4 filter
        is_nr4 = self._is_nr4(df5)
        indicators["is_nr4"] = is_nr4
        if not is_nr4:
            return self._skip("not_nr4", indicators)

        # VIX filter
        if vix > 28:
            return self._skip("vix_too_high", indicators)
        vix_mult = self._vix_multiplier(vix)

        spot = float(df5["close"].iloc[-1])
        indicators["spot"] = round(spot, 2)

        # Supertrend consensus
        self._compute_st_grid(df5)
        bar_idx = len(df5) - 1
        bull_pct, bear_pct = self._get_consensus(bar_idx)
        indicators["st_bull_pct"] = round(bull_pct, 2)
        indicators["st_bear_pct"] = round(bear_pct, 2)

        # Breakout detection
        if spot > self._or_high and bull_pct >= self.CONSENSUS_THRESHOLD:
            self._breakout_done = True
            confidence = self._calc_confidence(bull_pct, or_width, vix)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        if spot < self._or_low and bear_pct >= self.CONSENSUS_THRESHOLD:
            self._breakout_done = True
            confidence = self._calc_confidence(bear_pct, or_width, vix)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        return self._skip("no_breakout", indicators)

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

    def _calc_confidence(self, consensus_pct: float, or_width: float,
                         vix: float) -> float:
        base = 0.55
        if consensus_pct >= 0.90:
            base += 0.12
        elif consensus_pct >= 0.80:
            base += 0.08
        elif consensus_pct >= 0.70:
            base += 0.05

        if or_width > 50:
            base += 0.05
        elif or_width > 30:
            base += 0.03

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
