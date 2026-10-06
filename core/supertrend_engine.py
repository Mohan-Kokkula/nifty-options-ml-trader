"""
supertrend_engine.py — PSAR + Supertrend Sensitivity Engine (Strategy 5)
========================================================================
Dual-confirmation: PSAR flip + Supertrend Parameter Sensitivity consensus.

Supertrend Sensitivity tests a 10x10 grid of (ATR_period, Multiplier)
parameter combos and counts how many are bullish vs bearish. This gives
a robust consensus view that's less sensitive to any single parameter choice.

Backtest-optimized: 70% consensus + EMA40 + FLAT filter
  PF 1.33, WR 51.5%, +370pts (+Rs.27,754/lot) over 69 trading days

Signal logic:
  CALL: 5m PSAR flip bullish + ST consensus >=70% bullish
        + spot > EMA40
  PUT:  5m PSAR flip bearish + ST consensus >=70% bearish
        + spot < EMA40 + not FLAT day
  SKIP: no PSAR flip, consensus weak, EMA wrong side, or time filter
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine
from core.psar_engine import compute_psar

logger = logging.getLogger(__name__)


def compute_supertrend(df: pd.DataFrame, atr_period: int = 10,
                       multiplier: float = 3.0):
    """Compute Supertrend for a single parameter set.
    Returns direction Series: 1=bullish, -1=bearish."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    # ATR using Wilder's smoothing (RMA)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    atr = np.empty(n)
    atr[:atr_period] = np.nan
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    alpha = 1.0 / atr_period
    for i in range(atr_period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    avg_hl = (high + low) / 2.0
    basic_upper = avg_hl + multiplier * atr
    basic_lower = avg_hl - multiplier * atr

    upper = np.empty(n)
    lower = np.empty(n)
    direction = np.ones(n, dtype=int)

    upper[0] = basic_upper[0]
    lower[0] = basic_lower[0]

    for i in range(1, n):
        lower[i] = max(basic_lower[i], lower[i - 1]) if close[i - 1] > lower[i - 1] else basic_lower[i]
        upper[i] = min(basic_upper[i], upper[i - 1]) if close[i - 1] < upper[i - 1] else basic_upper[i]

        if direction[i - 1] == -1 and close[i] > upper[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return pd.Series(direction, index=df.index, name="st_dir")


def supertrend_consensus(df: pd.DataFrame,
                         len_start: int = 5, len_step: int = 1,
                         mult_start: float = 1.0, mult_step: float = 0.1,
                         grid_size: int = 10):
    """Compute Supertrend across a grid of parameters.
    Returns (bullish_pct, bearish_pct, stable_direction, stable_params)."""
    n_combos = grid_size * grid_size
    bullish = 0
    bearish = 0
    directions = np.zeros((grid_size, grid_size), dtype=int)

    for i in range(grid_size):
        atr_len = len_start + i * len_step
        for j in range(grid_size):
            mult = mult_start + j * mult_step
            st_dir = compute_supertrend(df, atr_len, mult)
            d = int(st_dir.iloc[-2]) if len(st_dir) > 1 else int(st_dir.iloc[-1])
            directions[i, j] = d
            if d == 1:
                bullish += 1
            else:
                bearish += 1

    bull_pct = bullish / n_combos
    bear_pct = bearish / n_combos

    # Find most stable area (3x3 neighborhood with highest avg - lowest variance)
    stable_i, stable_j = 1, 1
    max_score = -np.inf
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            neighborhood = directions[i-1:i+2, j-1:j+2].flatten().astype(float)
            avg = np.mean(neighborhood)
            std = np.std(neighborhood)
            score = abs(avg) - std * 1.5
            if score > max_score:
                max_score = score
                stable_i, stable_j = i, j

    stable_dir = directions[stable_i, stable_j]
    stable_params = {
        "atr_length": len_start + stable_i * len_step,
        "multiplier": round(mult_start + stable_j * mult_step, 1),
    }

    return bull_pct, bear_pct, stable_dir, stable_params


class SupertrendEngine(StrategyEngine):
    """PSAR + Supertrend Sensitivity dual-confirmation strategy.

    Waits for PSAR flip on 5m, then checks if Supertrend parameter
    sensitivity consensus (100 combos) agrees on direction.
    EMA40 side filter + FLAT day PUT filter (same as PSAR engine).
    """

    BASE_SL = 60
    BASE_TP = 120
    CONSENSUS_THRESHOLD = 0.70
    FLIP_MAX_BARS = 1
    EMA_PERIOD = 40
    FLAT_THRESHOLD = 40

    LEN_START = 5
    LEN_STEP = 1
    MULT_START = 1.0
    MULT_STEP = 0.1
    GRID_SIZE = 10

    def __init__(self):
        super().__init__("SupertrendSensitivity")
        self._last_vix_mult = 1.0

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {"strategy": "SupertrendSensitivity", "vix": round(vix, 2)}

        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        if len(df5) < 30:
            return self._skip("insufficient_data", indicators)

        # ── PSAR flip check on 5m ──
        _, psar_dir, psar_flip = compute_psar(df5, af_start=0.03, af_step=0.02, af_max=0.2)
        confirmed_idx = -2 if len(df5) > 5 else -1
        flip_val = int(psar_flip.iloc[confirmed_idx])
        psar_d = int(psar_dir.iloc[confirmed_idx])

        bars_since_flip = 0
        dirs = psar_dir.values[:confirmed_idx + len(df5) if confirmed_idx < 0 else confirmed_idx + 1]
        for i in range(len(dirs) - 2, -1, -1):
            if int(dirs[i]) != psar_d:
                break
            bars_since_flip += 1

        indicators["psar_dir"] = psar_d
        indicators["psar_flip"] = flip_val
        indicators["bars_since_flip"] = bars_since_flip

        if bars_since_flip > self.FLIP_MAX_BARS:
            return self._skip(f"no_psar_flip (bars={bars_since_flip})", indicators)

        # ── Supertrend Parameter Sensitivity consensus ──
        bull_pct, bear_pct, stable_dir, stable_params = supertrend_consensus(
            df5, self.LEN_START, self.LEN_STEP,
            self.MULT_START, self.MULT_STEP, self.GRID_SIZE
        )

        indicators["st_bull_pct"] = round(bull_pct, 2)
        indicators["st_bear_pct"] = round(bear_pct, 2)
        indicators["st_stable_dir"] = stable_dir
        indicators["st_stable_params"] = stable_params

        # ── VIX filter ──
        if vix > 28:
            return self._skip("vix_too_high", indicators)

        vix_mult = self._vix_multiplier(vix)
        self._last_vix_mult = vix_mult

        # ── EMA40 side filter ──
        ema_val = float(df5["close"].ewm(span=self.EMA_PERIOD, adjust=False).mean().iloc[-1])
        spot = float(df5["close"].iloc[-1])
        indicators["ema40"] = round(ema_val, 2)
        indicators["spot"] = round(spot, 2)

        # ── FLAT day detection for PUT filter ──
        today = df5.index[-1].date()
        today_bars = df5[df5.index.date == today]
        day_open = float(today_bars.iloc[0]["open"]) if not today_bars.empty else spot
        is_flat = abs(spot - day_open) <= self.FLAT_THRESHOLD
        indicators["day_open"] = round(day_open, 2)
        indicators["is_flat"] = is_flat

        # ── Dual confirmation: PSAR flip + Supertrend consensus + EMA40 ──
        if psar_d == 1 and bull_pct >= self.CONSENSUS_THRESHOLD:
            if spot < ema_val:
                return self._skip(f"ema_side_call (spot={spot:.0f} < EMA={ema_val:.0f})", indicators)
            confidence = self._calc_confidence(bull_pct, stable_dir == 1, vix)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        if psar_d == -1 and bear_pct >= self.CONSENSUS_THRESHOLD:
            if is_flat:
                return self._skip(f"flat_day_put (diff={abs(spot-day_open):.0f}pts)", indicators)
            if spot > ema_val:
                return self._skip(f"ema_side_put (spot={spot:.0f} > EMA={ema_val:.0f})", indicators)
            confidence = self._calc_confidence(bear_pct, stable_dir == -1, vix)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        return self._skip("psar_st_disagree", indicators)

    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0) -> tuple:
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

    def _calc_confidence(self, consensus_pct: float, stable_agrees: bool,
                         vix: float) -> float:
        """Confidence from consensus strength + stable parameter agreement."""
        base = 0.50

        if consensus_pct >= 0.90:
            base += 0.15
        elif consensus_pct >= 0.80:
            base += 0.10
        elif consensus_pct >= 0.70:
            base += 0.06

        if stable_agrees:
            base += 0.05

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
