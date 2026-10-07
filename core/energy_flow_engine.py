"""
energy_flow_engine.py — Market Energy Flow Engine (Strategy 12)
===============================================================
Sequential AND-gate filter chain. Every condition must pass before
the next is evaluated. If any step fails → no signal.

Flow:
  1. VOLUME      → Is volume expanding? (> 1.5x 20-bar avg)
  2. FORCE INDEX → Is volume moving price? (directional)
  3. ATR         → Is volatility expanding? (ATR > 20-bar SMA)
  4. RSI         → Is momentum aligned? (>50 rising = bull, <50 falling = bear)
  5. ADX         → Is trend strong enough? (ADX > 20)
  6. STRUCTURE   → Close breaks 20-bar swing high/low

All 6 gates pass → ENTRY SIGNAL.

SL/TP: Fixed 35/70 pts (2:1 R:R).
Filters: Max 1 per direction per day, max 2 total per day.
Cooldown: 5 bars after SL.
Trail-after-TP: 20 pts behind peak once TP reached.
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


class EnergyFlowEngine(StrategyEngine):
    """Sequential AND-gate energy flow engine."""

    BASE_SL = 35
    BASE_TP = 70
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20
    SWING_LOOKBACK = 20
    COOLDOWN_BARS = 5
    MAX_PER_DIR_DAY = 1
    MAX_PER_DAY = 2

    def __init__(self):
        super().__init__("EFlow")
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
        indicators = {"strategy": "EFlow", "vix": round(vix, 2)}

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

        direction, gate_detail = self._run_gates(df5)
        indicators.update(gate_detail)

        if direction is None:
            return self._skip(gate_detail.get("failed_gate", "no_signal"), indicators)

        indicators["direction"] = direction

        bars_since_sl = self._bar_counter - self._last_sl_bar[direction]
        if bars_since_sl < self.COOLDOWN_BARS:
            return self._skip(f"sl_cooldown ({bars_since_sl}/{self.COOLDOWN_BARS})", indicators)

        if self._daily_dir_count[direction] >= self.MAX_PER_DIR_DAY:
            return self._skip(f"max_dir_day ({direction})", indicators)

        if self._daily_total >= self.MAX_PER_DAY:
            return self._skip("max_daily_total", indicators)

        self._daily_dir_count[direction] += 1
        self._daily_total += 1

        confidence = self._calc_confidence(vix)
        indicators["confidence"] = round(confidence, 3)

        if direction == "CALL":
            p_call = confidence
            p_put = (1.0 - confidence) * 0.10
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators
        else:
            p_put = confidence
            p_call = (1.0 - confidence) * 0.10
            p_skip = 1.0 - p_put - p_call
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

    def _run_gates(self, df5: pd.DataFrame) -> tuple:
        """Run the 6-gate sequential AND chain. Returns (direction, detail)."""
        detail = {}
        close = float(df5["close"].iloc[-1])
        close_prev = float(df5["close"].iloc[-2])

        vol = df5["volume"] if "volume" in df5.columns else pd.Series(1000000, index=df5.index)
        vol_cur = float(vol.iloc[-1])
        vol_avg = float(vol.tail(20).mean())

        # ── Gate 1: VOLUME — Is volume expanding? ──
        vol_expand = vol_cur > 1.5 * vol_avg if vol_avg > 0 else False
        detail["g1_vol_expand"] = vol_expand
        if not vol_expand:
            detail["failed_gate"] = "g1_volume"
            return None, detail

        # ── Gate 2: FORCE INDEX — Is volume moving price? ──
        force = (close - close_prev) * vol_cur
        detail["g2_force_index"] = round(force, 0)
        bull_force = force > 0
        bear_force = force < 0
        if not (bull_force or bear_force):
            detail["failed_gate"] = "g2_force_index"
            return None, detail

        # ── Gate 3: ATR — Is volatility expanding? ──
        atr_series = _compute_atr(df5)
        atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0
        atr_avg = float(atr_series.tail(20).mean()) if not atr_series.tail(20).isna().all() else 0
        atr_expand = atr_val > atr_avg if atr_avg > 0 else False
        detail["g3_atr"] = round(atr_val, 2)
        detail["g3_atr_expand"] = atr_expand
        if not atr_expand:
            detail["failed_gate"] = "g3_atr"
            return None, detail

        # ── Gate 4: RSI — Is momentum aligned? ──
        rsi = _compute_rsi(df5["close"])
        rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
        rsi_prev = float(rsi.iloc[-2]) if not np.isnan(rsi.iloc[-2]) else 50
        detail["g4_rsi"] = round(rsi_val, 1)
        bull_rsi = rsi_val > 50 and rsi_val > rsi_prev
        bear_rsi = rsi_val < 50 and rsi_val < rsi_prev
        if not (bull_rsi or bear_rsi):
            detail["failed_gate"] = "g4_rsi"
            return None, detail

        # ── Gate 5: ADX — Is trend strong enough? ──
        adx = _compute_adx(df5)
        adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 15
        detail["g5_adx"] = round(adx_val, 1)
        if adx_val <= 20:
            detail["failed_gate"] = "g5_adx"
            return None, detail

        # ── Gate 6: STRUCTURE — Breakout of swing high/low ──
        swing_high = float(df5["high"].iloc[-self.SWING_LOOKBACK - 1:-1].max())
        swing_low = float(df5["low"].iloc[-self.SWING_LOOKBACK - 1:-1].min())
        detail["g6_swing_high"] = round(swing_high, 1)
        detail["g6_swing_low"] = round(swing_low, 1)

        bull_breakout = close > swing_high
        bear_breakout = close < swing_low

        # Direction: all gates must agree
        if bull_force and bull_rsi and bull_breakout:
            detail["gates_passed"] = 6
            return "CALL", detail
        elif bear_force and bear_rsi and bear_breakout:
            detail["gates_passed"] = 6
            return "PUT", detail

        detail["failed_gate"] = "g6_structure_or_direction_conflict"
        return None, detail

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

    def _calc_confidence(self, vix: float) -> float:
        base = 0.62
        if vix >= 22:
            base -= 0.03
        return min(0.82, max(0.45, round(base, 3)))
