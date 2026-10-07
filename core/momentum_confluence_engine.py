"""
momentum_confluence_engine.py — Momentum Confluence Engine (Strategy 10)
========================================================================
Trend-following momentum strategy using oscillator consensus.

Oscillators:
  1. Chaikin Oscillator (3,10) — accumulation/distribution momentum
  2. PMO (Price Momentum Oscillator 20,10,35) — double-smoothed ROC

Signal: Both oscillators must agree on direction.
  - CALL: Chaikin > 0 AND PMO line > PMO signal
  - PUT:  Chaikin < 0 AND PMO line < PMO signal

Filters:
  - ADX > 20 (trending market — opposite of S9's range requirement)
  - Cooldown: 5 bars after SL before re-entry in same direction
  - Max 1 trade per direction per day (quality over quantity)
  - No time filters (momentum persists through lunch)

SL/TP: Fixed 35/70 pts (no ATR scaling). Tight SL exits fast,
2:1 R:R. Backtested: 116 trades, WR 47.4%, PF 1.81, MaxDD -220.
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


def _compute_chaikin_osc(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, np.nan)
    clv = clv.fillna(0)
    vol = df["volume"] if "volume" in df.columns else pd.Series(1000000, index=df.index)
    ad = (clv * vol).cumsum()
    return ad.ewm(span=fast, adjust=False).mean() - ad.ewm(span=slow, adjust=False).mean()


def _compute_pmo(series: pd.Series, smooth1: int = 20, smooth2: int = 10,
                 signal_period: int = 35) -> tuple:
    roc = ((series / series.shift(1)) - 1) * 100
    pmo_raw = roc.ewm(span=smooth1, adjust=False).mean()
    pmo_line = pmo_raw.ewm(span=smooth2, adjust=False).mean()
    pmo_signal = pmo_line.ewm(span=signal_period, adjust=False).mean()
    return pmo_line, pmo_signal


class MomentumConfluenceEngine(StrategyEngine):
    """Chaikin + PMO dual momentum consensus."""

    BASE_SL = 35
    BASE_TP = 70
    SL_ATR_MULT = 0  # fixed SL/TP, no ATR scaling
    TP_ATR_MULT = 0

    ADX_MIN = 20
    COOLDOWN_BARS = 5
    MAX_PER_DIR_DAY = 1

    def __init__(self):
        super().__init__("MomConf")
        self._last_sl_bar = {"CALL": -999, "PUT": -999}
        self._daily_dir_count = {"CALL": 0, "PUT": 0}
        self._last_date = None
        self._bar_counter = 0

    def reset_daily(self):
        super().reset_daily()
        self._daily_dir_count = {"CALL": 0, "PUT": 0}

    def on_exit(self, direction: str, exit_reason: str):
        """Called by router when position exits — tracks SL cooldown."""
        if exit_reason == "SL":
            self._last_sl_bar[direction] = self._bar_counter

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        self._bar_counter += 1
        indicators = {"strategy": "MomConf", "vix": round(vix, 2)}

        if len(df5) < 60:
            return self._skip("insufficient_data", indicators)

        if vix > 28:
            return self._skip("vix_too_high", indicators)

        # No time filter except EOD cutoff
        if current_hm >= 1500:
            return self._skip("eod_cutoff", indicators)

        spot = float(df5["close"].iloc[-1])
        indicators["spot"] = round(spot, 2)

        today = df5.index[-1].date()
        if self._last_date != today:
            self._daily_dir_count = {"CALL": 0, "PUT": 0}
            self._last_date = today

        # ADX — must be trending
        adx_series = _compute_adx(df5)
        adx_val = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else 15.0
        indicators["adx"] = round(adx_val, 1)

        if adx_val < self.ADX_MIN:
            return self._skip(f"adx_low ({adx_val:.0f})", indicators)

        # ATR for SL/TP
        atr = _compute_atr(df5)
        atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0
        indicators["atr"] = round(atr_val, 2)

        # Chaikin Oscillator
        chaikin = _compute_chaikin_osc(df5)
        chaikin_val = float(chaikin.iloc[-1]) if not np.isnan(chaikin.iloc[-1]) else 0
        indicators["chaikin"] = round(chaikin_val, 2)

        # PMO
        pmo_line, pmo_signal = _compute_pmo(df5["close"])
        pmo_l = float(pmo_line.iloc[-1]) if not np.isnan(pmo_line.iloc[-1]) else 0
        pmo_s = float(pmo_signal.iloc[-1]) if not np.isnan(pmo_signal.iloc[-1]) else 0
        indicators["pmo_line"] = round(pmo_l, 4)
        indicators["pmo_signal"] = round(pmo_s, 4)

        # Consensus
        chaikin_bull = chaikin_val > 0
        chaikin_bear = chaikin_val < 0
        pmo_bull = pmo_l > pmo_s
        pmo_bear = pmo_l < pmo_s

        direction = None
        if chaikin_bull and pmo_bull:
            direction = "CALL"
        elif chaikin_bear and pmo_bear:
            direction = "PUT"

        if direction is None:
            return self._skip("no_consensus", indicators)

        indicators["direction"] = direction

        # Cooldown after SL
        bars_since_sl = self._bar_counter - self._last_sl_bar[direction]
        if bars_since_sl < self.COOLDOWN_BARS:
            indicators["cooldown_remaining"] = self.COOLDOWN_BARS - bars_since_sl
            return self._skip(f"sl_cooldown ({bars_since_sl}/{self.COOLDOWN_BARS})", indicators)

        # Max per direction per day
        if self._daily_dir_count[direction] >= self.MAX_PER_DIR_DAY:
            return self._skip(f"max_dir_day ({direction})", indicators)

        self._daily_dir_count[direction] += 1

        # Confidence
        confidence = self._calc_confidence(adx_val, chaikin_val, pmo_l, pmo_s, vix)
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

    def _calc_confidence(self, adx: float, chaikin: float,
                         pmo_line: float, pmo_signal: float,
                         vix: float) -> float:
        base = 0.52

        # ADX strength — stronger trend = higher confidence
        if adx > 35:
            base += 0.08
        elif adx > 28:
            base += 0.05
        elif adx > 22:
            base += 0.02

        # PMO gap — wider gap between line and signal = stronger momentum
        pmo_gap = abs(pmo_line - pmo_signal)
        if pmo_gap > 0.1:
            base += 0.06
        elif pmo_gap > 0.05:
            base += 0.03

        # Chaikin magnitude
        chaikin_abs = abs(chaikin)
        if chaikin_abs > 5000000:
            base += 0.05
        elif chaikin_abs > 2000000:
            base += 0.02

        if vix >= 22:
            base -= 0.03

        return min(0.82, max(0.40, round(base, 3)))
