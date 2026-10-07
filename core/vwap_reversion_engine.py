"""
vwap_reversion_engine.py — VWAP Mean Reversion Engine (Strategy 4)
==================================================================
Primary signal: Price deviates from VWAP by threshold, then reverses
Confirmation:   PSAR direction supports the reversion
Context:        VIX regime, volume confirmation

Signal logic:
  CALL: Price drops > N pts below VWAP
        + 5m candle shows reversal (close > open, bullish)
        + PSAR 15m is bullish (trend supports bounce)
  PUT:  Price rises > N pts above VWAP
        + 5m candle shows reversal (close < open, bearish)
        + PSAR 15m is bearish (trend supports fade)
  SKIP: no deviation, no reversal, or time filter

VWAP = cumulative (price * volume) / cumulative volume from day open.
Mean reversion: price tends to return to VWAP (institutional fair value).
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine
from core.psar_engine import compute_psar

logger = logging.getLogger(__name__)


def _compute_vwap(df_today: pd.DataFrame) -> pd.Series:
    """Compute VWAP from today's bars (anchored to day open)."""
    typical_price = (df_today["high"] + df_today["low"] + df_today["close"]) / 3
    cum_tp_vol = (typical_price * df_today["volume"]).cumsum()
    cum_vol = df_today["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _psar_direction(df: pd.DataFrame, af_start=0.03, af_step=0.02,
                    af_max=0.2) -> int:
    """Get PSAR direction: 1=bullish, -1=bearish, 0=unknown."""
    if df is None or df.empty or len(df) < 5:
        return 0
    psar_val, psar_dir, _ = compute_psar(df, af_start, af_step, af_max)
    idx = -2 if len(df) > 5 else -1
    return int(psar_dir.iloc[idx])


class VWAPReversionEngine(StrategyEngine):
    """VWAP mean reversion strategy with PSAR confirmation.

    Trades when price deviates significantly from VWAP and shows
    reversal candle pattern, confirmed by PSAR trend direction.
    """

    BASE_SL = 40   # tighter SL for reversion trades
    BASE_TP = 60   # target is VWAP (not a runaway move)
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20
    DEVIATION_PTS = 30  # min distance from VWAP to trigger
    MIN_BARS_TODAY = 6  # need ~30 min of data for meaningful VWAP

    def __init__(self):
        super().__init__("VWAP")
        self._last_vix_mult = 1.0

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {"strategy": "VWAP", "vix": round(vix, 2)}

        # Time filter
        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        # Need bars after open to compute VWAP — skip first 30 min
        if current_hm < 950:
            return self._skip("vwap_warmup", indicators)

        if len(df5) < 10 or len(df15) < 5:
            return self._skip("insufficient_data", indicators)

        # ── Compute VWAP from today's bars ──
        today = df5.index[-1].date()
        df_today = df5[df5.index.date == today].copy()
        if len(df_today) < self.MIN_BARS_TODAY:
            return self._skip("not_enough_today_bars", indicators)

        vwap = _compute_vwap(df_today)
        vwap_val = float(vwap.iloc[-1])
        spot = float(df5["close"].iloc[-1])
        deviation = spot - vwap_val
        dev_abs = abs(deviation)

        indicators["vwap"] = round(vwap_val, 2)
        indicators["spot"] = round(spot, 2)
        indicators["deviation"] = round(deviation, 2)
        indicators["deviation_abs"] = round(dev_abs, 2)

        # VIX-scaled deviation threshold
        vix_mult = self._vix_multiplier(vix)
        self._last_vix_mult = vix_mult
        threshold = self.DEVIATION_PTS * vix_mult

        indicators["threshold"] = round(threshold, 1)

        if dev_abs < threshold:
            return self._skip(f"deviation_small ({dev_abs:.0f} < {threshold:.0f})", indicators)

        # ── Reversal candle check ──
        # Current bar should show reversal toward VWAP
        cur_bar = df5.iloc[-1]
        bar_open = float(cur_bar["open"])
        bar_close = float(cur_bar["close"])
        bar_body = bar_close - bar_open  # positive = bullish candle

        # Previous bar for momentum check
        prev_close = float(df5["close"].iloc[-2])

        indicators["bar_body"] = round(bar_body, 2)

        # Below VWAP → need bullish reversal candle
        if deviation < -threshold and bar_body > 0 and bar_close > prev_close:
            reversal = "CALL"
        # Above VWAP → need bearish reversal candle
        elif deviation > threshold and bar_body < 0 and bar_close < prev_close:
            reversal = "PUT"
        else:
            return self._skip("no_reversal_candle", indicators)

        indicators["reversal"] = reversal

        # ── Confirmation: PSAR direction on 15m ──
        psar_dir_15m = _psar_direction(df15)
        psar_dir_30m = _psar_direction(df30)
        indicators["psar_15m_dir"] = psar_dir_15m
        indicators["psar_30m_dir"] = psar_dir_30m

        # VWAP reversion works WITH the trend, not against it
        # Below VWAP + bullish reversal needs PSAR bullish (trend supports bounce)
        if reversal == "CALL" and psar_dir_15m != 1:
            return self._skip("psar_disagrees_call", indicators)
        if reversal == "PUT" and psar_dir_15m != -1:
            return self._skip("psar_disagrees_put", indicators)

        # ── Confidence ──
        confidence = self._calc_confidence(dev_abs, threshold, vix,
                                            psar_dir_30m == (1 if reversal == "CALL" else -1))

        if reversal == "CALL":
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators
        else:
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

    SL_ATR_MULT = 1.3
    TP_ATR_MULT = 2.0

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

    def _calc_confidence(self, deviation: float, threshold: float,
                         vix: float, tf30_agrees: bool) -> float:
        """Confidence from deviation magnitude + PSAR agreement."""
        base = 0.50

        # Bigger deviation = stronger mean reversion pull
        ratio = deviation / threshold if threshold > 0 else 1.0
        if ratio > 2.0:
            base += 0.12
        elif ratio > 1.5:
            base += 0.08
        elif ratio > 1.0:
            base += 0.04

        # 30m PSAR also agrees
        if tf30_agrees:
            base += 0.05

        # VIX penalty
        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
