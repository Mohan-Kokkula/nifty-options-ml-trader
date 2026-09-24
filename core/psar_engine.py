"""
psar_engine.py — Parabolic SAR Signal Engine
=============================================
Replaces the 143-feature ML model with a simple, robust signal:
  - Multi-timeframe PSAR alignment (5m, 15m, 30m)
  - VIX regime as chop filter (from sentinel/regime engine)
  - Output: CALL / PUT / SKIP with confidence scores

Signal logic (VIX-adaptive alignment):
  CALL: 5m bullish + enough TFs agree (2/3 or 3/3 based on VIX)
  PUT:  5m bearish + enough TFs agree (2/3 or 3/3 based on VIX)
  SKIP: insufficient alignment or 5m disagrees

VIX-adaptive alignment:
  Normal/Low VIX (<22): 2/3 TF alignment — faster entries on clean trends
  High VIX (>=22): 3/3 required — avoid whipsaws in volatile markets
  5m must ALWAYS agree (fastest TF = entry trigger)
  Partial alignment (2/3): tighter SL/TP (0.85x) + reduced confidence (0.80x)
  VIX scales SL/TP: 1.0x normal, 1.2x elevated (17-22), 1.5x high (22+).
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def compute_psar(df: pd.DataFrame, af_start=0.03, af_step=0.02, af_max=0.2):
    """Compute Parabolic SAR matching MK-Narayanashram-I Pine Script exactly.
    Start=0.03, Increment=0.02, Maximum=0.2.
    Returns Series with SAR values, direction (1=bull, -1=bear), and flip flags."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    sar = np.full(n, np.nan)
    direction = np.zeros(n)
    flips = np.zeros(n)  # 1=bull flip, -1=bear flip, 0=no flip

    if n < 2:
        return (pd.Series(close, index=df.index),
                pd.Series(np.ones(n), index=df.index),
                pd.Series(np.zeros(n), index=df.index))

    # --- Initialization on bar 1 (matching Pine: bar_index == 1) ---
    uptrend = close[1] > close[0]
    if uptrend:
        ep = high[1]
        prev_sar = low[0]
        prev_ep = high[1]
    else:
        ep = low[1]
        prev_sar = high[0]
        prev_ep = low[1]

    af = af_start
    sar[0] = np.nan  # bar 0 has no SAR in Pine
    sar[1] = prev_sar + af_start * (prev_ep - prev_sar)
    direction[0] = 1 if uptrend else -1
    direction[1] = 1 if uptrend else -1
    flips[1] = 1 if uptrend else -1  # initial direction counts as a flip

    # next_bar_sar carries forward to the next iteration
    next_bar_sar = sar[1] + af * (ep - sar[1])

    for i in range(2, n):
        first_trend_bar = False
        current_sar = next_bar_sar

        # --- Detect flips ---
        bear_flip = uptrend and current_sar > low[i]
        bull_flip = (not uptrend) and current_sar < high[i]

        # --- Apply flips ---
        if bear_flip:
            first_trend_bar = True
            uptrend = False
            current_sar = max(ep, high[i])
            ep = low[i]
            af = af_start
            flips[i] = -1
        elif bull_flip:
            first_trend_bar = True
            uptrend = True
            current_sar = min(ep, low[i])
            ep = high[i]
            af = af_start
            flips[i] = 1

        # --- EP/AF update when trend continues ---
        if not first_trend_bar:
            if uptrend and high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            elif not uptrend and low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)

        # --- Clamp SAR to prior extremums ---
        if uptrend:
            current_sar = min(current_sar, low[i - 1])
            if i >= 2:
                current_sar = min(current_sar, low[i - 2])
        else:
            current_sar = max(current_sar, high[i - 1])
            if i >= 2:
                current_sar = max(current_sar, high[i - 2])

        sar[i] = current_sar
        direction[i] = 1 if uptrend else -1
        next_bar_sar = current_sar + af * (ep - current_sar)

    return (
        pd.Series(sar, index=df.index, name="psar"),
        pd.Series(direction, index=df.index, name="psar_dir"),
        pd.Series(flips, index=df.index, name="psar_flip"),
    )


def _psar_signal_for_tf(df: pd.DataFrame) -> dict:
    """Compute PSAR signal for one timeframe.
    Returns direction, SAR value, distance, bars since flip, and whether
    the last confirmed bar was a flip (matching Pine's barstate.isconfirmed)."""
    if df is None or df.empty or len(df) < 5:
        return {"direction": 0, "psar": 0, "distance_pts": 0,
                "bars_since_flip": 0, "is_flip": False}

    psar_vals, psar_dirs, psar_flips = compute_psar(df)

    # Use second-to-last bar as "confirmed" (last bar may still be forming)
    confirmed_idx = -2 if len(df) > 5 else -1
    last_close = float(df["close"].iloc[confirmed_idx])
    last_psar = float(psar_vals.iloc[confirmed_idx])
    last_dir = int(psar_dirs.iloc[confirmed_idx])
    distance = last_close - last_psar
    is_flip = int(psar_flips.iloc[confirmed_idx]) != 0

    # Count bars since last direction flip on confirmed bars
    bars_since_flip = 0
    dirs = psar_dirs.values[:confirmed_idx + len(df) if confirmed_idx < 0 else confirmed_idx + 1]
    for i in range(len(dirs) - 2, -1, -1):
        if int(dirs[i]) != last_dir:
            break
        bars_since_flip += 1

    return {
        "direction": last_dir,      # 1=bullish, -1=bearish
        "psar": round(last_psar, 2),
        "distance_pts": round(distance, 2),
        "bars_since_flip": bars_since_flip,
        "is_flip": is_flip,
    }


class PSAREngine:
    """Multi-timeframe PSAR signal engine with VIX-adaptive alignment.

    Settings:
      - VIX-adaptive alignment: 2/3 for VIX<22 (faster), 3/3 for VIX>=22 (safer)
      - 5m must always agree (entry trigger TF)
      - Flip-only: signal within 1 bar of 5m PSAR flip (not on every aligned bar)
      - Partial (2/3) alignment: 0.85x SL/TP, 0.80x confidence
      - Skip 09:15-09:30 (market open noise) and 12:00-13:30 (lunch chop)
      - SL=60pts, TP=120pts (1:2 R:R), VIX-scaled
    """

    # Backtest-proven SL/TP (points)
    BASE_SL = 60
    BASE_TP = 120
    OPEN_SETTLE = 930
    LUNCH_START = 1200
    LUNCH_END = 1330
    FLIP_MAX_BARS = 1

    def __init__(self):
        self._ready = False
        self._trades_today = 0
        self._today = None
        self._last_aligned = 3

    def is_ready(self) -> bool:
        return self._ready

    def reset_daily(self):
        self._trades_today = 0

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        """Run PSAR signal engine.

        Args:
            df5:  5-min OHLCV bars (need >= 20)
            df15: 15-min OHLCV bars (need >= 10)
            df30: 30-min OHLCV bars (need >= 10)
            vix:  current India VIX level
            current_hm: current time as HHMM (e.g. 1230 for 12:30)

        Returns:
            signal: 0=CALL, 1=PUT, 2=SKIP
            proba: [P(CALL), P(PUT), P(SKIP)] — synthetic confidence
            confidence: max probability
            indicators: dict with PSAR details for each TF
        """
        self._ready = True

        # Compute PSAR for each timeframe
        sig5 = _psar_signal_for_tf(df5)
        sig15 = _psar_signal_for_tf(df15)
        sig30 = _psar_signal_for_tf(df30)

        # ── VIX-adaptive alignment threshold ──
        # Normal/Low VIX (<22): 2/3 OK — trends are cleaner, enter faster
        # High VIX (>=22): 3/3 required — volatile, avoid whipsaws
        min_align = 3 if vix >= 22 else 2

        # SL/TP scaled by VIX regime
        vix_mult = 1.5 if vix >= 22 else (1.2 if vix >= 17 else 1.0)
        sl_pts = self.BASE_SL * vix_mult
        tp_pts = self.BASE_TP * vix_mult

        indicators = {
            "psar_5m": sig5,
            "psar_15m": sig15,
            "psar_30m": sig30,
            "vix": round(vix, 2),
            "sl_pts": round(sl_pts, 1),
            "tp_pts": round(tp_pts, 1),
        }

        dirs = [sig5["direction"], sig15["direction"], sig30["direction"]]
        bullish_count = sum(1 for d in dirs if d == 1)
        bearish_count = sum(1 for d in dirs if d == -1)

        indicators["bullish_count"] = bullish_count
        indicators["bearish_count"] = bearish_count
        indicators["min_aligned_required"] = min_align
        indicators["alignment_mode"] = "strict" if min_align == 3 else "fast"

        # ── Pre-signal filters ──

        # Market open settle (09:15-09:30) — direction not established yet
        if current_hm < self.OPEN_SETTLE:
            indicators["skip_reason"] = "market_open_settle"
            return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators

        # Lunch chop filter (12:00-13:30)
        if self.LUNCH_START <= current_hm <= self.LUNCH_END:
            indicators["skip_reason"] = "lunch_chop_zone"
            return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators

        # Flip-only: signal only when 5m PSAR has just flipped direction
        if sig5["bars_since_flip"] > self.FLIP_MAX_BARS:
            indicators["skip_reason"] = f"no_5m_flip (bars={sig5['bars_since_flip']})"
            return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators

        # ── Signal: VIX-adaptive alignment ──
        # 5m MUST agree with signal direction (fastest TF = entry trigger)

        if bullish_count >= min_align and sig5["direction"] == 1:
            self._last_aligned = bullish_count
            confidence = self._calc_confidence(sig5, sig15, sig30, vix)
            if bullish_count == 2:
                confidence *= 0.80
                sl_pts *= 0.85
                tp_pts *= 0.85
                indicators["partial_alignment"] = True
            indicators["aligned_count"] = bullish_count
            indicators["sl_pts"] = round(sl_pts, 1)
            indicators["tp_pts"] = round(tp_pts, 1)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        if bearish_count >= min_align and sig5["direction"] == -1:
            self._last_aligned = bearish_count
            confidence = self._calc_confidence(sig5, sig15, sig30, vix)
            if bearish_count == 2:
                confidence *= 0.80
                sl_pts *= 0.85
                tp_pts *= 0.85
                indicators["partial_alignment"] = True
            indicators["aligned_count"] = bearish_count
            indicators["sl_pts"] = round(sl_pts, 1)
            indicators["tp_pts"] = round(tp_pts, 1)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        # SKIP — insufficient alignment or 5m disagrees
        self._last_aligned = 3
        if bullish_count >= 2 or bearish_count >= 2:
            indicators["skip_reason"] = "5m_disagrees"
        else:
            indicators["skip_reason"] = "tf_disagreement"
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators

    def record_trade(self):
        """Call after a trade is taken to track daily count."""
        self._trades_today += 1

    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0) -> tuple:
        """Get SL/TP in points, scaled by VIX, alignment, and risk budget.

        If max_loss_budget and lot_size are provided, caps SL so that
        1 lot × SL ≤ budget.  TP scales proportionally to maintain R:R.
        """
        vix_mult = 1.5 if vix >= 22 else (1.2 if vix >= 17 else 1.0)
        align_mult = 0.85 if self._last_aligned < 3 else 1.0
        sl = self.BASE_SL * vix_mult * align_mult
        tp = self.BASE_TP * vix_mult * align_mult

        if max_loss_budget > 0 and lot_size > 0:
            max_sl = (max_loss_budget / lot_size) * 0.995
            if max_sl < sl:
                ratio = max_sl / sl
                sl = max_sl
                tp = tp * ratio
                logger.info(
                    f"PSAR SL capped by risk budget: SL={sl:.1f}pts TP={tp:.1f}pts "
                    f"(budget=₹{max_loss_budget:.0f}, lot={lot_size})"
                )

        return round(sl, 1), round(tp, 1)

    def _calc_confidence(self, sig5: dict, sig15: dict, sig30: dict,
                         vix: float) -> float:
        """Calculate confidence score from PSAR alignment strength."""
        base = 0.55  # 3/3 alignment baseline

        # Bonus for strong PSAR distance (price far from dots = strong trend)
        avg_distance = (abs(sig5["distance_pts"]) + abs(sig15["distance_pts"])
                        + abs(sig30["distance_pts"])) / 3.0
        if avg_distance > 50:
            base += 0.10
        elif avg_distance > 25:
            base += 0.05

        # Bonus for fresh PSAR flip (recent reversal = momentum)
        if sig5["bars_since_flip"] <= 3:
            base += 0.08
        elif sig5["bars_since_flip"] <= 6:
            base += 0.04

        # VIX adjustment
        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
