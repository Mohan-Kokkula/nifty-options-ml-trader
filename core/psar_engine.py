"""
psar_engine.py — Parabolic SAR Signal Engine
=============================================
Replaces the 143-feature ML model with a simple, robust signal:
  - Dual-timeframe PSAR alignment (5m + 15m)
  - VIX regime as chop filter (from sentinel/regime engine)
  - Output: CALL / PUT / SKIP with confidence scores

Signal logic:
  CALL: 5m PSAR flip bullish + 15m PSAR bullish + EMA40 side filter
  PUT:  5m PSAR flip bearish + 15m PSAR bearish + EMA40 side filter
  SKIP: 15m disagrees or filters not met

Backtest-optimized: 5m+15m only (PF 1.18, +218pts) outperforms
3-TF alignment which over-filters and misses entries.
VIX scales SL/TP: 1.0x normal, 1.2x elevated (17-22), 1.5x high (22+).
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def _compute_adx(df: pd.DataFrame, period: int = 14):
    """Compute ADX (Average Directional Index) for trend strength."""
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


def _psar_signal_for_tf(df: pd.DataFrame,
                        af_start=0.03, af_step=0.02, af_max=0.2) -> dict:
    """Compute PSAR signal for one timeframe.
    Returns direction, SAR value, distance, bars since flip, and whether
    the last confirmed bar was a flip (matching Pine's barstate.isconfirmed)."""
    if df is None or df.empty or len(df) < 5:
        return {"direction": 0, "psar": 0, "distance_pts": 0,
                "bars_since_flip": 0, "is_flip": False}

    psar_vals, psar_dirs, psar_flips = compute_psar(df, af_start, af_step, af_max)

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
    """Dual-timeframe PSAR signal engine (5m + 15m).

    Settings:
      - 5m PSAR flip as entry trigger + 15m PSAR direction must agree
      - Flip-only: signal within 1 bar of 5m PSAR flip
      - EMA40 side filter + FLAT day PUT filter
      - Skip 09:15-09:30 (market open noise) and 12:00-13:30 (lunch chop)
      - SL=60pts, TP=120pts (1:2 R:R), VIX-scaled
    """

    # Backtest-proven SL/TP (points)
    BASE_SL = 60
    BASE_TP = 120
    OPEN_SETTLE = 920
    LUNCH_START = 1200
    LUNCH_END = 1330
    FLIP_MAX_BARS = 1
    FLAT_THRESHOLD = 40
    EMA_PERIOD = 40

    def __init__(self):
        self._ready = False
        self._trades_today = 0
        self._today = None

    @staticmethod
    def _adx_adaptive_af(adx_val) -> tuple:
        """Return (af_start, af_step, af_max) based on ADX trend strength.
        Backtest-tuned: aggressive in normal trends, moderate in strong, conservative in ranging."""
        if adx_val is not None and not np.isnan(adx_val):
            if adx_val > 30:
                return 0.03, 0.025, 0.3
            elif adx_val > 20:
                return 0.045, 0.03, 0.3
            else:
                return 0.02, 0.015, 0.15
        return 0.03, 0.02, 0.2

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

        # ADX-adaptive AF: tune tracking speed based on trend strength
        adx_series = _compute_adx(df5)
        adx_val = float(adx_series.iloc[-2]) if len(adx_series) > 1 and not np.isnan(adx_series.iloc[-2]) else None
        af_start, af_step, af_max = self._adx_adaptive_af(adx_val)

        # Compute PSAR for each timeframe
        sig5 = _psar_signal_for_tf(df5, af_start, af_step, af_max)
        sig15 = _psar_signal_for_tf(df15, af_start, af_step, af_max)
        sig30 = _psar_signal_for_tf(df30, af_start, af_step, af_max)

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
            "adx": round(adx_val, 1) if adx_val is not None else None,
            "af_params": {"start": af_start, "step": af_step, "max": af_max},
        }

        # Day open for FLAT detection — first bar of today in df5
        today = df5.index[-1].date()
        today_bars = df5[df5.index.date == today]
        day_open = float(today_bars.iloc[0]["open"]) if not today_bars.empty else float(df5["open"].iloc[-1])
        spot = float(df5["close"].iloc[-1])
        is_flat = abs(spot - day_open) <= self.FLAT_THRESHOLD
        indicators["day_open"] = round(day_open, 2)
        indicators["is_flat"] = is_flat

        # EMA side filter — CALL only above EMA, PUT only below
        ema_val = float(df5["close"].ewm(span=self.EMA_PERIOD, adjust=False).mean().iloc[-1])
        indicators["ema"] = round(ema_val, 2)

        d5 = sig5["direction"]
        d15 = sig15["direction"]
        both_bull = d5 == 1 and d15 == 1
        both_bear = d5 == -1 and d15 == -1

        indicators["d5"] = d5
        indicators["d15"] = d15
        indicators["psar_30m"] = sig30
        indicators["alignment"] = "bull" if both_bull else ("bear" if both_bear else "none")

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

        # ── Signal: 5m + 15m must both agree ──

        if both_bull:
            if spot < ema_val:
                indicators["skip_reason"] = f"ema_side_call (spot={spot:.0f} < EMA={ema_val:.0f})"
                return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators
            confidence = self._calc_confidence(sig5, sig15, sig30, vix)
            indicators["sl_pts"] = round(sl_pts, 1)
            indicators["tp_pts"] = round(tp_pts, 1)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        if both_bear:
            if is_flat:
                indicators["skip_reason"] = f"flat_day_put (spot={spot:.0f} open={day_open:.0f} diff={abs(spot-day_open):.0f}pts)"
                return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators
            if spot > ema_val:
                indicators["skip_reason"] = f"ema_side_put (spot={spot:.0f} > EMA={ema_val:.0f})"
                return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators
            confidence = self._calc_confidence(sig5, sig15, sig30, vix)
            indicators["sl_pts"] = round(sl_pts, 1)
            indicators["tp_pts"] = round(tp_pts, 1)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        indicators["skip_reason"] = "5m_15m_disagree"
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators

    def record_trade(self):
        """Call after a trade is taken to track daily count."""
        self._trades_today += 1

    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0) -> tuple:
        """Get SL/TP in points, scaled by VIX and risk budget.

        If max_loss_budget and lot_size are provided, caps SL so that
        1 lot × SL ≤ budget.  TP scales proportionally to maintain R:R.
        """
        vix_mult = 1.5 if vix >= 22 else (1.2 if vix >= 17 else 1.0)
        sl = self.BASE_SL * vix_mult
        tp = self.BASE_TP * vix_mult

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
