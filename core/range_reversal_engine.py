"""
range_reversal_engine.py — Range Reversal Engine (Strategy 9)
=============================================================
ICT-based liquidity sweep reversal on range-bound days.
Combines Turtle Soup structure with ICT smart-money concepts.

Three-phase signal:
  Phase 1 (Setup): ADX < 25, price sweeps 20-bar high/low (liquidity grab),
                   RSI(2) extreme
  Phase 2 (Confirmation): Displacement candle + FVG or Order Block detected
  Phase 3 (Trigger): Market Structure Shift (break of recent swing)
                     OR reversal candle closing back inside range

ICT concepts used:
  - Liquidity sweep: stop hunt above/below range boundary
  - Displacement: candle body > 1.5x ATR = institutional intent
  - Fair Value Gap (FVG): 3-candle imbalance (gap between candle 1 and 3)
  - Order Block (OB): last opposing candle before displacement move
  - Market Structure Shift (MSS): break of most recent swing high/low
  - Premium/Discount: entry only in correct zone (above/below 50% of range)
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine

logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def _find_swing_highs(highs: np.ndarray, lows: np.ndarray, left: int = 3, right: int = 1):
    """Find swing highs: bar higher than `left` bars before and `right` bars after."""
    swings = []
    for i in range(left, len(highs) - right):
        is_swing = True
        for j in range(1, left + 1):
            if highs[i] <= highs[i - j]:
                is_swing = False
                break
        if is_swing:
            for j in range(1, right + 1):
                if i + j < len(highs) and highs[i] <= highs[i + j]:
                    is_swing = False
                    break
        if is_swing:
            swings.append((i, float(highs[i])))
    return swings


def _find_swing_lows(highs: np.ndarray, lows: np.ndarray, left: int = 3, right: int = 1):
    """Find swing lows: bar lower than `left` bars before and `right` bars after."""
    swings = []
    for i in range(left, len(lows) - right):
        is_swing = True
        for j in range(1, left + 1):
            if lows[i] >= lows[i - j]:
                is_swing = False
                break
        if is_swing:
            for j in range(1, right + 1):
                if i + j < len(lows) and lows[i] >= lows[i + j]:
                    is_swing = False
                    break
        if is_swing:
            swings.append((i, float(lows[i])))
    return swings


def _detect_fvg_bearish(df: pd.DataFrame, idx: int) -> dict | None:
    """Bearish FVG at idx: candle[idx-2].low > candle[idx].high (gap down)."""
    if idx < 2:
        return None
    c1_low = float(df["low"].iloc[idx - 2])
    c3_high = float(df["high"].iloc[idx])
    if c1_low > c3_high:
        return {"top": c1_low, "bottom": c3_high, "bar_idx": idx}
    return None


def _detect_fvg_bullish(df: pd.DataFrame, idx: int) -> dict | None:
    """Bullish FVG at idx: candle[idx-2].high < candle[idx].low (gap up)."""
    if idx < 2:
        return None
    c1_high = float(df["high"].iloc[idx - 2])
    c3_low = float(df["low"].iloc[idx])
    if c3_low > c1_high:
        return {"top": c3_low, "bottom": c1_high, "bar_idx": idx}
    return None


def _find_bearish_ob(df: pd.DataFrame, end_idx: int, lookback: int = 5) -> dict | None:
    """Last bullish candle before a bearish displacement — bearish order block."""
    for i in range(end_idx - 1, max(end_idx - lookback, 0), -1):
        c = float(df["close"].iloc[i])
        o = float(df["open"].iloc[i])
        if c > o:  # bullish candle = the order block
            return {
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "bar_idx": i,
            }
    return None


def _find_bullish_ob(df: pd.DataFrame, end_idx: int, lookback: int = 5) -> dict | None:
    """Last bearish candle before a bullish displacement — bullish order block."""
    for i in range(end_idx - 1, max(end_idx - lookback, 0), -1):
        c = float(df["close"].iloc[i])
        o = float(df["open"].iloc[i])
        if c < o:  # bearish candle = the order block
            return {
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "bar_idx": i,
            }
    return None


class RangeReversalEngine(StrategyEngine):
    """ICT liquidity sweep + displacement + FVG/OB + MSS."""

    BASE_SL = 45
    BASE_TP = 90
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 3.0

    ADX_MAX = 25
    RSI_PERIOD = 2
    RSI_OVERBOUGHT = 90
    RSI_OVERSOLD = 10
    LOOKBACK_BARS = 20
    SWEEP_MIN_PTS = 2
    SWEEP_MAX_PTS = 50
    TRIGGER_WINDOW = 5
    DISPLACEMENT_ATR_MULT = 1.5

    def __init__(self):
        super().__init__("RangeRev")
        self._setup = None
        self._setup_date = None
        self._setup_sl_ref = 0.0

    def reset_daily(self):
        super().reset_daily()
        self._setup = None

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {"strategy": "RangeRev", "vix": round(vix, 2)}

        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        if len(df5) < self.LOOKBACK_BARS + 15:
            return self._skip("insufficient_data", indicators)

        if vix > 28:
            return self._skip("vix_too_high", indicators)

        spot = float(df5["close"].iloc[-1])
        bar_open = float(df5["open"].iloc[-1])
        bar_high = float(df5["high"].iloc[-1])
        bar_low = float(df5["low"].iloc[-1])
        indicators["spot"] = round(spot, 2)

        today = df5.index[-1].date()
        if self._setup_date != today:
            self._setup = None
            self._setup_date = today

        # ADX — must be range-bound
        adx_series = _compute_adx(df5)
        adx_val = float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else 30.0
        indicators["adx"] = round(adx_val, 1)

        if adx_val >= self.ADX_MAX:
            self._setup = None
            return self._skip(f"adx_high ({adx_val:.0f})", indicators)

        # RSI(2)
        rsi = _compute_rsi(df5["close"], self.RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        indicators["rsi2"] = round(rsi_val, 2)

        # ATR for displacement detection
        atr = _compute_atr(df5)
        atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0
        indicators["atr"] = round(atr_val, 2)

        # 20-bar range boundary
        lookback = df5.iloc[-(self.LOOKBACK_BARS + 1):-1]
        range_high = float(lookback["high"].max())
        range_low = float(lookback["low"].min())
        range_mid = (range_high + range_low) / 2
        indicators["range_high"] = round(range_high, 2)
        indicators["range_low"] = round(range_low, 2)
        indicators["range_mid"] = round(range_mid, 2)

        # PDH/PDL — Previous Day High/Low (major BSL/SSL pools)
        pdh, pdl = None, None
        prev_day_bars = df5[df5.index.date < today]
        if len(prev_day_bars) > 0:
            last_day = prev_day_bars.index[-1].date()
            day_bars = prev_day_bars[prev_day_bars.index.date == last_day]
            if len(day_bars) > 0:
                pdh = float(day_bars["high"].max())
                pdl = float(day_bars["low"].min())
                indicators["pdh"] = round(pdh, 2)
                indicators["pdl"] = round(pdl, 2)

        bar_idx = len(df5) - 1
        bar_body = abs(spot - bar_open)
        is_bearish = spot < bar_open
        is_bullish = spot > bar_open
        is_displacement = atr_val > 0 and bar_body > self.DISPLACEMENT_ATR_MULT * atr_val

        # ICT: detect FVG on current bar
        fvg_bear = _detect_fvg_bearish(df5, bar_idx)
        fvg_bull = _detect_fvg_bullish(df5, bar_idx)

        # ICT: swing structure for MSS
        highs = df5["high"].values
        lows = df5["low"].values
        recent_swing_lows = _find_swing_lows(highs, lows)
        recent_swing_highs = _find_swing_highs(highs, lows)

        # ── Phase 2/3: Trigger on active setup ──
        if self._setup is not None:
            bars_since = bar_idx - self._setup["bar_idx"]
            indicators["setup_active"] = True
            indicators["setup_dir"] = self._setup["dir"]
            indicators["bars_since_setup"] = bars_since

            if bars_since > self.TRIGGER_WINDOW:
                self._setup = None
                return self._skip("setup_expired", indicators)

            if self._setup["dir"] == "PUT":
                # ICT triggers (strongest to weakest):
                # 1. MSS: price breaks below most recent swing low
                mss = False
                if recent_swing_lows:
                    last_swing_low = recent_swing_lows[-1][1]
                    mss = spot < last_swing_low
                    indicators["last_swing_low"] = round(last_swing_low, 2)

                # 2. Displacement + FVG/OB
                has_displacement = is_bearish and is_displacement
                has_fvg = fvg_bear is not None
                ob = _find_bearish_ob(df5, bar_idx)
                has_ob = ob is not None

                # 3. Reversal candle back inside range
                reversal_inside = is_bearish and spot < self._setup["range_high"]

                # Premium zone check: for PUT, entry should be in premium (above mid)
                in_premium = spot > range_mid

                # Determine trigger and confidence boost
                trigger = None
                ict_boost = 0.0
                if mss and (has_displacement or has_fvg):
                    trigger = "MSS+displacement"
                    ict_boost = 0.12
                elif mss:
                    trigger = "MSS"
                    ict_boost = 0.08
                elif has_displacement and has_fvg:
                    trigger = "displacement+FVG"
                    ict_boost = 0.10
                elif has_displacement and has_ob:
                    trigger = "displacement+OB"
                    ict_boost = 0.08
                elif has_displacement and reversal_inside:
                    trigger = "displacement_reversal"
                    ict_boost = 0.06
                elif reversal_inside:
                    trigger = "reversal_inside"
                    ict_boost = 0.0

                if trigger:
                    setup_rsi = self._setup["rsi"]
                    setup_sweep = self._setup["sweep_pts"]
                    pdh_swept = self._setup.get("pdh_swept", False)
                    sl_ref = self._setup_sl_ref
                    self._setup = None
                    confidence = self._calc_confidence(
                        setup_rsi, adx_val, setup_sweep, vix,
                        is_put=True, ict_boost=ict_boost, in_zone=in_premium,
                        pdhl_swept=pdh_swept)
                    p_put = confidence
                    p_call = (1.0 - confidence) * 0.15
                    p_skip = 1.0 - p_put - p_call
                    indicators["trigger"] = trigger
                    indicators["sl_ref"] = round(sl_ref, 2)
                    indicators["has_fvg"] = has_fvg
                    indicators["has_ob"] = has_ob is not None
                    indicators["mss"] = mss
                    indicators["displacement"] = has_displacement
                    indicators["in_premium"] = in_premium
                    indicators["pdh_swept"] = pdh_swept
                    return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

            elif self._setup["dir"] == "CALL":
                # ICT triggers for CALL
                mss = False
                if recent_swing_highs:
                    last_swing_high = recent_swing_highs[-1][1]
                    mss = spot > last_swing_high
                    indicators["last_swing_high"] = round(last_swing_high, 2)

                has_displacement = is_bullish and is_displacement
                has_fvg = fvg_bull is not None
                ob = _find_bullish_ob(df5, bar_idx)
                has_ob = ob is not None

                reversal_inside = is_bullish and spot > self._setup["range_low"]

                # Discount zone check: for CALL, entry should be in discount (below mid)
                in_discount = spot < range_mid

                trigger = None
                ict_boost = 0.0
                if mss and (has_displacement or has_fvg):
                    trigger = "MSS+displacement"
                    ict_boost = 0.12
                elif mss:
                    trigger = "MSS"
                    ict_boost = 0.08
                elif has_displacement and has_fvg:
                    trigger = "displacement+FVG"
                    ict_boost = 0.10
                elif has_displacement and has_ob:
                    trigger = "displacement+OB"
                    ict_boost = 0.08
                elif has_displacement and reversal_inside:
                    trigger = "displacement_reversal"
                    ict_boost = 0.06
                elif reversal_inside:
                    trigger = "reversal_inside"
                    ict_boost = 0.0

                if trigger:
                    setup_rsi = self._setup["rsi"]
                    setup_sweep = self._setup["sweep_pts"]
                    pdl_swept = self._setup.get("pdl_swept", False)
                    sl_ref = self._setup_sl_ref
                    self._setup = None
                    confidence = self._calc_confidence(
                        setup_rsi, adx_val, setup_sweep, vix,
                        is_put=False, ict_boost=ict_boost, in_zone=in_discount,
                        pdhl_swept=pdl_swept)
                    p_call = confidence
                    p_put = (1.0 - confidence) * 0.15
                    p_skip = 1.0 - p_call - p_put
                    indicators["trigger"] = trigger
                    indicators["sl_ref"] = round(sl_ref, 2)
                    indicators["has_fvg"] = has_fvg
                    indicators["has_ob"] = has_ob is not None
                    indicators["mss"] = mss
                    indicators["displacement"] = has_displacement
                    indicators["in_discount"] = in_discount
                    indicators["pdl_swept"] = pdl_swept
                    return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

            return self._skip(f"setup_waiting ({bars_since}/{self.TRIGGER_WINDOW})", indicators)

        # ── Phase 1: Detect new setup (liquidity sweep) ──
        sweep_above = bar_high - range_high
        sweep_below = range_low - bar_low
        indicators["sweep_above"] = round(sweep_above, 2)
        indicators["sweep_below"] = round(sweep_below, 2)

        # PDH/PDL sweep (major BSL/SSL pools)
        pdh_swept = pdh is not None and bar_high > pdh and self.SWEEP_MIN_PTS <= (bar_high - pdh) <= self.SWEEP_MAX_PTS
        pdl_swept = pdl is not None and bar_low < pdl and self.SWEEP_MIN_PTS <= (pdl - bar_low) <= self.SWEEP_MAX_PTS

        # PUT setup: sweep above range (buy-side liquidity grab) + RSI exhausted
        if self.SWEEP_MIN_PTS <= sweep_above <= self.SWEEP_MAX_PTS:
            if rsi_val >= self.RSI_OVERBOUGHT:
                self._setup = {
                    "dir": "PUT",
                    "bar_idx": bar_idx,
                    "sweep_pts": sweep_above,
                    "rsi": rsi_val,
                    "range_high": range_high,
                    "range_low": range_low,
                    "pdh_swept": pdh_swept,
                }
                self._setup_sl_ref = bar_high + 5
                indicators["setup_created"] = "PUT"
                indicators["pdh_swept"] = pdh_swept
                return self._skip("setup_PUT_created", indicators)

        # CALL setup: sweep below range (sell-side liquidity grab) + RSI exhausted
        if self.SWEEP_MIN_PTS <= sweep_below <= self.SWEEP_MAX_PTS:
            if rsi_val <= self.RSI_OVERSOLD:
                self._setup = {
                    "dir": "CALL",
                    "bar_idx": bar_idx,
                    "sweep_pts": sweep_below,
                    "rsi": rsi_val,
                    "range_high": range_high,
                    "range_low": range_low,
                    "pdl_swept": pdl_swept,
                }
                self._setup_sl_ref = bar_low - 5
                indicators["setup_created"] = "CALL"
                indicators["pdl_swept"] = pdl_swept
                return self._skip("setup_CALL_created", indicators)

        return self._skip("no_range_sweep", indicators)

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

    def _calc_confidence(self, rsi_val: float, adx: float,
                         sweep_pts: float, vix: float, is_put: bool,
                         ict_boost: float = 0.0, in_zone: bool = False,
                         pdhl_swept: bool = False) -> float:
        base = 0.50

        # RSI extremity
        rsi_extreme = (rsi_val - self.RSI_OVERBOUGHT) if is_put else (self.RSI_OVERSOLD - rsi_val)
        if rsi_extreme > 5:
            base += 0.08
        elif rsi_extreme > 2:
            base += 0.04

        # Range regime strength
        if adx < 15:
            base += 0.06
        elif adx < 20:
            base += 0.03

        # Sweep depth — shallow = better trap
        if sweep_pts < 10:
            base += 0.04
        elif sweep_pts < 25:
            base += 0.02

        # ICT confluence boost (MSS, displacement, FVG, OB)
        base += ict_boost

        # Premium/Discount zone bonus
        if in_zone:
            base += 0.04

        # PDH/PDL sweep — major institutional liquidity pool
        if pdhl_swept:
            base += 0.05

        if vix >= 22:
            base -= 0.03

        return min(0.85, max(0.35, round(base, 3)))
