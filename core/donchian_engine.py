"""
donchian_engine.py — Donchian Channel Breakout Engine (Strategy 13)
===================================================================
Adapted from neurotrader888's MCPT framework for Nifty 5min intraday.

Signal: Donchian channel breakout with walk-forward adaptive lookback.
  CALL: close breaks above upper channel (rolling max) + EMA confirms uptrend
  PUT:  close breaks below lower channel (rolling min) + EMA confirms downtrend
  SKIP: no breakout, wrong EMA side, or filters not met

The lookback period is re-optimized daily using the prior N days of data
(walk-forward style), so the strategy adapts to changing volatility regimes
without overfitting to a single parameter.

Statistical validation: use backtest_s13.py with MCPT bar permutation
to confirm the strategy has a statistically significant edge (p < 0.05).
"""

import logging
import numpy as np
import pandas as pd

from core.strategy_base import StrategyEngine

logger = logging.getLogger(__name__)


def donchian_channel(closes: pd.Series, lookback: int):
    """Compute Donchian upper/lower bands from close prices.
    Upper = rolling max of prior (lookback-1) bars, shifted by 1.
    Lower = rolling min of prior (lookback-1) bars, shifted by 1."""
    upper = closes.rolling(lookback - 1).max().shift(1)
    lower = closes.rolling(lookback - 1).min().shift(1)
    return upper, lower


def donchian_signal(closes: pd.Series, lookback: int) -> pd.Series:
    """Generate persistent Donchian breakout signal.
    +1 when close > upper channel (bullish breakout)
    -1 when close < lower channel (bearish breakout)
    Forward-fills between breakouts."""
    upper, lower = donchian_channel(closes, lookback)
    signal = pd.Series(np.nan, index=closes.index)
    signal.loc[closes > upper] = 1
    signal.loc[closes < lower] = -1
    signal = signal.ffill()
    return signal


def optimize_lookback(df: pd.DataFrame, lb_min: int = 10, lb_max: int = 60) -> tuple:
    """Find the lookback that maximizes profit factor on the given data.
    Returns (best_lookback, best_pf)."""
    closes = df["close"]
    log_r = np.log(closes).diff().shift(-1)
    best_pf = 0.0
    best_lb = lb_min

    for lb in range(lb_min, lb_max + 1):
        sig = donchian_signal(closes, lb)
        rets = sig * log_r
        wins = rets[rets > 0].sum()
        losses = rets[rets < 0].abs().sum()
        if losses > 0:
            pf = wins / losses
        else:
            pf = 0.0
        if pf > best_pf:
            best_pf = pf
            best_lb = lb

    return best_lb, best_pf


class DonchianEngine(StrategyEngine):
    """Donchian channel breakout — walk-forward adaptive lookback.

    Default lookback=20 (100min on 5m bars). Re-optimizes daily using
    the last TRAIN_DAYS of 5m data when available.

    Filters: EMA40 trend direction, ADX > 15, time filters.
    """

    BASE_SL = 45
    BASE_TP = 70
    TRAIL_AFTER_TP = True
    TRAIL_AFTER_TP_STEP = 20
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 3.5

    DEFAULT_LOOKBACK = 20
    LB_MIN = 10
    LB_MAX = 60
    TRAIN_DAYS = 20
    EMA_PERIOD = 40
    ADX_MIN = 15
    FLIP_MAX_BARS = 2

    def __init__(self):
        super().__init__("DONCH")
        self._lookback = self.DEFAULT_LOOKBACK
        self._last_opt_date = None
        self._last_vix_mult = 1.0

    def _maybe_reoptimize(self, df5: pd.DataFrame):
        """Re-optimize lookback at the start of each new day using recent data."""
        today = df5.index[-1].date()
        if self._last_opt_date == today:
            return

        n_bars_needed = self.TRAIN_DAYS * 75
        if len(df5) >= n_bars_needed:
            train_data = df5.iloc[-n_bars_needed:]
            best_lb, best_pf = optimize_lookback(
                train_data, self.LB_MIN, self.LB_MAX
            )
            if best_pf > 1.0:
                self._lookback = best_lb
                logger.info(
                    f"DONCH: re-optimized lookback={best_lb} "
                    f"(PF={best_pf:.2f} on {self.TRAIN_DAYS}d train)"
                )
            else:
                self._lookback = self.DEFAULT_LOOKBACK
                logger.info(
                    f"DONCH: optimization PF={best_pf:.2f} < 1.0, "
                    f"using default lookback={self.DEFAULT_LOOKBACK}"
                )
        self._last_opt_date = today

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        self._ready = True
        indicators = {
            "strategy": "DONCH",
            "vix": round(vix, 2),
            "lookback": self._lookback,
        }

        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        min_bars = max(self._lookback + 10, 30)
        if len(df5) < min_bars:
            return self._skip("insufficient_data", indicators)

        self._maybe_reoptimize(df5)
        indicators["lookback"] = self._lookback

        closes = df5["close"]
        spot = float(closes.iloc[-1])

        # Donchian channel
        upper, lower = donchian_channel(closes, self._lookback)
        upper_val = float(upper.iloc[-1])
        lower_val = float(lower.iloc[-1])
        indicators["upper"] = round(upper_val, 2)
        indicators["lower"] = round(lower_val, 2)
        indicators["channel_width"] = round(upper_val - lower_val, 2)

        # Compute signal for recent bars to detect flips
        sig = donchian_signal(closes, self._lookback)
        cur_sig = float(sig.iloc[-1]) if not np.isnan(sig.iloc[-1]) else 0
        indicators["donch_signal"] = int(cur_sig)

        if np.isnan(cur_sig) or cur_sig == 0:
            return self._skip("no_donch_signal", indicators)

        # Detect flip: signal changed direction within last FLIP_MAX_BARS
        bars_since_flip = 0
        for i in range(1, min(len(sig), 20)):
            prev_val = sig.iloc[-(i + 1)]
            cur_val = sig.iloc[-i]
            if not np.isnan(prev_val) and not np.isnan(cur_val) and prev_val != cur_val:
                bars_since_flip = i
                break

        indicators["bars_since_flip"] = bars_since_flip

        if bars_since_flip == 0 or bars_since_flip > self.FLIP_MAX_BARS:
            return self._skip(
                f"no_donch_flip (bars={bars_since_flip})", indicators
            )

        # EMA trend filter on 5m
        ema = closes.ewm(span=self.EMA_PERIOD, adjust=False).mean()
        ema_val = float(ema.iloc[-1])
        indicators["ema40"] = round(ema_val, 2)

        ema_bull = spot > ema_val
        ema_bear = spot < ema_val

        # ADX trend strength
        adx_val = self._compute_adx(df5)
        indicators["adx"] = round(adx_val, 1)

        if adx_val < self.ADX_MIN:
            return self._skip(f"adx_too_low ({adx_val:.0f})", indicators)

        # VIX multiplier
        vix_mult = self._vix_multiplier(vix)
        self._last_vix_mult = vix_mult

        # 15m trend confirmation
        ema15 = df15["close"].ewm(span=20, adjust=False).mean()
        trend_15m = 1 if float(df15["close"].iloc[-1]) > float(ema15.iloc[-1]) else -1
        indicators["trend_15m"] = trend_15m

        # CALL: bullish breakout + EMA confirms + 15m agrees
        if cur_sig == 1 and ema_bull and trend_15m == 1:
            conf = self._calc_confidence(
                spot, upper_val, lower_val, adx_val, vix
            )
            return (
                0,
                np.array([conf, (1 - conf) * 0.2, 1 - conf - (1 - conf) * 0.2]),
                conf,
                indicators,
            )

        # PUT: bearish breakout + EMA confirms + 15m agrees
        if cur_sig == -1 and ema_bear and trend_15m == -1:
            conf = self._calc_confidence(
                spot, upper_val, lower_val, adx_val, vix
            )
            return (
                1,
                np.array([(1 - conf) * 0.2, conf, 1 - conf - (1 - conf) * 0.2]),
                conf,
                indicators,
            )

        return self._skip("filter_mismatch", indicators)

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

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()
        up = high.diff()
        dn = -low.diff()
        plus_dm = ((up > dn) & (up > 0)).astype(float) * up
        minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()
        val = float(adx.iloc[-2]) if len(adx) > 1 and not np.isnan(adx.iloc[-2]) else 25.0
        return val

    @staticmethod
    def _calc_confidence(spot: float, upper: float, lower: float,
                         adx: float, vix: float) -> float:
        """Confidence from breakout strength + ADX + VIX."""
        base = 0.50

        # Breakout magnitude relative to channel width
        ch_width = upper - lower
        if ch_width > 0:
            breakout_pct = abs(spot - (upper + lower) / 2) / ch_width
            if breakout_pct > 0.6:
                base += 0.10
            elif breakout_pct > 0.4:
                base += 0.05

        if adx > 30:
            base += 0.08
        elif adx > 20:
            base += 0.04

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
