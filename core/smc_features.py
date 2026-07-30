"""
smc_features.py — Smart Money Concepts (SMC) Feature Engineering
=================================================================
Replaces retail-indicator-only filters (EMA, RSI, ADX) with three
mathematically-defined SMC patterns added directly as ML features.

Pattern definitions:
──────────────────────────────────────────────────────────────────
1. FAIR VALUE GAP (FVG)
   A 3-candle imbalance where one side of the market is missing.
   Bullish FVG: candle[i].low > candle[i-2].high
   Bearish FVG: candle[i].high < candle[i-2].low
   Price statistically returns to "fill" these gaps.

2. LIQUIDITY SWEEP
   Price wicks through previous day's high (PDH) or low (PDL) —
   triggering retail stop-losses — then reverses sharply.
   Bullish sweep: wick below PDL → close above PDL  (long setup)
   Bearish sweep: wick above PDH → close below PDH  (short setup)

3. ORDER BLOCK (OB)
   The last opposing candle before a strong impulse move, confirmed
   by a volume spike. Represents where institutions entered.
   Bullish OB: last bearish candle before bullish engulfing impulse + vol spike
   Bearish OB: last bullish candle before bearish engulfing impulse + vol spike

Usage:
  from core.smc_features import extract_latest_smc_features
  feats = extract_latest_smc_features(df_15m, pdh=prev_high, pdl=prev_low)
  # Returns flat dict — merge into ML feature row
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. FAIR VALUE GAP
# ──────────────────────────────────────────────────────────────────────────────

def calc_fair_value_gaps(
    df: pd.DataFrame,
    min_gap_pts: float = 5.0,
) -> pd.DataFrame:
    """
    Compute FVG features on a 15m OHLCV DataFrame.

    Parameters
    ----------
    df          : DataFrame with columns [open, high, low, close]
    min_gap_pts : Minimum gap size in Nifty points to qualify (default 5)

    Added columns
    -------------
    fvg_bull        : 1 if this candle creates a bullish FVG, else 0
    fvg_bear        : 1 if this candle creates a bearish FVG, else 0
    fvg_bull_size   : gap size in points (0 if no bullish FVG)
    fvg_bear_size   : gap size in points (0 if no bearish FVG)
    fvg_bull_recent : 1 if any bullish FVG in last 5 bars
    fvg_bear_recent : 1 if any bearish FVG in last 5 bars
    fvg_net         : +1 = more bullish, -1 = more bearish, 0 = neutral
    """
    df = df.copy()
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    # Two-bar look-back: current candle vs candle two positions earlier
    high_2 = high.shift(2)
    low_2  = low.shift(2)

    bull_gap = (low - high_2).clip(lower=0)        # current low > prev-prev high
    bear_gap = (low_2 - high).clip(lower=0)        # current high < prev-prev low

    df["fvg_bull"]        = (bull_gap >= min_gap_pts).astype(int)
    df["fvg_bear"]        = (bear_gap >= min_gap_pts).astype(int)
    df["fvg_bull_size"]   = bull_gap.where(bull_gap >= min_gap_pts, 0.0).round(2)
    df["fvg_bear_size"]   = bear_gap.where(bear_gap >= min_gap_pts, 0.0).round(2)
    df["fvg_bull_recent"] = df["fvg_bull"].rolling(5, min_periods=1).max().fillna(0).astype(int)
    df["fvg_bear_recent"] = df["fvg_bear"].rolling(5, min_periods=1).max().fillna(0).astype(int)
    df["fvg_net"]         = df["fvg_bull_recent"] - df["fvg_bear_recent"]

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. LIQUIDITY SWEEP
# ──────────────────────────────────────────────────────────────────────────────

def calc_liquidity_sweeps(
    df: pd.DataFrame,
    pdh: float,
    pdl: float,
    min_pierce_pts: float = 2.0,
    sweep_window: int = 6,
) -> pd.DataFrame:
    """
    Detect liquidity sweeps of previous day high/low.

    Parameters
    ----------
    df              : 15m OHLCV DataFrame
    pdh             : Previous day's high in Nifty points
    pdl             : Previous day's low in Nifty points
    min_pierce_pts  : Minimum wick beyond PDH/PDL to count as sweep
    sweep_window    : Lookback bars for "recent sweep" flag

    Added columns
    -------------
    liq_sweep_bull      : 1 if this bar swept below PDL and closed above
    liq_sweep_bear      : 1 if this bar swept above PDH and closed below
    liq_sweep_bull_pts  : how many points below PDL the wick went
    liq_sweep_bear_pts  : how many points above PDH the wick went
    liq_sweep_recent    : 1 if any sweep in last `sweep_window` bars
    liq_sweep_dir       : +1 bullish sweep, -1 bearish sweep, 0 none
    """
    df = df.copy()
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    if pdl > 0:
        bull_pierce  = (pdl - low).clip(lower=0)                  # wick below PDL
        bull_reversal = close > pdl                                # closed back above
        bull_sweep   = ((bull_pierce >= min_pierce_pts) & bull_reversal).astype(int)
    else:
        bull_pierce = pd.Series(0.0, index=df.index)
        bull_sweep  = pd.Series(0,   index=df.index)

    if pdh > 0:
        bear_pierce  = (high - pdh).clip(lower=0)                 # wick above PDH
        bear_reversal = close < pdh                                # closed back below
        bear_sweep   = ((bear_pierce >= min_pierce_pts) & bear_reversal).astype(int)
    else:
        bear_pierce = pd.Series(0.0, index=df.index)
        bear_sweep  = pd.Series(0,   index=df.index)

    df["liq_sweep_bull"]      = bull_sweep
    df["liq_sweep_bear"]      = bear_sweep
    df["liq_sweep_bull_pts"]  = bull_pierce.where(bull_sweep == 1, 0.0).round(2)
    df["liq_sweep_bear_pts"]  = bear_pierce.where(bear_sweep == 1, 0.0).round(2)
    df["liq_sweep_recent"]    = (
        (bull_sweep | bear_sweep)
        .rolling(sweep_window, min_periods=1).max()
        .fillna(0).astype(int)
    )
    df["liq_sweep_dir"] = bull_sweep - bear_sweep   # +1 / -1 / 0

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. ORDER BLOCK
# ──────────────────────────────────────────────────────────────────────────────

def calc_order_blocks(
    df: pd.DataFrame,
    vol_spike_mult: float = 1.5,
    lookback: int = 20,
    ob_window: int = 8,
) -> pd.DataFrame:
    """
    Detect Order Blocks — institutional accumulation/distribution zones.

    Logic
    -----
    Bullish OB: last BEARISH candle before a bullish engulfing impulse
                + volume >= vol_spike_mult × rolling avg volume
    Bearish OB: last BULLISH candle before a bearish engulfing impulse
                + volume >= vol_spike_mult × rolling avg volume

    Parameters
    ----------
    df            : 15m OHLCV DataFrame (must have 'volume' column)
    vol_spike_mult: volume must exceed this × rolling mean to qualify
    lookback      : rolling window for volume baseline
    ob_window     : bars to look back for "recent OB" flag

    Added columns
    -------------
    ob_bull          : 1 if this candle is a bullish order block
    ob_bear          : 1 if this candle is a bearish order block
    ob_bull_strength : vol_ratio × candle_body_size (quality score)
    ob_bear_strength : vol_ratio × candle_body_size (quality score)
    ob_recent        : 1 if any OB in last `ob_window` bars
    ob_net           : +1 more bullish OBs, -1 more bearish OBs
    """
    df = df.copy()
    open_  = df["open"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    close  = df["close"].astype(float)

    if "volume" in df.columns:
        volume = df["volume"].astype(float).clip(lower=1.0)
    else:
        logger.debug("smc_features: no volume column — OB strength will be uniform")
        volume = pd.Series(1.0, index=df.index)

    vol_ma    = volume.rolling(lookback, min_periods=5).mean().replace(0, 1.0)
    vol_ratio = (volume / vol_ma).fillna(1.0)
    vol_spike = vol_ratio >= vol_spike_mult

    body_size = (close - open_).abs()
    is_bear   = close < open_
    is_bull   = close > open_

    # Next candle info for engulfing check
    next_close = close.shift(-1)
    next_open  = open_.shift(-1)

    bull_engulf = (next_close > open_) & (next_open < close)   # next swallows current
    bear_engulf = (next_close < open_) & (next_open > close)

    ob_bull_flag = is_bear & vol_spike & bull_engulf
    ob_bear_flag = is_bull & vol_spike & bear_engulf

    df["ob_bull"]          = ob_bull_flag.astype(int)
    df["ob_bear"]          = ob_bear_flag.astype(int)
    df["ob_bull_strength"] = (vol_ratio * body_size).where(ob_bull_flag, 0.0).round(2)
    df["ob_bear_strength"] = (vol_ratio * body_size).where(ob_bear_flag, 0.0).round(2)
    df["ob_recent"]        = (
        (ob_bull_flag | ob_bear_flag)
        .rolling(ob_window, min_periods=1).max()
        .fillna(0).astype(int)
    )
    df["ob_net"] = (
        df["ob_bull"].rolling(ob_window, min_periods=1).sum() -
        df["ob_bear"].rolling(ob_window, min_periods=1).sum()
    ).fillna(0).astype(int)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API — compute all + extract for ML
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_smc_features(
    df_15m: pd.DataFrame,
    pdh: float = 0.0,
    pdl: float = 0.0,
) -> pd.DataFrame:
    """
    Compute all three SMC feature sets on a 15m DataFrame and return it
    with all new columns appended.

    Parameters
    ----------
    df_15m : 15-minute OHLCV DataFrame
    pdh    : previous day's high (0 = skip liquidity sweep)
    pdl    : previous day's low  (0 = skip liquidity sweep)
    """
    if df_15m is None or df_15m.empty:
        return df_15m

    df = calc_fair_value_gaps(df_15m)
    df = calc_order_blocks(df)

    if pdh > 0 and pdl > 0:
        df = calc_liquidity_sweeps(df, pdh=pdh, pdl=pdl)
    else:
        # PDH/PDL unavailable — zero-fill
        for col in _SWEEP_COLS:
            df[col] = 0

    return df


def extract_latest_smc_features(
    df_15m: pd.DataFrame,
    pdh: float = 0.0,
    pdl: float = 0.0,
) -> dict:
    """
    Run all SMC calculations and return the LAST ROW as a flat dict
    suitable for injection into the ML feature row.

    NaN values are replaced with 0.0.
    Returns a zeroed dict if input is empty or computation fails.
    """
    try:
        df = compute_all_smc_features(df_15m, pdh=pdh, pdl=pdl)
        if df is None or df.empty:
            return _zero_smc_features()

        all_cols = _FVG_COLS + _OB_COLS + _SWEEP_COLS
        row      = df.iloc[-1]
        result   = {}
        for col in all_cols:
            if col in row.index:
                v = row[col]
                result[col] = 0.0 if (
                    v is None or (isinstance(v, float) and np.isnan(v))
                ) else float(v)
            else:
                result[col] = 0.0
        return result

    except Exception as e:
        logger.error(f"extract_latest_smc_features failed: {e}")
        return _zero_smc_features()


def get_smc_signal(feats: dict) -> str:
    """
    Collapse SMC features into a directional signal string.
    Used for logging / Claude post-analysis context.

    Returns: "BULLISH" | "BEARISH" | "NEUTRAL"
    """
    bull_score = (
        feats.get("fvg_bull_recent", 0) * 1 +
        feats.get("ob_bull", 0) * 2 +
        feats.get("liq_sweep_bull", 0) * 2
    )
    bear_score = (
        feats.get("fvg_bear_recent", 0) * 1 +
        feats.get("ob_bear", 0) * 2 +
        feats.get("liq_sweep_bear", 0) * 2
    )
    if bull_score > bear_score:
        return "BULLISH"
    if bear_score > bull_score:
        return "BEARISH"
    return "NEUTRAL"


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_FVG_COLS = [
    "fvg_bull", "fvg_bear", "fvg_bull_size", "fvg_bear_size",
    "fvg_bull_recent", "fvg_bear_recent", "fvg_net",
]
_OB_COLS = [
    "ob_bull", "ob_bear", "ob_bull_strength", "ob_bear_strength",
    "ob_recent", "ob_net",
]
_SWEEP_COLS = [
    "liq_sweep_bull", "liq_sweep_bear", "liq_sweep_bull_pts",
    "liq_sweep_bear_pts", "liq_sweep_recent", "liq_sweep_dir",
]


def _zero_smc_features() -> dict:
    return {c: 0.0 for c in _FVG_COLS + _OB_COLS + _SWEEP_COLS}
