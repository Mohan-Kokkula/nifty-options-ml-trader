"""
train_model_v8.py  -  Multi-Timeframe ML Model for Nifty
=========================================================
New in V8 vs V7:
  - Multi-timeframe features: 15-min, 30-min, 60-min, Daily
  - ADX (14) on 15-min — trend vs sideways filter
  - Previous day High / Low / Close (key S/R levels)
  - Day open gap (vs prev close) — directional bias
  - 15-min RSI, MACD, EMA alignment — confirmation
  - 30-min trend direction (DI+/DI-) — intermediate bias
  - 60-min major trend (EMA 9/21) — session bias
  - Daily EMA200 distance — macro trend
  - Entries still on 5-min bars (no lookahead)

Usage:
    uv run scripts/train_model_v8.py
    uv run scripts/train_model_v8.py --csv5 data/nifty_5min.csv

All higher-TF features are merged with merge_asof (last completed bar only
— zero lookahead guaranteed).
"""

import warnings; warnings.filterwarnings("ignore")
import argparse, logging, sys, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH   = "models/nifty_v8_models.pkl"
SCALER_PATH  = "models/nifty_v8_scaler.pkl"
FCOLS_PATH   = "models/feature_cols_v8.pkl"

CSV_5M   = "data/nifty_5min.csv"
CSV_15M  = "data/nifty_15min.csv"
CSV_30M  = "data/nifty_30min.csv"
CSV_60M  = "data/nifty_60min.csv"
CSV_DAY  = "data/nifty_day.csv"
CSV_VIX  = "data/india_vix.csv"       # India VIX daily — download from NSE

FWD_BARS = 3      # 5-min bars forward for label (= 15 minutes) — V8 original

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_XLG = True
except ImportError:
    HAS_XLG = False
    print("INFO: xgboost/lightgbm not found — using HistGBM (still good)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [V8] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==================================================================
# DATA LOADING
# ==================================================================

def _load_raw(path, tz_strip=True):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    dt = next(c for c in df.columns if any(x in c for x in ["date","time","timestamp"]))
    df[dt] = pd.to_datetime(df[dt], format="mixed", dayfirst=False, utc=False)
    # Strip timezone if present
    if hasattr(df[dt].dt, "tz") and df[dt].dt.tz is not None:
        df[dt] = df[dt].dt.tz_localize(None)
    df = df.set_index(dt).sort_index()
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if   "open"   in lc: col_map[col] = "open"
        elif "high"   in lc: col_map[col] = "high"
        elif "low"    in lc: col_map[col] = "low"
        elif "close"  in lc: col_map[col] = "close"
    df = df.rename(columns=col_map)
    # Keep volume when present — used by intraday VWAP feature.
    # Falls back to equal-weight VWAP when volume column missing.
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[keep].astype(float).dropna(subset=[c for c in ["open","high","low","close"] if c in df.columns])


def load_5min(path):
    df = _load_raw(path)
    df = df.between_time("09:15", "15:30")
    log.info(f"5-min : {len(df):,} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_higher(path, freq_label):
    """Load a higher-TF csv, filter market hours, drop after-hours garbage."""
    df = _load_raw(path)
    # Daily data has midnight timestamps — skip between_time filter
    times = df.index.time
    has_midnight = (times == pd.Timestamp("00:00").time()).mean() > 0.5
    if not has_midnight:
        df = df.between_time("09:15", "15:30")
        # Drop duplicate-price after-hours rows
        df = df[df["high"] != df["low"]]
    df = df[~df.index.duplicated(keep="last")]
    log.info(f"{freq_label}: {len(df):,} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def resample_from_5min(df5, rule):
    """Resample 5-min to higher TF (used when uploaded file has gaps)."""
    ohlc = df5.resample(rule, label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last")
    ).dropna()
    ohlc = ohlc.between_time("09:15", "15:30")
    return ohlc


# ==================================================================
# ADX HELPER (works on any TF dataframe)
# ==================================================================

def compute_adx(df, period=14):
    """Return ADX, DI+, DI- as a DataFrame aligned to df.index."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    up   = h - h.shift(1)
    down = l.shift(1) - l
    pdm  = up.where((up > down) & (up > 0), 0.0)
    ndm  = down.where((down > up) & (down > 0), 0.0)

    atr14  = tr.ewm(span=period, adjust=False).mean()
    pdi14  = 100 * pdm.ewm(span=period, adjust=False).mean() / (atr14 + 1e-9)
    ndi14  = 100 * ndm.ewm(span=period, adjust=False).mean() / (atr14 + 1e-9)
    dx     = 100 * (pdi14 - ndi14).abs() / (pdi14 + ndi14 + 1e-9)
    adx    = dx.ewm(span=period, adjust=False).mean()
    return pd.DataFrame({"adx": adx, "diplus": pdi14, "diminus": ndi14},
                        index=df.index)


# ==================================================================
# FEATURE ENGINEERING — 5-MIN BASE (V7 CORE, TRIMMED)
# ==================================================================

def features_5min(df):
    """All V7 5-min features. Returns feature dataframe."""
    df = df.copy()
    o, h, l, c = df.open, df.high, df.low, df.close
    fr = h - l + 1e-8
    body = c - o

    # Candlestick
    df["body_pct"]    = body / o
    df["body_ratio"]  = body.abs() / fr
    df["upper_wick"]  = (h - c.clip(lower=o)) / fr
    df["lower_wick"]  = (o.clip(upper=c) - l) / fr
    df["is_bullish"]  = (c > o).astype(int)
    df["hl_range"]    = fr / c
    df["close_pos"]   = (c - l) / fr
    df["is_doji"]     = (df["body_ratio"] < 0.10).astype(int)
    df["is_marubozu"] = (df["body_ratio"] > 0.85).astype(int)
    df["prev_cp"]     = df["close_pos"].shift(1)
    df["prev_br"]     = df["body_ratio"].shift(1)

    # Returns
    df["log_ret"] = np.log(c / c.shift(1))
    df["gap_up"]  = ((o - c.shift(1)) / c.shift(1)).clip(lower=0)
    df["gap_dn"]  = ((c.shift(1) - o) / c.shift(1)).clip(lower=0)

    # Consecutive
    bull = (c > o).astype(int); bear = (c < o).astype(int)
    df["consec_bull"] = bull.groupby((bull != bull.shift()).cumsum()).cumcount().where(bull==1, 0)
    df["consec_bear"] = bear.groupby((bear != bear.shift()).cumsum()).cumcount().where(bear==1, 0)

    # Patterns
    df["bull_eng"]   = ((c>o)&(c.shift(1)<o.shift(1))&(c>o.shift(1))&(o<c.shift(1))).astype(int)
    df["bear_eng"]   = ((c<o)&(c.shift(1)>o.shift(1))&(c<o.shift(1))&(o>c.shift(1))).astype(int)
    df["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    # Lag returns
    for lag in [1, 2, 3, 5, 8, 13]:
        df[f"rl{lag}"] = df["log_ret"].shift(lag)

    # Momentum
    for w in [3, 6, 9, 12, 24]:
        df[f"m{w}"] = c.pct_change(w)
    df["roc"] = df["m3"] - df["m6"]

    # Distance from high/low
    for w in [5, 10, 20]:
        df[f"dh{w}"] = (c - h.rolling(w).max()) / c
        df[f"dl{w}"] = (c - l.rolling(w).min()) / c

    # EMA distances
    for w in [5, 9, 13, 21, 50]:
        df[f"de{w}"] = (c - c.ewm(span=w, adjust=False).mean()) / c

    # EMA crossovers
    e9  = c.ewm(span=9,  adjust=False).mean()
    e13 = c.ewm(span=13, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    df["e9a21"] = (e9 > e21).astype(int)
    df["e9s"]   = (e9 - e13) / c
    df["e13s"]  = (e13 - e21) / c
    df["eslp"]  = e9.diff(3) / c

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    a10 = tr.rolling(10).mean(); a14 = tr.rolling(14).mean(); a50 = tr.rolling(50).mean()
    df["a10"] = a10/c; df["a14"] = a14/c
    df["ar"]  = a10 / a14.replace(0, np.nan)
    df["ava"] = a10 / a50.replace(0, np.nan)

    # RSI
    for p in [7, 14]:
        d2 = c.diff()
        g  = d2.clip(lower=0).ewm(span=p, adjust=False).mean()
        ls = (-d2.clip(upper=0)).ewm(span=p, adjust=False).mean()
        df[f"rsi{p}"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))
    df["rsis"] = df["rsi14"].diff(3)

    # MACD
    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ms  = mac.ewm(span=9, adjust=False).mean()
    df["mh"]   = (mac - ms) / c
    df["mx"]   = np.sign(mac - ms)
    df["mslp"] = df["mh"].diff(2)

    # Stochastic
    for k in [5, 14]:
        lk = l.rolling(k).min(); hk = h.rolling(k).max()
        df[f"st{k}"] = 100 * (c - lk) / (hk - lk + 1e-9)
    df["stx"] = np.sign(df["st5"] - df["st14"])

    # Williams %R
    df["wr14"] = -100 * (h.rolling(14).max() - c) / (
        h.rolling(14).max() - l.rolling(14).min() + 1e-9)

    # CCI
    tp2 = (h + l + c) / 3
    df["cci20"] = (tp2 - tp2.rolling(20).mean()) / (
        0.015 * tp2.rolling(20).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True) + 1e-9)

    # Realized volatility
    for w in [5, 10, 20]:
        df[f"rv{w}"] = df["log_ret"].rolling(w).std() * np.sqrt(252*75)

    # Bollinger Bands
    bm = c.rolling(20).mean(); bs = c.rolling(20).std() + 1e-9
    df["bbw"]  = 4*bs/bm
    df["bbp"]  = (c - (bm - 2*bs)) / (4*bs)
    df["bbsq"] = (df["bbw"] < df["bbw"].rolling(50).quantile(0.2)).astype(int)

    # Parkinson & GK volatility
    df["park"] = np.sqrt((1/(4*np.log(2))) * (np.log(h/l)**2).rolling(10).mean() * 252*75)
    gk = 0.5*np.log(h/l)**2 - (2*np.log(2)-1)*np.log(c/o)**2
    df["gkv"]  = np.sqrt(gk.rolling(10).mean() * 252*75)
    df["vov"]  = df["rv10"].rolling(20).std()

    # SMC: BOS, CHoCH, FVG, Order Blocks, Liquidity Sweeps
    swing_h = h.rolling(10).max(); swing_l = l.rolling(10).min()
    df["bos_bull"] = (c > swing_h.shift(1)).astype(int)
    df["bos_bear"] = (c < swing_l.shift(1)).astype(int)
    hh = h.rolling(5).max(); ll = l.rolling(5).min()
    prev_hh = hh.shift(5);   prev_ll = ll.shift(5)
    df["choch_bull"] = ((ll > prev_ll) & (hh.shift(1) < prev_hh.shift(1))).astype(int)
    df["choch_bear"] = ((hh < prev_hh) & (ll.shift(1) > prev_ll.shift(1))).astype(int)
    df["fvg_bull_prev"] = (l.shift(-1) > h.shift(1)).astype(int).shift(2)
    df["fvg_bear_prev"] = (h.shift(-1) < l.shift(1)).astype(int).shift(2)
    range_vol = fr.rolling(10).std() / c
    range_narrow = range_vol < range_vol.rolling(50).quantile(0.2)
    df["ob_bull"] = (range_narrow.shift(1) & (c > h.rolling(10).max().shift(1))).astype(int)
    df["ob_bear"] = (range_narrow.shift(1) & (c < l.rolling(10).min().shift(1))).astype(int)
    p20h = h.rolling(20).max().shift(1); p20l = l.rolling(20).min().shift(1)
    df["liq_sweep_high"] = ((h > p20h) & (c < p20h)).astype(int)
    df["liq_sweep_low"]  = ((l < p20l) & (c > p20l)).astype(int)

    # Volume proxy (CMF)
    mfm = ((c-l) - (h-c)) / fr
    df["cmf_proxy"] = (mfm*fr).rolling(20).sum() / (fr.rolling(20).sum() + 1e-9)

    # Z-score mean reversion
    df["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)
    df["zscore_50"] = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-9)

    # Volatility regime
    rv20 = df["rv20"]
    rv_med = rv20.rolling(252).median()
    df["vol_regime"]      = np.where(rv20 > rv_med*1.5, 2, np.where(rv20 < rv_med*0.7, 0, 1))
    df["vol_percentile"]  = rv20.rolling(252).rank(pct=True)

    # Intraday trend (5-min bars)
    # EMA 50  = 250 min = ~3.5 hours (session trend)
    # EMA 200 = 1000 min = ~2.5 days (multi-day trend)
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df["dow_primary"]   = (ema50 > ema200).astype(int)
    df["dow_secondary"] = (c > ema50).astype(int)

    # Session + time
    ts = df.index.strftime("%H:%M")
    df["is_morning"]   = ((ts >= "09:15") & (ts <= "10:30")).astype(int)
    df["is_midday"]    = ((ts >  "10:30") & (ts <= "13:30")).astype(int)
    df["is_afternoon"] = ((ts >  "13:30") & (ts <= "15:30")).astype(int)
    # Expiry day & DTE — derived from real historical expiry calendar
    # (handles holiday shifts + Sept 2025 Thursday→Tuesday transition).
    # File: data/nifty_expiry_history.csv
    df["dow"] = df.index.dayofweek
    try:
        exp_df = pd.read_csv("data/nifty_expiry_history.csv", parse_dates=["expiry_date"])
        expiries = sorted(exp_df["expiry_date"].dt.date.tolist())

        def _dte_for_date(bar_date):
            for exp in expiries:
                if exp >= bar_date:
                    return min(5, (exp - bar_date).days)
            return 3

        def _exp_dow_for_date(bar_date):
            # weekday of the *next* expiry (calendar-driven, not bar's own dow)
            for exp in expiries:
                if exp >= bar_date:
                    return exp.weekday()
            return 1  # default Tue

        bar_dates = df.index.normalize()
        unique_dates = pd.Series(bar_dates.unique()).dt.date
        dte_lookup = {d: _dte_for_date(d) for d in unique_dates}
        edow_lookup = {d: _exp_dow_for_date(d) for d in unique_dates}
        df["dte"] = bar_dates.date
        df["expiry_dow"] = df["dte"].map(edow_lookup).astype(int)
        df["dte"] = df["dte"].map(dte_lookup).astype(int)
        df["is_expiry"] = (df["dte"] == 0).astype(int)
        # Era flag derived from expiry calendar (Tue era = expiry_dow==1)
        df["expiry_is_tue"] = (df["expiry_dow"] == 1).astype(int)
        log.info(f"  DTE+expiry_dow computed from {len(expiries)} real historical expiries")
    except Exception as e:
        log.warning(f"  Real-expiry DTE failed ({e}); falling back to Tuesday weekday")
        df["is_expiry"] = (df.index.dayofweek == 1).astype(int)
        dte_map = {0: 1, 1: 0, 2: 5, 3: 4, 4: 3}
        df["dte"] = df.index.dayofweek.map(dte_map).fillna(3).astype(int)
        df["expiry_dow"] = 1
        df["expiry_is_tue"] = 1
    df["dte_norm"] = df["dte"] / 6.0

    log.info(f"  5-min features computed: {len([c for c in df.columns if c not in ['open','high','low','close']])} cols")
    return df


# ==================================================================
# HIGHER-TF FEATURES (15-min, 30-min, 60-min, Daily)
# ==================================================================

def features_15min(df15):
    """
    15-min features — ADX trend state, RSI, MACD, EMA.
    These are the most important: tell us if market is trending or ranging.
    """
    c, h, l = df15["close"], df15["high"], df15["low"]
    out = pd.DataFrame(index=df15.index)

    # ADX (14) — core trend/sideways detector
    adx_df = compute_adx(df15, period=14)
    out["tf15_adx"]       = adx_df["adx"]
    out["tf15_diplus"]    = adx_df["diplus"]
    out["tf15_diminus"]   = adx_df["diminus"]
    out["tf15_trending"]  = (adx_df["adx"] > 25).astype(int)   # 1 = trend
    out["tf15_sideways"]  = (adx_df["adx"] < 20).astype(int)   # 1 = range
    out["tf15_bull_trend"]= ((adx_df["adx"] > 25) & (adx_df["diplus"] > adx_df["diminus"])).astype(int)
    out["tf15_bear_trend"]= ((adx_df["adx"] > 25) & (adx_df["diminus"] > adx_df["diplus"])).astype(int)

    # RSI 14 on 15-min
    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["tf15_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    # MACD histogram on 15-min
    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ms  = mac.ewm(span=9, adjust=False).mean()
    out["tf15_macd_hist"] = (mac - ms) / c

    # EMA alignment on 15-min
    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    out["tf15_ema_bull"]  = (e9 > e21).astype(int)
    out["tf15_ema_dist"]  = (c - e21) / c

    # ATR ratio (volatility expanding/contracting)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    a5  = tr.rolling(5).mean()
    a14 = tr.rolling(14).mean()
    out["tf15_atr_ratio"] = a5 / a14.replace(0, np.nan)

    # Bollinger squeeze on 15-min
    bm = c.rolling(20).mean(); bs = c.rolling(20).std() + 1e-9
    bbw = 4*bs/bm
    out["tf15_bb_squeeze"] = (bbw < bbw.rolling(30).quantile(0.2)).astype(int)
    out["tf15_bb_pos"]     = (c - (bm - 2*bs)) / (4*bs)

    log.info(f"  15-min features: {len(out.columns)} cols")
    return out


def features_30min(df30):
    """30-min features — intermediate trend direction."""
    c, h, l = df30["close"], df30["high"], df30["low"]
    out = pd.DataFrame(index=df30.index)

    # ADX on 30-min
    adx_df = compute_adx(df30, period=14)
    out["tf30_adx"]        = adx_df["adx"]
    out["tf30_bull_trend"] = ((adx_df["adx"] > 25) & (adx_df["diplus"] > adx_df["diminus"])).astype(int)
    out["tf30_bear_trend"] = ((adx_df["adx"] > 25) & (adx_df["diminus"] > adx_df["diplus"])).astype(int)

    # RSI on 30-min
    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["tf30_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    # EMA alignment
    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    out["tf30_ema_bull"]  = (e9 > e21).astype(int)
    out["tf30_ema_dist"]  = (c - e21) / c

    # 30-min high/low range (session range so far)
    out["tf30_close_pos"] = (c - l.rolling(5).min()) / (
        h.rolling(5).max() - l.rolling(5).min() + 1e-9)

    log.info(f"  30-min features: {len(out.columns)} cols")
    return out


def features_60min(df60):
    """60-min features — session-level trend."""
    c, h, l = df60["close"], df60["high"], df60["low"]
    out = pd.DataFrame(index=df60.index)

    # ADX on 60-min
    adx_df = compute_adx(df60, period=14)
    out["tf60_adx"]        = adx_df["adx"]
    out["tf60_bull_trend"] = ((adx_df["adx"] > 25) & (adx_df["diplus"] > adx_df["diminus"])).astype(int)
    out["tf60_bear_trend"] = ((adx_df["adx"] > 25) & (adx_df["diminus"] > adx_df["diplus"])).astype(int)

    # RSI on 60-min
    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=9, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=9, adjust=False).mean()
    out["tf60_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    # EMA distance on 60-min
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    out["tf60_ema_bull"]  = (e21 > e50).astype(int)
    out["tf60_ema_dist"]  = (c - e50) / c

    # 60-min session high/low position
    out["tf60_close_pos"] = (c - l.rolling(4).min()) / (
        h.rolling(4).max() - l.rolling(4).min() + 1e-9)

    log.info(f"  60-min features: {len(out.columns)} cols")
    return out


def features_daily(df_day):
    """
    Daily features — previous day H/L/C, gap, weekly trend.
    EMAs aligned to options timeframes (max 50 days expiry):
      EMA 9  = 2-week trend  (most relevant for weekly options)
      EMA 20 = 1-month trend (relevant for current expiry)
      EMA 50 = 50-day trend  (matches max options expiry)
    EMA 200 removed — irrelevant for options under 50 days.
    """
    c, h, l, o = df_day["close"], df_day["high"], df_day["low"], df_day["open"]
    out = pd.DataFrame(index=df_day.index)

    # Previous day levels (shift by 1 — no lookahead)
    out["day_prev_high"]  = h.shift(1)
    out["day_prev_low"]   = l.shift(1)
    out["day_prev_close"] = c.shift(1)

    # Day open gap vs previous close
    out["day_gap_pct"]   = (o - c.shift(1)) / c.shift(1)
    out["day_gap_up"]    = out["day_gap_pct"].clip(lower=0)
    out["day_gap_down"]  = out["day_gap_pct"].clip(upper=0).abs()
    out["day_big_gap"]   = (out["day_gap_pct"].abs() > 0.005).astype(int)

    # Weekly high/low (5-day)
    out["week_high"] = h.rolling(5).max().shift(1)
    out["week_low"]  = l.rolling(5).min().shift(1)

    # Options-relevant EMAs on daily
    e9  = c.ewm(span=9,  adjust=False).mean()   # 2-week — weekly options
    e20 = c.ewm(span=20, adjust=False).mean()   # 1-month — current expiry
    e50 = c.ewm(span=50, adjust=False).mean()   # 50-day  — max expiry

    out["day_ema9_bull"]  = (e9  > e20).astype(int)   # short-term bull
    out["day_ema_bull"]   = (e20 > e50).astype(int)   # medium-term bull
    out["day_ema9_dist"]  = (c - e9)  / c             # distance from 2-week EMA
    out["day_ema_dist"]   = (c - e20) / c             # distance from 1-month EMA
    out["day_ema50_dist"] = (c - e50) / c             # distance from expiry EMA

    # Position within weekly range
    out["day_week_pos"] = (c - out["week_low"]) / (
        out["week_high"] - out["week_low"] + 1e-9)

    # Daily RSI
    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["day_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    # Daily ATR (normalized)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out["day_atr_pct"] = tr.rolling(14).mean() / c

    log.info(f"  Daily features: {len(out.columns)} cols")
    return out


# ==================================================================
# INDIA VIX FEATURES
# ==================================================================

def load_vix(path):
    """
    Load India VIX daily CSV.
    NSE format: Date, Open, High, Low, Close (VIX level)
    Download from: https://www.nseindia.com/reports-indices-historical-vix
    Or use: data/india_vix.csv fetched via update_data.py
    """
    if not Path(path).exists():
        log.warning(f"VIX file not found: {path} — VIX features will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    # Handle NSE date format (e.g. "20-Mar-2026" or "2026-03-20")
    dt = next((c for c in df.columns if "date" in c), None)
    if not dt:
        log.warning("VIX CSV has no date column")
        return pd.DataFrame()
    df[dt] = pd.to_datetime(df[dt], format="mixed", dayfirst=True)
    df = df.set_index(dt).sort_index()
    # Get close column
    close_col = next((c for c in df.columns if "close" in c), None)
    if not close_col:
        # Some NSE files have just one data column
        df.columns = ["vix"]
    else:
        df = df.rename(columns={close_col: "vix"})
    df = df[["vix"]].astype(float).dropna()
    df = df[~df.index.duplicated(keep="last")]
    log.info(f"VIX : {len(df):,} days | {df.index[0].date()} → {df.index[-1].date()} | "
             f"range={df['vix'].min():.1f}–{df['vix'].max():.1f}")
    return df


def features_vix(df_vix):
    """
    India VIX features merged onto daily bars.
    VIX is the single most important missing feature:
      - High VIX = options expensive + wide moves + SL hits more often
      - Low VIX  = options cheap + small moves + TP rarely hit
      - Ideal VIX (12-18) = best option buying conditions

    Features:
      vix_level     : raw VIX value
      vix_regime    : 0=calm(<12) 1=normal(12-16) 2=elevated(16-20) 3=high(20-25) 4=very_high(25-30) 5=extreme(>30)
      vix_pct5      : VIX percentile over 5-day rolling window
      vix_pct20     : VIX percentile over 20-day rolling window
      vix_rising    : VIX going up (fear increasing)
      vix_spike     : VIX > previous 5-day high (sudden fear spike)
      vix_ideal     : VIX in 12-18 range (best for option buying)
      vix_danger    : VIX > 20 (avoid buying options)
      vix_change1   : 1-day VIX change (rate of fear change)
      vix_change5   : 5-day VIX change (trend of fear)
    """
    v = df_vix["vix"]
    out = pd.DataFrame(index=df_vix.index)

    out["vix_level"]  = v
    out["vix_regime"] = pd.cut(v,
        bins=[0, 12, 16, 20, 25, 30, 999],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)
    out["vix_pct5"]   = v.rolling(5).rank(pct=True)
    out["vix_pct20"]  = v.rolling(20).rank(pct=True)
    out["vix_rising"] = (v > v.shift(1)).astype(int)
    out["vix_spike"]  = (v > v.rolling(5).max().shift(1)).astype(int)
    out["vix_ideal"]  = ((v >= 12) & (v <= 18)).astype(int)
    out["vix_danger"] = (v > 20).astype(int)
    out["vix_change1"]= v.pct_change(1)
    out["vix_change5"]= v.pct_change(5)

    log.info(f"  VIX features: {len(out.columns)} cols")
    return out


def merge_vix_onto_5min(df5, vix_feat):
    """
    Merge VIX daily features onto 5-min bars.
    Each 5-min bar gets the PREVIOUS day's VIX (no lookahead).
    """
    if vix_feat.empty:
        return df5
    df5 = df5.copy()
    df5["_date"] = df5.index.normalize()
    vix_feat = vix_feat.copy()
    vix_feat.index = pd.to_datetime(vix_feat.index).normalize()
    # Shift by 1 day — use yesterday's VIX (published same day, but use previous)
    vix_feat = vix_feat.shift(1)
    vix_feat = vix_feat[~vix_feat.index.duplicated(keep="last")]

    for col in vix_feat.columns:
        df5[col] = df5["_date"].map(vix_feat[col])

    df5.drop(columns=["_date"], inplace=True)
    return df5

def merge_htf(df5, htf_feat, prefix):
    """
    Merge higher-TF features onto 5-min bars.
    Uses merge_asof so each 5-min bar gets the LAST COMPLETED
    higher-TF bar — zero lookahead.
    """
    htf_reset = htf_feat.reset_index().rename(columns={"date": "ts", htf_feat.index.name: "ts"})
    if "ts" not in htf_reset.columns:
        htf_reset = htf_feat.reset_index()
        htf_reset.columns = ["ts"] + list(htf_feat.columns)

    df5_reset = df5.reset_index()
    df5_reset.columns = ["ts"] + list(df5.columns[0:])
    if df5_reset.columns[0] != "ts":
        df5_reset = df5_reset.rename(columns={df5_reset.columns[0]: "ts"})

    merged = pd.merge_asof(
        df5_reset.sort_values("ts"),
        htf_reset.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged = merged.set_index("ts")
    # Keep only new HTF columns
    new_cols = [c for c in merged.columns if c not in df5.columns]
    result = df5.copy()
    for col in new_cols:
        result[col] = merged[col].values
    return result


def merge_daily_onto_5min(df5, day_feat):
    """
    Special daily merge: each 5-min bar on day D gets the features
    from the PREVIOUS day's close (day D-1). No lookahead.
    """
    df5 = df5.copy()
    df5["_date"] = df5.index.normalize()
    day_feat = day_feat.copy()
    day_feat.index = pd.to_datetime(day_feat.index).normalize()
    # Drop duplicate dates — keep last (most complete bar of that day)
    day_feat = day_feat[~day_feat.index.duplicated(keep="last")]

    for col in day_feat.columns:
        df5[col] = df5["_date"].map(day_feat[col])

    df5.drop(columns=["_date"], inplace=True)
    return df5

def add_intraday_context(df5):
    """
    Add features that need the current day's 5-min data:
    - Distance from day open
    - Distance from prev day H/L (already merged from daily)
    - Position within day's range so far
    """
    c = df5["close"]

    # Day open (first bar of each day)
    df5["day_open"] = df5.groupby(df5.index.date)["open"].transform("first")
    df5["dist_from_open"]    = (c - df5["day_open"]) / df5["day_open"]

    # Position in today's range (cumulative high/low within day)
    df5["intraday_high"] = df5.groupby(df5.index.date)["high"].cummax()
    df5["intraday_low"]  = df5.groupby(df5.index.date)["low"].cummin()
    df5["intraday_pos"]  = (c - df5["intraday_low"]) / (
        df5["intraday_high"] - df5["intraday_low"] + 1e-9)

    # Distance from previous day levels (if available)
    if "day_prev_high" in df5.columns:
        df5["dist_prev_high"] = (c - df5["day_prev_high"]) / c
        df5["dist_prev_low"]  = (c - df5["day_prev_low"])  / c
        df5["dist_prev_close"]= (c - df5["day_prev_close"])/ c
        df5["above_prev_high"]= (c > df5["day_prev_high"]).astype(int)
        df5["below_prev_low"] = (c < df5["day_prev_low"]).astype(int)

    # Distance from week high/low
    if "week_high" in df5.columns:
        df5["dist_week_high"] = (c - df5["week_high"]) / c
        df5["dist_week_low"]  = (c - df5["week_low"])  / c

    # ── Intraday VWAP features ───────────────────────────────────────
    # Added 2026-04-16 to fix downtrend blindness: on trending days the model
    # was predicting CALL while price dropped 200+ pts below VWAP. Teaching
    # it "we've been below VWAP for 18 bars" gives the trend context it needs.
    #
    # Nifty is an INDEX — no real volume. Historical CSVs have volume=0 or
    # missing. We use weights per-day: real volume if present, else equal
    # weight (typical-price cumulative mean). Same math either way — just
    # whether we re-weight bars by turnover.
    typ = (df5["high"] + df5["low"] + df5["close"]) / 3.0
    if "volume" in df5.columns:
        vol_raw = df5["volume"].astype(float).fillna(0.0)
    else:
        vol_raw = pd.Series(0.0, index=df5.index)

    _date = pd.Series(df5.index.date, index=df5.index, name="_d")

    # Per-day mask: does this day have any real volume?
    day_vol_sum = vol_raw.groupby(_date).transform("sum")
    # Where day has no volume, use 1.0 (equal weight); else use real volume
    vol = np.where(day_vol_sum > 0, vol_raw, 1.0)
    vol = pd.Series(vol, index=df5.index)

    cum_tpv = (typ * vol).groupby(_date).cumsum()
    cum_vol = vol.groupby(_date).cumsum().replace(0, np.nan)
    vwap = cum_tpv / cum_vol
    df5["vwap_session"]    = vwap
    df5["dist_vwap"]       = (c - vwap) / c                       # signed % distance
    df5["vwap_dist_pts"]   = c - vwap                             # signed point distance
    df5["above_vwap"]      = (c > vwap).astype(int)
    df5["below_vwap"]      = (c < vwap).astype(int)

    # Bars-since-VWAP-cross (how long have we been on this side?)
    # Streak counter resets at each day boundary and at each VWAP cross.
    side = np.sign((c - vwap).fillna(0)).astype(int)             # +1 above, -1 below, 0 unknown
    # Build a streak key that changes whenever day OR side changes
    side_change = (side != side.shift()) | (_date != _date.shift())
    streak_grp = side_change.cumsum()
    streak = side.groupby(streak_grp).cumcount() + 1
    df5["bars_above_vwap"] = np.where(side > 0, streak, 0)
    df5["bars_below_vwap"] = np.where(side < 0, streak, 0)

    # Session range expansion (intraday_range grows = trending day, stalls = range day)
    day_range_pts = df5["intraday_high"] - df5["intraday_low"]
    df5["intraday_range_pts"] = day_range_pts
    df5["intraday_range_pct"] = day_range_pts / c

    # Signed session move so far (close - day_open) — positive bull day, negative bear day
    if "day_open" in df5.columns:
        df5["day_move_pts"] = c - df5["day_open"]
        df5["day_move_pct"] = (c - df5["day_open"]) / df5["day_open"]

    return df5


# ==================================================================
# LABELS — V8 original forward-return method
# Simple and effective: proved 86%+ win rate in backtest
# VIX is now a feature — model learns when not to trade from data
# ==================================================================

def compute_thresholds(df5, quantile=0.70):
    fwd = df5["close"].shift(-FWD_BARS) / df5["close"] - 1
    ts  = df5.index.strftime("%H:%M")
    am  = (ts >= "09:15") & (ts <= "10:30")
    mid = (ts >  "10:30") & (ts <= "13:30")
    pm  = (ts >  "13:30") & (ts <= "15:30")
    def _q(arr, mask):
        v = arr[mask].dropna().abs()
        return float(np.quantile(v, quantile)) if len(v) > 0 else 0.001
    am_t, mid_t, pm_t = _q(fwd, am), _q(fwd, mid), _q(fwd, pm)
    log.info(f"  Thresholds: AM={am_t:.5f} MID={mid_t:.5f} PM={pm_t:.5f}")
    return am_t, mid_t, pm_t


def create_labels(df, df15=None):
    """
    V8.1 Range-based RSI+ADX labels:

    RSI Zone + ADX Zone = Trading Decision:
      RSI 0-30  + ADX > 40  -> PUT allowed  (strong downtrend, not just dip)
      RSI 0-30  + ADX < 25  -> SKIP or CALL (weak dip, possible bounce)
      RSI 30-70 + ADX > 20  -> CALL/PUT     (normal setup, trend present)
      RSI 30-70 + ADX < 20  -> SKIP         (flat, no direction)
      RSI 70-100+ ADX > 40  -> CALL allowed (strong uptrend, not just spike)
      RSI 70-100+ ADX < 25  -> SKIP or PUT  (overbought spike, possible fade)

    2026-03-30 fix: RSI=25 + ADX=63 = strong downtrend, old model said SKIP.
    """
    fwd = df["close"].shift(-FWD_BARS) / df["close"] - 1
    df["label"] = 2  # default SKIP
    ts  = df.index.strftime("%H:%M")
    am  = (ts >= "09:15") & (ts <= "10:30")
    mid = (ts >  "10:30") & (ts <= "13:30")
    pm  = (ts >  "13:30") & (ts <  "15:00")
    in_session = am | mid | pm

    # Session-specific forward-return thresholds
    am_t, mid_t, pm_t = compute_thresholds(df)
    bar_thresh = pd.Series(0.001, index=df.index)
    bar_thresh[am]  = am_t
    bar_thresh[mid] = mid_t
    bar_thresh[pm]  = pm_t

    # --- RSI (5-min) ---
    rsi = df["rsi14"] if "rsi14" in df.columns else pd.Series(50.0, index=df.index)

    # --- ADX: use 15-min if available, else compute from 5-min ---
    if "tf15_adx" in df.columns:
        adx = df["tf15_adx"]
    else:
        adx_df = compute_adx(df, period=14)
        adx = adx_df["adx"]

    # --- RSI+ADX range zones ---
    rsi_oversold   = rsi < 30
    rsi_overbought = rsi > 70
    rsi_normal     = (~rsi_oversold) & (~rsi_overbought)

    adx_strong     = adx > 40
    adx_trending   = adx > 20
    adx_weak       = adx <= 25

    # ZONE 1: Normal RSI (30-70) — standard labeling
    normal_call = in_session & rsi_normal & adx_trending & (fwd > bar_thresh)
    normal_put  = in_session & rsi_normal & adx_trending & (fwd < -bar_thresh)
    df.loc[normal_call, "label"] = 0
    df.loc[normal_put,  "label"] = 1

    # ZONE 2: Oversold RSI (<30) — PUT if strong trend, CALL if weak (bounce)
    oversold_put = (in_session & rsi_oversold & adx_strong
                    & (fwd < -bar_thresh * 0.7))
    df.loc[oversold_put, "label"] = 1

    oversold_call = (in_session & rsi_oversold & adx_weak
                     & (fwd > bar_thresh * 1.2))
    df.loc[oversold_call, "label"] = 0

    # ZONE 3: Overbought RSI (>70) — CALL if strong trend, PUT if weak (fade)
    overbought_call = (in_session & rsi_overbought & adx_strong
                       & (fwd > bar_thresh * 0.7))
    df.loc[overbought_call, "label"] = 0

    overbought_put = (in_session & rsi_overbought & adx_weak
                      & (fwd < -bar_thresh * 1.2))
    df.loc[overbought_put, "label"] = 1

    # VIX: only EXTREME (>35) forces SKIP — rest handled by dynamic regime
    if "vix_level" in df.columns:
        extreme_vix = df["vix_level"] > 35.0
        df.loc[extreme_vix, "label"] = 2
        log.info(f"  VIX>35 forced SKIP: {extreme_vix.sum():,} bars")

    # Log zone statistics
    log.info(f"  RSI+ADX zones: normal={normal_call.sum()+normal_put.sum():,} | "
             f"oversold->PUT={oversold_put.sum():,} oversold->CALL={oversold_call.sum():,} | "
             f"overbought->CALL={overbought_call.sum():,} overbought->PUT={overbought_put.sum():,}")

    lc = df["label"].value_counts().sort_index()
    log.info(f"  Labels: CALL={lc.get(0,0):,} PUT={lc.get(1,0):,} SKIP={lc.get(2,0):,}")
    return df


# ==================================================================
# MAIN TRAINING
# ==================================================================

def train(csv5, csv15=None, csv30=None, csv60=None, csv_day=None, csv_vix=None):
    log.info("=" * 62)
    log.info("  NIFTY V8 — MULTI-TIMEFRAME MODEL TRAINING")
    log.info("  15-min ADX · 30-min trend · 60-min bias · Daily S/R · VIX")
    log.info("=" * 62)

    # ── Load 5-min base ──────────────────────────────────────────
    df5 = load_5min(csv5)

    # ── Load / resample higher TFs ───────────────────────────────
    # 15-min (combine uploaded file + resample recent from 5-min)
    if csv15 and Path(csv15).exists():
        df15_upload = load_higher(csv15, "15-min (uploaded)")
        df15_recent = resample_from_5min(
            df5[df5.index > df15_upload.index[-1]], "15min"
        )
        df15 = pd.concat([df15_upload, df15_recent]).sort_index()
        log.info(f"15-min combined: {len(df15):,} bars → {df15.index[-1].date()}")
    else:
        log.info("15-min CSV not found — resampling from 5-min")
        df15 = resample_from_5min(df5, "15min")

    # 30-min
    if csv30 and Path(csv30).exists():
        df30 = load_higher(csv30, "30-min")
    else:
        log.info("30-min CSV not found — resampling from 5-min")
        df30 = resample_from_5min(df5, "30min")

    # 60-min
    if csv60 and Path(csv60).exists():
        df60 = load_higher(csv60, "60min")
    else:
        log.info("60-min CSV not found — resampling from 5-min")
        df60 = resample_from_5min(df5, "60min")

    # Daily
    if csv_day and Path(csv_day).exists():
        df_day = load_higher(csv_day, "Daily")
        df_day = df_day[~df_day.index.duplicated(keep="last")]
    else:
        log.info("Daily CSV not found — resampling from 5-min")
        df_day = resample_from_5min(df5, "D")

    # ── Compute higher-TF feature tables ─────────────────────────
    log.info("\nComputing higher-TF features...")
    feat15   = features_15min(df15)
    feat30   = features_30min(df30)
    feat60   = features_60min(df60)
    feat_day = features_daily(df_day)

    # ── India VIX features ────────────────────────────────────────
    vix_path = csv_vix or CSV_VIX
    df_vix   = load_vix(vix_path)
    feat_vix = features_vix(df_vix) if not df_vix.empty else pd.DataFrame()

    # ── Compute 5-min base features ──────────────────────────────
    log.info("\nComputing 5-min base features...")
    df_feat = features_5min(df5)

    # ── Merge higher-TF onto 5-min ───────────────────────────────
    log.info("\nMerging timeframes (no-lookahead merge_asof)...")
    df_feat = merge_htf(df_feat, feat15,  "15m")
    df_feat = merge_htf(df_feat, feat30,  "30m")
    df_feat = merge_htf(df_feat, feat60,  "60m")
    df_feat = merge_daily_onto_5min(df_feat, feat_day)
    if not feat_vix.empty:
        df_feat = merge_vix_onto_5min(df_feat, feat_vix)
        log.info(f"  VIX features merged onto 5-min bars")
    else:
        log.warning("  VIX features NOT merged — download india_vix.csv from NSE")
    df_feat = add_intraday_context(df_feat)

    log.info(f"Total columns after merge: {len(df_feat.columns)}")

    # ── Labels — V8 original forward-return ─────────────────────
    df_feat = create_labels(df_feat)
    df_feat.dropna(inplace=True)

    # Feature columns
    SKIP = {"label","open","high","low","close","day_open",
            "intraday_high","intraday_low"}
    FCOLS = [c for c in df_feat.columns if c not in SKIP
             and df_feat[c].dtype in [np.float64, np.int64, np.int32, float, int]]
    X = df_feat[FCOLS].values
    y = df_feat["label"].values

    log.info(f"\nTraining on {len(df_feat):,} rows | {len(FCOLS)} features")
    log.info(f"  5-min base features + {len([c for c in FCOLS if c.startswith('tf')])} HTF features")

    # ── Train/val split (97/3 time-based — keep Tuesday era IN training) ──
    sp = int(len(X) * 0.97)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[:sp]);  Xvl = sc.transform(X[sp:])
    ytr, yvl = y[:sp], y[sp:]

    # Class weights
    skip_pct  = (ytr == 2).mean()
    trade_pct = 1 - skip_pct
    trade_w   = skip_pct / trade_pct if trade_pct > 0 else 2.0
    sample_w  = np.where(ytr == 2, 1.0, trade_w)

    # ── Calendar-driven expiry-day boost (uses expiry_dow from real calendar) ──
    # Boosts rows whose NEXT expiry is Tuesday — works for any future day-change
    # because the source of truth is data/nifty_expiry_history.csv.
    try:
        if "expiry_is_tue" in df_feat.columns:
            _tue_era = df_feat["expiry_is_tue"].values[:sp].astype(bool)
        else:
            _tue_era = np.zeros(sp, dtype=bool)
        _boost = np.where(_tue_era, 6.0, 1.0)
        sample_w = sample_w * _boost
        log.info(f"Expiry-calendar boost (next expiry == Tue): {_tue_era.sum():,}/{len(_tue_era):,} rows × 6.0x")
    except Exception as _e:
        log.warning(f"Expiry-calendar boost skipped: {_e}")

    # ── Trending-day boost ──────────────────────────────────────────────
    # 2026-04-16 lesson: model missed a clean 250pt downtrend because trending
    # days are the minority in training data. Give them 3× weight so the
    # model pays attention to "price has been below VWAP for 18 bars and
    # day_move is -200pts" setups. This is SAMPLE weighting during training
    # only — no inference-time lookahead.
    try:
        df_train_slice = df_feat.iloc[:sp]
        bar_dates = df_train_slice.index.date
        # Compute each day's absolute open→close swing (|full-day move|)
        day_open_s  = df_train_slice.groupby(bar_dates)["open"].first()
        day_close_s = df_train_slice.groupby(bar_dates)["close"].last()
        day_move_s  = (day_close_s - day_open_s).abs()
        trending_days = set(day_move_s[day_move_s > 150.0].index)
        trend_mask = np.array([d in trending_days for d in bar_dates], dtype=bool)
        trend_boost = np.where(trend_mask, 3.0, 1.0)
        sample_w = sample_w * trend_boost
        log.info(
            f"Trending-day boost (|day_move|>150pts): "
            f"{trend_mask.sum():,}/{len(trend_mask):,} rows × 3.0x "
            f"({len(trending_days)} trending days)"
        )
    except Exception as _e:
        log.warning(f"Trending-day boost skipped: {_e}")

    log.info(f"Class weights: CALL/PUT={trade_w:.2f}x  SKIP=1.0x")

    models = {}

    if HAS_XLG:
        log.info("\nTraining XGBoost...")
        m = xgb.XGBClassifier(
            n_estimators=700, max_depth=5, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.5, min_child_weight=15,
            gamma=0.3, reg_alpha=1.5, reg_lambda=3.0,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", early_stopping_rounds=50,
            verbosity=0, n_jobs=-1,
        )
        m.fit(Xtr, ytr, sample_weight=sample_w,
              eval_set=[(Xvl, yvl)], verbose=False)
        models["xgb"] = m
        log.info("  XGBoost [OK]")

        log.info("Training LightGBM...")
        m = lgb.LGBMClassifier(
            n_estimators=700, max_depth=5, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.5, min_child_samples=30,
            reg_alpha=1.5, reg_lambda=3.0, num_leaves=28,
            objective="multiclass", num_class=3, verbose=-1, n_jobs=-1,
        )
        m.fit(Xtr, ytr, sample_weight=sample_w,
              eval_set=[(Xvl, yvl)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        models["lgb"] = m
        log.info("  LightGBM [OK]")

    else:
        log.info("\nUsing HistGBM (no xgboost/lightgbm)...")
        for i, name in enumerate(["hgb_a", "hgb_b", "hgb_c"]):
            log.info(f"  Training {name}...")
            m = HistGradientBoostingClassifier(
                max_iter=600, max_depth=5,
                learning_rate=0.02 + i*0.007,
                min_samples_leaf=30, l2_regularization=3.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=40, random_state=42 + i*17,
            )
            m.fit(Xtr, ytr)
            models[name] = m
            log.info(f"  {name} [OK]")

    # ── Save ─────────────────────────────────────────────────────
    Path(MODEL_PATH).parent.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    joblib.dump(sc,     SCALER_PATH)
    joblib.dump(FCOLS,  FCOLS_PATH)
    log.info(f"\n  Saved model  -> {MODEL_PATH}")
    log.info(f"  Saved scaler -> {SCALER_PATH}")
    log.info(f"  Saved fcols  -> {FCOLS_PATH}  ({len(FCOLS)} features)")

    # ── Validation ───────────────────────────────────────────────
    CONF = 0.22  # lower threshold for trade-sim trained model
    preds = np.mean([m.predict_proba(Xvl) for m in models.values()], axis=0)
    sigs  = np.where(preds.max(axis=1) >= CONF, preds.argmax(axis=1), 2)
    traded = sigs != 2

    log.info(f"\n{'='*62}")
    log.info("  VALIDATION RESULTS")
    log.info(f"{'='*62}")
    if traded.any():
        acc     = (yvl[traded] == sigs[traded]).mean()
        dir_msk = traded & (yvl != 2)
        dir_acc = (yvl[dir_msk] == sigs[dir_msk]).mean() if dir_msk.any() else 0
        nc = (sigs==0).sum(); np_=(sigs==1).sum(); ns=(sigs==2).sum()
        log.info(f"  Signals : CALL={nc}  PUT={np_}  SKIP={ns}")
        log.info(f"  Overall accuracy   : {acc:.1%}")
        log.info(f"  Direction accuracy : {dir_acc:.1%}")

        # Feature importance
        model_for_imp = models.get("xgb") or models.get("lgb") or list(models.values())[0]
        if hasattr(model_for_imp, "feature_importances_"):
            imp = model_for_imp.feature_importances_
            top = np.argsort(imp)[-25:][::-1]
            log.info(f"\n  Top 25 features:")
            htf_in_top = 0
            for i in top:
                is_htf = any(FCOLS[i].startswith(p) for p in
                             ["tf15","tf30","tf60","day_","dist_","week_",
                              "intraday","above_","below_"])
                marker = " <-- HTF" if is_htf else ""
                log.info(f"    {FCOLS[i]:30s} {imp[i]:.4f}{marker}")
                if is_htf: htf_in_top += 1
            log.info(f"\n  HTF features in top 25: {htf_in_top}/25")
    else:
        log.warning("  No signals in validation — CONFIDENCE may be too high")

    log.info(f"\n{'='*62}")
    log.info("  V8 Training complete!")
    log.info(f"  Model files saved to: models/")
    log.info(f"  To use in backtest: update MODEL_PATH in backtest.py")
    log.info(f"  To use live: update MODEL_PATH in ml_engine.py")
    log.info(f"{'='*62}\n")


# ==================================================================
# ENTRY POINT
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nifty V8 multi-timeframe model")
    parser.add_argument("--csv5",   default=CSV_5M,   help="5-min CSV path")
    parser.add_argument("--csv15",  default=CSV_15M,  help="15-min CSV path")
    parser.add_argument("--csv30",  default=CSV_30M,  help="30-min CSV path")
    parser.add_argument("--csv60",  default=CSV_60M,  help="60-min CSV path")
    parser.add_argument("--csvday", default=CSV_DAY,  help="Daily CSV path")
    parser.add_argument("--csvvix", default=CSV_VIX,  help="India VIX daily CSV path")
    args = parser.parse_args()

    train(
        csv5    = args.csv5,
        csv15   = args.csv15,
        csv30   = args.csv30,
        csv60   = args.csv60,
        csv_day = args.csvday,
        csv_vix = args.csvvix,
    )
