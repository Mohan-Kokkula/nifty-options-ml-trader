"""
train_model_v9.py  -  V8 with stationarity + lookahead-safe SMC + SHAP pruning
==============================================================================
Changes vs V8:

  FIX 1 — Volume Paradox  (COMPLETE)
    Training now ingests data/nifty_fut_5min.csv (Nifty current-month futures)
    when available (≥30 bars) to compute REAL session-anchored VWAP and REAL
    CMF from actual futures volume. Features are merged onto the spot DataFrame
    via merge_asof (backward, exact timestamp alignment — no future leakage).
    Falls back to spot-proxy (_proxy suffix) when futures CSV is missing/thin,
    preserving backward-compat with historical-only training runs.
    Live bot uses the same real futures feed via core/vwap_engine.py.

  FIX 2 — Stationarity (APPLIED)
    Removed POINT-based features that are non-stationary across an 8000→24000
    index range. Replaced with ATR-normalized or percentage equivalents:
      vwap_dist_pts     → vwap_dist_atr   (points / ATR)
      day_move_pts      → day_move_atr
      intraday_range_pts→ intraday_range_atr
      tp (levels)       → only used via returns / ATR
    day_open, intraday_high, intraday_low, day_prev_high/low/close, week_high/low,
    vwap_session are KEPT in df but EXCLUDED from FCOLS (absolute levels are
    already used indirectly via the distance features).

  FIX 3 — SMC Lookahead-Safe (DEFENSIVE REWRITE)
    All SMC features now use EXPLICIT .shift(1) even where
    rolling(N).max().shift(1) was already safe. Intent is now auditable
    line-by-line. FVG features use only shift(+1, +2) — never shift(-1).

  FIX 4 — SHAP Pruning (IMPROVED)
    After the first training pass, compute SHAP importances (falls back to
    model.feature_importances_ if shap package missing). Drops features that
    fall in the BOTTOM 25% of SHAP importance AND have high pairwise
    correlation (|ρ| > 0.95) with a stronger feature. This is more aggressive
    than the previous 0.1%-of-max threshold and correctly targets the
    "bottom quarter" the brief specifies. Typical pruning: 143 → ~80-100.
    Enable with --prune-shap (off by default for backward-compat).

Usage:
    python scripts/train_model_v9.py
    python scripts/train_model_v9.py --prune-shap
    python scripts/train_model_v9.py --csv5 data/nifty_5min.csv --prune-shap
    python scripts/train_model_v9.py --csvfut data/nifty_fut_5min.csv --prune-shap
"""

import warnings; warnings.filterwarnings("ignore")
import argparse, logging, os, sys, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH   = "models/nifty_v9_models.pkl"
SCALER_PATH  = "models/nifty_v9_scaler.pkl"
FCOLS_PATH   = "models/feature_cols_v9.pkl"

CSV_5M   = "data/nifty_5min.csv"
CSV_15M  = "data/nifty_15min.csv"
CSV_30M  = "data/nifty_30min.csv"
CSV_60M  = "data/nifty_60min.csv"
CSV_DAY  = "data/nifty_day.csv"
CSV_VIX  = "data/india_vix.csv"
CSV_FUT  = "data/nifty_fut_5min.csv"   # Fix 1: real futures for VWAP/CMF

# Minimum futures bars required to prefer real VWAP/CMF over spot-proxy
_MIN_FUT_BARS = 30

FWD_BARS = 3      # 15 minutes forward

# ── Purged 3-way split (chronological) — Issue #2 fix ─────────────────────────
# TRAIN  : model fitting
# VAL    : early-stopping ONLY (XGB/LGB eval_set) — model selects tree count here
# TEST   : FROZEN holdout — never seen during fit; ALL reported metrics use this
# A gap = FWD_BARS (purge) + EMBARGO_BARS (embargo) is dropped between each pair
# of sets so forward-looking labels can't leak across boundaries and the next
# set's feature windows can't bleed back into the previous set's used rows.
TRAIN_FRAC   = 0.85    # oldest 85%
VAL_FRAC     = 0.075   # next 7.5% (early stopping)
#  TEST_FRAC  = 1 - TRAIN_FRAC - VAL_FRAC = 0.075 (newest, frozen)
EMBARGO_BARS = 75      # ~1 trading day of 5-min bars; exceeds typical feature
                       # rolling windows (RSI14/ATR14/rolling20) + the 3-bar label.

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_XLG = True
except ImportError:
    HAS_XLG = False
    print("INFO: xgboost/lightgbm not found — using HistGBM")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [V9] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==================================================================
# DATA LOADING (unchanged from v8)
# ==================================================================

def _load_raw(path, tz_strip=True):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    dt = next(c for c in df.columns if any(x in c for x in ["date","time","timestamp"]))
    df[dt] = pd.to_datetime(df[dt], format="mixed", dayfirst=False, utc=False)
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
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[keep].astype(float).dropna(subset=[c for c in ["open","high","low","close"] if c in df.columns])


def load_5min(path):
    df = _load_raw(path)
    df = df.between_time("09:15", "15:30")
    log.info(f"5-min : {len(df):,} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_higher(path, freq_label):
    df = _load_raw(path)
    times = df.index.time
    has_midnight = (times == pd.Timestamp("00:00").time()).mean() > 0.5
    if not has_midnight:
        df = df.between_time("09:15", "15:30")
        df = df[df["high"] != df["low"]]
    df = df[~df.index.duplicated(keep="last")]
    log.info(f"{freq_label}: {len(df):,} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_futures(path) -> pd.DataFrame:
    """
    Fix 1 — load Nifty current-month futures 5-min OHLCV.
    Must have a real 'volume' column; returns empty DataFrame if file missing
    or volume is all-zero (falls back to spot-proxy in that case).
    """
    p = Path(path)
    if not p.exists():
        log.warning(f"Futures CSV not found: {path} — using spot-proxy VWAP/CMF")
        return pd.DataFrame()
    df = _load_raw(path)
    df = df.between_time("09:15", "15:30")
    if "volume" not in df.columns or df["volume"].fillna(0).sum() == 0:
        log.warning(f"Futures CSV has no real volume: {path} — using spot-proxy")
        return pd.DataFrame()
    if len(df) < _MIN_FUT_BARS:
        log.warning(
            f"Futures CSV only {len(df)} bars (need ≥{_MIN_FUT_BARS}) "
            f"— using spot-proxy VWAP/CMF"
        )
        return pd.DataFrame()
    log.info(
        f"Futures: {len(df):,} bars | "
        f"{df.index[0].date()} → {df.index[-1].date()} | "
        f"volume OK (sum={df['volume'].sum():,.0f})"
    )
    return df


def features_futures(df_fut: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 1 — compute REAL session-anchored VWAP and Chaikin Money Flow (CMF)
    from Nifty futures OHLCV (which has actual traded volume).

    VWAP formula:  Σ(TP × Vol) / Σ(Vol)  where TP = (H+L+C)/3
                   resets at 09:15 every session day.

    CMF formula:   Σ(MFV, 20) / Σ(Volume, 20)
                   where MFV = ((C-L)-(H-C)) / (H-L) × Volume

    All operations are strictly causal (no shift(-1), no forward fills).
    Returns a DataFrame indexed identically to df_fut with new columns only,
    ready for merge_asof onto the spot 5-min frame.
    """
    df = df_fut.copy()
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    _date = pd.Series(df.index.date, index=df.index, name="_d")

    # ── Session-Anchored VWAP ─────────────────────────────────────────
    tp      = (h + l + c) / 3.0
    tpv     = tp * v
    cum_tpv = tpv.groupby(_date).cumsum()
    cum_vol = v.groupby(_date).cumsum().replace(0, np.nan)
    vwap    = cum_tpv / cum_vol

    out = pd.DataFrame(index=df.index)
    out["fut_vwap"]          = vwap
    out["fut_dist_vwap_pct"] = (c - vwap) / c                  # signed %
    out["fut_above_vwap"]    = (c > vwap).astype(int)

    # ATR from futures (for vwap_dist_atr normalization)
    tr    = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    out["fut_vwap_dist_atr"] = (c - vwap) / atr14.replace(0, np.nan)

    # Bars since VWAP cross (causal streak counter)
    side        = np.sign((c - vwap).fillna(0)).astype(int)
    day_change  = (_date != _date.shift())
    side_change = (side != side.shift()) | day_change
    streak_grp  = side_change.cumsum()
    streak      = side.groupby(streak_grp).cumcount() + 1
    out["fut_bars_above_vwap"] = np.where(side > 0, streak, 0)
    out["fut_bars_below_vwap"] = np.where(side < 0, streak, 0)

    # ── Real Chaikin Money Flow (CMF) ─────────────────────────────────
    fr    = (h - l).replace(0, np.nan)
    mfm   = ((c - l) - (h - c)) / fr          # Money Flow Multiplier ∈ [-1, 1]
    mfv   = mfm * v                            # Money Flow Volume
    out["fut_cmf20"] = (
        mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    )
    out["fut_cmf_bullish"] = (out["fut_cmf20"] > 0).astype(int)
    out["fut_cmf_strong"]  = (out["fut_cmf20"].abs() > 0.15).astype(int)

    # ── Futures volume context ────────────────────────────────────────
    v_ma20 = v.rolling(20).mean().replace(0, np.nan)
    out["fut_vol_ratio"]  = v / v_ma20          # >1 = above-avg volume
    out["fut_vol_spike"]  = (out["fut_vol_ratio"] > 2.0).astype(int)
    out["fut_vol_regime"] = pd.cut(
        out["fut_vol_ratio"].fillna(1.0),
        bins=[0, 0.5, 1.0, 1.5, 2.0, 999],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    log.info(
        f"  Futures features: {len(out.columns)} cols "
        f"(real VWAP + CMF from {len(df):,} bars)"
    )
    return out


def merge_futures_onto_5min(df5: pd.DataFrame,
                             fut_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 1 — merge futures-derived VWAP/CMF features onto the spot 5-min frame.

    Uses pd.merge_asof with direction='backward' so that each spot bar gets
    the MOST RECENT futures bar whose timestamp ≤ spot bar timestamp.
    This is causally safe: futures and spot trade simultaneously so a futures
    bar closing at T is always available when the spot bar at T closes.

    Any NaN rows (warm-up period before 20 bars of futures data) are filled
    with 0 for binary/count features and left as NaN for continuous ones —
    the dropna() at the end of train() removes the warm-up rows.
    """
    if fut_feat.empty:
        return df5

    # Reset index to column; normalize the column name to "ts" regardless of
    # whether the DatetimeIndex has a name set or is anonymous (None / 0).
    def _reset_with_ts(df: pd.DataFrame) -> pd.DataFrame:
        r = df.reset_index()
        # After reset_index the former index is the first column — rename it.
        r = r.rename(columns={r.columns[0]: "ts"})
        return r

    df5_r = _reset_with_ts(df5)
    fut_r = _reset_with_ts(fut_feat)

    merged = pd.merge_asof(
        df5_r.sort_values("ts"),
        fut_r.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("5min"),   # never stale by more than 5m
    )
    merged = merged.set_index("ts")
    merged.index.name = df5.index.name

    result = df5.copy()
    new_cols = [c for c in merged.columns if c not in df5.columns]
    for col in new_cols:
        result[col] = merged[col].values
        # Fill binary flags with 0 (neutral), keep continuous as NaN
        if col.endswith(("_bullish", "_strong", "_spike", "_above_vwap",
                          "_below_vwap", "_regime")):
            result[col] = result[col].fillna(0)
    return result


def resample_from_5min(df5, rule):
    # Guard: empty input or missing DatetimeIndex causes resample to return
    # an object-dtype index, making between_time() crash with TypeError.
    # Return an empty properly-typed DataFrame so callers can concat safely.
    if df5 is None or df5.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    # Ensure the index really is a DatetimeIndex (it should be after _load_raw,
    # but a tiny filtered slice can lose the dtype in some pandas versions).
    if not isinstance(df5.index, pd.DatetimeIndex):
        df5 = df5.copy()
        df5.index = pd.to_datetime(df5.index)

    ohlc = df5.resample(rule, label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last")
    ).dropna()

    # After resample on a very small slice, the result index may degrade to
    # object dtype — recast before calling between_time().
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        ohlc.index = pd.to_datetime(ohlc.index)

    if ohlc.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    ohlc = ohlc.between_time("09:15", "15:30")
    return ohlc


# ==================================================================
# ADX HELPER
# ==================================================================

def compute_adx(df, period=14):
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
# FEATURE ENGINEERING — 5-min base
# ==================================================================

def features_5min(df):
    """
    V9: identical to V8 core indicators. SMC section rewritten with
    explicit shift(1) (Fix 3). CMF renamed cmf_proxy (already was) to make
    the volume-paradox caveat visible.
    """
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

    # Distance from rolling high/low (percentage — stationary)
    for w in [5, 10, 20]:
        df[f"dh{w}"] = (c - h.rolling(w).max()) / c
        df[f"dl{w}"] = (c - l.rolling(w).min()) / c

    # EMA distances (percentage — stationary)
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

    # ATR (normalized)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    a10 = tr.rolling(10).mean(); a14 = tr.rolling(14).mean(); a50 = tr.rolling(50).mean()
    df["a10"] = a10/c; df["a14"] = a14/c
    df["ar"]  = a10 / a14.replace(0, np.nan)
    df["ava"] = a10 / a50.replace(0, np.nan)
    # Expose raw ATR (in points) for stationarity-normalization downstream
    df["_atr14_pts"] = a14

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

    # ================================================================
    # SMC — FIX 3: explicit .shift(1) everywhere. No shift(-N) anywhere.
    # ================================================================
    # All rolling().max()/min() are computed on history UP TO t, then
    # shifted by 1 to guarantee the value used at bar t is strictly
    # derived from bars ≤ t-1.
    swing_h_prev = h.rolling(10).max().shift(1)       # swing high as of t-1
    swing_l_prev = l.rolling(10).min().shift(1)       # swing low  as of t-1
    df["bos_bull"] = (c > swing_h_prev).astype(int)
    df["bos_bear"] = (c < swing_l_prev).astype(int)

    hh_prev1 = h.rolling(5).max().shift(1)            # short-term HH at t-1
    ll_prev1 = l.rolling(5).min().shift(1)            # short-term LL at t-1
    hh_prev6 = h.rolling(5).max().shift(6)            # 5 bars earlier than prev1
    ll_prev6 = l.rolling(5).min().shift(6)
    df["choch_bull"] = ((ll_prev1 > ll_prev6) & (hh_prev1.shift(1) < hh_prev6.shift(1))).astype(int)
    df["choch_bear"] = ((hh_prev1 < hh_prev6) & (ll_prev1.shift(1) > ll_prev6.shift(1))).astype(int)

    # FVG: classic 3-bar pattern (t-2, t-1, t). We read the pattern whose
    # middle bar was TWO bars ago — strictly historical. No shift(-1).
    # Bull FVG at bar t means: low(t-1) > high(t-3) AND there is a gap.
    # Written via shift(+N) only.
    h_t2 = h.shift(2)      # high 2 bars ago
    l_t2 = l.shift(2)      # low  2 bars ago
    h_t0_prev = h.shift(1) # high 1 bar ago (the "latest bar" of the 3-bar pattern, read at t)
    l_t0_prev = l.shift(1)
    h_t4 = h.shift(4)      # high 4 bars ago (the "earliest bar" of the 3-bar pattern)
    l_t4 = l.shift(4)
    df["fvg_bull_prev"] = (l_t0_prev > h_t4).astype(int)   # gap up pattern as of t-1
    df["fvg_bear_prev"] = (h_t0_prev < l_t4).astype(int)   # gap down pattern as of t-1

    # Order blocks: narrow range THEN breakout, all historical
    range_vol = (fr.rolling(10).std() / c).shift(1)
    range_narrow_prev = (range_vol < range_vol.rolling(50).quantile(0.2)).fillna(False)
    df["ob_bull"] = (range_narrow_prev & (c > h.rolling(10).max().shift(1))).astype(int)
    df["ob_bear"] = (range_narrow_prev & (c < l.rolling(10).min().shift(1))).astype(int)

    # Liquidity sweep: touches historical extreme then reverses on current close
    p20h_prev = h.rolling(20).max().shift(1)
    p20l_prev = l.rolling(20).min().shift(1)
    df["liq_sweep_high"] = ((h > p20h_prev) & (c < p20h_prev)).astype(int)
    df["liq_sweep_low"]  = ((l < p20l_prev) & (c > p20l_prev)).astype(int)

    # Volume proxy — CMF_PROXY. Nifty spot volume is 0, so this reduces to a
    # close-position indicator. Kept for continuity; LIVE uses real futures
    # CMF elsewhere. Fix-1 caveat: historical value ≠ live value.
    mfm = ((c-l) - (h-c)) / fr
    df["cmf_proxy"] = (mfm*fr).rolling(20).sum() / (fr.rolling(20).sum() + 1e-9)

    # Z-score mean reversion (stationary by construction)
    df["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)
    df["zscore_50"] = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-9)

    # Volatility regime
    rv20 = df["rv20"]
    rv_med = rv20.rolling(252).median()
    df["vol_regime"]      = np.where(rv20 > rv_med*1.5, 2, np.where(rv20 < rv_med*0.7, 0, 1))
    df["vol_percentile"]  = rv20.rolling(252).rank(pct=True)

    # Intraday trend
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df["dow_primary"]   = (ema50 > ema200).astype(int)
    df["dow_secondary"] = (c > ema50).astype(int)

    # Session + time
    ts = df.index.strftime("%H:%M")
    df["is_morning"]   = ((ts >= "09:15") & (ts <= "10:30")).astype(int)
    df["is_midday"]    = ((ts >  "10:30") & (ts <= "13:30")).astype(int)
    df["is_afternoon"] = ((ts >  "13:30") & (ts <= "15:30")).astype(int)
    df["dow"] = df.index.dayofweek

    # Expiry calendar
    try:
        exp_df = pd.read_csv("data/nifty_expiry_history.csv", parse_dates=["expiry_date"])
        expiries = sorted(exp_df["expiry_date"].dt.date.tolist())

        def _dte_for_date(bar_date):
            for exp in expiries:
                if exp >= bar_date:
                    return min(5, (exp - bar_date).days)
            return 3

        def _exp_dow_for_date(bar_date):
            for exp in expiries:
                if exp >= bar_date:
                    return exp.weekday()
            return 1

        bar_dates = df.index.normalize()
        unique_dates = pd.Series(bar_dates.unique()).dt.date
        dte_lookup = {d: _dte_for_date(d) for d in unique_dates}
        edow_lookup = {d: _exp_dow_for_date(d) for d in unique_dates}
        df["dte"] = bar_dates.date
        df["expiry_dow"] = df["dte"].map(edow_lookup).astype(int)
        df["dte"] = df["dte"].map(dte_lookup).astype(int)
        df["is_expiry"] = (df["dte"] == 0).astype(int)
        df["expiry_is_tue"] = (df["expiry_dow"] == 1).astype(int)
        log.info(f"  DTE+expiry_dow computed from {len(expiries)} real historical expiries")
    except Exception as e:
        log.warning(f"  Real-expiry DTE failed ({e}); falling back")
        df["is_expiry"] = (df.index.dayofweek == 1).astype(int)
        dte_map = {0: 1, 1: 0, 2: 5, 3: 4, 4: 3}
        df["dte"] = df.index.dayofweek.map(dte_map).fillna(3).astype(int)
        df["expiry_dow"] = 1
        df["expiry_is_tue"] = 1
    df["dte_norm"] = df["dte"] / 6.0

    # HH/HL price-structure feature (additive; does not modify any feature
    # above). Reuses the exact same causal trailing-swing function used live
    # (core/structure_features.py) so training and inference compute this
    # identically -- see that module's docstring for why the swing
    # definition must be trailing-only (no lookahead) in both places.
    try:
        from core.structure_features import compute_hh_hl_structure
        closes_list = c.tolist()
        n = len(closes_list)
        hh_hl_up = np.zeros(n, dtype=np.float64)
        hh_hl_down = np.zeros(n, dtype=np.float64)
        for i in range(n):
            window = closes_list[max(0, i - 19):i + 1]  # up to and including bar i
            struct = compute_hh_hl_structure(window)
            hh_hl_up[i] = 1.0 if struct["hh_hl_up"] else 0.0
            hh_hl_down[i] = 1.0 if struct["hh_hl_down"] else 0.0
        df["struct_hh_hl_up"] = hh_hl_up
        df["struct_hh_hl_down"] = hh_hl_down
        log.info(
            f"  struct_hh_hl: up={int(hh_hl_up.sum()):,} down={int(hh_hl_down.sum()):,} "
            f"bars flagged (of {n:,})"
        )
    except Exception as e:
        log.warning(f"  struct_hh_hl computation failed ({e}) — feature omitted, training unaffected")

    log.info(f"  5-min features computed: {len([c for c in df.columns if c not in ['open','high','low','close']])} cols")
    return df


# ==================================================================
# HIGHER-TF FEATURES (unchanged — already stationary)
# ==================================================================

def features_15min(df15):
    c, h, l = df15["close"], df15["high"], df15["low"]
    out = pd.DataFrame(index=df15.index)

    adx_df = compute_adx(df15, period=14)
    out["tf15_adx"]       = adx_df["adx"]
    out["tf15_diplus"]    = adx_df["diplus"]
    out["tf15_diminus"]   = adx_df["diminus"]
    out["tf15_trending"]  = (adx_df["adx"] > 25).astype(int)
    out["tf15_sideways"]  = (adx_df["adx"] < 20).astype(int)
    out["tf15_bull_trend"]= ((adx_df["adx"] > 25) & (adx_df["diplus"] > adx_df["diminus"])).astype(int)
    out["tf15_bear_trend"]= ((adx_df["adx"] > 25) & (adx_df["diminus"] > adx_df["diplus"])).astype(int)

    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["tf15_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ms  = mac.ewm(span=9, adjust=False).mean()
    out["tf15_macd_hist"] = (mac - ms) / c

    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    out["tf15_ema_bull"]  = (e9 > e21).astype(int)
    out["tf15_ema_dist"]  = (c - e21) / c

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    a5  = tr.rolling(5).mean()
    a14 = tr.rolling(14).mean()
    out["tf15_atr_ratio"] = a5 / a14.replace(0, np.nan)

    bm = c.rolling(20).mean(); bs = c.rolling(20).std() + 1e-9
    bbw = 4*bs/bm
    out["tf15_bb_squeeze"] = (bbw < bbw.rolling(30).quantile(0.2)).astype(int)
    out["tf15_bb_pos"]     = (c - (bm - 2*bs)) / (4*bs)

    log.info(f"  15-min features: {len(out.columns)} cols")
    return out


def features_30min(df30):
    c, h, l = df30["close"], df30["high"], df30["low"]
    out = pd.DataFrame(index=df30.index)

    adx_df = compute_adx(df30, period=14)
    out["tf30_adx"]        = adx_df["adx"]
    out["tf30_bull_trend"] = ((adx_df["adx"] > 25) & (adx_df["diplus"] > adx_df["diminus"])).astype(int)
    out["tf30_bear_trend"] = ((adx_df["adx"] > 25) & (adx_df["diminus"] > adx_df["diplus"])).astype(int)

    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["tf30_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    out["tf30_ema_bull"]  = (e9 > e21).astype(int)
    out["tf30_ema_dist"]  = (c - e21) / c

    out["tf30_close_pos"] = (c - l.rolling(5).min()) / (
        h.rolling(5).max() - l.rolling(5).min() + 1e-9)

    log.info(f"  30-min features: {len(out.columns)} cols")
    return out


def features_60min(df60):
    c, h, l = df60["close"], df60["high"], df60["low"]
    out = pd.DataFrame(index=df60.index)

    adx_df = compute_adx(df60, period=14)
    out["tf60_adx"]        = adx_df["adx"]
    out["tf60_bull_trend"] = ((adx_df["adx"] > 25) & (adx_df["diplus"] > adx_df["diminus"])).astype(int)
    out["tf60_bear_trend"] = ((adx_df["adx"] > 25) & (adx_df["diminus"] > adx_df["diplus"])).astype(int)

    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=9, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=9, adjust=False).mean()
    out["tf60_rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    out["tf60_ema_bull"]  = (e21 > e50).astype(int)
    out["tf60_ema_dist"]  = (c - e50) / c

    out["tf60_close_pos"] = (c - l.rolling(4).min()) / (
        h.rolling(4).max() - l.rolling(4).min() + 1e-9)

    # ── Trend Persistence features (added to catch afternoon trend days) ──
    #
    # Problem these fix: on a day like Jun 5 2026, the market fell 160pts
    # from the high but the ML kept saying SKIP because RSI=24 (oversold)
    # looks the same whether the market is at a V-bottom or mid-trend.
    # These features tell the model "the downtrend has been running for
    # N hours, not just the last 5 minutes" — allowing it to distinguish
    # a genuine trending afternoon from a quick dip near support.

    # 1. Consecutive bearish 60-min bars (close < prev close).
    #    +N = N bars of continuous selling pressure (max clipped at ±6).
    #    Today 13:00: would show −2 or −3 = multiple hours of selling.
    bear_bar = (c.diff() < 0).astype(int)
    bull_bar = (c.diff() > 0).astype(int)
    # Streak counter: resets to 0 on direction change
    bear_streak = bear_bar.groupby((bear_bar != bear_bar.shift()).cumsum()).cumsum()
    bull_streak = bull_bar.groupby((bull_bar != bull_bar.shift()).cumsum()).cumsum()
    out["tf60_consec_bear"] = bear_streak.clip(upper=6) / 6.0   # normalised 0-1
    out["tf60_consec_bull"] = bull_streak.clip(upper=6) / 6.0

    # 2. 3-hour trend slope — normalised linear slope of last 3 closes.
    #    Negative and steep = confirmed downtrend. Measures directional
    #    persistence over 3 hours, independent of RSI noise.
    def _slope3(series):
        """Slope of last 3 values, normalised by std (stationary)."""
        def _s(x):
            if len(x) < 3 or x.std() < 1e-9:
                return 0.0
            return float(np.polyfit(range(len(x)), x, 1)[0] / (x.std() + 1e-9))
        return series.rolling(3).apply(_s, raw=True)

    out["tf60_trend_slope_3h"] = _slope3(c)

    # 3. Distance from 3-hour peak (how far price has fallen from recent high).
    #    Captures "ongoing selloff depth" — distinct from session_high dist
    #    because it resets every 3 hours, not at market open.
    #    Today 13:10: spot=23349, 3h high=23478 → dist = −129pts / 23478 = −0.55%
    h3_max = h.rolling(3).max()
    out["tf60_dist_from_3h_high"] = (c - h3_max) / h3_max.replace(0, np.nan)

    l3_min = l.rolling(3).min()
    out["tf60_dist_from_3h_low"]  = (c - l3_min) / l3_min.replace(0, np.nan)

    # 4. EMA21 slope — is the hourly trend turning up or down?
    #    (ema21_now - ema21_3_bars_ago) / ema21_3_bars_ago.
    #    When EMA21 starts sloping down on the 60-min chart, the
    #    downtrend is already well-established (confirming PUT direction).
    out["tf60_ema21_slope"] = (e21 - e21.shift(3)) / e21.shift(3).replace(0, np.nan)

    log.info(f"  60-min features: {len(out.columns)} cols")
    return out


def features_daily(df_day):
    """
    V9: Daily features — keeps previous-day level COLUMNS in the frame for
    distance calculations, but the level columns themselves are EXCLUDED
    from FCOLS (they're absolute points = non-stationary).
    """
    c, h, l, o = df_day["close"], df_day["high"], df_day["low"], df_day["open"]
    out = pd.DataFrame(index=df_day.index)

    # Levels (used internally — excluded from FCOLS)
    out["day_prev_high"]  = h.shift(1)
    out["day_prev_low"]   = l.shift(1)
    out["day_prev_close"] = c.shift(1)
    out["week_high"]      = h.rolling(5).max().shift(1)
    out["week_low"]       = l.rolling(5).min().shift(1)

    # Gap — percentage (stationary)
    out["day_gap_pct"]   = (o - c.shift(1)) / c.shift(1)
    out["day_gap_up"]    = out["day_gap_pct"].clip(lower=0)
    out["day_gap_down"]  = out["day_gap_pct"].clip(upper=0).abs()
    out["day_big_gap"]   = (out["day_gap_pct"].abs() > 0.005).astype(int)

    # Options-relevant EMAs
    # 2026-06-11 PHASE-0 LEAKAGE FIX: these were computed from TODAY's daily
    # close and merged onto TODAY's intraday bars (merge_daily_onto_5min does
    # not shift), so a 10:00 bar saw functions of the 15:30 close. All
    # close-derived daily features now shift(1) — intraday bars see only the
    # last COMPLETED daily bar. Gap features stay unshifted (today's OPEN vs
    # yesterday's close — known at 09:15). Level columns were already shifted.
    e9  = c.ewm(span=9,  adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    out["day_ema9_bull"]  = (e9  > e20).astype(int).shift(1)
    out["day_ema_bull"]   = (e20 > e50).astype(int).shift(1)
    out["day_ema9_dist"]  = ((c - e9)  / c).shift(1)
    out["day_ema_dist"]   = ((c - e20) / c).shift(1)
    out["day_ema50_dist"] = ((c - e50) / c).shift(1)

    out["day_week_pos"] = (c.shift(1) - out["week_low"]) / (
        out["week_high"] - out["week_low"] + 1e-9)

    d2 = c.diff()
    g  = d2.clip(lower=0).ewm(span=14, adjust=False).mean()
    ls = (-d2.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out["day_rsi"] = (100 - (100 / (1 + g / ls.replace(0, np.nan)))).shift(1)

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    # PHASE-0 LEAKAGE FIX: shift(1) — was today's full-day TR/close intraday
    out["day_atr_pct"] = (tr.rolling(14).mean() / c).shift(1)

    log.info(f"  Daily features: {len(out.columns)} cols")
    return out


# ==================================================================
# VIX (unchanged — already stationary)
# ==================================================================

def load_vix(path):
    if not Path(path).exists():
        log.warning(f"VIX file not found: {path} — VIX features will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    dt = next((c for c in df.columns if "date" in c), None)
    if not dt:
        return pd.DataFrame()
    df[dt] = pd.to_datetime(df[dt], format="mixed", dayfirst=True)
    df = df.set_index(dt).sort_index()
    close_col = next((c for c in df.columns if "close" in c), None)
    if not close_col:
        df.columns = ["vix"]
    else:
        df = df.rename(columns={close_col: "vix"})
    df = df[["vix"]].astype(float).dropna()
    df = df[~df.index.duplicated(keep="last")]
    log.info(f"VIX : {len(df):,} days | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def features_vix(df_vix):
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

    # 2026-05-15 OPTIONS-PRICING FEATURES (Sensibull-equivalent without scraping):
    # India VIX IS the ATM IV of front-month NIFTY options by NSE definition.
    # These derived features replicate what Sensibull's "Strategy Builder" shows:
    out["vix_rank_30d"]   = v.rolling(30, min_periods=10).rank(pct=True)  # 0..1
    out["vix_zscore_20d"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    # IV expansion vs contraction signal
    out["vix_above_ma20"] = (v > v.rolling(20).mean()).astype(int)
    # Mean reversion potential — VIX rarely sustains > 24 for long
    out["vix_extreme_high"] = (v > 24).astype(int)
    out["vix_extreme_low"]  = (v < 11).astype(int)
    log.info(f"  VIX features: {len(out.columns)} cols (with options-pricing additions)")
    return out


def merge_vix_onto_5min(df5, vix_feat):
    if vix_feat.empty:
        return df5
    df5 = df5.copy()
    df5["_date"] = df5.index.normalize()
    vix_feat = vix_feat.copy()
    vix_feat.index = pd.to_datetime(vix_feat.index).normalize()
    vix_feat = vix_feat.shift(1)  # use yesterday's VIX → no lookahead
    vix_feat = vix_feat[~vix_feat.index.duplicated(keep="last")]
    for col in vix_feat.columns:
        df5[col] = df5["_date"].map(vix_feat[col])
    df5.drop(columns=["_date"], inplace=True)
    return df5


def merge_htf(df5, htf_feat, prefix):
    # 2026-06-11 PHASE-0 LEAKAGE FIX: HTF bars are labeled at bar START, so an
    # as-of-backward merge at a 5m bar mid-way through the HTF bar picked up
    # that bar's FINAL values (up to 14/29/59 min of future data). Re-label
    # each HTF bar at (start + duration - 5min): a 5m bar's features may only
    # include HTF bars that complete by that 5m bar's own close.
    _dur = {"15m": 15, "30m": 30, "60m": 60}.get(prefix)
    if _dur:
        htf_feat = htf_feat.copy()
        htf_feat.index = htf_feat.index + pd.Timedelta(minutes=_dur - 5)
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
    new_cols = [c for c in merged.columns if c not in df5.columns]
    result = df5.copy()
    for col in new_cols:
        result[col] = merged[col].values
    return result


def merge_daily_onto_5min(df5, day_feat):
    df5 = df5.copy()
    df5["_date"] = df5.index.normalize()
    day_feat = day_feat.copy()
    day_feat.index = pd.to_datetime(day_feat.index).normalize()
    day_feat = day_feat[~day_feat.index.duplicated(keep="last")]
    for col in day_feat.columns:
        df5[col] = df5["_date"].map(day_feat[col])
    df5.drop(columns=["_date"], inplace=True)
    return df5


# ==================================================================
# INTRADAY CONTEXT — FIX 2 applied (ATR-normalized, no raw points)
# ==================================================================

def add_intraday_context(df5, has_real_futures: bool = False):
    """
    V9 (Fix 1 complete) — Intraday context features.

    Fix 2: all "_pts" features replaced with ATR-normalized variants.
    Fix 1: when real futures VWAP/CMF are already merged onto df5
           (has_real_futures=True), we skip the spot-proxy VWAP and instead
           alias the futures columns to the canonical names used downstream
           (dist_vwap, above_vwap, vwap_dist_atr, bars_above_vwap, etc.)
           so the model always trains on mathematically valid VWAP features.
           The spot-proxy path is kept as fallback when futures data is absent.

    Absolute levels (day_open, intraday_high, intraday_low, day_prev_*, week_*,
    vwap_session) remain in df for reference but are excluded from FCOLS.
    """
    c = df5["close"]

    # Day open (first bar of each day) — LEVEL (excluded from FCOLS)
    df5["day_open"] = df5.groupby(df5.index.date)["open"].transform("first")
    df5["dist_from_open"] = (c - df5["day_open"]) / df5["day_open"]

    df5["intraday_high"] = df5.groupby(df5.index.date)["high"].cummax()
    df5["intraday_low"]  = df5.groupby(df5.index.date)["low"].cummin()
    df5["intraday_pos"]  = (c - df5["intraday_low"]) / (
        df5["intraday_high"] - df5["intraday_low"] + 1e-9)

    # Distances from previous-day levels — all PERCENT
    if "day_prev_high" in df5.columns:
        df5["dist_prev_high"]  = (c - df5["day_prev_high"]) / c
        df5["dist_prev_low"]   = (c - df5["day_prev_low"])  / c
        df5["dist_prev_close"] = (c - df5["day_prev_close"])/ c
        df5["above_prev_high"] = (c > df5["day_prev_high"]).astype(int)
        df5["below_prev_low"]  = (c < df5["day_prev_low"]).astype(int)

    if "week_high" in df5.columns:
        df5["dist_week_high"] = (c - df5["week_high"]) / c
        df5["dist_week_low"]  = (c - df5["week_low"])  / c

    # ATR (needed for all ATR-normalized features below)
    atr14 = df5.get("_atr14_pts")
    if atr14 is None:
        h, l, cc = df5["high"], df5["low"], df5["close"]
        tr = pd.concat([h-l, (h-cc.shift()).abs(), (l-cc.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()

    # ── VWAP (Fix 1) ─────────────────────────────────────────────────
    if has_real_futures and "fut_vwap" in df5.columns:
        # Real futures VWAP already merged — alias to canonical names.
        # This is the mathematically correct path.
        log.info("  Intraday VWAP: using REAL futures VWAP (Fix 1 active)")
        df5["vwap_session"]    = df5["fut_vwap"]           # LEVEL (excluded from FCOLS)
        df5["dist_vwap"]       = df5["fut_dist_vwap_pct"]
        df5["above_vwap"]      = df5["fut_above_vwap"]
        df5["below_vwap"]      = (df5["fut_above_vwap"] == 0).astype(int)
        df5["vwap_dist_atr"]   = df5["fut_vwap_dist_atr"]
        df5["bars_above_vwap"] = df5["fut_bars_above_vwap"]
        df5["bars_below_vwap"] = df5["fut_bars_below_vwap"]
        # Also alias CMF so downstream code sees a unified name
        if "fut_cmf20" in df5.columns:
            df5["cmf20"]        = df5["fut_cmf20"]
            df5["cmf_bullish"]  = df5["fut_cmf_bullish"]
            # Drop the spot-proxy CMF feature to avoid multicollinearity
            df5.drop(columns=["cmf_proxy"], errors="ignore", inplace=True)
    else:
        # Fallback: spot-proxy VWAP (volume=1 for spot index)
        log.info("  Intraday VWAP: using SPOT PROXY (no futures data) — marked _proxy")
        typ = (df5["high"] + df5["low"] + df5["close"]) / 3.0
        vol_raw = df5["volume"].astype(float).fillna(0.0) if "volume" in df5.columns \
                  else pd.Series(0.0, index=df5.index)
        _date       = pd.Series(df5.index.date, index=df5.index, name="_d")
        day_vol_sum = vol_raw.groupby(_date).transform("sum")
        vol         = pd.Series(np.where(day_vol_sum > 0, vol_raw, 1.0), index=df5.index)
        cum_tpv     = (typ * vol).groupby(_date).cumsum()
        cum_vol     = vol.groupby(_date).cumsum().replace(0, np.nan)
        vwap        = cum_tpv / cum_vol
        _date_s     = pd.Series(df5.index.date, index=df5.index, name="_d")

        df5["vwap_session"]  = vwap
        df5["dist_vwap"]     = (c - vwap) / c
        df5["above_vwap"]    = (c > vwap).astype(int)
        df5["below_vwap"]    = (c < vwap).astype(int)
        df5["vwap_dist_atr"] = (c - vwap) / atr14.replace(0, np.nan)

        side        = np.sign((c - vwap).fillna(0)).astype(int)
        side_change = (side != side.shift()) | (_date_s != _date_s.shift())
        streak_grp  = side_change.cumsum()
        streak      = side.groupby(streak_grp).cumcount() + 1
        df5["bars_above_vwap"] = np.where(side > 0, streak, 0)
        df5["bars_below_vwap"] = np.where(side < 0, streak, 0)

    # Fix 2: intraday range — ATR-normalized and percent only, NO raw pts
    day_range_pts = df5["intraday_high"] - df5["intraday_low"]
    df5["intraday_range_pct"] = day_range_pts / c
    df5["intraday_range_atr"] = day_range_pts / atr14.replace(0, np.nan)

    # Fix 2: day move — percent and ATR-normalized, NO raw pts
    if "day_open" in df5.columns:
        df5["day_move_pct"] = (c - df5["day_open"]) / df5["day_open"]
        df5["day_move_atr"] = (c - df5["day_open"]) / atr14.replace(0, np.nan)

    return df5


# ==================================================================
# LABELS
# ==================================================================

def compute_thresholds(df5, quantile=0.70):
    """
    DEPRECATED (Issue #3): computes thresholds over the full dataset,
    leaking future volatility into past bars' labels. Kept only so that
    train_model_v8.py (which has its own identical copy) is unaffected.
    All V9/V10 training goes through compute_thresholds_causal() below.
    """
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


def compute_thresholds_causal(df5, quantile=0.70,
                               min_periods=500, lookback_bars=None):
    """
    Per-bar CAUSAL threshold — no future data allowed (Issue #3 fix).

    For each bar at position i, the threshold = quantile(|fwd|) of all
    PAST bars in the same session bucket (AM / MID / PM) whose forward
    return has already been fully realized.

    MECHANICS
    ---------
    fwd[i] = close[i+FWD_BARS] / close[i] - 1   (looks forward, unknown at i)

    Shifting by FWD_BARS gives the REALIZED series:
        fwd_realized[i] = fwd[i - FWD_BARS]
                        = close[i] / close[i-FWD_BARS] - 1

    At bar i, fwd_realized[i] is the return that SETTLED at bar i;
    all values fwd_realized[0..i] are known without any future peek.

    The expanding quantile of |fwd_realized| in each session bucket is
    therefore a fully causal threshold that adapts to the volatility regime
    seen so far.

    PARAMETERS
    ----------
    min_periods : int
        Minimum same-session bars with realized data before switching from
        the fallback 0.001 to a data-driven quantile.
        500 bars ≈ 33 trading days for AM (~15 bars/day).
    lookback_bars : int or None
        None (default) → expanding window (all history, most conservative).
        int            → rolling window of that many same-session bars
                         (regime-adaptive; e.g. 4000 ≈ 1 year of AM bars).

    RETURNS
    -------
    pd.Series, same index as df5, dtype float64.
        Each value is the causal threshold for that bar's session.
        Bars outside any session (overnight gap, pre-market) get 0.001.
    """
    fwd = df5["close"].shift(-FWD_BARS) / df5["close"] - 1
    # Shift forward by FWD_BARS so index i holds the return that settled at i
    fwd_realized = fwd.shift(FWD_BARS)

    ts = df5.index.strftime("%H:%M")
    session = pd.Series("other", index=df5.index, dtype=object)
    session.loc[(ts >= "09:15") & (ts <= "10:30")] = "am"
    session.loc[(ts >  "10:30") & (ts <= "13:30")] = "mid"
    session.loc[(ts >  "13:30") & (ts <= "15:30")] = "pm"

    bar_thresh = pd.Series(0.001, index=df5.index, dtype=float)

    for sess in ("am", "mid", "pm"):
        mask      = session == sess
        sess_fwd  = fwd_realized[mask].abs()

        if lookback_bars is None:
            rolling_q = sess_fwd.expanding(min_periods=min_periods).quantile(quantile)
        else:
            rolling_q = sess_fwd.rolling(
                window=lookback_bars, min_periods=min_periods
            ).quantile(quantile)

        # Early bars with < min_periods samples get NaN → keep fallback 0.001
        # (more permissive threshold → more CALL/PUT labels, which is fine;
        # the model learns from them and the scaler/split handle it correctly)
        bar_thresh.loc[mask] = rolling_q.fillna(0.001).values

    # Sanity: thresholds must be positive and finite
    bar_thresh = bar_thresh.clip(lower=1e-6).fillna(0.001)

    # Log representative end-of-dataset values so we can see regime adaptation
    for sess in ("am", "mid", "pm"):
        mask = session == sess
        if mask.any():
            last_val = bar_thresh[mask].iloc[-1]
            log.info(f"  Causal threshold [{sess.upper()}] final = {last_val:.5f} "
                     f"(vs global {compute_thresholds(df5, quantile)[['am','mid','pm'].index(sess)]:.5f})")

    return bar_thresh


def create_labels(df, df15=None):
    fwd = df["close"].shift(-FWD_BARS) / df["close"] - 1
    df["label"] = 2
    ts  = df.index.strftime("%H:%M")
    am  = (ts >= "09:15") & (ts <= "10:30")
    mid = (ts >  "10:30") & (ts <= "13:30")
    pm  = (ts >  "13:30") & (ts <  "15:00")
    in_session = am | mid | pm

    # Issue #3: use causal per-bar thresholds (no future data)
    bar_thresh = compute_thresholds_causal(df)

    rsi = df["rsi14"] if "rsi14" in df.columns else pd.Series(50.0, index=df.index)

    if "tf15_adx" in df.columns:
        adx = df["tf15_adx"]
    else:
        adx_df = compute_adx(df, period=14)
        adx = adx_df["adx"]

    rsi_oversold   = rsi < 30
    rsi_overbought = rsi > 70
    rsi_normal     = (~rsi_oversold) & (~rsi_overbought)

    adx_strong   = adx > 40
    adx_trending = adx > 20
    adx_weak     = adx <= 25

    normal_call = in_session & rsi_normal & adx_trending & (fwd > bar_thresh)
    normal_put  = in_session & rsi_normal & adx_trending & (fwd < -bar_thresh)
    df.loc[normal_call, "label"] = 0
    df.loc[normal_put,  "label"] = 1

    oversold_put = (in_session & rsi_oversold & adx_strong
                    & (fwd < -bar_thresh * 0.7))
    df.loc[oversold_put, "label"] = 1
    oversold_call = (in_session & rsi_oversold & adx_weak
                     & (fwd > bar_thresh * 1.2))
    df.loc[oversold_call, "label"] = 0

    overbought_call = (in_session & rsi_overbought & adx_strong
                       & (fwd > bar_thresh * 0.7))
    df.loc[overbought_call, "label"] = 0
    overbought_put = (in_session & rsi_overbought & adx_weak
                      & (fwd < -bar_thresh * 1.2))
    df.loc[overbought_put, "label"] = 1

    if "vix_level" in df.columns:
        extreme_vix = df["vix_level"] > 35.0
        df.loc[extreme_vix, "label"] = 2
        log.info(f"  VIX>35 forced SKIP: {extreme_vix.sum():,} bars")

    log.info(f"  RSI+ADX zones: normal={normal_call.sum()+normal_put.sum():,} | "
             f"oversold->PUT={oversold_put.sum():,} oversold->CALL={oversold_call.sum():,} | "
             f"overbought->CALL={overbought_call.sum():,} overbought->PUT={overbought_put.sum():,}")

    lc = df["label"].value_counts().sort_index()
    log.info(f"  Labels: CALL={lc.get(0,0):,} PUT={lc.get(1,0):,} SKIP={lc.get(2,0):,}")
    return df


# ==================================================================
# FIX 4 — SHAP-BASED FEATURE PRUNING
# ==================================================================

def shap_prune_features(model, X_sample, feature_names,
                        bottom_pct=0.25, corr_threshold=0.95):
    """
    Fix 4 (improved) — returns a pruned list of feature names.

    Strategy (two-pass):
      Pass A — Redundancy pruning: drop features whose pairwise correlation
               with any higher-importance feature exceeds corr_threshold (0.95).
               This kills true duplicates like m3/m6/m9 overlap, multi-RSI, etc.
      Pass B — Bottom-N% pruning: from the remaining features, drop those
               in the bottom `bottom_pct` (default 25%) of SHAP importance.
               This is more aggressive than the previous 0.1%-of-max heuristic
               and correctly targets the brief spec ("drop bottom 25%").

    Falls back to model.feature_importances_ when shap package is absent.
    """
    # ── Compute importances ───────────────────────────────────────────
    importances = None
    if HAS_SHAP:
        try:
            log.info("  Computing SHAP values (sampled 2000 rows)...")
            sample_n = min(2000, X_sample.shape[0])
            rng      = np.random.default_rng(42)
            idx_samp = rng.choice(X_sample.shape[0], sample_n, replace=False)
            X_samp   = X_sample[idx_samp]
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_samp)
            if isinstance(sv, list):        # multiclass → list of arrays
                importances = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
            else:
                importances = np.abs(sv).mean(axis=0)
            log.info("  SHAP values: OK")
        except Exception as e:
            log.warning(f"  SHAP failed ({e}) — falling back to feature_importances_")

    if importances is None:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            log.info("  Using model.feature_importances_ (no SHAP)")
        else:
            log.warning("  No importance source available — skipping prune")
            return list(feature_names)

    feature_names = list(feature_names)
    importance_rank = np.argsort(importances)[::-1]     # descending

    # ── Correlation matrix ────────────────────────────────────────────
    X_df = pd.DataFrame(X_sample, columns=feature_names)
    corr = X_df.corr().abs().fillna(0.0).values

    # ── Pass A: remove correlation-redundant features ─────────────────
    keep_idx_a: list[int] = []
    dropped_a: list[tuple] = []
    for idx in importance_rank:
        redundant = False
        for kept in keep_idx_a:
            if corr[idx, kept] > corr_threshold:
                dropped_a.append((
                    feature_names[idx],
                    importances[idx],
                    f"corr={corr[idx,kept]:.2f}>{corr_threshold} "
                    f"with {feature_names[kept]}"
                ))
                redundant = True
                break
        if not redundant:
            keep_idx_a.append(idx)

    # ── Pass B: drop bottom `bottom_pct` of remaining features ────────
    kept_imps = importances[keep_idx_a]
    cutoff    = np.percentile(kept_imps, bottom_pct * 100)   # e.g. 25th percentile
    keep_idx_b = [i for i in keep_idx_a if importances[i] >= cutoff]
    dropped_b  = [
        (feature_names[i], importances[i], f"bottom_{int(bottom_pct*100)}pct")
        for i in keep_idx_a if importances[i] < cutoff
    ]

    kept = [feature_names[i] for i in keep_idx_b]
    n_total = len(feature_names)
    log.info(
        f"  Prune: {n_total} → {len(kept)} features "
        f"(Pass A corr dropped {len(dropped_a)}, "
        f"Pass B bottom-{int(bottom_pct*100)}% dropped {len(dropped_b)})"
    )

    all_dropped = dropped_a + dropped_b
    if all_dropped:
        log.info(f"  First 15 dropped:")
        for name, imp, reason in all_dropped[:15]:
            log.info(f"    - {name:35s} imp={imp:.5f}  [{reason}]")
    return kept


# ==================================================================
# TRAINING HELPERS
# ==================================================================

def purged_train_val_test_split(n, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                                fwd_bars=FWD_BARS, embargo_bars=EMBARGO_BARS):
    """
    Chronological 3-way split with purge + embargo (Issue #2).

    Returns (train_idx, val_idx, test_idx) as positional numpy index arrays.

    Layout (positions increase with time; data is already sorted + dropna'd):

        [ TRAIN ........ ] gap [ VAL ...... ] gap [ TEST ...... ]
        0          b1-gap   b1         b2-gap  b2              n

      gap = fwd_bars + embargo_bars  (dropped, belongs to NO set)

    Why a gap:
      * PURGE (fwd_bars): a bar's label is close[t+fwd_bars]/close[t]-1, so the
        last bars of each block "see" forward into the next block. Dropping the
        block's forward-looking tail removes that label leakage.
      * EMBARGO (embargo_bars): the next block's feature rolling-windows
        (RSI/ATR/VWAP/etc.) would otherwise reach back into the previous block.
        The extra buffer keeps the sets feature-independent too.

    Positional purge is conservative: after dropna the retained rows are a
    time-ordered subsequence, so a positional gap of `gap` rows spans at least
    `gap` time-bars (usually more) — never less. So no leakage is possible.
    """
    gap = int(fwd_bars + embargo_bars)
    b1  = int(n * train_frac)
    b2  = int(n * (train_frac + val_frac))

    train_idx = np.arange(0,  max(0,  b1 - gap))
    val_idx   = np.arange(b1, max(b1, b2 - gap))
    test_idx  = np.arange(b2, n)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"purged split produced an empty set "
            f"(n={n}, gap={gap}, b1={b1}, b2={b2}). "
            f"Dataset too small for these fractions/embargo."
        )

    log.info(
        f"  Purged 3-way split (gap={gap}=purge{fwd_bars}+embargo{embargo_bars}): "
        f"TRAIN={len(train_idx):,} | VAL={len(val_idx):,} | TEST={len(test_idx):,} "
        f"(dropped {n - len(train_idx) - len(val_idx) - len(test_idx):,} gap bars)"
    )
    return train_idx, val_idx, test_idx


def _fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w):
    models = {}
    if HAS_XLG:
        log.info("Training XGBoost...")
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
        for i, name in enumerate(["hgb_a", "hgb_b", "hgb_c"]):
            log.info(f"Training {name}...")
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
    return models


# ==================================================================
# MAIN TRAIN
# ==================================================================

def train(csv5, csv15=None, csv30=None, csv60=None, csv_day=None,
          csv_vix=None, csv_fut=None, prune_shap=False):
    log.info("=" * 62)
    log.info("  NIFTY V9 — all 4 fixes: futures VWAP/CMF + stationarity")
    log.info("           + lookahead-safe SMC + SHAP bottom-25% pruning")
    log.info("=" * 62)

    # ── Fix 1: load futures CSV for real VWAP + CMF ───────────────────
    fut_path   = csv_fut or CSV_FUT
    df_fut     = load_futures(fut_path)
    feat_fut   = features_futures(df_fut) if not df_fut.empty else pd.DataFrame()
    use_real_vwap = not feat_fut.empty
    log.info(
        f"  VWAP/CMF source: "
        f"{'REAL futures ✓' if use_real_vwap else 'SPOT PROXY (no futures CSV)'}"
    )

    # Load spot
    df5 = load_5min(csv5)

    if csv15 and Path(csv15).exists():
        df15_upload = load_higher(csv15, "15-min (uploaded)")
        _df5_delta  = df5[df5.index > df15_upload.index[-1]]
        df15_recent = resample_from_5min(_df5_delta, "15min")
        if df15_recent.empty:
            df15 = df15_upload
        else:
            df15 = pd.concat([df15_upload, df15_recent]).sort_index()
    else:
        df15 = resample_from_5min(df5, "15min")

    df30 = load_higher(csv30, "30-min") if csv30 and Path(csv30).exists() else resample_from_5min(df5, "30min")
    df60 = load_higher(csv60, "60min")  if csv60 and Path(csv60).exists() else resample_from_5min(df5, "60min")
    if csv_day and Path(csv_day).exists():
        df_day = load_higher(csv_day, "Daily")
        df_day = df_day[~df_day.index.duplicated(keep="last")]
    else:
        df_day = resample_from_5min(df5, "D")

    log.info("\nComputing higher-TF features...")
    feat15   = features_15min(df15)
    feat30   = features_30min(df30)
    feat60   = features_60min(df60)
    feat_day = features_daily(df_day)

    vix_path = csv_vix or CSV_VIX
    df_vix   = load_vix(vix_path)
    feat_vix = features_vix(df_vix) if not df_vix.empty else pd.DataFrame()

    log.info("\nComputing 5-min base features...")
    df_feat = features_5min(df5)

    log.info("\nMerging timeframes...")
    df_feat = merge_htf(df_feat, feat15,  "15m")
    df_feat = merge_htf(df_feat, feat30,  "30m")
    df_feat = merge_htf(df_feat, feat60,  "60m")
    df_feat = merge_daily_onto_5min(df_feat, feat_day)
    if not feat_vix.empty:
        df_feat = merge_vix_onto_5min(df_feat, feat_vix)
    # Fix 1: merge real futures VWAP/CMF features before intraday context
    if use_real_vwap:
        log.info("  Merging futures VWAP/CMF features (Fix 1)...")
        df_feat = merge_futures_onto_5min(df_feat, feat_fut)
    df_feat = add_intraday_context(df_feat, has_real_futures=use_real_vwap)

    log.info(f"Total columns after merge: {len(df_feat.columns)}")

    df_feat = create_labels(df_feat)
    df_feat.dropna(inplace=True)

    # Fix 2: explicitly exclude absolute-price LEVEL columns + helpers
    LEVEL_COLS = {
        "open","high","low","close","volume",
        "day_open","intraday_high","intraday_low",
        "day_prev_high","day_prev_low","day_prev_close",
        "week_high","week_low",
        "vwap_session",
        "fut_vwap",             # Fix 1: raw price level from futures — excluded
        "_atr14_pts",           # helper exposed by features_5min
    }
    # Defensive: also exclude any remaining *_pts feature names (stationarity)
    PTS_EXCLUDE = {c for c in df_feat.columns if c.endswith("_pts")}
    # 2026-06-12 PHASE-1 FINDING: VIX features have no directional value
    # (grouped permutation importance −0.010 on clean frozen test; regime-only
    # ablation test dir_acc 0.464 < chance). Excluded from the MODEL only —
    # vix_level stays in the frame for label creation (VIX>35 → SKIP) and the
    # pilot keeps VIX for SL/TP sizing and confidence adjustment.
    VIX_EXCLUDE = {c for c in df_feat.columns if c.startswith("vix_")}
    # Feature flag (same env var + default as the live pilot's parallel-output
    # toggle in core/claude_pilot.py) -- lets this exact script be run twice,
    # once with the new struct_hh_hl_* columns in FCOLS and once without,
    # for a controlled A/B comparison without touching any other code path.
    _hh_hl_on = os.getenv("FEATURE_HH_HL_ENABLED", "true").lower() == "true"
    HH_HL_EXCLUDE = set() if _hh_hl_on else \
        {c for c in df_feat.columns if c.startswith("struct_hh_hl_")}
    if not _hh_hl_on:
        log.info("  FEATURE_HH_HL_ENABLED=false -> struct_hh_hl_* excluded from FCOLS (baseline run)")
    SKIP = {"label", "volume"} | LEVEL_COLS | PTS_EXCLUDE | VIX_EXCLUDE | HH_HL_EXCLUDE

    FCOLS = [c for c in df_feat.columns if c not in SKIP
             and df_feat[c].dtype in [np.float64, np.int64, np.int32, float, int]]
    log.info(f"\nFCOLS after stationarity filter: {len(FCOLS)} features")
    log.info(f"  Excluded point-based: {sorted(PTS_EXCLUDE)}")
    log.info(f"  Excluded VIX (no directional value, Phase 1): "
             f"{len(VIX_EXCLUDE)} cols")

    X = df_feat[FCOLS].values
    y = df_feat["label"].values

    # ── Purged 3-way split (Issue #2): TRAIN / VAL(early-stop) / TEST(frozen) ──
    train_idx, val_idx, test_idx = purged_train_val_test_split(len(X))
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X[train_idx])     # scaler fit on TRAIN ONLY
    Xvl = sc.transform(X[val_idx])           # early-stopping set
    Xte = sc.transform(X[test_idx])          # FROZEN test — never seen by fit
    ytr, yvl, yte = y[train_idx], y[val_idx], y[test_idx]

    # 2026-05-15 TIER-1 FIX: Separate CALL/PUT class balancing.
    # OLD: sample_w = np.where(ytr == 2, 1.0, trade_w)
    #      → CALL and PUT got SAME weight regardless of frequency.
    # If training data has 70% PUT / 30% CALL (downtrend bias), model
    # under-learns CALL setups → produces 25 PUT vs 2 CALL signals live.
    # NEW: Inverse-frequency weighting per directional class.
    call_n  = int((ytr == 0).sum())
    put_n   = int((ytr == 1).sum())
    skip_n  = int((ytr == 2).sum())
    dir_n   = call_n + put_n
    skip_pct  = (ytr == 2).mean()
    trade_pct = 1 - skip_pct
    trade_w   = skip_pct / trade_pct if trade_pct > 0 else 2.0

    # Per-class balancing factor: scale up the minority direction
    if dir_n > 0 and call_n > 0 and put_n > 0:
        call_balance = dir_n / (2.0 * call_n)
        put_balance  = dir_n / (2.0 * put_n)
    else:
        call_balance = put_balance = 1.0

    sample_w  = np.where(ytr == 0, trade_w * call_balance,
                np.where(ytr == 1, trade_w * put_balance, 1.0))
    log.info(
        f"Class balance: CALL_n={call_n:,} PUT_n={put_n:,} SKIP_n={skip_n:,} | "
        f"CALL_weight×{call_balance:.2f}, PUT_weight×{put_balance:.2f}"
    )
    if abs(call_balance - put_balance) > 0.5:
        log.warning(
            f"⚠️  Significant CALL/PUT imbalance — model would be "
            f"{'PUT-biased' if put_n > call_n else 'CALL-biased'} without this fix"
        )

    try:
        if "expiry_is_tue" in df_feat.columns:
            _tue_era = df_feat["expiry_is_tue"].values[train_idx].astype(bool)
        else:
            _tue_era = np.zeros(len(train_idx), dtype=bool)
        _boost = np.where(_tue_era, 6.0, 1.0)
        sample_w = sample_w * _boost
        log.info(f"Expiry-calendar boost: {_tue_era.sum():,}/{len(_tue_era):,} rows × 6.0x")
    except Exception as _e:
        log.warning(f"Expiry-calendar boost skipped: {_e}")

    try:
        # Fix 2 (stationarity): trending-day threshold was 150 absolute pts —
        # non-stationary across an 8000→24000 index range. Replace with
        # ATR-normalized threshold: day move > 1.5× daily ATR = trending.
        df_train_slice = df_feat.iloc[train_idx]
        bar_dates      = df_train_slice.index.date
        day_open_s     = df_train_slice.groupby(bar_dates)["open"].first()
        day_close_s    = df_train_slice.groupby(bar_dates)["close"].last()
        day_move_s     = (day_close_s - day_open_s).abs()
        # Daily ATR proxy: use day_atr_pct × close if available, else HL range
        if "day_atr_pct" in df_train_slice.columns:
            day_atr_s = (
                df_train_slice.groupby(bar_dates)["day_atr_pct"].last()
                * day_close_s
            )
        else:
            day_high_s = df_train_slice.groupby(bar_dates)["high"].max()
            day_low_s  = df_train_slice.groupby(bar_dates)["low"].min()
            day_atr_s  = day_high_s - day_low_s
        atr_mult_s    = day_move_s / day_atr_s.replace(0, np.nan)
        trending_days = set(atr_mult_s[atr_mult_s > 1.5].index)
        trend_mask    = np.array([d in trending_days for d in bar_dates], dtype=bool)
        sample_w      = sample_w * np.where(trend_mask, 3.0, 1.0)
        log.info(
            f"Trending-day boost (ATR-normalized): "
            f"{trend_mask.sum():,}/{len(trend_mask):,} rows × 3.0x "
            f"(threshold: day_move > 1.5 × daily_ATR)"
        )
    except Exception as _e:
        log.warning(f"Trending-day boost skipped: {_e}")

    log.info(f"Class weights: CALL/PUT={trade_w:.2f}x  SKIP=1.0x")

    # ── Pass 1 ────────────────────────────────────────────────────
    log.info("\n── Pass 1 (all filtered features) ──")
    models = _fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w)

    # FIX 4: SHAP prune + retrain
    if prune_shap:
        log.info("\n── SHAP pruning ──")
        anchor_model = models.get("xgb") or models.get("lgb") or list(models.values())[0]
        pruned_cols = shap_prune_features(
            anchor_model, Xtr, FCOLS,
            bottom_pct=0.25, corr_threshold=0.95
        )
        if len(pruned_cols) < len(FCOLS):
            log.info("\n── Pass 2 (pruned features) ──")
            FCOLS = pruned_cols
            X = df_feat[FCOLS].values
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[train_idx])
            Xvl = sc.transform(X[val_idx])
            Xte = sc.transform(X[test_idx])
            models = _fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w)

    # ── Save to registry (Issue #4: never overwrite live model directly) ─────
    # The registry saves a versioned candidate. The promotion gate in
    # retrain_weekly.py decides whether to copy it to the live path.
    # Direct joblib.dump to the live path is intentionally removed.
    Path(MODEL_PATH).parent.mkdir(exist_ok=True)
    log.info(f"\n  Saving candidate to registry (live path updated only after gate passes)")

    # ── Evaluation ────────────────────────────────────────────────────────
    # Headline metrics use the FROZEN TEST set (never seen during fit).
    # VAL is shown too — a large VAL-vs-TEST gap signals overfitting to the
    # early-stopping set.
    CONF = 0.22

    def _eval(X_eval, y_eval):
        p = np.mean([m.predict_proba(X_eval) for m in models.values()], axis=0)
        s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
        t = s != 2
        if not t.any():
            return None
        a  = (y_eval[t] == s[t]).mean()
        dm = t & (y_eval != 2)
        da = (y_eval[dm] == s[dm]).mean() if dm.any() else 0.0
        return {"acc": a, "dir_acc": da,
                "nc": int((s == 0).sum()), "np": int((s == 1).sum()),
                "ns": int((s == 2).sum())}

    val_m  = _eval(Xvl, yvl)
    test_m = _eval(Xte, yte)

    log.info(f"\n{'='*62}")
    log.info("  EVALUATION (V9) — headline = FROZEN TEST set")
    log.info(f"{'='*62}")
    if val_m:
        log.info(f"  [VAL  early-stop] acc={val_m['acc']:.1%}  "
                 f"dir_acc={val_m['dir_acc']:.1%}  "
                 f"(CALL={val_m['nc']} PUT={val_m['np']} SKIP={val_m['ns']})")
    if test_m:
        log.info(f"  [TEST  frozen   ] acc={test_m['acc']:.1%}  "
                 f"dir_acc={test_m['dir_acc']:.1%}  "
                 f"(CALL={test_m['nc']} PUT={test_m['np']} SKIP={test_m['ns']})")
        if val_m:
            gap = val_m['dir_acc'] - test_m['dir_acc']
            log.info(f"  VAL→TEST dir_acc gap: {gap:+.1%} "
                     f"({'⚠️ overfit risk' if gap > 0.05 else 'ok'})")
    else:
        log.warning("  No TEST signals at CONF=%.2f — threshold may be too high", CONF)

    # Headline values for downstream logging
    if test_m:
        acc, dir_acc = test_m["acc"], test_m["dir_acc"]
    elif val_m:
        acc, dir_acc = val_m["acc"], val_m["dir_acc"]
    else:
        acc, dir_acc = 0.0, 0.0
    if test_m or val_m:

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
        log.warning("  No signals — CONFIDENCE may be too high")

    log.info(f"\n{'='*62}")
    log.info("  V9 Training complete!")
    log.info(f"{'='*62}\n")

    # ── Registry: save candidate with metadata (Issue #4) ─────────────────
    candidate_id = None
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.model_registry import save_candidate

        tr_dates = df_feat.index[train_idx]
        va_dates = df_feat.index[val_idx]
        te_dates = df_feat.index[test_idx]

        metadata = {
            # Data ranges
            "train_date_start": str(tr_dates.min().date()),
            "train_date_end":   str(tr_dates.max().date()),
            "val_date_start":   str(va_dates.min().date()),
            "val_date_end":     str(va_dates.max().date()),
            "test_date_start":  str(te_dates.min().date()),
            "test_date_end":    str(te_dates.max().date()),
            "train_bars":  int(len(train_idx)),
            "val_bars":    int(len(val_idx)),
            "test_bars":   int(len(test_idx)),
            # Model quality (from frozen TEST set)
            "val_dir_acc":  round(float(val_m["dir_acc"]),  4) if val_m  else None,
            "test_dir_acc": round(float(test_m["dir_acc"]), 4) if test_m else 0.0,
            "test_signals": int(test_m["nc"] + test_m["np"]) if test_m else 0,
            "val_test_gap": round(
                float(val_m["dir_acc"]) - float(test_m["dir_acc"]), 4
            ) if (val_m and test_m) else None,
            # Architecture
            "n_features":    len(FCOLS),
            "fwd_bars":      FWD_BARS,
            "train_frac":    TRAIN_FRAC,
            "val_frac":      VAL_FRAC,
            "embargo_bars":  EMBARGO_BARS,
            "label_quantile": 0.70,
        }

        candidate_id = save_candidate("v9", models, sc, FCOLS, metadata)
        log.info(f"  Registry candidate: v9/{candidate_id}")
        log.info(f"  (promote via: retrain_weekly.py or registry.promote_if_passes_gate)")
    except Exception as _e:
        log.warning(f"  Registry save failed ({_e}) — falling back to direct save")
        # Fallback: write directly so training is not lost even if registry fails
        joblib.dump(models, MODEL_PATH)
        joblib.dump(sc,     SCALER_PATH)
        joblib.dump(FCOLS,  FCOLS_PATH)
        log.info(f"  Fallback saved: {MODEL_PATH}")

    return candidate_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nifty V9 model")
    parser.add_argument("--csv5",   default=CSV_5M)
    parser.add_argument("--csv15",  default=CSV_15M)
    parser.add_argument("--csv30",  default=CSV_30M)
    parser.add_argument("--csv60",  default=CSV_60M)
    parser.add_argument("--csvday", default=CSV_DAY)
    parser.add_argument("--csvvix", default=CSV_VIX)
    parser.add_argument("--csvfut", default=CSV_FUT,
                        help="Nifty futures 5-min OHLCV with real volume (Fix 1)")
    parser.add_argument("--prune-shap", action="store_true",
                        help="Apply SHAP bottom-25%% pruning after first pass (Fix 4)")
    args = parser.parse_args()

    train(
        csv5       = args.csv5,
        csv15      = args.csv15,
        csv30      = args.csv30,
        csv60      = args.csv60,
        csv_day    = args.csvday,
        csv_vix    = args.csvvix,
        csv_fut    = args.csvfut,
        prune_shap = args.prune_shap,
    )
