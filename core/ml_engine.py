"""
ml_engine.py  -  V8 ML Model for Nifty option signals
------------------------------------------------------
Multi-timeframe ensemble (XGBoost + LightGBM) that generates
CALL/PUT/SKIP signals from 5-min OHLC bars enriched with
15-min, 30-min, 60-min and daily features.

143 features:
  - 5-min base: candlesticks, RSI, MACD, EMA, ATR, BB, SMC, vol
  - 15-min: ADX trend state, RSI, MACD, EMA alignment
  - 30-min: ADX, RSI, EMA, close position
  - 60-min: ADX, RSI, EMA, session bias
  - Daily: prev day H/L/C, gap, weekly H/L, EMA200
"""

import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

log = logging.getLogger(__name__)

# -- Module state --
_models = None
_scaler = None
_fcols  = None
_loaded = False

# ── Model paths (auto-selected: V10 > V9 > V8 > V7) ──────────────
MODEL_PATH        = "models/nifty_v10_models.pkl"
SCALER_PATH       = "models/nifty_v10_scaler.pkl"
FEATURE_COLS_PATH = "models/feature_cols_v10.pkl"

# Populated by load_model() — used to pick matching feature-engineering fn
# and to label log lines ("ML V8" vs "ML V9"). Falls back to "V8" so old
# log greps still match until a V9 pickle is found.
_loaded_version: str = "V8"

# ── 2026-06-12 PROBABILITY CALIBRATION (parallel output, NOT gate input) ──
# Loads models/calibration/calibrator.pkl (isotonic/Platt fit on the VAL set
# only). Calibrated P(class) is computed each cycle and exposed in the
# summary + log line so downstream consumers and the shadow analysis can use
# an honest probability. The gate-facing `conf` blend is DELIBERATELY
# unchanged: calibrated class probabilities live on the true base-rate scale
# (~5-30%), so feeding them into the 72/58 bars without a remap would
# silently halt trading — that remap is a separate, explicitly-approved step.
_calibrator = None
_calib_tried = False


def _get_calibrator():
    global _calibrator, _calib_tried
    if _calib_tried:
        return _calibrator
    _calib_tried = True
    try:
        import joblib as _jl
        p = Path("models/calibration/calibrator.pkl")
        if p.exists():
            _calibrator = _jl.load(p)
            log.info(f"🎯 Probability calibrator loaded "
                     f"({_calibrator.get('method')}) — parallel output only")
    except Exception as e:
        log.debug(f"Calibrator load failed (parallel output disabled): {e}")
    return _calibrator


def _calibrate(proba):
    """Return calibrated [call, put, skip] or None. Never raises."""
    try:
        cal = _get_calibrator()
        if cal is None:
            return None
        out = []
        for k in range(3):
            c = cal["calibrators"][k]
            if hasattr(c, "predict_proba"):              # Platt
                out.append(float(c.predict_proba([[proba[k]]])[0, 1]))
            else:                                         # isotonic
                out.append(float(c.predict([proba[k]])[0]))
        s = sum(out)
        return [x / s for x in out] if s > 0 else None
    except Exception:
        return None


# Signal filter thresholds — tuned for trade-simulation trained model
# Trade-sim labels are conservative so we lower thresholds slightly
#
# 2026-06-11 ASYMMETRIC THRESHOLDS: live CALL WR=22.2% (n=18) vs PUT near
# backtest — the shared 0.30 threshold was too loose for CALL and too tight
# for PUT (Jun 10 PM: PUT prob peaked at 0.205 in a weakening market, zero
# trades taken). CALL raised 0.30→0.32, PUT lowered 0.30→0.25.
# SKIP_CEIL 0.60→0.65: SKIP prob is miscalibrated-high (0.74-0.80 all day);
# allow directional signals through when SKIP is only moderately dominant.
CONFIDENCE_CALL = 0.32
CONFIDENCE_PUT  = 0.25
MIN_EDGE        = 0.05
SKIP_CEIL       = 0.65


# ── Feature engineering — always V9 (V10 extends V9, V8 no longer used) ──────
def _get_engineer_fn():
    """
    Always return V9 feature engineering.
    V10 = V9 base features + 13 option chain features.
    V8 is no longer trained or deployed.
    """
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts.train_model_v9 import features_5min
        log.debug("Using V9 feature engineering (5-min base)")
        return features_5min
    except Exception as e:
        log.warning(f"V9 features import failed ({e}) — using V6 inline fallback")
        return _engineer_features_v6


def load_model(model_dir: str = "") -> bool:
    """Load pre-trained ML models. Prefers V10 → V9. V8/V7 no longer used."""
    global _models, _scaler, _fcols, _loaded, _loaded_version

    try:
        import joblib
    except ImportError:
        log.error("joblib not installed — ML engine disabled")
        return False

    base = Path(model_dir) if model_dir else Path(".")

    # V11 preferred (26 features: the co-movement-pruned trader set).
    # V10 next (188 features: V9 base + option chain).
    # V9 fallback if neither is trained/promoted.
    # V8 and V7 removed — no longer trained or maintained.
    #
    # Listing V11 first is inert until something writes its live paths.
    # Only model_registry.promote_if_passes_gate() does that, and it
    # refuses to compare across metric_version, so V11 (metric_version 2)
    # can only displace a champion scored the same way.
    candidates = [
        ("V11",
         base / "models/nifty_v11_models.pkl",
         base / "models/nifty_v11_scaler.pkl",
         base / "models/feature_cols_v11.pkl"),
        ("V10",
         base / "models/nifty_v10_models.pkl",
         base / "models/nifty_v10_scaler.pkl",
         base / "models/feature_cols_v10.pkl"),
        ("V9",
         base / "models/nifty_v9_models.pkl",
         base / "models/nifty_v9_scaler.pkl",
         base / "models/feature_cols_v9.pkl"),
    ]

    # 2026-08-06: optional pin. Default "" keeps the V10→V9 preference
    # above exactly as-is (no behavior change).
    #
    # Why this exists: a head-to-head on identical, fully out-of-sample
    # bars (2026-05-18→2026-07-31), both scored with this module's own
    # live decision rule, found V9 strictly better than V10 —
    #   V9 : recall 54.5% of real directional bars, dir_acc 60.3% (n=146)
    #   V10: recall  7.1% of real directional bars, dir_acc 73.7% (n=19)
    # V10's edge in dir_acc is on 19 signals (high variance) while its
    # recall is 7.7x worse, because requiring option/IV context on every
    # row cuts training data from ~212k bars to ~29k (13.8%). Set
    # ML_MODEL_VERSION=v9 to pin V9 and skip V10 entirely.
    # 2026-08-07: V11 added to the pin set. A controlled ablation holding
    # the whole pipeline constant found the 163-feature set generalizes
    # worse than a 26-feature one — every subset at <=50 features had
    # POSITIVE VAL->TEST drift, every subset at >=99 had NEGATIVE drift:
    #   V9  (163 feat): TEST dir_acc 60.0%, recall 63.7%, drift -0.015
    #   V11 ( 26 feat): TEST dir_acc 64.9%, recall 48.9%, drift +0.007
    # V11 fires ~24% fewer signals at ~5pp higher accuracy, which is the
    # right trade under friction. See scripts/train_model_v11.py.
    _pin = os.getenv("ML_MODEL_VERSION", "").strip().lower()
    if _pin in ("v9", "v10", "v11"):
        _want = _pin.upper()
        candidates = [c for c in candidates if c[0] == _want]
        log.info(f"ML_MODEL_VERSION={_pin} — pinned to {_want}, other versions skipped")
    elif _pin:
        log.warning(
            f"ML_MODEL_VERSION={_pin!r} not recognised (expected 'v9', 'v10' or "
            f"'v11') — ignoring, using default V11→V10→V9 preference"
        )

    for version, mpath, spath, fpath in candidates:
        if mpath.exists() and spath.exists() and fpath.exists():
            try:
                _models = joblib.load(mpath)
                _scaler = joblib.load(spath)
                _fcols  = joblib.load(fpath)
                _loaded = True
                _loaded_version = version
                log.info(
                    f"🤖 ML model loaded [{version}]: {list(_models.keys())} | "
                    f"{len(_fcols)} features"
                )
                # 2026-06-12 PIPELINE-COMPATIBILITY GUARD: the feature
                # pipeline was leak-fixed on 2026-06-11. Models trained
                # before that receive shifted daily/HTF inputs they never
                # saw in training (deployment-readiness config B measured
                # PF 0.73 / net-negative in that state). Warn loudly.
                try:
                    from datetime import datetime as _dt
                    _fix_date = _dt(2026, 6, 11, 12, 0)
                    _mtime = _dt.fromtimestamp(mpath.stat().st_mtime)
                    if _mtime < _fix_date:
                        log.warning(
                            f"⚠️ MODEL/PIPELINE MISMATCH: {mpath.name} was "
                            f"trained {_mtime:%Y-%m-%d %H:%M} — BEFORE the "
                            f"2026-06-11 leakage fix. It now receives "
                            f"shifted daily/HTF features it was not trained "
                            f"on. Promote a clean-trained model."
                        )
                except Exception:
                    pass
                return True
            except Exception as e:
                log.warning(f"Model load failed ({mpath.name}): {e}")
                continue

    log.warning("🤖 No ML model found — ML signals disabled")
    return False


def is_ready() -> bool:
    return _loaded


def predict(bar_buffer: pd.DataFrame) -> tuple:
    """
    Run V8 model on 5-min bar buffer.

    Returns:
        signal: 0=CALL, 1=PUT, 2=SKIP
        proba: [P(CALL), P(PUT), P(SKIP)]
        confidence: max probability
        features_summary: dict with key indicator values
    """
    if not _loaded:
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}

    try:
        # Use V8 feature engineering (imports from train_model_v8)
        eng_fn = _get_engineer_fn()
        df_feat = eng_fn(bar_buffer)
        df_feat.dropna(inplace=True)

        if len(df_feat) == 0:
            return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}

        missing = [c for c in _fcols if c not in df_feat.columns]
        if missing:
            log.error(f"Missing features: {missing[:5]}")
            return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}

        X        = df_feat[_fcols].iloc[[-1]].values
        X_scaled = _scaler.transform(X)

        # 2026-05-15 TIER-4 #4: Ensemble disagreement check.
        # XGB + LGB disagreement >0.20 = unreliable signal → force SKIP.
        individual_probas = [m.predict_proba(X_scaled) for m in _models.values()]
        probas = np.mean(individual_probas, axis=0)
        proba = probas[0]
        conf  = float(proba.max())
        call_p, put_p, skip_p = proba[0], proba[1], proba[2]

        disagreement = 0.0
        if len(individual_probas) >= 2:
            preds = [p[0] for p in individual_probas]
            disagreement = max(
                max(p[0] for p in preds) - min(p[0] for p in preds),
                max(p[1] for p in preds) - min(p[1] for p in preds),
            )

        if (call_p >= CONFIDENCE_CALL and call_p - put_p >= MIN_EDGE
                and skip_p < SKIP_CEIL):
            signal = 0
        elif (put_p >= CONFIDENCE_PUT and put_p - call_p >= MIN_EDGE
                and skip_p < SKIP_CEIL):
            signal = 1
        else:
            signal = 2

        # Force SKIP when models disagree significantly
        if signal in (0, 1) and disagreement > 0.20:
            log.info(
                f"🤖 ML ENSEMBLE DISAGREE: spread={disagreement:.2f}>0.20 "
                f"→ forcing SKIP (was {['CALL','PUT'][signal]})"
            )
            signal = 2

        # 2026-04-27 EXPERT FIX: directional probability vs argmax probability
        # Why: 3-class softmax (CALL/PUT/SKIP) splits prob mass 3 ways.
        # Pure argmax under-reports conviction.
        #
        # P(CALL | direction) = call/(call+put) is the right probability,
        # BUT it can over-amplify when one side is ~0:
        #   call=0.50 put=0.01 skip=0.49 → 98% (not real conviction!)
        # So we blend with raw call_p (anchors against SKIP-suction):
        #   conf = 0.65 × dir_prob + 0.35 × raw_prob_×_2
        # This preserves the directional fix but caps inflation.
        # Final hard cap: 90% — preserves "very high" but blocks fake 99%.
        if signal == 0:
            dir_p = call_p / max(call_p + put_p, 1e-9)
            anchor = min(1.0, call_p * 2.0)   # raw scaled to same range
            conf = 0.65 * dir_p + 0.35 * anchor
            conf = min(0.90, conf)
        elif signal == 1:
            dir_p = put_p / max(call_p + put_p, 1e-9)
            anchor = min(1.0, put_p * 2.0)
            conf = 0.65 * dir_p + 0.35 * anchor
            conf = min(0.90, conf)
        # else: leave conf as proba.max() (= skip_p, already correct)

        # Key indicators for Claude context
        last = df_feat.iloc[-1]
        summary = _build_summary(last)

        sig_name = "CALL" if signal == 0 else ("PUT" if signal == 1 else "SKIP")
        log.info(
            f"🤖 ML {_loaded_version}: {sig_name} | CALL={call_p:.3f} PUT={put_p:.3f} "
            f"SKIP={skip_p:.3f} | dir_conf={conf:.3f} | RSI={summary['rsi14']} "
            f"ADX={summary['tf15_adx']}"
        )
        return signal, proba, conf, summary

    except Exception as e:
        log.error(f"ML prediction error: {e}", exc_info=True)
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}


def predict_precomputed(df_feat: pd.DataFrame) -> tuple:
    """
    Run V8 model on a DataFrame that ALREADY has all features engineered.
    Called by the pilot after it does multi-TF feature engineering itself.

    This skips the feature engineering step — goes straight to the model.
    Accepts the full feature DataFrame and uses only the last row.

    Returns: (signal, proba, confidence, features_summary)
    """
    if not _loaded:
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}

    try:
        if len(df_feat) == 0:
            return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}

        # Fill missing features with 0 (some TFs may not have enough bars)
        missing = [c for c in _fcols if c not in df_feat.columns]
        if missing:
            log.debug(f"predict_precomputed: {len(missing)} features missing — filling with 0")
            for col in missing:
                df_feat[col] = 0.0

        X        = df_feat[_fcols].iloc[[-1]].values
        X_scaled = _scaler.transform(X)

        # 2026-05-15 TIER-4 #4: Ensemble disagreement check.
        # XGB + LGB disagreement >0.20 = unreliable signal → force SKIP.
        individual_probas = [m.predict_proba(X_scaled) for m in _models.values()]
        probas = np.mean(individual_probas, axis=0)
        proba = probas[0]
        conf  = float(proba.max())
        call_p, put_p, skip_p = proba[0], proba[1], proba[2]

        disagreement = 0.0
        if len(individual_probas) >= 2:
            preds = [p[0] for p in individual_probas]
            disagreement = max(
                max(p[0] for p in preds) - min(p[0] for p in preds),
                max(p[1] for p in preds) - min(p[1] for p in preds),
            )

        if (call_p >= CONFIDENCE_CALL and call_p - put_p >= MIN_EDGE
                and skip_p < SKIP_CEIL):
            signal = 0
        elif (put_p >= CONFIDENCE_PUT and put_p - call_p >= MIN_EDGE
                and skip_p < SKIP_CEIL):
            signal = 1
        else:
            signal = 2

        # Force SKIP when models disagree significantly
        if signal in (0, 1) and disagreement > 0.20:
            log.info(
                f"🤖 ML ENSEMBLE DISAGREE: spread={disagreement:.2f}>0.20 "
                f"→ forcing SKIP (was {['CALL','PUT'][signal]})"
            )
            signal = 2

        # 2026-04-27 EXPERT FIX: directional probability vs argmax probability
        # Why: 3-class softmax (CALL/PUT/SKIP) splits prob mass 3 ways.
        # Pure argmax under-reports conviction.
        #
        # P(CALL | direction) = call/(call+put) is the right probability,
        # BUT it can over-amplify when one side is ~0:
        #   call=0.50 put=0.01 skip=0.49 → 98% (not real conviction!)
        # So we blend with raw call_p (anchors against SKIP-suction):
        #   conf = 0.65 × dir_prob + 0.35 × raw_prob_×_2
        # This preserves the directional fix but caps inflation.
        # Final hard cap: 90% — preserves "very high" but blocks fake 99%.
        if signal == 0:
            dir_p = call_p / max(call_p + put_p, 1e-9)
            anchor = min(1.0, call_p * 2.0)   # raw scaled to same range
            conf = 0.65 * dir_p + 0.35 * anchor
            conf = min(0.90, conf)
        elif signal == 1:
            dir_p = put_p / max(call_p + put_p, 1e-9)
            anchor = min(1.0, put_p * 2.0)
            conf = 0.65 * dir_p + 0.35 * anchor
            conf = min(0.90, conf)
        # else: leave conf as proba.max() (= skip_p, already correct)

        # Key indicators for Claude context
        last = df_feat.iloc[-1]
        summary = _build_summary(last)

        # Calibrated probabilities — parallel output, never gates
        calib = _calibrate([call_p, put_p, skip_p])
        calib_str = ""
        if calib is not None:
            summary["calib_call_p"] = round(calib[0], 4)
            summary["calib_put_p"] = round(calib[1], 4)
            summary["calib_skip_p"] = round(calib[2], 4)
            calib_str = (f" | calib C={calib[0]:.3f} P={calib[1]:.3f} "
                         f"S={calib[2]:.3f}")

        sig_name = "CALL" if signal == 0 else ("PUT" if signal == 1 else "SKIP")
        log.info(
            f"🤖 ML {_loaded_version}: {sig_name} | CALL={call_p:.3f} PUT={put_p:.3f} "
            f"SKIP={skip_p:.3f} | dir_conf={conf:.3f} | RSI={summary['rsi14']} "
            f"ADX={summary['tf15_adx']}{calib_str}"
        )
        return signal, proba, conf, summary

    except Exception as e:
        log.error(f"predict_precomputed error: {e}", exc_info=True)
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, {}


def _build_summary(last) -> dict:
    """Extract key indicator values from the last row for Claude context."""
    return {
        "rsi14":         round(float(last.get("rsi14",    0)), 1),
        "rsi7":          round(float(last.get("rsi7",     0)), 1),
        "macd_hist":     round(float(last.get("mh",       0)) * 100, 4),
        "ema_aligned":   bool(last.get("e9a21",   0)),
        "ema_slope":     round(float(last.get("eslp",     0)) * 10000, 2),
        "stoch_5":       round(float(last.get("st5",      0)), 1),
        "stoch_14":      round(float(last.get("st14",     0)), 1),
        "bb_position":   round(float(last.get("bbp",      0)), 2),
        "bb_squeeze":    bool(last.get("bbsq",    0)),
        "atr_ratio":     round(float(last.get("ar",       0)), 2),
        "vol_regime":    round(float(last.get("vol_regime", 0)), 2),
        "cci20":         round(float(last.get("cci20",    0)), 1),
        "williams_r":    round(float(last.get("wr14",     0)), 1),
        "body_ratio":    round(float(last.get("body_ratio", 0)), 2),
        "is_doji":       bool(last.get("is_doji",  0)),
        "bull_engulfing":bool(last.get("bull_eng", 0)),
        "bear_engulfing":bool(last.get("bear_eng", 0)),
        "consec_bull":   int(last.get("consec_bull", 0)),
        "consec_bear":   int(last.get("consec_bear", 0)),
        # VIX features
        "vix_level":     round(float(last.get("vix_level",  0)), 2),
        "vix_regime":    int(last.get("vix_regime", -1)),
        "vix_ideal":     bool(last.get("vix_ideal",  0)),
        "vix_danger":    bool(last.get("vix_danger", 0)),
        "vix_rising":    bool(last.get("vix_rising", 0)),
        "vix_spike":     bool(last.get("vix_spike",  0)),
        # Multi-TF
        "tf15_adx":      round(float(last.get("tf15_adx",      0)), 1),
        "tf15_trending": bool(last.get("tf15_trending",  0)),
        "tf15_sideways": bool(last.get("tf15_sideways",  0)),
        "day_gap_pct":   round(float(last.get("day_gap_pct",   0)) * 100, 2),
        "dist_prev_high":round(float(last.get("dist_prev_high",0)) * 100, 2),
        "dist_prev_low": round(float(last.get("dist_prev_low", 0)) * 100, 2),
        "intraday_pos":  round(float(last.get("intraday_pos",  0)), 2),
    }


def predict_batch(df: pd.DataFrame) -> np.ndarray:
    """
    Batch predict for backtesting. Returns array of signals.
    """
    if not _loaded:
        return np.full(len(df), 2)

    feat = _get_engineer_fn()(df)
    feat.dropna(inplace=True)

    missing = [c for c in _fcols if c not in feat.columns]
    if missing:
        return np.full(len(feat), 2)

    X = feat[_fcols].values
    X_scaled = _scaler.transform(X)
    probas = np.mean(
        [m.predict_proba(X_scaled) for m in _models.values()], axis=0
    )

    signals = np.full(len(probas), 2)
    signals[
        (probas[:, 0] >= CONFIDENCE_CALL)
        & (probas[:, 0] - probas[:, 1] >= MIN_EDGE)
        & (probas[:, 2] < SKIP_CEIL)
    ] = 0
    signals[
        (probas[:, 1] >= CONFIDENCE_PUT)
        & (probas[:, 1] - probas[:, 0] >= MIN_EDGE)
        & (probas[:, 2] < SKIP_CEIL)
    ] = 1

    return signals, probas, feat


# ======================================================================
# FEATURE ENGINEERING (identical to training — must stay in sync)
# ======================================================================
def _engineer_features_v6(df):
    df = df.copy()
    o, h, l, c = df.open, df.high, df.low, df.close
    fr = h - l + 1e-8
    body = c - o

    # Candlestick features
    df["body_pct"] = body / o
    df["body_ratio"] = body.abs() / fr
    df["upper_wick"] = (h - c.clip(lower=o)) / fr
    df["lower_wick"] = (o.clip(upper=c) - l) / fr
    df["is_bullish"] = (c > o).astype(int)
    df["hl_range"] = fr / c
    df["close_pos"] = (c - l) / fr
    df["is_doji"] = (df["body_ratio"] < 0.10).astype(int)
    df["is_marubozu"] = (df["body_ratio"] > 0.85).astype(int)
    df["prev_cp"] = df["close_pos"].shift(1)
    df["prev_br"] = df["body_ratio"].shift(1)

    # Returns
    df["log_ret"] = np.log(c / c.shift(1))
    df["gap_up"] = ((o - c.shift(1)) / c.shift(1)).clip(lower=0)
    df["gap_dn"] = ((c.shift(1) - o) / c.shift(1)).clip(lower=0)

    # Consecutive candles
    bull = (c > o).astype(int)
    bear = (c < o).astype(int)
    df["consec_bull"] = bull.groupby(
        (bull != bull.shift()).cumsum()
    ).cumcount().where(bull == 1, 0)
    df["consec_bear"] = bear.groupby(
        (bear != bear.shift()).cumsum()
    ).cumcount().where(bear == 1, 0)

    # Patterns
    df["bull_eng"] = (
        (c > o) & (c.shift(1) < o.shift(1))
        & (c > o.shift(1)) & (o < c.shift(1))
    ).astype(int)
    df["bear_eng"] = (
        (c < o) & (c.shift(1) > o.shift(1))
        & (c < o.shift(1)) & (o > c.shift(1))
    ).astype(int)
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

    # EMA alignment & slope
    e9 = c.ewm(span=9, adjust=False).mean()
    e13 = c.ewm(span=13, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    df["e9a21"] = (e9 > e21).astype(int)
    df["e9s"] = (e9 - e13) / c
    df["e13s"] = (e13 - e21) / c
    df["eslp"] = e9.diff(3) / c

    # ATR variants (VOLUME PROXY — measures volatility without volume)
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    a10 = tr.rolling(10).mean()
    a14 = tr.rolling(14).mean()
    a50 = tr.rolling(50).mean()
    df["a10"] = a10 / c
    df["a14"] = a14 / c
    df["ar"] = a10 / a14.replace(0, np.nan)
    df["ava"] = a10 / a50.replace(0, np.nan)

    # RSI
    for p in [7, 14]:
        d2 = c.diff()
        g = d2.clip(lower=0).ewm(span=p, adjust=False).mean()
        ls = (-d2.clip(upper=0)).ewm(span=p, adjust=False).mean()
        df[f"rsi{p}"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))
    df["rsis"] = df["rsi14"].diff(3)

    # MACD
    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    ms = mac.ewm(span=9, adjust=False).mean()
    df["mh"] = (mac - ms) / c
    df["mx"] = np.sign(mac - ms)
    df["mslp"] = df["mh"].diff(2)

    # Stochastic
    for k in [5, 14]:
        lk = l.rolling(k).min()
        hk = h.rolling(k).max()
        df[f"st{k}"] = 100 * (c - lk) / (hk - lk + 1e-9)
    df["stx"] = np.sign(df["st5"] - df["st14"])

    # Williams %R
    df["wr14"] = -100 * (h.rolling(14).max() - c) / (
        h.rolling(14).max() - l.rolling(14).min() + 1e-9
    )

    # CCI
    tp2 = (h + l + c) / 3
    df["cci20"] = (tp2 - tp2.rolling(20).mean()) / (
        0.015 * tp2.rolling(20).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        ) + 1e-9
    )

    # Realized volatility (VOLUME PROXY)
    for w in [5, 10, 20]:
        df[f"rv{w}"] = df["log_ret"].rolling(w).std() * np.sqrt(252 * 75)

    # Bollinger Bands
    bm = c.rolling(20).mean()
    bs = c.rolling(20).std() + 1e-9
    df["bbw"] = 4 * bs / bm
    df["bbp"] = (c - (bm - 2 * bs)) / (4 * bs)
    df["bbsq"] = (df["bbw"] < df["bbw"].rolling(50).quantile(0.2)).astype(int)

    # Parkinson volatility (VOLUME PROXY from OHLC)
    df["park"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(h / l) ** 2).rolling(10).mean() * 252 * 75
    )

    # Garman-Klass volatility (VOLUME PROXY from OHLC)
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    df["gkv"] = np.sqrt(gk.rolling(10).mean() * 252 * 75)

    # Volatility of volatility
    df["vov"] = df["rv10"].rolling(20).std()

    # Session flags
    ts = df.index.strftime("%H:%M")
    df["is_morning"] = ((ts >= "09:15") & (ts <= "10:30")).astype(int)
    df["is_midday"] = ((ts > "10:30") & (ts <= "13:30")).astype(int)
    df["is_afternoon"] = ((ts > "13:30") & (ts <= "15:30")).astype(int)
    # Expiry day & DTE — derived from real historical expiry calendar
    # (data/nifty_expiry_history.csv) so that backtest features match training
    # exactly. Falls back to NIFTY_EXPIRY env var, then to Tuesday weekday.
    df["dow"] = df.index.dayofweek
    try:
        import os
        cal_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "nifty_expiry_history.csv",
        )
        if os.path.exists(cal_path):
            exp_df = pd.read_csv(cal_path, parse_dates=["expiry_date"])
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
        else:
            raise FileNotFoundError("expiry calendar missing")
    except Exception:
        # Fallback: NIFTY_EXPIRY env var (current expiry only)
        try:
            from core.expiry_utils import get_expiry_date
            exp_date = get_expiry_date()
            if exp_date is not None:
                expiry_dow = exp_date.weekday()
                df["is_expiry"] = (df.index.dayofweek == expiry_dow).astype(int)
                df["dte"] = ((expiry_dow - df.index.dayofweek) % 7).astype(int).clip(upper=5)
                df["expiry_dow"] = int(expiry_dow)
                df["expiry_is_tue"] = int(expiry_dow == 1)
            else:
                raise ValueError("No expiry")
        except Exception:
            # Last resort: Tuesday weekday
            df["is_expiry"] = (df.index.dayofweek == 1).astype(int)
            dte_map = {0: 1, 1: 0, 2: 5, 3: 4, 4: 3}
            df["dte"] = df.index.dayofweek.map(dte_map).fillna(3).astype(int)
            df["expiry_dow"] = 1
            df["expiry_is_tue"] = 1
    df["dte_norm"] = df["dte"] / 6.0

    return df
