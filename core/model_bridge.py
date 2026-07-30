"""
model_bridge.py — Connects the trained V7 ML model to the Claude autopilot.

Loads the pre-trained XGBoost/LightGBM ensemble and provides:
  - Real-time CALL/PUT/SKIP signal with confidence scores
  - Feature importance context for Claude to reason about
  - The ML signal becomes additional data in Claude's analysis prompt

Usage in Claude pilot:
  1. Every 5 minutes, fetch latest 5-min bars from OpenAlgo
  2. Run model_bridge.predict() to get ML signal
  3. Include ML signal + confidence in the data sent to Claude
  4. Claude weighs ML signal alongside OI, PCR, and its own reasoning
  5. Combined decision is more accurate than either alone

The bridge also works standalone without Claude — can be used
directly by the level monitor or manual trading endpoints.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime

# Use the same feature engineering as training
from scripts.train_model_v7 import engineer_features

log = logging.getLogger("ModelBridge")

# Default paths (override via config if available)
DEFAULT_MODEL_PATH = "models/nifty_v7_models.pkl"
DEFAULT_SCALER_PATH = "models/nifty_v7_scaler.pkl"
DEFAULT_FCOLS_PATH = "models/feature_cols_v7.pkl"


class ModelBridge:
    """
    Loads the trained V7 model and runs inference on live bar data.
    Returns structured predictions for the Claude analyzer.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        fcols_path: str = DEFAULT_FCOLS_PATH,
        confidence_threshold: float = 0.30,
        min_edge: float = 0.05,
        skip_ceil: float = 0.60,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.fcols_path = fcols_path
        self.confidence_threshold = confidence_threshold
        self.min_edge = min_edge
        self.skip_ceil = skip_ceil

        self._models = None
        self._scaler = None
        self._fcols = None
        self._loaded = False

    def load(self) -> bool:
        """Load pre-trained models from disk."""
        try:
            mp = Path(self.model_path)
            if not mp.exists():
                log.warning(f"Model not found: {mp} — run train_model_v7.py first")
                return False

            self._models = joblib.load(self.model_path)
            self._scaler = joblib.load(self.scaler_path)
            self._fcols = joblib.load(self.fcols_path)
            self._loaded = True

            log.info(
                f"V7 model loaded | Models: {list(self._models.keys())} "
                f"| Features: {len(self._fcols)}"
            )
            return True
        except Exception as e:
            log.error(f"Model load failed: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def predict(self, bar_buffer: pd.DataFrame) -> dict:
        """
        Run V7 model inference on a buffer of 5-min bars.

        Args:
            bar_buffer: DataFrame with columns [open, high, low, close]
                       and DatetimeIndex. Needs at least 200 bars for
                       feature engineering.

        Returns:
            dict with:
                signal: "CALL" | "PUT" | "SKIP"
                signal_code: 0 | 1 | 2
                confidence: float (0-1)
                probabilities: {"call": float, "put": float, "skip": float}
                edge: float (how much winning direction beats the other)
                session: "MORNING" | "MIDDAY" | "AFTERNOON"
                features_summary: dict of key feature values
        """
        if not self._loaded:
            return self._empty_result("Model not loaded")

        if len(bar_buffer) < 200:
            return self._empty_result(f"Need 200+ bars, got {len(bar_buffer)}")

        try:
            # Feature engineering
            df_feat = engineer_features(bar_buffer)
            df_feat.dropna(inplace=True)

            if len(df_feat) == 0:
                return self._empty_result("No data after feature engineering")

            # Check feature alignment
            missing = [c for c in self._fcols if c not in df_feat.columns]
            if missing:
                log.warning(f"Missing features: {missing[:5]}... Using zeros")
                for col in missing:
                    df_feat[col] = 0.0

            # Extract last row
            X_live = df_feat[self._fcols].iloc[[-1]].values
            X_scaled = self._scaler.transform(X_live)

            # Ensemble predict
            probas = np.mean(
                [m.predict_proba(X_scaled) for m in self._models.values()],
                axis=0
            )
            proba = probas[0]
            call_p, put_p, skip_p = float(proba[0]), float(proba[1]), float(proba[2])

            # Signal logic (same as V6)
            if (call_p >= self.confidence_threshold
                    and call_p - put_p >= self.min_edge
                    and skip_p < self.skip_ceil):
                signal = "CALL"
                signal_code = 0
            elif (put_p >= self.confidence_threshold
                    and put_p - call_p >= self.min_edge
                    and skip_p < self.skip_ceil):
                signal = "PUT"
                signal_code = 1
            else:
                signal = "SKIP"
                signal_code = 2

            confidence = float(proba.max())
            edge = float(abs(call_p - put_p))

            # Session detection
            now = datetime.now()
            hour_min = now.strftime("%H:%M")
            if hour_min <= "10:30":
                session = "MORNING"
            elif hour_min <= "13:30":
                session = "MIDDAY"
            else:
                session = "AFTERNOON"

            # Key feature values for Claude context
            last_row = df_feat.iloc[-1]
            features_summary = {
                "rsi14": round(float(last_row.get("rsi14", 0)), 1),
                "macd_hist": round(float(last_row.get("mh", 0)) * 10000, 2),
                "bbp": round(float(last_row.get("bbp", 0.5)), 3),
                "atr_ratio": round(float(last_row.get("ar", 1)), 3),
                "vol_regime": int(last_row.get("vol_regime", 1)),
                "zscore_20": round(float(last_row.get("zscore_20", 0)), 2),
                "bos_bull": int(last_row.get("bos_bull", 0)),
                "bos_bear": int(last_row.get("bos_bear", 0)),
                "choch_bull": int(last_row.get("choch_bull", 0)),
                "choch_bear": int(last_row.get("choch_bear", 0)),
                "ob_bull": int(last_row.get("ob_bull", 0)),
                "ob_bear": int(last_row.get("ob_bear", 0)),
                "liq_sweep_high": int(last_row.get("liq_sweep_high", 0)),
                "liq_sweep_low": int(last_row.get("liq_sweep_low", 0)),
                "cmf_proxy": round(float(last_row.get("cmf_proxy", 0)), 3),
                "dow_primary": int(last_row.get("dow_primary", 0)),
            }

            log.info(
                f"ML Signal: {signal} | "
                f"CALL={call_p:.3f} PUT={put_p:.3f} SKIP={skip_p:.3f} | "
                f"Edge={edge:.3f} | Session={session}"
            )

            return {
                "signal": signal,
                "signal_code": signal_code,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "call": round(call_p, 4),
                    "put": round(put_p, 4),
                    "skip": round(skip_p, 4),
                },
                "edge": round(edge, 4),
                "session": session,
                "features_summary": features_summary,
                "model_version": "V7",
                "models_used": list(self._models.keys()),
                "features_count": len(self._fcols),
                "bars_used": len(bar_buffer),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            log.error(f"Prediction error: {e}", exc_info=True)
            return self._empty_result(str(e))

    def _empty_result(self, reason: str) -> dict:
        return {
            "signal": "SKIP",
            "signal_code": 2,
            "confidence": 0.0,
            "probabilities": {"call": 0.0, "put": 0.0, "skip": 1.0},
            "edge": 0.0,
            "session": "UNKNOWN",
            "features_summary": {},
            "model_version": "V7",
            "error": reason,
            "timestamp": datetime.now().isoformat(),
        }

    def get_feature_importance(self, top_n: int = 20) -> list:
        """Return top N features by importance."""
        if not self._loaded:
            return []

        # Try XGBoost first, then LightGBM, then HistGBM
        for name, model in self._models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[-top_n:][::-1]
                return [
                    {"feature": self._fcols[i], "importance": round(float(imp[i]), 4)}
                    for i in top_idx
                ]
        return []
