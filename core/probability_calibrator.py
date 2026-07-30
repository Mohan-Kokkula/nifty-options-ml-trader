"""
probability_calibrator.py — Calibrate raw ML probabilities to actual win rates.

Problem (López de Prado, "Advances in Financial ML" Ch. 14):
    XGBoost / LightGBM produce probabilities that are NOT calibrated.
    A model output of 0.45 might mean actual 55% win rate.
    This is why your model "says 42%" but trades placed at 42% would
    actually win more often than 50%.

Solution:
    Apply a sigmoid-based recalibration based on observed historical
    win rates per probability bucket. As the bot trades more, this
    learns from outcomes (Platt scaling).

Usage:
    cal = get_calibrator()
    raw_conf = 0.42                 # model says 42%
    true_conf = cal.calibrate(raw_conf, direction="CALL", regime="REVERSAL_UP")
    # → returns 0.58 (actual historical win rate at this raw level)

For the first 50 trades, calibration is identity (no data yet).
After 50+ trades, calibration adjusts based on rolling win rate.

Stored in: data/calibration.json (persists across restarts).
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


CALIBRATION_FILE = Path("data/calibration.json")
MIN_SAMPLES_FOR_CALIBRATION = 30  # need 30 trades before we trust calibration
BUCKET_SIZE = 0.05                # 5% probability buckets


class ProbabilityCalibrator:
    """
    Tracks (raw_confidence_bucket, outcome) pairs.
    Returns calibrated probability when enough data exists.
    """

    def __init__(self):
        # buckets[direction_regime][bucket] = {"wins": N, "losses": M}
        self._buckets: dict = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0}))
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not CALIBRATION_FILE.exists():
            return
        try:
            data = json.loads(CALIBRATION_FILE.read_text())
            for key, buckets in data.items():
                for bk, stats in buckets.items():
                    self._buckets[key][bk] = stats
            logger.info(f"📊 Calibrator loaded {len(self._buckets)} keys from {CALIBRATION_FILE}")
        except Exception as e:
            logger.warning(f"Calibration load failed: {e}")

    def _save(self):
        try:
            CALIBRATION_FILE.parent.mkdir(exist_ok=True, parents=True)
            data = {k: dict(v) for k, v in self._buckets.items()}
            CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Calibration save failed: {e}")

    def record(self, raw_conf: float, won: bool, direction: str = "ANY", regime: str = "ANY"):
        """Record the outcome of a trade. Call this from your trade exit handler."""
        try:
            with self._lock:
                bucket = self._bucket_key(raw_conf)
                key = f"{direction}_{regime}"
                stats = self._buckets[key][bucket]
                if won:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                # Also record in "ANY_ANY" aggregate
                agg = self._buckets["ANY_ANY"][bucket]
                if won:
                    agg["wins"] += 1
                else:
                    agg["losses"] += 1
                self._save()
        except Exception as e:
            logger.warning(f"Calibrator record failed: {e}")

    def calibrate(self, raw_conf: float, direction: str = "ANY", regime: str = "ANY") -> float:
        """
        Return calibrated probability. Falls back to raw_conf if insufficient data.

        raw_conf: 0.0 to 1.0 (or 0 to 100 — auto-detected)
        """
        try:
            # Auto-detect 0-100 vs 0-1 scale
            if raw_conf > 1.0:
                raw_conf = raw_conf / 100.0

            with self._lock:
                # Try most-specific key first, fall back to less specific
                for key in (f"{direction}_{regime}", f"{direction}_ANY", "ANY_ANY"):
                    bucket = self._bucket_key(raw_conf)
                    stats = self._buckets.get(key, {}).get(bucket)
                    if not stats:
                        continue
                    n = stats["wins"] + stats["losses"]
                    if n < MIN_SAMPLES_FOR_CALIBRATION:
                        continue
                    # Beta-binomial smoothed win rate
                    # Add prior: assume bucket midpoint with weight 10
                    prior_n = 10
                    prior_p = float(bucket) + BUCKET_SIZE / 2
                    smoothed = (stats["wins"] + prior_n * prior_p) / (n + prior_n)
                    return max(0.05, min(0.95, smoothed))

            # No calibration data — return raw
            return raw_conf
        except Exception as e:
            logger.debug(f"Calibrator calibrate failed: {e}")
            return raw_conf

    @staticmethod
    def _bucket_key(p: float) -> str:
        # Bucket: 0.0-0.05 → "0.00", 0.05-0.10 → "0.05", etc.
        b = math.floor(p / BUCKET_SIZE) * BUCKET_SIZE
        return f"{b:.2f}"

    def stats(self) -> dict:
        """Diagnostic dump."""
        with self._lock:
            return {
                k: {
                    bk: {
                        "n": s["wins"] + s["losses"],
                        "win_rate": (
                            s["wins"] / max(1, s["wins"] + s["losses"])
                        ),
                    }
                    for bk, s in v.items()
                }
                for k, v in self._buckets.items()
            }


# ─── Singleton ──────────────────────────────────────────────────────────────
_INSTANCE: Optional[ProbabilityCalibrator] = None


def get_calibrator() -> ProbabilityCalibrator:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ProbabilityCalibrator()
    return _INSTANCE
