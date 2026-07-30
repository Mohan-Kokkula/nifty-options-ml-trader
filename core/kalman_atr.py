"""
kalman_atr.py — Adaptive ATR using Kalman filter.

Replaces fixed-window ATR(14) that lags 7-14 bars during regime shifts.
Kalman ATR adapts in 2-3 bars by combining prediction + measurement
with optimal weighting.

WHY: 13-day log analysis showed 4/10 losses occurred on volatility-shift
days where ATR(14) was stale → SL was sized for OLD volatility → got
stopped by NEW volatility.

USAGE:
    from core.kalman_atr import get_kalman_atr
    katr = get_kalman_atr()
    katr.update(true_range_5m)   # call on every new 5-min bar
    adaptive_atr = katr.x         # use in SL calculation
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KalmanATR:
    """
    1D Kalman filter for ATR estimation.

    Process model:  x_t = x_{t-1} + noise_q
    Observation:    z_t = x_t + noise_r

    State: x (estimated ATR), p (uncertainty)
    Tunable: q (process noise, default 0.5), r (measurement noise, default 5.0)

    Higher q = faster adaptation (more weight to new bars)
    Higher r = smoother (more weight to history)
    """
    def __init__(self, q: float = 0.5, r: float = 5.0, initial_atr: float = 30.0):
        self.x = float(initial_atr)
        self.p = 10.0      # initial uncertainty
        self.q = float(q)
        self.r = float(r)
        self._initialized = False
        self._n_updates = 0

    def update(self, true_range: float) -> float:
        """
        Process a new true_range observation. Returns updated ATR estimate.
        """
        if not isinstance(true_range, (int, float)) or true_range <= 0:
            return self.x   # invalid input, return current estimate

        # First-time bootstrap: use observation directly
        if not self._initialized:
            self.x = float(true_range)
            self._initialized = True
            self._n_updates = 1
            return self.x

        # Predict
        self.p += self.q

        # Update
        kalman_gain = self.p / (self.p + self.r)
        self.x = self.x + kalman_gain * (true_range - self.x)
        self.p = (1 - kalman_gain) * self.p

        self._n_updates += 1
        return self.x

    def get(self) -> float:
        return self.x

    def reset(self):
        self.x = 30.0
        self.p = 10.0
        self._initialized = False
        self._n_updates = 0


# ─── Singleton ──────────────────────────────────────────────────────────────
_INSTANCE: Optional[KalmanATR] = None


def get_kalman_atr() -> KalmanATR:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = KalmanATR()
    return _INSTANCE
