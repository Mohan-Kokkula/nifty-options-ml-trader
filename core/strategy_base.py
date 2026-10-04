"""
strategy_base.py — Base class for all signal engines
=====================================================
All strategies return the same output format so the SignalRouter
can treat them interchangeably.

Output:  signal (0=CALL, 1=PUT, 2=SKIP),
         proba  [P(CALL), P(PUT), P(SKIP)],
         confidence (float 0-1),
         indicators (dict with strategy-specific details)
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StrategyEngine(ABC):
    """Base class for all independent signal engines."""

    OPEN_SETTLE = 920
    LUNCH_START = 1200
    LUNCH_END = 1330

    def __init__(self, name: str):
        self.name = name
        self._ready = False
        self._trades_today = 0

    def is_ready(self) -> bool:
        return self._ready

    def reset_daily(self):
        self._trades_today = 0

    def record_trade(self):
        self._trades_today += 1

    def _time_filter(self, current_hm: int) -> str | None:
        """Common time filters. Returns skip reason or None if OK."""
        if current_hm < self.OPEN_SETTLE:
            return "market_open_settle"
        if self.LUNCH_START <= current_hm <= self.LUNCH_END:
            return "lunch_chop_zone"
        return None

    @abstractmethod
    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000) -> tuple:
        """Generate signal.

        Returns:
            signal: 0=CALL, 1=PUT, 2=SKIP
            proba: np.array([P(CALL), P(PUT), P(SKIP)])
            confidence: float 0-1
            indicators: dict with strategy details
        """
        ...

    @abstractmethod
    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0) -> tuple:
        """Return (sl_pts, tp_pts) for this strategy."""
        ...

    def _skip(self, reason: str, indicators: dict) -> tuple:
        indicators["skip_reason"] = reason
        return 2, np.array([0.0, 0.0, 1.0]), 0.0, indicators

    def _vix_multiplier(self, vix: float) -> float:
        if vix >= 22:
            return 1.5
        elif vix >= 17:
            return 1.2
        return 1.0
