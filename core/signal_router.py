"""
signal_router.py — Multi-Strategy Signal Router
================================================
Orchestrates 8 independent signal engines, manages conflict resolution,
and enforces risk budgets.

Rules:
  - One position at a time (NIFTY options, single underlying)
  - Same direction as existing position → HOLD (no action)
  - Opposite direction → CLOSE existing + OPEN new
  - Multiple strategies fire same bar → highest confidence wins
  - Each strategy has its own max-loss budget; total daily cap applies
"""

import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DIRECTION_MAP = {0: "CALL", 1: "PUT", 2: "SKIP"}


@dataclass
class ActivePosition:
    direction: str           # "CALL" or "PUT"
    strategy: str            # which engine opened it
    entry_time: str = ""
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0


@dataclass
class StrategyBudget:
    name: str
    max_trades_per_day: int = 3
    trades_today: int = 0
    pnl_today: float = 0.0
    max_daily_loss: float = -5000.0  # per-strategy daily loss cap


class SignalRouter:
    """Routes signals from multiple engines, enforces one-position-at-a-time."""

    TOTAL_MAX_DAILY_LOSS = -10000.0  # hard cap across all strategies

    def __init__(self, engines: dict, budgets: dict = None):
        """
        Args:
            engines: {"PSAR": psar_engine, "MACD": macd_engine, ...}
            budgets: {"PSAR": StrategyBudget(...), ...} — optional overrides
        """
        self.engines = engines
        self.budgets: dict[str, StrategyBudget] = {}
        for name in engines:
            if budgets and name in budgets:
                self.budgets[name] = budgets[name]
            else:
                self.budgets[name] = StrategyBudget(name=name)

        self.active_position: ActivePosition | None = None
        self._total_pnl_today = 0.0
        self._today_date = ""
        self._signal_log: list = []

    def reset_daily(self):
        """Reset all daily counters. Called at start of each trading day."""
        self._today_date = datetime.now().strftime("%Y-%m-%d")
        self._total_pnl_today = 0.0
        for b in self.budgets.values():
            b.trades_today = 0
            b.pnl_today = 0.0
        for eng in self.engines.values():
            if hasattr(eng, "reset_daily"):
                eng.reset_daily()
        self._signal_log = []
        logger.info(f"SignalRouter: daily reset for {self._today_date}")

    def record_trade_result(self, strategy_name: str, pnl: float):
        """Record a closed trade's PnL for budget tracking."""
        if strategy_name in self.budgets:
            self.budgets[strategy_name].pnl_today += pnl
        self._total_pnl_today += pnl

    def evaluate(self, df5, df15, df30, vix: float = 15.0,
                 current_hm: int = 1000, chain_data: dict = None) -> dict:
        """Run all engines and return the best actionable signal.

        Returns dict with:
            action: "OPEN" | "CLOSE_AND_OPEN" | "HOLD" | "SKIP"
            direction: "CALL" | "PUT" | "SKIP"
            strategy: engine name that produced the signal
            confidence: float
            sl: float (points)
            tp: float (points)
            close_reason: str (if action is CLOSE_AND_OPEN)
            all_signals: dict of all engine results
            indicators: dict from winning engine
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today_date:
            self.reset_daily()

        # ── Total daily loss check ──
        if self._total_pnl_today <= self.TOTAL_MAX_DAILY_LOSS:
            return {
                "action": "SKIP",
                "direction": "SKIP",
                "strategy": "ROUTER",
                "confidence": 0.0,
                "skip_reason": f"total_daily_loss_hit ({self._total_pnl_today:.0f})",
                "all_signals": {},
            }

        # ── Run all engines ──
        results = {}
        for name, engine in self.engines.items():
            try:
                if name == "OI":
                    sig, proba, conf, ind = engine.predict(
                        df5, df15, df30, vix=vix, current_hm=current_hm,
                        chain_data=chain_data
                    )
                else:
                    sig, proba, conf, ind = engine.predict(
                        df5, df15, df30, vix=vix, current_hm=current_hm
                    )
                results[name] = {
                    "signal": sig,
                    "direction": DIRECTION_MAP.get(sig, "SKIP"),
                    "proba": list(proba),
                    "confidence": conf,
                    "indicators": ind,
                }
            except Exception as e:
                logger.warning(f"Engine {name} failed: {e}")
                results[name] = {
                    "signal": 2,
                    "direction": "SKIP",
                    "proba": [0.0, 0.0, 1.0],
                    "confidence": 0.0,
                    "indicators": {"error": str(e)},
                }

        # ── Filter actionable signals (not SKIP) ──
        actionable = []
        for name, r in results.items():
            if r["signal"] == 2:
                continue
            budget = self.budgets.get(name)
            if budget:
                if budget.trades_today >= budget.max_trades_per_day:
                    logger.debug(f"{name}: max trades reached ({budget.trades_today})")
                    continue
                if budget.pnl_today <= budget.max_daily_loss:
                    logger.debug(f"{name}: daily loss cap hit ({budget.pnl_today:.0f})")
                    continue
            actionable.append((name, r))

        if not actionable:
            return {
                "action": "SKIP",
                "direction": "SKIP",
                "strategy": "ROUTER",
                "confidence": 0.0,
                "skip_reason": "no_actionable_signal",
                "all_signals": results,
            }

        # ── Pick highest confidence ──
        actionable.sort(key=lambda x: x[1]["confidence"], reverse=True)
        best_name, best_result = actionable[0]
        best_dir = best_result["direction"]

        # Get SL/TP from the winning engine
        engine = self.engines[best_name]
        sl, tp = engine.get_sl_tp(vix=vix)

        # ── Conflict resolution with active position ──
        if self.active_position is not None:
            if best_dir == self.active_position.direction:
                # Same direction → HOLD, don't stack
                return {
                    "action": "HOLD",
                    "direction": best_dir,
                    "strategy": best_name,
                    "confidence": best_result["confidence"],
                    "hold_reason": f"same_direction_as_{self.active_position.strategy}",
                    "all_signals": results,
                    "indicators": best_result["indicators"],
                }
            else:
                # Opposite direction → close existing + open new
                close_reason = (
                    f"{best_name}({best_dir}) vs "
                    f"{self.active_position.strategy}({self.active_position.direction})"
                )
                return {
                    "action": "CLOSE_AND_OPEN",
                    "direction": best_dir,
                    "strategy": best_name,
                    "confidence": best_result["confidence"],
                    "sl": sl,
                    "tp": tp,
                    "close_reason": close_reason,
                    "all_signals": results,
                    "indicators": best_result["indicators"],
                }

        # ── No active position → OPEN ──
        return {
            "action": "OPEN",
            "direction": best_dir,
            "strategy": best_name,
            "confidence": best_result["confidence"],
            "sl": sl,
            "tp": tp,
            "all_signals": results,
            "indicators": best_result["indicators"],
        }

    def on_position_opened(self, direction: str, strategy: str,
                           entry_price: float = 0.0, sl: float = 0.0,
                           tp: float = 0.0):
        """Call after a position is successfully opened."""
        self.active_position = ActivePosition(
            direction=direction,
            strategy=strategy,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            entry_price=entry_price,
            sl=sl,
            tp=tp,
        )
        if strategy in self.budgets:
            self.budgets[strategy].trades_today += 1
        logger.info(
            f"SignalRouter: position OPENED {direction} by {strategy} "
            f"@ {entry_price:.2f} SL={sl:.1f} TP={tp:.1f}"
        )

    def on_position_closed(self, pnl: float = 0.0):
        """Call after a position is closed (SL/TP/signal/manual)."""
        if self.active_position:
            strat = self.active_position.strategy
            self.record_trade_result(strat, pnl)
            logger.info(
                f"SignalRouter: position CLOSED ({self.active_position.direction} "
                f"by {strat}) PnL={pnl:+.0f} | "
                f"day_total={self._total_pnl_today:+.0f}"
            )
            self.active_position = None
        else:
            self._total_pnl_today += pnl

    def get_status(self) -> dict:
        """Return current router state for logging/display."""
        return {
            "date": self._today_date,
            "total_pnl": round(self._total_pnl_today, 2),
            "active_position": {
                "direction": self.active_position.direction,
                "strategy": self.active_position.strategy,
                "entry_time": self.active_position.entry_time,
            } if self.active_position else None,
            "budgets": {
                name: {
                    "trades": b.trades_today,
                    "max_trades": b.max_trades_per_day,
                    "pnl": round(b.pnl_today, 2),
                    "max_loss": b.max_daily_loss,
                }
                for name, b in self.budgets.items()
            },
            "engines": list(self.engines.keys()),
        }
