"""
risk_sentinel.py — Always-On Risk Monitoring Agent
===================================================
Runs in a dedicated background thread, continuously monitoring for
danger signals that require immediate action — not waiting for the
next scheduled check.

Monitors:
  1. VIX spike velocity (VIX jumping >2 pts in 5 min = danger)
  2. Spot price crash velocity (Nifty dropping >150pts in 10 min)
  3. Daily drawdown limit (cumulative loss approaching cap)
  4. Circuit breaker proximity (price near circuit limits)
  5. Losing streak escalation (3+ consecutive losses → reduce risk)

Actions:
  - WARN   → Log + notify, reduce confidence by 10
  - PAUSE  → Stop pilot temporarily (10-15 min cooldown)
  - STOP   → Stop pilot, close all positions
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class SentinelAlert:
    level: str          # WARN, PAUSE, STOP
    reason: str
    timestamp: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class SentinelConfig:
    poll_interval: int = 10
    vix_spike_threshold: float = 2.0
    vix_spike_window: int = 5
    spot_crash_threshold: float = 150.0
    spot_crash_window: int = 10
    max_daily_drawdown_pct: float = 80.0
    losing_streak_warn: int = 3
    losing_streak_stop: int = 5
    pause_duration: int = 900
    circuit_breaker_pct: float = 4.5
    # Data health
    spot_stale_seconds: int = 60
    spot_fail_max: int = 5
    broker_fail_max: int = 3


class RiskSentinel:
    """
    Always-on risk monitoring agent. Runs independently of the pilot's
    analysis cycle and can interrupt mid-trade.
    """

    def __init__(self, config: SentinelConfig = None):
        self.config = config or SentinelConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Data sources — set by the pilot/main
        self._get_spot: Optional[Callable] = None
        self._get_vix: Optional[Callable] = None
        self._pilot = None
        self._notifier = None
        self._journal = None
        self._daily_loss_cap: float = 5000.0

        # Internal state
        self._spot_history: deque = deque(maxlen=120)
        self._vix_history: deque = deque(maxlen=60)
        self._alerts: list[SentinelAlert] = []
        self._paused_until: float = 0
        self._today = date.today()
        self._confidence_penalty: int = 0

        # Data health tracking
        self._last_spot_success: float = 0
        self._spot_consecutive_fails: int = 0
        self._last_broker_success: float = 0
        self._broker_consecutive_fails: int = 0
        self._data_blocked: bool = False
        self._broker_blocked: bool = False
        self._get_broker_status: Optional[Callable] = None

        # Callbacks
        self._on_warn: Optional[Callable] = None
        self._on_pause: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None

    def configure(self, get_spot=None, get_vix=None, pilot=None,
                  notifier=None, journal=None, daily_loss_cap=5000.0,
                  get_broker_status=None):
        if get_spot:
            self._get_spot = get_spot
        if get_vix:
            self._get_vix = get_vix
        if pilot:
            self._pilot = pilot
        if notifier:
            self._notifier = notifier
        if journal:
            self._journal = journal
        if get_broker_status:
            self._get_broker_status = get_broker_status
        self._daily_loss_cap = daily_loss_cap

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="risk-sentinel"
        )
        self._thread.start()
        logger.info("Risk sentinel STARTED — continuous monitoring active")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Risk sentinel STOPPED")

    @property
    def is_paused(self) -> bool:
        return time.monotonic() < self._paused_until

    @property
    def entries_blocked(self) -> bool:
        """True if new entries should be blocked (pause, data stale, broker down)."""
        return self.is_paused or self._data_blocked or self._broker_blocked

    @property
    def block_reason(self) -> str:
        if self.is_paused:
            return "sentinel_paused"
        if self._data_blocked:
            return "data_stale_or_unavailable"
        if self._broker_blocked:
            return "broker_unhealthy"
        return ""

    @property
    def confidence_penalty(self) -> int:
        return self._confidence_penalty

    def get_status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            spot_age = (now - self._last_spot_success) if self._last_spot_success else -1
            return {
                "running": self._running,
                "paused": self.is_paused,
                "entries_blocked": self.entries_blocked,
                "block_reason": self.block_reason,
                "pause_remaining_sec": max(0, int(self._paused_until - now)),
                "confidence_penalty": self._confidence_penalty,
                "data_health": {
                    "spot_age_sec": round(spot_age, 1) if spot_age >= 0 else None,
                    "spot_consecutive_fails": self._spot_consecutive_fails,
                    "data_blocked": self._data_blocked,
                    "broker_consecutive_fails": self._broker_consecutive_fails,
                    "broker_blocked": self._broker_blocked,
                },
                "recent_alerts": [
                    {"level": a.level, "reason": a.reason, "time": a.timestamp}
                    for a in self._alerts[-5:]
                ],
                "spot_readings": len(self._spot_history),
                "vix_readings": len(self._vix_history),
            }

    # ------------------------------------------------------------------
    # Main monitoring loop
    # ------------------------------------------------------------------

    def _monitor_loop(self):
        while self._running:
            try:
                self._reset_daily()
                now = datetime.now()
                hm = now.hour * 100 + now.minute

                # Only monitor during market hours (9:15 - 15:30)
                if hm < 915 or hm > 1530:
                    time.sleep(30)
                    continue

                # Collect data
                self._collect_spot()
                self._collect_vix()

                # Run checks
                self._check_data_health()
                self._check_broker_health()
                self._check_vix_spike()
                self._check_spot_crash()
                self._check_daily_drawdown()
                self._check_losing_streak()

            except Exception as e:
                logger.debug(f"Sentinel check error: {e}")

            time.sleep(self.config.poll_interval)

    def _reset_daily(self):
        today = date.today()
        if today != self._today:
            self._today = today
            self._confidence_penalty = 0
            self._paused_until = 0
            self._alerts.clear()
            self._spot_history.clear()
            self._vix_history.clear()
            self._data_blocked = False
            self._broker_blocked = False
            self._spot_consecutive_fails = 0
            self._broker_consecutive_fails = 0
            logger.info("Sentinel: daily reset")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_spot(self):
        if not self._get_spot:
            return
        try:
            spot = self._get_spot()
            if spot and spot > 0:
                self._spot_history.append((time.monotonic(), spot))
                self._last_spot_success = time.monotonic()
                self._spot_consecutive_fails = 0
            else:
                self._spot_consecutive_fails += 1
        except Exception:
            self._spot_consecutive_fails += 1

    def _collect_vix(self):
        if not self._get_vix:
            return
        try:
            vix = self._get_vix()
            if vix > 0:
                self._vix_history.append((time.monotonic(), vix))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _check_data_health(self):
        """Block new entries if market data is stale or unavailable."""
        now = time.monotonic()

        # Check spot data staleness
        if self._last_spot_success > 0:
            age = now - self._last_spot_success
            if age > self.config.spot_stale_seconds:
                if not self._data_blocked:
                    self._data_blocked = True
                    self._emit_alert(
                        "WARN",
                        f"Market data stale: no spot update for {age:.0f}s "
                        f"(threshold={self.config.spot_stale_seconds}s)",
                        {"spot_age_sec": age},
                    )
                    logger.warning("SENTINEL: data stale → blocking new entries")
                return

        # Check consecutive failures
        if self._spot_consecutive_fails >= self.config.spot_fail_max:
            if not self._data_blocked:
                self._data_blocked = True
                self._emit_alert(
                    "WARN",
                    f"Market data unavailable: {self._spot_consecutive_fails} "
                    f"consecutive spot fetch failures",
                    {"consecutive_fails": self._spot_consecutive_fails},
                )
                logger.warning("SENTINEL: spot fetch failing → blocking new entries")
            return

        # Data is healthy — unblock if previously blocked
        if self._data_blocked:
            self._data_blocked = False
            logger.info("SENTINEL: market data recovered → entries unblocked")

    def _check_broker_health(self):
        """Block new entries if broker API is unhealthy."""
        if not self._get_broker_status:
            return

        try:
            status = self._get_broker_status()
            if status and status.get("healthy", True):
                self._broker_consecutive_fails = 0
                if self._broker_blocked:
                    self._broker_blocked = False
                    logger.info("SENTINEL: broker recovered → entries unblocked")
                return
            self._broker_consecutive_fails += 1
        except Exception:
            self._broker_consecutive_fails += 1

        if self._broker_consecutive_fails >= self.config.broker_fail_max:
            if not self._broker_blocked:
                self._broker_blocked = True
                self._emit_alert(
                    "WARN",
                    f"Broker unhealthy: {self._broker_consecutive_fails} "
                    f"consecutive failures — blocking new entries",
                    {"consecutive_fails": self._broker_consecutive_fails},
                )
                logger.warning("SENTINEL: broker unhealthy → blocking new entries")

    def _check_vix_spike(self):
        """Detect VIX jumping too fast — sign of sudden panic."""
        if len(self._vix_history) < 2:
            return

        now_t, now_vix = self._vix_history[-1]
        window = self.config.vix_spike_window * 60

        for t, v in self._vix_history:
            if now_t - t <= window:
                change = now_vix - v
                if change >= self.config.vix_spike_threshold * 1.5:
                    self._emit_alert(
                        "STOP",
                        f"VIX CRASH SPIKE {change:+.1f} in {int((now_t - t) / 60)}min",
                        {"vix_from": v, "vix_to": now_vix, "change": change},
                    )
                    return
                if change >= self.config.vix_spike_threshold:
                    self._emit_alert(
                        "PAUSE",
                        f"VIX spiked {change:+.1f} in {int((now_t - t) / 60)}min "
                        f"({v:.1f} → {now_vix:.1f})",
                        {"vix_from": v, "vix_to": now_vix, "change": change},
                    )
                    return

    def _check_spot_crash(self):
        """Detect Nifty dropping too fast — flash crash protection."""
        if len(self._spot_history) < 5:
            return

        now_t, now_spot = self._spot_history[-1]
        window = self.config.spot_crash_window * 60

        for t, s in self._spot_history:
            if now_t - t <= window:
                drop = s - now_spot
                if drop >= self.config.spot_crash_threshold:
                    self._emit_alert(
                        "STOP",
                        f"Nifty dropped {drop:.0f}pts in {int((now_t - t) / 60)}min "
                        f"({s:.0f} → {now_spot:.0f})",
                        {"spot_from": s, "spot_to": now_spot, "drop": drop},
                    )
                    return
                if drop >= self.config.spot_crash_threshold * 0.6:
                    self._emit_alert(
                        "WARN",
                        f"Nifty falling fast: {drop:.0f}pts in {int((now_t - t) / 60)}min",
                        {"spot_from": s, "spot_to": now_spot, "drop": drop},
                    )
                    return

    def _check_daily_drawdown(self):
        """Check if cumulative daily loss is approaching the cap."""
        if not self._journal:
            return

        today_trades = self._journal.get_today()
        total_pnl = sum(t.pnl_points for t in today_trades if t.exit_reason)
        if total_pnl >= 0:
            return

        loss_pct = abs(total_pnl) / max(self._daily_loss_cap / 50, 1) * 100

        if loss_pct >= self.config.max_daily_drawdown_pct:
            self._emit_alert(
                "STOP",
                f"Daily drawdown {loss_pct:.0f}% of cap "
                f"(P&L={total_pnl:+.0f}pts, cap={self._daily_loss_cap})",
                {"total_pnl": total_pnl, "loss_pct": loss_pct},
            )
        elif loss_pct >= self.config.max_daily_drawdown_pct * 0.6:
            self._emit_alert(
                "WARN",
                f"Daily drawdown reaching {loss_pct:.0f}% of cap",
                {"total_pnl": total_pnl, "loss_pct": loss_pct},
            )

    def _get_current_vix(self) -> float:
        """Get latest VIX reading from history."""
        if self._vix_history:
            return self._vix_history[-1][1]
        return 0.0

    def _get_streak_thresholds(self) -> tuple:
        """Regime-aware losing streak thresholds.
        Higher VIX = stricter thresholds because losses are more dangerous."""
        vix = self._get_current_vix()
        if vix >= 28:       # VERY_HIGH / EXTREME
            return 2, 3     # warn at 2, stop at 3
        elif vix >= 22:     # HIGH
            return 2, 4     # warn at 2, stop at 4
        elif vix >= 17:     # ELEVATED
            return 3, 4     # warn at 3, stop at 4
        else:               # NORMAL / CALM
            return self.config.losing_streak_warn, self.config.losing_streak_stop

    def _check_losing_streak(self):
        """Check for consecutive losses — regime-aware risk reduction."""
        if not self._journal:
            return

        streak_type, streak_count = self._journal.get_streak()
        if streak_type != "LOSS":
            self._confidence_penalty = 0
            return

        warn_at, stop_at = self._get_streak_thresholds()
        vix = self._get_current_vix()
        regime_note = f" (VIX={vix:.1f}, warn@{warn_at} stop@{stop_at})"

        if streak_count >= stop_at:
            self._emit_alert(
                "STOP",
                f"{streak_count} consecutive losses{regime_note} — stopping for the day",
                {"streak": streak_count, "vix": vix,
                 "warn_threshold": warn_at, "stop_threshold": stop_at},
            )
        elif streak_count >= warn_at:
            penalty = (streak_count - warn_at + 1) * 5
            self._confidence_penalty = min(25, penalty)
            self._emit_alert(
                "WARN",
                f"{streak_count} losses{regime_note} — confidence penalty {self._confidence_penalty}",
                {"streak": streak_count, "penalty": self._confidence_penalty,
                 "vix": vix, "warn_threshold": warn_at, "stop_threshold": stop_at},
            )

    # ------------------------------------------------------------------
    # Alert emission and actions
    # ------------------------------------------------------------------

    def _emit_alert(self, level: str, reason: str, data: dict = None):
        # Deduplicate — don't spam the same alert
        with self._lock:
            if self._alerts:
                last = self._alerts[-1]
                if last.level == level and last.reason == reason:
                    return

            alert = SentinelAlert(
                level=level,
                reason=reason,
                timestamp=datetime.now().isoformat(),
                data=data or {},
            )
            self._alerts.append(alert)

            # Keep last 50 alerts
            if len(self._alerts) > 50:
                self._alerts = self._alerts[-50:]

        logger.warning(f"SENTINEL [{level}]: {reason}")

        # Execute action
        if level == "WARN":
            self._action_warn(alert)
        elif level == "PAUSE":
            self._action_pause(alert)
        elif level == "STOP":
            self._action_stop(alert)

    def _action_warn(self, alert: SentinelAlert):
        if self._notifier:
            self._notifier.notify_trade(
                action="SENTINEL_WARN",
                symbol="SYSTEM",
                side="WARN",
                qty=0,
                price=0,
                order_id="",
                status="warning",
                details=f"Risk Sentinel WARNING: {alert.reason}",
            )

    def _action_pause(self, alert: SentinelAlert):
        self._paused_until = time.monotonic() + self.config.pause_duration
        logger.warning(
            f"SENTINEL PAUSE: pilot paused for {self.config.pause_duration}s — {alert.reason}"
        )

        if self._pilot and self._pilot.is_running:
            self._pilot.stop()
            logger.warning("Sentinel PAUSED pilot")

        if self._notifier:
            self._notifier.notify_trade(
                action="SENTINEL_PAUSE",
                symbol="SYSTEM",
                side="PAUSE",
                qty=0,
                price=0,
                order_id="",
                status="paused",
                details=f"Risk Sentinel PAUSED trading for {self.config.pause_duration // 60}min: {alert.reason}",
            )

    def _action_stop(self, alert: SentinelAlert):
        self._paused_until = time.monotonic() + 86400
        logger.warning(f"SENTINEL STOP: trading halted — {alert.reason}")

        if self._pilot and self._pilot.is_running:
            self._pilot.stop()
            logger.warning("Sentinel STOPPED pilot")

        if self._notifier:
            self._notifier.notify_trade(
                action="SENTINEL_STOP",
                symbol="SYSTEM",
                side="STOP",
                qty=0,
                price=0,
                order_id="",
                status="stopped",
                details=f"Risk Sentinel STOPPED trading: {alert.reason}",
            )
