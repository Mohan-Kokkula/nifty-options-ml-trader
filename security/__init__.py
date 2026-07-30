"""
Security module — audit logging, rate limiting, and withdrawal guard.
Every trade action passes through this layer before execution.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, date
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class WithdrawalGuard:
    """
    Hard block on any withdrawal or fund-transfer operations.
    This class inspects every outgoing API call and rejects anything
    that could move money out of the trading account.
    """

    BLOCKED_ENDPOINTS = [
        "/funds/withdrawal",
        "/funds/transfer",
        "/withdraw",
        "/payout",
        "/bank_transfer",
    ]

    BLOCKED_KEYWORDS = [
        "withdraw",
        "payout",
        "fund_transfer",
        "bank_transfer",
    ]

    @classmethod
    def check(cls, endpoint: str, payload: dict) -> bool:
        """
        Returns True if the request is SAFE, False if it's a blocked withdrawal.
        """
        endpoint_lower = endpoint.lower()

        # Block known withdrawal endpoints
        for blocked in cls.BLOCKED_ENDPOINTS:
            if blocked in endpoint_lower:
                logger.critical(
                    f"🚫 WITHDRAWAL BLOCKED — endpoint: {endpoint}"
                )
                return False

        # Block suspicious keywords in payload
        payload_str = str(payload).lower()
        for keyword in cls.BLOCKED_KEYWORDS:
            if keyword in payload_str:
                logger.critical(
                    f"🚫 WITHDRAWAL BLOCKED — keyword '{keyword}' in payload"
                )
                return False

        return True


class RateLimiter:
    """
    Token-bucket rate limiter to stay within Kotak Neo's 10 req/sec limit.
    Default: 8/sec to leave headroom.
    """

    def __init__(self, max_per_sec: int = 8):
        self.max_per_sec = max_per_sec
        self.tokens = max_per_sec
        self.last_refill = time.monotonic()
        self._lock = Lock()

    def acquire(self) -> bool:
        """Returns True if request is allowed, False if rate-limited."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_per_sec, self.tokens + elapsed * self.max_per_sec
            )
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait_and_acquire(self):
        """Block until a token is available."""
        while not self.acquire():
            time.sleep(0.05)


class RiskManager:
    """
    Enforces per-trade and per-day risk limits.
    Tracks daily P&L and open position count.
    """

    def __init__(
        self,
        max_daily_loss: float,
        max_loss_per_trade: float,
        max_open_positions: int,
        state_file: str = "data/risk_state.json",
        drawdown_pause_pct: float = 0.70,
        consecutive_loss_pause_threshold: int = 4,
        pause_cooldown_min: int = 60,
        notifier=None,
    ):
        self.max_daily_loss = max_daily_loss
        self.max_loss_per_trade = max_loss_per_trade
        self.max_open_positions = max_open_positions
        self.daily_pnl: float = 0.0
        self.open_positions: int = 0
        self._today: date = date.today()
        self._lock = Lock()
        self._state_file = Path(state_file)
        # ── Automatic strategy health (pause on excess drawdown/streak) ──
        self.drawdown_pause_pct = drawdown_pause_pct
        self.consecutive_loss_pause_threshold = consecutive_loss_pause_threshold
        self.pause_cooldown_min = pause_cooldown_min
        self.notifier = notifier
        self.consecutive_losses: int = 0
        self.paused_until: float = 0.0   # epoch seconds; 0 = not paused
        self.pause_reason: str = ""
        # ── Operational-safety fix #6 (2026-07-21): fail-safe persistence ──
        self._persist_failures: int = 0
        self._persistence_locked_out: bool = False
        self._load_state()

    def set_notifier(self, notifier):
        """Deferred notifier injection (mirrors KotakNeoClient.set_notifier) —
        RiskManager is constructed before the NotificationManager exists in
        main.py's bootstrap order."""
        self.notifier = notifier

    def _load_state(self):
        """Restore daily_pnl/open_positions from a same-day restart so a mid-
        session process restart doesn't silently zero the real risk budget.

        Independent-verification fix #7 (2026-07-21) — design choice A:
        the persistence-failure lockout (_persistence_locked_out /
        _persist_failures) is restored HERE, before the same-day date
        check below, and is deliberately NOT subject to it. It is an
        infrastructure-health signal (disk full, permissions broken), not
        a daily trading counter like daily_pnl/consecutive_losses — a new
        calendar day does not mean a broken disk fixed itself, so unlike
        those fields this one must survive across a day boundary too.
        This makes a bare process restart (with no other action taken)
        UNABLE to silently clear a fail-safe lockout, matching the
        original spec's "requires manual recovery": recovery now means
        actually editing/deleting risk_state.json's persistence_locked_out
        field (or fixing the underlying I/O issue AND doing that), not
        just restarting the process.
        """
        try:
            if not self._state_file.exists():
                return
            state = json.loads(self._state_file.read_text())

            self._persistence_locked_out = bool(state.get("persistence_locked_out", False))
            self._persist_failures = int(state.get("persist_failures", 0))
            if self._persistence_locked_out:
                logger.critical(
                    "🚨 Risk state persistence lockout RESTORED from disk — "
                    "new trade entries remain DISABLED pending manual "
                    "recovery (edit/delete risk_state.json's "
                    "persistence_locked_out field once the underlying issue "
                    "is actually fixed)."
                )

            if state.get("date") != self._today.isoformat():
                logger.debug("Risk state from different day — not restoring daily counters")
                return
            self.daily_pnl = float(state.get("daily_pnl", 0.0))
            self.open_positions = int(state.get("open_positions", 0))
            self.consecutive_losses = int(state.get("consecutive_losses", 0))
            self.paused_until = float(state.get("paused_until", 0.0))
            self.pause_reason = str(state.get("pause_reason", ""))
            logger.info(
                f"📊 Risk state restored: daily_pnl=₹{self.daily_pnl:.2f} "
                f"open_positions={self.open_positions} "
                f"consecutive_losses={self.consecutive_losses}"
                + (f" | PAUSED until {datetime.fromtimestamp(self.paused_until)} ({self.pause_reason})"
                   if self.paused_until and time.time() < self.paused_until else "")
            )
        except Exception as e:
            logger.debug(f"Risk state load failed: {e}")

    # Operational-safety fix #6: this many CONSECUTIVE save failures counts
    # as "persistent" (vs. one transient I/O blip) before the fail-safe
    # lockout engages.
    _PERSIST_FAILURE_LOCKOUT_THRESHOLD = 3

    def _save_state(self):
        """
        Operational-safety fix #6 (2026-07-21): atomic write (temp file +
        fsync + os.replace) so a crash or power loss mid-write can never
        leave risk_state.json truncated or corrupt — the file on disk is
        always either the old complete state or the new complete state,
        never a partial one.

        On PERSISTENT write failure (_PERSIST_FAILURE_LOCKOUT_THRESHOLD
        consecutive failures — a single transient blip does not trip this):
        new trade entries are disabled via _persistence_locked_out (checked
        in can_open_trade()) and a Telegram alert is sent. This lockout does
        NOT auto-clear on a later successful write, AND (independent-
        verification fix #7, 2026-07-21) it is now included in the
        persisted state below, so it also does NOT clear on a bare process
        restart — see _load_state() for why. Manual recovery now means
        actually editing/deleting risk_state.json (or otherwise fixing the
        underlying I/O issue), not just restarting the process. Unlike the
        self-healing strategy-health pause above, nothing here auto-clears
        this lockout.
        """
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "date": self._today.isoformat(),
                "daily_pnl": self.daily_pnl,
                "open_positions": self.open_positions,
                "consecutive_losses": self.consecutive_losses,
                "paused_until": self.paused_until,
                "pause_reason": self.pause_reason,
                "persistence_locked_out": self._persistence_locked_out,
                "persist_failures": self._persist_failures,
            }
            tmp = self._state_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                f.write(json.dumps(state))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._state_file)
            self._persist_failures = 0   # reset the streak on a real success
        except Exception as e:
            logger.error(f"Risk state save failed: {e}")
            self._persist_failures += 1
            if (not self._persistence_locked_out
                    and self._persist_failures >= self._PERSIST_FAILURE_LOCKOUT_THRESHOLD):
                self._persistence_locked_out = True
                logger.critical(
                    f"🚨 RISK STATE PERSISTENCE FAILED {self._persist_failures}x "
                    f"consecutively ({e}) — new trade entries DISABLED. "
                    f"Requires MANUAL recovery."
                )
                if self.notifier:
                    try:
                        self.notifier.notify(
                            subject="🚨 Risk State Persistence FAILED",
                            message=(
                                f"risk_state.json could not be written "
                                f"{self._persist_failures} times in a row "
                                f"({e}). New trade entries are now DISABLED "
                                f"as a fail-safe (existing positions keep "
                                f"being monitored/closed normally). This "
                                f"requires MANUAL recovery — check disk "
                                f"space/permissions on {self._state_file}, "
                                f"then restart the bot."
                            ),
                        )
                    except Exception:
                        pass

    def can_open_trade(self) -> tuple[bool, str]:
        """Check if a new trade is allowed under risk limits."""
        with self._lock:
            self._reset_if_new_day()

            # Operational-safety fix #6: persistent risk-state write failure
            # fail-safe. Does not auto-clear — requires manual recovery.
            if self._persistence_locked_out:
                msg = "Risk-state persistence failed repeatedly — entries disabled pending manual recovery"
                logger.critical(f"🚨 RISK BLOCK: {msg}")
                return False, msg

            # Strategy-health pause — self-resolving once the cooldown
            # elapses (no explicit "resume" action needed, same style as
            # _reset_if_new_day's own self-healing check below).
            if self.paused_until and time.time() < self.paused_until:
                resume_at = datetime.fromtimestamp(self.paused_until).strftime("%H:%M:%S")
                msg = f"Strategy paused ({self.pause_reason}) until {resume_at}"
                logger.warning(f"⚠️ RISK BLOCK: {msg}")
                return False, msg

            if self.open_positions >= self.max_open_positions:
                msg = (
                    f"Max open positions reached ({self.max_open_positions})"
                )
                logger.warning(f"⚠️ RISK BLOCK: {msg}")
                return False, msg

            # 2026-05-07 BUG FIX: was using abs(daily_pnl) which blocked
            # PROFITS too. May 6 won ₹5,262 then risk cap fired and blocked
            # subsequent winners. Risk cap should ONLY trigger on losses.
            if self.daily_pnl <= -self.max_daily_loss:
                msg = (
                    f"Daily loss limit reached (₹{self.daily_pnl:.0f} / "
                    f"₹-{self.max_daily_loss:.0f})"
                )
                logger.warning(f"⚠️ RISK BLOCK: {msg}")
                return False, msg

            return True, "OK"

    def record_trade_open(self):
        with self._lock:
            self.open_positions += 1
            self._save_state()

    def record_trade_close(self, pnl: float):
        with self._lock:
            self.open_positions = max(0, self.open_positions - 1)
            self.daily_pnl += pnl
            logger.info(
                f"📊 Trade closed — P&L: ₹{pnl:.2f} | "
                f"Daily P&L: ₹{self.daily_pnl:.2f} | "
                f"Open positions: {self.open_positions}"
            )
            # 2026-05-05: Early-warning thresholds — log loud at 50% and 80%
            if self.daily_pnl < 0:
                pct = abs(self.daily_pnl) / max(self.max_daily_loss, 1) * 100
                if pct >= 80 and not getattr(self, "_warned_80", False):
                    self._warned_80 = True
                    logger.warning(
                        f"⚠️  DAILY LOSS at 80% of cap "
                        f"(₹{self.daily_pnl:.0f} / ₹{self.max_daily_loss:.0f}) — "
                        f"only ₹{self.max_daily_loss + self.daily_pnl:.0f} left"
                    )
                elif pct >= 50 and not getattr(self, "_warned_50", False):
                    self._warned_50 = True
                    logger.warning(
                        f"⚠️  DAILY LOSS at 50% of cap "
                        f"(₹{self.daily_pnl:.0f} / ₹{self.max_daily_loss:.0f})"
                    )

            # Consecutive-loss streak — canonical tracker for auto-pause
            # (mirrors the loss-streak logic already duplicated in
            # claude_pilot.py for size-halving; this is the risk-gating half).
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            self._check_health_pause()
            self._save_state()

    def _check_health_pause(self):
        """
        Automatic strategy health: pause new-trade entry when either the
        daily-loss drawdown crosses drawdown_pause_pct of MAX_DAILY_LOSS
        (tighter than the hard 100% stop in can_open_trade) OR consecutive
        losses reach consecutive_loss_pause_threshold. Auto-resumes once
        pause_cooldown_min elapses — no explicit "resume" action needed,
        can_open_trade() simply stops blocking once time.time() passes
        paused_until. Must be called with self._lock already held.
        """
        if self.paused_until and time.time() < self.paused_until:
            return  # already paused, nothing new to trigger

        drawdown_hit = (
            self.max_daily_loss > 0
            and self.daily_pnl <= -(self.drawdown_pause_pct * self.max_daily_loss)
        )
        streak_hit = self.consecutive_losses >= self.consecutive_loss_pause_threshold

        if not (drawdown_hit or streak_hit):
            return

        reason = (
            f"drawdown {abs(self.daily_pnl):.0f}/{self.max_daily_loss:.0f} "
            f"({self.drawdown_pause_pct:.0%} threshold)"
            if drawdown_hit else
            f"{self.consecutive_losses} consecutive losses"
        )
        self.paused_until = time.time() + self.pause_cooldown_min * 60
        self.pause_reason = reason
        resume_at = datetime.fromtimestamp(self.paused_until).strftime("%H:%M:%S")
        logger.critical(
            f"🛑 STRATEGY AUTO-PAUSED: {reason} — new entries blocked until {resume_at} "
            f"({self.pause_cooldown_min}min cooldown)"
        )
        if self.notifier:
            try:
                self.notifier.notify(
                    subject="Strategy Auto-Paused",
                    message=(
                        f"🛑 Strategy paused: {reason}\n"
                        f"New trade entry blocked until {resume_at} "
                        f"({self.pause_cooldown_min}min cooldown). "
                        f"Existing positions continue to be monitored."
                    ),
                )
            except Exception as e:
                logger.debug(f"Auto-pause notification failed: {e}")

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._today:
            logger.info("📅 New trading day — resetting daily P&L tracker")
            self.daily_pnl = 0.0
            self.open_positions = 0
            self._today = today
            # Reset warning flags
            self._warned_50 = False
            self._warned_80 = False
            # New day = fresh start for the strategy-health pause too —
            # a drawdown/streak from yesterday shouldn't block today.
            self.consecutive_losses = 0
            self.paused_until = 0.0
            self.pause_reason = ""
            self._save_state()


class AuditLogger:
    """
    Append-only CSV audit trail for every trade action.
    Columns: timestamp, action, symbol, side, qty, price, order_id, status, details
    """

    def __init__(self, log_path: str, enabled: bool = True):
        self.enabled = enabled
        self.log_path = Path(log_path)
        self._lock = Lock()

        if self.enabled:
            self._ensure_header()

    def _ensure_header(self):
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "action",
                        "symbol",
                        "side",
                        "qty",
                        "price",
                        "order_id",
                        "status",
                        "details",
                    ]
                )

    def log(
        self,
        action: str,
        symbol: str = "",
        side: str = "",
        qty: int = 0,
        price: float = 0.0,
        order_id: str = "",
        status: str = "",
        details: str = "",
    ):
        if not self.enabled:
            return

        with self._lock:
            try:
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            datetime.now().isoformat(),
                            action,
                            symbol,
                            side,
                            qty,
                            f"{price:.2f}",
                            order_id,
                            status,
                            details,
                        ]
                    )
            except Exception as e:
                logger.error(f"Audit log write failed: {e}")
