"""
Strategy Monitor — watches Nifty levels in real-time and triggers
trades when entry conditions are met.

Supports two concurrent setups:
  LONG  — Buy CE on dip to support with bounce confirmation
  SHORT — Buy PE on rally to resistance with rejection confirmation

Runs as a background thread, polls spot every N seconds,
and fires notifications + optional auto-execution.
"""

import logging
import time
import threading
from datetime import datetime, date, time as dtime
from typing import Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Setup definitions
# ------------------------------------------------------------------

@dataclass
class LevelSetup:
    """A trade setup watching a specific price level."""

    name: str                       # e.g. "LONG_BOUNCE", "SHORT_REJECTION"
    enabled: bool = True
    direction: str = "LONG"         # LONG or SHORT
    option_type: str = "CE"         # CE or PE
    strike_mode: str = "ATM"        # ATM, OTM_1, OTM_2, etc.

    # Level watching
    watch_zone_low: float = 0.0     # Price zone to watch (low end)
    watch_zone_high: float = 0.0    # Price zone to watch (high end)
    trigger_type: str = "BOUNCE"    # BOUNCE = enters zone then leaves upward
                                    # REJECTION = enters zone then leaves downward
                                    # TOUCH = just touches the zone

    # Confirmation
    confirm_candles: int = 2        # How many polls must confirm the move
    confirm_direction: str = "UP"   # UP = price moving up, DOWN = price moving down

    # Trade params
    qty: int = 50
    stop_loss: float = 0.0
    target: float = 0.0

    # Execution
    auto_execute: bool = False      # If True, places order automatically
    max_triggers: int = 1           # Max times this setup can trigger (prevents re-entry)

    # State (internal)
    triggered_count: int = 0
    in_zone: bool = False
    confirm_count: int = 0
    last_spot: float = 0.0
    triggered: bool = False


# ------------------------------------------------------------------
# Pre-built setups from today's analysis
# ------------------------------------------------------------------

def build_todays_setups() -> list[LevelSetup]:
    """
    Build the two setups from today's (17 Mar 2026) analysis.
    Call this at startup — levels should be refreshed daily.
    """
    return [
        LevelSetup(
            name="LONG_BOUNCE",
            direction="LONG",
            option_type="CE",
            strike_mode="ATM",
            watch_zone_low=23350,
            watch_zone_high=23450,
            trigger_type="BOUNCE",
            confirm_candles=2,
            confirm_direction="UP",
            qty=50,
            stop_loss=23300,
            target=23750,
            auto_execute=False,
            max_triggers=1,
        ),
        LevelSetup(
            name="SHORT_REJECTION",
            direction="SHORT",
            option_type="PE",
            strike_mode="OTM_1",
            watch_zone_low=23650,
            watch_zone_high=23800,
            trigger_type="REJECTION",
            confirm_candles=2,
            confirm_direction="DOWN",
            qty=50,
            stop_loss=23850,
            target=23100,
            auto_execute=False,
            max_triggers=1,
        ),
    ]


# ------------------------------------------------------------------
# Monitor engine
# ------------------------------------------------------------------

class StrategyMonitor:
    """
    Background thread that polls Nifty spot and checks setups.

    Flow per poll:
      1. Fetch spot
      2. For each setup:
         a. Check if spot is in the watch zone
         b. If it was in zone and now left in the confirm direction → trigger
         c. If triggered → notify + optionally execute
    """

    def __init__(
        self,
        trader,             # NiftyOptionsTrader instance
        notifier,           # NotificationManager instance
        poll_interval: int = 15,  # seconds between checks
    ):
        self.trader = trader
        self.notifier = notifier
        self.poll_interval = poll_interval
        self.setups: list[LevelSetup] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def add_setup(self, setup: LevelSetup):
        """Add a setup to monitor."""
        with self._lock:
            self.setups.append(setup)
            logger.info(
                f"📌 Setup added: {setup.name} — "
                f"Watch {setup.watch_zone_low}-{setup.watch_zone_high} "
                f"for {setup.trigger_type} → "
                f"{'AUTO' if setup.auto_execute else 'ALERT'} "
                f"{setup.direction} {setup.option_type} {setup.strike_mode}"
            )

    def load_setups(self, setups: list[LevelSetup]):
        """Load multiple setups."""
        for s in setups:
            self.add_setup(s)

    def start(self):
        """Start monitoring in a background thread."""
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="strategy-monitor",
        )
        self._thread.start()
        logger.info(
            f"🔍 Strategy monitor started — "
            f"{len(self.setups)} setups, polling every {self.poll_interval}s"
        )

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🔍 Strategy monitor stopped")

    @property
    def status(self) -> dict:
        """Get current monitor status."""
        with self._lock:
            return {
                "running": self._running,
                "poll_interval": self.poll_interval,
                "setups": [
                    {
                        "name": s.name,
                        "enabled": s.enabled,
                        "direction": s.direction,
                        "option_type": s.option_type,
                        "strike_mode": s.strike_mode,
                        "watch_zone": f"{s.watch_zone_low}-{s.watch_zone_high}",
                        "trigger_type": s.trigger_type,
                        "auto_execute": s.auto_execute,
                        "triggered": s.triggered,
                        "triggered_count": s.triggered_count,
                        "in_zone": s.in_zone,
                        "confirm_count": s.confirm_count,
                        "last_spot": s.last_spot,
                        "stop_loss": s.stop_loss,
                        "target": s.target,
                    }
                    for s in self.setups
                ],
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_market_hours(self) -> bool:
        """Check if we're within NSE trading hours (9:15 AM - 3:30 PM IST)."""
        now = datetime.now().time()
        market_open = dtime(9, 15)
        market_close = dtime(15, 30)
        return market_open <= now <= market_close

    def _run_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                if not self._is_market_hours():
                    time.sleep(30)
                    continue

                # Fetch spot
                try:
                    spot = self.trader.get_nifty_spot()
                except Exception as e:
                    logger.debug(f"Spot fetch failed: {e}")
                    time.sleep(self.poll_interval)
                    continue

                # Check each setup
                with self._lock:
                    for setup in self.setups:
                        if not setup.enabled:
                            continue
                        if setup.triggered_count >= setup.max_triggers:
                            continue

                        self._check_setup(setup, spot)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            time.sleep(self.poll_interval)

    def _check_setup(self, setup: LevelSetup, spot: float):
        """Check a single setup against current spot."""

        in_zone = setup.watch_zone_low <= spot <= setup.watch_zone_high
        prev_spot = setup.last_spot
        setup.last_spot = spot

        if prev_spot == 0:
            # First poll — just record
            setup.in_zone = in_zone
            return

        # Track zone entry/exit
        if in_zone and not setup.in_zone:
            # Just entered zone
            setup.in_zone = True
            setup.confirm_count = 0
            logger.info(
                f"👀 {setup.name}: Nifty entered watch zone "
                f"{setup.watch_zone_low}-{setup.watch_zone_high} "
                f"(spot: {spot:.2f})"
            )

        elif not in_zone and setup.in_zone:
            # Just left zone — check direction
            setup.in_zone = False

            if setup.trigger_type == "BOUNCE":
                # Entered zone and left upward
                if spot > setup.watch_zone_high:
                    setup.confirm_count += 1
                else:
                    setup.confirm_count = 0

            elif setup.trigger_type == "REJECTION":
                # Entered zone and left downward
                if spot < setup.watch_zone_low:
                    setup.confirm_count += 1
                else:
                    setup.confirm_count = 0

        elif not in_zone:
            # Outside zone — check continuing confirmation
            if setup.confirm_count > 0:
                if setup.confirm_direction == "UP" and spot > prev_spot:
                    setup.confirm_count += 1
                elif setup.confirm_direction == "DOWN" and spot < prev_spot:
                    setup.confirm_count += 1
                else:
                    # Direction reversed — reset
                    setup.confirm_count = 0

        elif in_zone and setup.trigger_type == "TOUCH":
            # Touch mode — trigger just by entering zone
            setup.confirm_count = setup.confirm_candles

        # Check if confirmed
        if setup.confirm_count >= setup.confirm_candles:
            self._trigger_setup(setup, spot)

    def _trigger_setup(self, setup: LevelSetup, spot: float):
        """Fire the setup — notify and optionally execute."""
        setup.triggered = True
        setup.triggered_count += 1
        setup.confirm_count = 0

        action = "BUY" if setup.direction == "LONG" else "BUY"  # buying options both ways
        details = (
            f"🎯 STRATEGY TRIGGER: {setup.name}\n"
            f"Nifty spot: ₹{spot:.2f}\n"
            f"Signal: {setup.trigger_type} at "
            f"{setup.watch_zone_low}-{setup.watch_zone_high}\n"
            f"Trade: BUY {setup.option_type} ({setup.strike_mode})\n"
            f"SL: {setup.stop_loss} | Target: {setup.target}\n"
            f"Auto-execute: {'YES' if setup.auto_execute else 'NO — manual confirmation needed'}"
        )

        logger.info(f"🎯 TRIGGERED: {setup.name} — {details}")

        # Notify
        self.notifier.notify_trade(
            action="STRATEGY_TRIGGER",
            symbol=f"NIFTY {setup.option_type}",
            side=action,
            qty=setup.qty,
            price=spot,
            order_id="",
            status="triggered",
            details=details,
        )

        # Auto-execute if enabled
        if setup.auto_execute:
            try:
                result = self.trader.smart_trade(
                    action=action,
                    option_type=setup.option_type,
                    strike_mode=setup.strike_mode,
                    qty=setup.qty,
                    price_type="MARKET",
                )
                logger.info(
                    f"🚀 Auto-executed: {setup.name} — {result}"
                )
            except Exception as e:
                logger.error(f"Auto-execution failed for {setup.name}: {e}")
                self.notifier.notify_trade(
                    action="EXECUTION_FAILED",
                    symbol=f"NIFTY {setup.option_type}",
                    side=action,
                    qty=setup.qty,
                    price=spot,
                    order_id="",
                    status="error",
                    details=f"Auto-execution failed: {e}",
                )
