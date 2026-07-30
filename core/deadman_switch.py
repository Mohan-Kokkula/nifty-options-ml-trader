"""
deadman_switch.py — Independent monitoring daemon.

Catches silent failures BEFORE they cost trades. Specifically designed to
prevent the 2026-04-29 incident where a deadlock silently froze the
Kotak login for hours, costing a 200pt move.

Three checks fire at fixed times:

  09:11 IST → "Is Kotak logged in?"
              If NO → Telegram CRITICAL alert + try emergency login

  09:30 IST → "Has the pilot fired any cycle since 09:15?"
              If NO → Telegram CRITICAL alert (pilot loop is dead)

  Every 5min during market hours →
              "Have I heard a heartbeat from the pilot in last 6 minutes?"
              If NO → Telegram alert (pilot stalled)

Pilot writes its heartbeat to a tiny file every cycle:
    data/pilot_heartbeat.txt  (epoch timestamp + cycle count)

This module reads that file. Independent thread → can't be deadlocked
by the pilot's own bugs.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


HEARTBEAT_FILE = Path("data/pilot_heartbeat.txt")
PILOT_STALE_SEC = 6 * 60   # 6 min without heartbeat = stale (cycles are 5min)


def _setup_log_file():
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True, parents=True)
        log_path = log_dir / "deadman.log"
        if any(
            isinstance(h, RotatingFileHandler) and
            getattr(h, "baseFilename", "").endswith("deadman.log")
            for h in logger.handlers
        ):
            return
        h = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024,
                                backupCount=5, encoding="utf-8")
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(h)
        logger.propagate = False
    except Exception as e:
        logger.warning(f"deadman log file setup failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Heartbeat write/read
# ──────────────────────────────────────────────────────────────────────

def write_heartbeat(cycle: int = 0, status: str = "ok") -> None:
    """Pilot calls this every cycle. Stamps current epoch + cycle count."""
    try:
        HEARTBEAT_FILE.parent.mkdir(exist_ok=True, parents=True)
        HEARTBEAT_FILE.write_text(f"{int(time.time())},{cycle},{status}")
    except Exception as e:
        logger.debug(f"heartbeat write failed: {e}")


def read_heartbeat() -> Optional[dict]:
    try:
        if not HEARTBEAT_FILE.exists():
            return None
        parts = HEARTBEAT_FILE.read_text().strip().split(",")
        if len(parts) >= 3:
            return {"ts": int(parts[0]), "cycle": int(parts[1]), "status": parts[2]}
    except Exception as e:
        logger.debug(f"heartbeat read failed: {e}")
    return None


# ──────────────────────────────────────────────────────────────────────
# Telegram alert helper
# ──────────────────────────────────────────────────────────────────────

def _alert(notifier, text: str, level: str = "WARN") -> None:
    """Send urgent Telegram alert."""
    prefix = "🚨🚨🚨 CRITICAL" if level == "CRITICAL" else "⚠️"
    msg = f"{prefix}\n\n{text}\n\n⏰ {datetime.now().strftime('%H:%M:%S')} IST"
    logger.warning(f"DEADMAN ALERT [{level}]: {text}")
    try:
        if notifier and hasattr(notifier, "telegram") and notifier.telegram and getattr(notifier.telegram, "enabled", False):
            notifier.telegram.send_message(msg)
    except Exception as e:
        logger.error(f"Telegram alert send failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Watchdog checks
# ──────────────────────────────────────────────────────────────────────

def _check_kotak_login(client, notifier) -> bool:
    """At 09:11 — verify Kotak is logged in."""
    try:
        is_valid = bool(getattr(client, "_session_valid", False))
        if is_valid:
            logger.info("✅ 09:11 check: Kotak login confirmed")
            return True

        # Not logged in! Try emergency login
        logger.error("❌ 09:11 check: Kotak NOT logged in — attempting emergency login")
        _alert(notifier,
               "Kotak NOT logged in at 09:11 IST!\n"
               "Attempting emergency login. Check kotak_login.log for details.",
               level="CRITICAL")
        try:
            if hasattr(client, "ensure_logged_in"):
                # Watchdog the call — 30s max
                result = {"ok": False, "done": False}
                def _try_login():
                    try:
                        result["ok"] = client.ensure_logged_in()
                    finally:
                        result["done"] = True
                t = threading.Thread(target=_try_login, daemon=True)
                t.start()
                t.join(timeout=30)
                if not result["done"]:
                    _alert(notifier,
                           "Emergency login HUNG (30s timeout). Bot is broken.\n"
                           "MANUAL RESTART REQUIRED: sudo systemctl restart openclaw-v9-kotak",
                           level="CRITICAL")
                    return False
                if result["ok"]:
                    _alert(notifier, "Emergency login succeeded ✅", level="WARN")
                    return True
                else:
                    _alert(notifier, "Emergency login returned False — check Kotak portal",
                           level="CRITICAL")
        except Exception as e:
            _alert(notifier, f"Emergency login raised: {e}", level="CRITICAL")
        return False
    except Exception as e:
        logger.error(f"_check_kotak_login error: {e}")
        return False


def _check_pilot_alive(notifier, since_when: str) -> bool:
    """Verify pilot has emitted a heartbeat recently."""
    try:
        hb = read_heartbeat()
        if hb is None:
            _alert(notifier,
                   f"Pilot heartbeat MISSING ({since_when}).\n"
                   f"Pilot loop is dead — no trades will fire.\n"
                   f"MANUAL ACTION: sudo systemctl restart openclaw-v9-kotak",
                   level="CRITICAL")
            return False

        age = time.time() - hb["ts"]
        if age > PILOT_STALE_SEC:
            _alert(notifier,
                   f"Pilot heartbeat STALE — last beat {age/60:.1f} min ago "
                   f"(cycle #{hb['cycle']}, status={hb['status']}).\n"
                   f"Pilot may be hung. MANUAL ACTION: "
                   f"sudo systemctl restart openclaw-v9-kotak",
                   level="CRITICAL")
            return False

        logger.info(
            f"✅ Pilot heartbeat OK: cycle={hb['cycle']} age={age:.0f}s status={hb['status']}"
        )
        return True
    except Exception as e:
        logger.error(f"_check_pilot_alive error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Main scheduler
# ──────────────────────────────────────────────────────────────────────

_NSE_HOLIDAYS: set = {
    # 2026 NSE holidays — keep in sync with claude_pilot.NSE_HOLIDAYS
    "2026-01-26",  # Republic Day
    "2026-02-19",  # Mahashivratri
    "2026-03-03",  # Holi
    "2026-03-31",  # Id-Ul-Fitr
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id (Eid ul-Adha) — NSE confirmed holiday
    "2026-08-15",  # Independence Day
    "2026-08-26",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-21",  # Diwali – Laxmi Pujan
    "2026-11-04",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}


def _is_market_day() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    if now.strftime("%Y-%m-%d") in _NSE_HOLIDAYS:
        return False
    return True


def _seconds_until(target_h: int, target_m: int) -> float:
    now = datetime.now()
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _scheduler_loop(client, notifier):
    logger.info("🐕 Deadman scheduler started")

    # Track if we've already done today's checks (avoid double-firing on restart)
    state = {"login_check_done": "", "pilot_check_done": ""}

    while True:
        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # 09:11 IST — Kotak login check
            if (_is_market_day() and now.hour == 9 and 11 <= now.minute <= 14
                    and state["login_check_done"] != today_str):
                logger.info("⏰ 09:11 trigger — checking Kotak login")
                _check_kotak_login(client, notifier)
                state["login_check_done"] = today_str

            # 09:30 IST — Pilot alive check
            if (_is_market_day() and now.hour == 9 and 30 <= now.minute <= 33
                    and state["pilot_check_done"] != today_str):
                logger.info("⏰ 09:30 trigger — checking pilot heartbeat")
                _check_pilot_alive(notifier, since_when="since 09:15 market open")
                state["pilot_check_done"] = today_str

            # Continuous heartbeat check (every cycle, but only during market)
            if _is_market_day() and (
                (now.hour == 9 and now.minute >= 30) or
                (10 <= now.hour <= 14) or
                (now.hour == 15 and now.minute <= 14)
            ):
                hb = read_heartbeat()
                if hb is not None:
                    age = time.time() - hb["ts"]
                    if age > PILOT_STALE_SEC:
                        # Throttle: only alert once per 5min
                        cache_key = f"stale_{int(time.time() / 300)}"
                        if state.get("last_stale_alert") != cache_key:
                            state["last_stale_alert"] = cache_key
                            _alert(notifier,
                                   f"Pilot heartbeat STALE for {age/60:.1f} min "
                                   f"(last cycle #{hb['cycle']}). Pilot likely hung.",
                                   level="CRITICAL")

            # Sleep 60s between checks
            time.sleep(60)
        except Exception as e:
            logger.error(f"deadman scheduler error: {e}")
            time.sleep(60)


def start_deadman(client, notifier=None) -> threading.Thread:
    """Start the deadman switch daemon. Call from main.py."""
    _setup_log_file()
    t = threading.Thread(
        target=_scheduler_loop,
        args=(client, notifier),
        daemon=True,
        name="DeadmanSwitch",
    )
    t.start()
    logger.info("🐕 Deadman switch armed: 09:11 login check + 09:30 pilot check + continuous heartbeat")
    return t
