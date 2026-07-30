"""
kotak_login_scheduler.py — Dedicated daemon that calls Kotak Neo login
EXACTLY at 09:10 IST every weekday. Does NOT depend on the pilot loop
being healthy — runs in its own thread.

Why this exists:
  - Original design relied on the pilot loop firing ensure_logged_in()
    at the first cycle (09:15:00). If the pilot thread dies between
    bot start (04:00) and market open (09:15), Kotak login never fires
    and the trader is silently locked out for the entire session.
  - This scheduler is independent: dies separately, alerts loudly.

Behaviour:
  - Sleeps until 09:10:00 IST each weekday.
  - Calls client.ensure_logged_in().
  - On failure, retries every 60s up to 5 attempts.
  - Sends Telegram alert on success and on persistent failure.
  - Logs to logs/kotak_login.log (separate file).

Usage:
  from core.kotak_login_scheduler import schedule_daily_login
  schedule_daily_login(client, notifier=notification_manager)
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


def _setup_log_file():
    """Dedicated log file: logs/kotak_login.log"""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True, parents=True)
        log_path = log_dir / "kotak_login.log"
        if any(
            isinstance(h, RotatingFileHandler) and
            getattr(h, "baseFilename", "").endswith("kotak_login.log")
            for h in logger.handlers
        ):
            return
        handler = RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=7, encoding="utf-8"
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.propagate = False
    except Exception as e:
        # Fall back to default
        logger.warning(f"Log file setup failed: {e}")


def _login_with_retries(client, notifier=None,
                         max_attempts: int = 5,
                         retry_delay_sec: int = 60,
                         per_attempt_timeout_sec: int = 60) -> bool:
    """
    Call ensure_logged_in() with retries + watchdog timeout per attempt.
    Returns True on success.

    Watchdog: each login attempt runs in a separate thread with a hard
    timeout. If it hangs (Kotak network issue, deadlock, etc.) we kill
    the wait and retry rather than blocking forever.
    """
    for attempt in range(1, max_attempts + 1):
        logger.info(f"🔐 Login attempt {attempt}/{max_attempts} starting...")
        result_box = {"ok": False, "exc": None, "done": False}

        def _do_call():
            try:
                if hasattr(client, "ensure_logged_in"):
                    logger.info(f"  → calling ensure_logged_in()")
                    result_box["ok"] = client.ensure_logged_in()
                    logger.info(f"  ← ensure_logged_in() returned {result_box['ok']}")
                elif hasattr(client, "_do_login"):
                    logger.info(f"  → calling _do_login()")
                    client._do_login()
                    result_box["ok"] = True
                    logger.info(f"  ← _do_login() returned")
                else:
                    result_box["exc"] = RuntimeError("client has no login method")
            except Exception as e:
                result_box["exc"] = e
            finally:
                result_box["done"] = True

        worker = threading.Thread(target=_do_call, daemon=True, name="KotakLoginWorker")
        worker.start()
        worker.join(timeout=per_attempt_timeout_sec)

        if not result_box["done"]:
            logger.error(
                f"❌ Login attempt {attempt} TIMED OUT after {per_attempt_timeout_sec}s "
                f"(thread still running — possible Kotak API hang or deadlock)"
            )
        elif result_box["exc"] is not None:
            logger.error(f"❌ Login attempt {attempt} raised: {result_box['exc']}")
        elif result_box["ok"]:
            logger.info(f"✅ Kotak login SUCCESS (attempt {attempt})")
            _telegram(notifier, f"🔐 Kotak login OK at {datetime.now().strftime('%H:%M:%S')} IST (attempt {attempt})")
            return True
        else:
            logger.error(f"❌ Login attempt {attempt} returned False")

        if attempt < max_attempts:
            logger.warning(f"Retrying in {retry_delay_sec}s...")
            time.sleep(retry_delay_sec)

    # All attempts failed
    msg = f"❌ Kotak login FAILED after {max_attempts} attempts at {datetime.now().strftime('%H:%M:%S')} IST"
    logger.critical(msg)
    _telegram(notifier, msg + "\n\nMARKET OPENS SOON — manual intervention needed!")
    return False


def _telegram(notifier, text: str) -> None:
    """Send a Telegram alert (best-effort)."""
    try:
        if notifier is None:
            return
        # Try multiple known interfaces
        if hasattr(notifier, "telegram") and notifier.telegram and getattr(notifier.telegram, "enabled", False):
            notifier.telegram.send_message(text)
        elif hasattr(notifier, "send_message"):
            notifier.send_message(text)
    except Exception as e:
        logger.debug(f"Telegram alert failed: {e}")


def _next_target_time(target_hour: int, target_min: int) -> datetime:
    """
    Return the next datetime when target HH:MM occurs on a weekday.
    Skips weekends. Always returns a future time.
    """
    now = datetime.now()
    target_today = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)

    # If target already passed today, move to tomorrow
    if target_today <= now:
        target_today += timedelta(days=1)

    # Skip weekends
    while target_today.weekday() >= 5:   # 5=Sat, 6=Sun
        target_today += timedelta(days=1)

    return target_today


def _scheduler_loop(client, notifier, target_hour: int, target_min: int):
    """Background daemon — fires login at target HH:MM each weekday."""
    logger.info(
        f"🔐 Kotak login scheduler started — will fire at "
        f"{target_hour:02d}:{target_min:02d} IST every weekday"
    )

    # If we're already past target time TODAY and today is a weekday and
    # the market window is upcoming or current → fire now
    try:
        now = datetime.now()
        if now.weekday() < 5:
            target_today = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
            market_close_today = now.replace(hour=15, minute=30, second=0, microsecond=0)
            # If we're between target and market close on a weekday, log in now
            if target_today <= now <= market_close_today:
                logger.warning(
                    f"⚡ Started past target time ({now.strftime('%H:%M:%S')}) "
                    f"on a weekday — firing immediate login"
                )
                _login_with_retries(client, notifier)
    except Exception as e:
        logger.error(f"Immediate-login check failed: {e}")

    # Main loop — sleep until next target, fire, repeat
    while True:
        try:
            target = _next_target_time(target_hour, target_min)
            wait_sec = (target - datetime.now()).total_seconds()
            logger.info(
                f"💤 Sleeping {wait_sec/3600:.1f}h until next login at "
                f"{target.strftime('%Y-%m-%d %H:%M:%S')} IST"
            )

            # Sleep in chunks to allow process to die cleanly on signal
            end_at = time.monotonic() + wait_sec
            while time.monotonic() < end_at:
                chunk = min(300, end_at - time.monotonic())   # 5 min chunks
                if chunk > 0:
                    time.sleep(chunk)

            # Fire login
            now = datetime.now()
            logger.info(f"⏰ Target time reached: {now.strftime('%Y-%m-%d %H:%M:%S')} IST → logging in")
            _login_with_retries(client, notifier)

            # Avoid double-fire if the loop returns immediately
            time.sleep(60)

        except Exception as e:
            logger.error(f"Scheduler loop error: {e} — recovering in 60s")
            time.sleep(60)


def schedule_daily_login(client, notifier=None,
                          target_hour: Optional[int] = None,
                          target_min: Optional[int] = None) -> threading.Thread:
    """
    Start the daily Kotak login scheduler in a background daemon.

    target_hour / target_min: defaults from env (KOTAK_LOGIN_TIME=09:10) or 09:10.
    """
    _setup_log_file()

    if target_hour is None or target_min is None:
        env_time = os.getenv("KOTAK_LOGIN_TIME", "09:10")
        try:
            h_str, m_str = env_time.split(":")
            target_hour = int(h_str)
            target_min = int(m_str)
        except Exception:
            target_hour, target_min = 9, 10

    t = threading.Thread(
        target=_scheduler_loop,
        args=(client, notifier, target_hour, target_min),
        daemon=True,
        name="KotakLoginScheduler",
    )
    t.start()
    return t
