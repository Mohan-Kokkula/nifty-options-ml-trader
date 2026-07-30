"""
shadow_scheduler.py — daily report, auto-cleanup, and health monitoring for
the production shadow framework (core/shadow_framework.py).

Once per weekday at SHADOW_REPORT_TIME (default 15:35 IST — just after the
15:30 EOD summary):
  1. generates today's report  -> data/shadow/reports/report_<date>.json
  2. runs cleanup_old_files()     (auto cleanup, SHADOW_RETENTION_DAYS)
  3. checks shadow-framework health and sends a Telegram alert if the
     shadow model failed to load or has repeated errors

Mirrors core/eod_summary.py's scheduler pattern exactly. Restart-safe: a
restart re-derives `sent_today` from the wall clock; at worst the daily
report is regenerated (idempotent — overwrites the same file).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _seconds_until(target_h: int, target_m: int) -> float:
    now = datetime.now()
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _alert(notifier, text: str) -> None:
    logger.warning(f"SHADOW FRAMEWORK ALERT: {text}")
    try:
        if notifier and getattr(notifier, "telegram", None) and notifier.telegram.enabled:
            notifier.telegram.send_message(f"⚠️ <b>Shadow Framework</b>\n\n{text}")
    except Exception as e:
        logger.debug(f"Shadow alert send failed: {e}")


def run_daily_tasks(notifier=None) -> Optional[dict]:
    """Generate today's report, clean up old files, and check health.
    Exposed as a standalone function so it can also be called manually
    (e.g. from a cron job) without starting the scheduler thread.
    """
    from core.shadow_framework import cleanup_old_files, health_snapshot, is_enabled
    from scripts.shadow_daily_report import generate_report

    if not is_enabled():
        logger.info("Shadow framework disabled (SHADOW_FRAMEWORK_ENABLED=false) — skipping daily tasks")
        return None

    report = None
    try:
        report = generate_report()
        logger.info(
            f"\U0001F4CB Shadow daily report: {report.get('date')} "
            f"n_cycles={report.get('n_cycles')} "
            f"agreement_rate={report.get('agreement_rate')}"
        )
    except Exception as e:
        logger.warning(f"Shadow daily report failed: {e}")

    try:
        result = cleanup_old_files()
        if result["removed"]:
            logger.info(f"\U0001F9F9 Shadow cleanup removed {len(result['removed'])} file(s)")
    except Exception as e:
        logger.warning(f"Shadow cleanup failed: {e}")

    try:
        health = health_snapshot()
        if health and health.get("model_loaded") is False:
            _alert(
                notifier,
                "Shadow model not loaded — comparison logging is inactive. "
                "Check SHADOW_FRAMEWORK_MODEL_DIR artifacts "
                f"({health.get('model_dir')})."
            )
        if health and health.get("consecutive_errors", 0) >= 5:
            _alert(
                notifier,
                f"{health['consecutive_errors']} consecutive observe() "
                f"errors. Last: {health.get('last_error')} "
                f"@ {health.get('last_error_ts')}"
            )
    except Exception as e:
        logger.debug(f"Shadow health check failed: {e}")

    return report


def _scheduler_loop(notifier, target_h: int, target_m: int):
    logger.info(
        f"\U0001F4CB Shadow framework scheduler started — daily "
        f"report/cleanup at {target_h:02d}:{target_m:02d} IST weekdays"
    )
    sent_today = ""
    while True:
        try:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            if now.weekday() >= 5:
                time.sleep(3600)
                continue

            target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
            if now >= target and sent_today != today:
                run_daily_tasks(notifier)
                sent_today = today
                time.sleep(12 * 3600)
                continue

            wait = (target - now).total_seconds()
            time.sleep(min(wait, 1800))
        except Exception as e:
            logger.error(f"Shadow scheduler error: {e}")
            time.sleep(300)


def schedule_shadow_tasks(notifier=None, target_time: Optional[str] = None) -> Optional[threading.Thread]:
    """Start the shadow framework's daily report/cleanup/health scheduler
    as a daemon thread. No-op if SHADOW_FRAMEWORK_ENABLED=false."""
    from core.shadow_framework import is_enabled
    if not is_enabled():
        logger.info("Shadow framework disabled — scheduler not started")
        return None

    target_time = target_time or os.getenv("SHADOW_REPORT_TIME", "15:35")
    try:
        h, m = map(int, target_time.split(":"))
    except Exception:
        h, m = 15, 35

    t = threading.Thread(
        target=_scheduler_loop, args=(notifier, h, m),
        daemon=True, name="ShadowFrameworkScheduler",
    )
    t.start()
    return t
