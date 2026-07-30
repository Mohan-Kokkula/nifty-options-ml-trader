"""
shadow_health_check.py — standalone health check for the production shadow
framework (core/shadow_framework.py).

Reads data/shadow/health.json (written every cycle by
core/shadow_framework.observe) and prints one status line.

Exit codes (cron / monitoring-friendly):
  0 = OK        running normally
  1 = WARN      no observations yet, or stale during market hours
  2 = CRITICAL  model failed to load, or repeated observe() errors

Usage:
  python scripts/shadow_health_check.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.shadow_framework import health_snapshot, is_enabled  # noqa: E402

STALE_MIN = 30


def market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= hm <= (15 * 60 + 30)


def main() -> int:
    if not is_enabled():
        print("DISABLED: SHADOW_FRAMEWORK_ENABLED=false")
        return 0

    snap = health_snapshot()
    now = datetime.now()

    if not snap:
        print("WARN: no health record yet (no cycles observed since startup)")
        return 1

    if snap.get("model_loaded") is False:
        print(
            f"CRITICAL: shadow model not loaded "
            f"(dir={snap.get('model_dir')}, last_error={snap.get('last_error')})"
        )
        return 2

    if snap.get("consecutive_errors", 0) >= 5:
        print(
            f"CRITICAL: {snap['consecutive_errors']} consecutive errors "
            f"(last: {snap.get('last_error')} @ {snap.get('last_error_ts')})"
        )
        return 2

    last_update = snap.get("last_update")
    if not last_update:
        print("WARN: no last_update timestamp in health record")
        return 1

    age_min = (now - datetime.fromisoformat(last_update)).total_seconds() / 60
    if market_open(now) and age_min > STALE_MIN:
        print(f"WARN: stale — last observation {age_min:.1f} min ago (market open)")
        return 1

    print(
        f"OK: model={snap.get('model_dir')} "
        f"n_features={snap.get('n_features')} "
        f"total_observations={snap.get('total_observations')} "
        f"last_update={last_update} ({age_min:.1f} min ago)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
