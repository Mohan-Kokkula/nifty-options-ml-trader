#!/usr/bin/env python3
"""
Regression test for C5: NotificationManager.notify() was called by three
sites (kotak_neo_client.py relogin-failed alert, kotak_neo_client.py
broker-unhealthy alert, security/__init__.py strategy-auto-pause alert)
but the method didn't exist, raising AttributeError — silently swallowed
by each call site's own try/except, so the alerts never reached Telegram.

Standalone script (repo convention), run with:
    python scripts/test_notify_method.py

Covers notifications/__init__.py::NotificationManager.notify().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications import NotificationManager


class _StubTelegram:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sent = []

    def send_message(self, text, parse_mode="HTML"):
        self.sent.append((text, parse_mode))
        return True


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True

    # notify() must exist and not raise (the exact bug: AttributeError on
    # the missing method, previously swallowed by the caller's try/except).
    telegram = _StubTelegram(enabled=True)
    mgr = NotificationManager(telegram_notifier=telegram)
    try:
        mgr.notify(subject="Kotak Neo — Broker Unhealthy", message="test body")
        raised = False
    except Exception as e:
        raised = True
        print(f"    unexpected exception: {e}")
    all_ok &= check("notify() does not raise", raised is False)

    mgr.shutdown()  # wait for the background executor to finish
    all_ok &= check("notify() reached telegram.send_message", len(telegram.sent) == 1)
    if telegram.sent:
        text, parse_mode = telegram.sent[0]
        all_ok &= check("subject present in sent text", "Broker Unhealthy" in text)
        all_ok &= check("message present in sent text", "test body" in text)
        all_ok &= check("parse_mode defaults to HTML", parse_mode == "HTML")

    # Disabled telegram -> no send attempted, no exception either.
    telegram_off = _StubTelegram(enabled=False)
    mgr2 = NotificationManager(telegram_notifier=telegram_off)
    try:
        mgr2.notify(subject="X", message="Y")
        raised2 = False
    except Exception as e:
        raised2 = True
        print(f"    unexpected exception: {e}")
    mgr2.shutdown()
    all_ok &= check("notify() does not raise when telegram disabled", raised2 is False)
    all_ok &= check("no send attempted when telegram disabled", len(telegram_off.sent) == 0)

    # No telegram configured at all (None) -> same safe no-op.
    mgr3 = NotificationManager()
    try:
        mgr3.notify(subject="X", message="Y")
        raised3 = False
    except Exception as e:
        raised3 = True
        print(f"    unexpected exception: {e}")
    mgr3.shutdown()
    all_ok &= check("notify() does not raise when telegram is None", raised3 is False)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
