#!/usr/bin/env python3
"""
Regression test for C3: a failed/rejected close_position()/cancel_all()
result must NOT be treated as a confirmed close.

Standalone script (repo convention), run with:
    python scripts/test_close_confirmation.py

Covers core/claude_pilot.py::_close_confirmed(), the pure decision function
extracted from _position_monitor_loop's close-handling block specifically
so this logic is testable without mocking the full threaded monitor loop
(broker client, notifier, journal, WS feed).

Bug being regression-tested: previously, self.trader.close_position()'s
return value was discarded entirely — a broker REJECTION (status="failed",
no exception raised) was silently treated as a successful close, clearing
_live_position and marking the position CLOSED while it was still open
and unprotected at the broker.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claude_pilot import _close_confirmed


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True

    # The exact real-world shape returned by trader.py::close_position() on
    # a clean broker rejection (no exception raised) — this is the case
    # that was previously silently treated as success.
    all_ok &= check(
        "broker-rejected close (status=failed) is NOT confirmed",
        _close_confirmed({"status": "failed", "error": "insufficient margin"}) is False,
    )

    # The exact shape on a genuine success.
    all_ok &= check(
        "successful close (status=success) IS confirmed",
        _close_confirmed({"status": "success", "orderid": "123"}) is True,
    )

    # cancel_all_orders()'s "error" status (network/unexpected failure caught
    # internally, still returned as a dict rather than raised).
    all_ok &= check(
        "cancel_all error status is NOT confirmed",
        _close_confirmed({"status": "error", "message": "timeout"}) is False,
    )

    # Defensive cases: nothing returned / unexpected shape must never be
    # mistaken for confirmation.
    all_ok &= check("empty dict is NOT confirmed", _close_confirmed({}) is False)
    all_ok &= check("None is NOT confirmed", _close_confirmed(None) is False)
    all_ok &= check(
        "dict with no status key is NOT confirmed",
        _close_confirmed({"orderid": "123"}) is False,
    )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
