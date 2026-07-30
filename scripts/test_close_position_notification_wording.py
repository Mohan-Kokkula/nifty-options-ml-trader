#!/usr/bin/env python3
"""
Regression test for C11: NiftyOptionsTrader.close_position()'s notification
must not claim the position was squared off when it wasn't.

Standalone script (repo convention), run with:
    python scripts/test_close_position_notification_wording.py

Covers core/trader.py::NiftyOptionsTrader.close_position().

Bug being regression-tested: previously, the notify_trade() call's
details= argument was hardcoded to "Position squared off" regardless of
the computed status, so a failed/rejected close (status="failed") still
sent a notification claiming the position was closed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trader import NiftyOptionsTrader


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _make_trader(client_result: dict):
    trader = object.__new__(NiftyOptionsTrader)
    trader.client = MagicMock()
    trader.client.close_position.return_value = client_result
    trader.exchange = "NFO"
    trader.product = "MIS"
    trader.strategy = "OpenClawNifty"
    trader.audit = MagicMock()
    trader.notif = MagicMock()
    return trader


def main():
    all_ok = True

    # ── Success: details must remain "Position squared off" ────────────
    trader_ok = _make_trader({"status": "success", "orderid": "ORD1"})
    trader_ok.close_position("NIFTY2660923400PE")
    kwargs_ok = trader_ok.notif.notify_trade.call_args.kwargs
    all_ok &= check(
        "success details is exactly 'Position squared off'",
        kwargs_ok.get("details") == "Position squared off",
    )
    all_ok &= check("success status kwarg is 'success'", kwargs_ok.get("status") == "success")

    # ── Failure: details must NOT claim the position was squared off ───
    trader_bad = _make_trader({"status": "failed", "message": "RMS block"})
    trader_bad.close_position("NIFTY2660923400PE")
    kwargs_bad = trader_bad.notif.notify_trade.call_args.kwargs
    all_ok &= check(
        "failure details does NOT say 'Position squared off'",
        "Position squared off" not in kwargs_bad.get("details", ""),
    )
    all_ok &= check(
        "failure details indicates the position remains open",
        "remains open" in kwargs_bad.get("details", ""),
    )
    all_ok &= check("failure status kwarg is unchanged ('failed')", kwargs_bad.get("status") == "failed")

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
