#!/usr/bin/env python3
"""
Regression test for C6: _smart_execute() must not fire a second (IOC-2)
order when IOC-1 raised an exception but the broker actually filled it --
that would create a duplicate position.

Standalone script (repo convention), run with:
    python scripts/test_ioc_duplicate_order_guard.py

Covers core/claude_pilot.py::ClaudePilot._smart_execute().

Bug being regression-tested: previously, _ioc_attempt() caught ANY
exception from trader.smart_trade() and treated it identically to a clean
non-fill, unconditionally firing IOC-2 -- even if the broker had actually
accepted and filled IOC-1 and only the response was lost (e.g. a network
blip after the order was placed).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claude_pilot import ClaudePilot


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _make_pilot():
    pilot = object.__new__(ClaudePilot)
    pilot.config = SimpleNamespace(max_spread_pct=0.02)
    pilot.trader = MagicMock()
    pilot.trader.strikes.get_next_expiry.return_value = "23JUL26"
    pilot.trader.get_nifty_spot.return_value = 24000.0
    pilot.trader.client.get_option_quote.return_value = {
        "ask": 100.0, "bid": 99.0, "ltp": 99.5,
    }
    return pilot


def main():
    all_ok = True

    # ── IOC-1 raises, but a stray position IS found: no IOC-2, treated as filled ──
    pilot = _make_pilot()
    pilot.trader.smart_trade.side_effect = Exception("Response lost after send")
    pilot.trader.client.get_open_nifty_positions.return_value = [
        {"symbol": "NIFTY2660924000CE", "direction": "CALL", "net_qty": 65, "ltp": 100.0},
    ]
    result = pilot._smart_execute(action="BUY", option_type="CE", strike_mode="ATM", qty=65)
    all_ok &= check(
        "stray fill found -> only ONE smart_trade call (no IOC-2)",
        pilot.trader.smart_trade.call_count == 1,
    )
    all_ok &= check(
        "stray fill found -> status is RECONCILED_FILLED, not ABORTED",
        result.get("status") == "RECONCILED_FILLED",
    )
    all_ok &= check(
        "stray fill found -> real broker symbol is used, not a placeholder",
        result.get("symbol") == "NIFTY2660924000CE",
    )

    # ── IOC-1 raises, no stray position: proceeds to IOC-2 exactly as before ──
    pilot2 = _make_pilot()
    pilot2.trader.smart_trade.side_effect = Exception("Genuine network error")
    pilot2.trader.client.get_open_nifty_positions.return_value = []
    result2 = pilot2._smart_execute(action="BUY", option_type="CE", strike_mode="ATM", qty=65)
    all_ok &= check(
        "no stray fill -> TWO smart_trade calls (IOC-1 then IOC-2)",
        pilot2.trader.smart_trade.call_count == 2,
    )
    all_ok &= check(
        "no stray fill, both attempts raise -> final status is ABORTED",
        result2.get("status") == "ABORTED",
    )

    # ── IOC-1 raises, reconciliation check itself fails: abort, no IOC-2 ──
    pilot3 = _make_pilot()
    pilot3.trader.smart_trade.side_effect = Exception("Response lost after send")
    pilot3.trader.client.get_open_nifty_positions.side_effect = Exception("Broker unreachable")
    result3 = pilot3._smart_execute(action="BUY", option_type="CE", strike_mode="ATM", qty=65)
    all_ok &= check(
        "reconciliation itself fails -> only ONE smart_trade call (no IOC-2)",
        pilot3.trader.smart_trade.call_count == 1,
    )
    all_ok &= check(
        "reconciliation itself fails -> status is ABORTED (safe default)",
        result3.get("status") == "ABORTED",
    )

    # ── IOC-1 cleanly rejected (no exception): unaffected, proceeds to IOC-2 ──
    pilot4 = _make_pilot()
    pilot4.trader.smart_trade.side_effect = [
        {"status": "blocked", "reason": "Risk limit"},
        {"trade": {"orderid": "O2", "status": "success"}, "symbol": "NIFTY2660924000CE"},
    ]
    result4 = pilot4._smart_execute(action="BUY", option_type="CE", strike_mode="ATM", qty=65)
    all_ok &= check(
        "clean rejection (not an exception) -> reconciliation NOT triggered",
        not pilot4.trader.client.get_open_nifty_positions.called,
    )
    all_ok &= check(
        "clean rejection -> IOC-2 still fires and can fill normally",
        pilot4.trader.smart_trade.call_count == 2 and result4.get("trade", {}).get("orderid") == "O2",
    )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
