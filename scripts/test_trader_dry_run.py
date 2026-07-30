#!/usr/bin/env python3
"""
Regression test for C2: NiftyOptionsTrader's order-placing methods must
not reach the real broker when DRY_RUN=true.

Standalone script (repo convention), run with:
    python scripts/test_trader_dry_run.py

Covers core/trader.py::NiftyOptionsTrader.smart_trade/close_position/cancel_all.

Bug being regression-tested: previously, trader.py had zero DRY_RUN
awareness. The automated signal-generation cycle in claude_pilot.py already
short-circuits before reaching these methods when DRY_RUN=true, but the
manual HTTP endpoints (/trade, /close, /cancel_all in main.py) call these
methods directly, bypassing that check entirely -- meaning a manual API
call could place a real order even while DRY_RUN=true.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trader import NiftyOptionsTrader


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _make_trader():
    trader = object.__new__(NiftyOptionsTrader)
    trader.client = MagicMock()
    trader.strikes = MagicMock()
    trader.strikes.get_next_expiry.return_value = "23JUL26"
    trader.risk = MagicMock()
    trader.risk.can_open_trade.return_value = (True, "")
    trader.exchange = "NFO"
    trader.product = "MIS"
    trader.strategy = "OpenClawNifty"
    trader.default_qty = 65
    trader.audit = MagicMock()
    trader.notif = MagicMock()
    return trader


def main():
    all_ok = True

    with patch.dict(os.environ, {"DRY_RUN": "true"}):
        # ── smart_trade: no real order placed ───────────────────────────
        trader = _make_trader()
        result = trader.smart_trade(action="BUY", option_type="CE", strike_mode="ATM")
        all_ok &= check(
            "smart_trade: client.place_option_order NOT called in DRY_RUN",
            not trader.client.place_option_order.called,
        )
        all_ok &= check(
            "smart_trade: risk.can_open_trade NOT called in DRY_RUN",
            not trader.risk.can_open_trade.called,
        )
        all_ok &= check(
            "smart_trade: simulated result has trade.status == simulated",
            result.get("trade", {}).get("status") == "simulated",
        )

        # ── close_position: no real close attempted ─────────────────────
        trader2 = _make_trader()
        result2 = trader2.close_position("NIFTY2660923400PE")
        all_ok &= check(
            "close_position: client.close_position NOT called in DRY_RUN",
            not trader2.client.close_position.called,
        )
        all_ok &= check(
            "close_position: simulated result has status=success",
            result2.get("status") == "success",
        )

        # ── cancel_all: no real cancel attempted ────────────────────────
        trader3 = _make_trader()
        result3 = trader3.cancel_all()
        all_ok &= check(
            "cancel_all: client.cancel_all_orders NOT called in DRY_RUN",
            not trader3.client.cancel_all_orders.called,
        )
        all_ok &= check(
            "cancel_all: simulated result has status=success",
            result3.get("status") == "success",
        )

    # ── Live mode (DRY_RUN=false): real broker calls still happen ──────
    with patch.dict(os.environ, {"DRY_RUN": "false"}):
        trader4 = _make_trader()
        trader4.client.place_option_order.return_value = {
            "orderid": "ORD1", "status": "success", "symbol": "NIFTY24000CE",
        }
        trader4.smart_trade(action="BUY", option_type="CE", strike_mode="ATM")
        all_ok &= check(
            "smart_trade: client.place_option_order IS called when DRY_RUN=false",
            trader4.client.place_option_order.called,
        )

        trader5 = _make_trader()
        trader5.client.close_position.return_value = {"status": "success", "orderid": "O1"}
        trader5.close_position("NIFTY2660923400PE")
        all_ok &= check(
            "close_position: client.close_position IS called when DRY_RUN=false",
            trader5.client.close_position.called,
        )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
