#!/usr/bin/env python3
"""
Regression test for C9: cancel_all()/cancel_all_orders() must report an
honest status instead of a hardcoded one.

Standalone script (repo convention), run with:
    python scripts/test_cancel_all_status.py

Covers:
  core/kotak_neo_client.py::cancel_all_orders() — previously returned
  status="success" unconditionally on its happy path, even when individual
  order cancels failed (non-empty "errors").

  core/trader.py::NiftyOptionsTrader.cancel_all() — previously hardcoded
  status="executed"/"success" for its audit log and notification,
  ignoring the actual result from cancel_all_orders().
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kotak_neo_client import KotakNeoClient
from core.trader import NiftyOptionsTrader


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _make_kotak_client():
    """Same construction idiom as scripts/test_phase2_exchange_sl.py."""
    client = object.__new__(KotakNeoClient)
    client._neo = MagicMock()
    client.rate_limiter = MagicMock()
    client.rate_limiter.wait_and_acquire = MagicMock()
    client._rate_limiter = client.rate_limiter
    client._consecutive_failures = 0
    client._session_valid = True
    client._lock = threading.Lock()
    client._login_lock = threading.RLock()
    client._rate_gate = lambda endpoint, payload: None
    return client


def main():
    all_ok = True

    # ── cancel_all_orders(): all cancels succeed → status="success" ──────
    client_ok = _make_kotak_client()
    client_ok._neo.order_report.return_value = {"data": [
        {"nOrdNo": "A1", "ordSt": "open"},
        {"nOrdNo": "A2", "ordSt": "pending"},
    ]}
    client_ok._neo.cancel_order.return_value = {}
    result_ok = client_ok.cancel_all_orders()
    all_ok &= check(
        "cancel_all_orders: no errors -> status=success",
        result_ok["status"] == "success",
    )
    all_ok &= check("cancel_all_orders: both orders cancelled", len(result_ok["cancelled"]) == 2)

    # ── cancel_all_orders(): one cancel fails → status must NOT be success ──
    client_partial = _make_kotak_client()
    client_partial._neo.order_report.return_value = {"data": [
        {"nOrdNo": "B1", "ordSt": "open"},
        {"nOrdNo": "B2", "ordSt": "open"},
    ]}

    def _cancel_side_effect(order_id):
        if order_id == "B2":
            raise Exception("RMS block")
        return {}

    client_partial._neo.cancel_order.side_effect = _cancel_side_effect
    result_partial = client_partial.cancel_all_orders()
    all_ok &= check(
        "cancel_all_orders: partial failure -> status is NOT success",
        result_partial["status"] != "success",
    )
    all_ok &= check("cancel_all_orders: successful cancel still recorded", "B1" in result_partial["cancelled"])
    all_ok &= check("cancel_all_orders: failed cancel recorded in errors", len(result_partial["errors"]) == 1)

    # ── NiftyOptionsTrader.cancel_all(): success result -> status=success ──
    trader_ok = object.__new__(NiftyOptionsTrader)
    trader_ok.client = MagicMock()
    trader_ok.client.cancel_all_orders.return_value = {
        "status": "success", "cancelled": ["X1"], "errors": [],
    }
    trader_ok.strategy = "OpenClawNifty"
    trader_ok.audit = MagicMock()
    trader_ok.notif = MagicMock()

    trader_ok.cancel_all()
    audit_status_ok = trader_ok.audit.log.call_args.kwargs.get("status")
    notif_status_ok = trader_ok.notif.notify_trade.call_args.kwargs.get("status")
    all_ok &= check("trader.cancel_all: audit.log status=success on success", audit_status_ok == "success")
    all_ok &= check("trader.cancel_all: notify_trade status=success on success", notif_status_ok == "success")

    # ── NiftyOptionsTrader.cancel_all(): partial-failure result -> NOT success ──
    trader_bad = object.__new__(NiftyOptionsTrader)
    trader_bad.client = MagicMock()
    trader_bad.client.cancel_all_orders.return_value = {
        "status": "error", "cancelled": ["Y1"], "errors": ["Y2: RMS block"],
    }
    trader_bad.strategy = "OpenClawNifty"
    trader_bad.audit = MagicMock()
    trader_bad.notif = MagicMock()

    trader_bad.cancel_all()
    audit_status_bad = trader_bad.audit.log.call_args.kwargs.get("status")
    notif_status_bad = trader_bad.notif.notify_trade.call_args.kwargs.get("status")
    all_ok &= check("trader.cancel_all: audit.log status != success on partial failure", audit_status_bad != "success")
    all_ok &= check("trader.cancel_all: notify_trade status != success on partial failure", notif_status_bad != "success")

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
