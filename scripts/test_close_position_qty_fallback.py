#!/usr/bin/env python3
"""
Regression test for C10: close_position() must not guess a quantity and
place an order when it cannot determine the real open quantity.

Standalone script (repo convention), run with:
    python scripts/test_close_position_qty_fallback.py

Covers core/kotak_neo_client.py::close_position().

Bug being regression-tested: previously, if self._neo.positions() raised
(e.g. a network error, or — since C4 — a timeout), close_position() fell
back to a hardcoded qty=1 guess and placed a real order for that guessed
quantity. If the broker accepted it, the result reported status="success",
which _close_confirmed() (C3) would treat as a confirmed close even though
the guessed quantity might not match the true open quantity, leaving a
residual position at the broker.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kotak_neo_client import KotakNeoClient


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

    # ── positions() raises -> no order placed, status="error" ──────────
    client = _make_kotak_client()
    client._neo.positions.side_effect = Exception("Broker connection lost")
    client._neo.place_order = MagicMock()

    result = client.close_position(
        symbol="NIFTY2660923400PE", exchange="NFO", product="MIS",
    )

    all_ok &= check(
        "close_position returns status=error when positions() raises",
        result.get("status") == "error",
    )
    all_ok &= check(
        "error message is preserved",
        "Broker connection lost" in result.get("message", ""),
    )
    all_ok &= check(
        "no order placement was attempted (place_order never called)",
        not client._neo.place_order.called,
    )

    # ── Successful path is unaffected: positions() succeeds -> order placed ──
    client_ok = _make_kotak_client()
    client_ok._neo.positions.return_value = {"data": [
        {"trdSym": "NIFTY2660923400PE", "flBuyQty": "65", "flSellQty": "0"},
    ]}
    client_ok._call_with_retry = lambda fn, *a, **kw: {"nOrdNo": "ORD1", "stat": "Ok"}
    client_ok._normalize_order_response = lambda raw, **kw: {
        "orderid": raw.get("nOrdNo", ""), "status": "success",
    }

    result_ok = client_ok.close_position(
        symbol="NIFTY2660923400PE", exchange="NFO", product="MIS",
    )
    all_ok &= check(
        "successful path still returns status=success when positions() succeeds",
        result_ok.get("status") == "success",
    )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
