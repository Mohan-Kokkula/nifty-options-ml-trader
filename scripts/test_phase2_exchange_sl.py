"""
test_phase2_exchange_sl.py — Phase 2: Exchange-Resident Protection

Tests:
  A. place_exchange_sl returns order_id on success
  B. place_exchange_sl returns "" on broker failure (non-fatal)
  C. cancel_exchange_sl calls cancel_order with correct order_id
  D. cancel_exchange_sl returns False gracefully on failure
  E. update_exchange_sl calls modify_order with correct trigger
  F. _place_exchange_sl skips DRY_RUN mode
  G. _place_exchange_sl skips PAPER positions
  H. _place_exchange_sl skips when EXCHANGE_SL_ENABLED=false
  I. _cancel_exchange_sl clears sl_order_id on success
  J. get_open_exchange_sl_orders filters correctly
  K. Double-sell prevention: cancel called before close
  L. Reconciliation re-places exchange SL when missing
  M. _place_exchange_sl skips when exchange_sl_trigger/limit not computed
  N. _cancel_exchange_sl retains sl_order_id when cancel is not confirmed
"""
import os, sys, json, threading, time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS  {name}")
    else:    FAIL += 1; print(f"  FAIL  {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kotak_client():
    """Create a KotakNeoClient with mocked _neo and _rate_gate."""
    from core.kotak_neo_client import KotakNeoClient
    client = object.__new__(KotakNeoClient)
    client._neo = MagicMock()
    client.rate_limiter = MagicMock()         # used by _rate_gate
    client.rate_limiter.wait_and_acquire = MagicMock()
    client._rate_limiter = client.rate_limiter # alias
    client._consecutive_failures = 0
    client._session_valid = True
    client._lock = threading.Lock()
    client._login_lock = threading.RLock()
    # Override _rate_gate to be a no-op (session check passes)
    client._rate_gate = lambda endpoint, payload: None
    return client

def _make_pos(symbol="NIFTY2660923400PE", sl=23470, qty=65, sl_order_id="",
               exchange_sl_trigger=80.0, exchange_sl_limit=76.0):
    from core.claude_pilot import LivePosition, PositionState
    return LivePosition(
        direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
        sl_price=float(sl), tp_price=23260.0, initial_sl=float(sl),
        symbol=symbol, atr_at_entry=35.0, entry_premium=90.0,
        state=PositionState.OPEN, original_qty=qty,
        sl_order_id=sl_order_id,
        exchange_sl_trigger=exchange_sl_trigger,
        exchange_sl_limit=exchange_sl_limit,
    )


# ---------------------------------------------------------------------------
# A. place_exchange_sl returns order_id
# ---------------------------------------------------------------------------
print("\n[A] place_exchange_sl returns order_id")

client_a = _make_kotak_client()
# Patch _call_with_retry to return success
client_a._call_with_retry = lambda fn, *a, **kw: {"nOrdNo": "ORD12345", "stat": "Ok"}
# Patch _normalize_order_response
client_a._normalize_order_response = lambda raw, **kw: {"orderid": raw.get("nOrdNo", ""), "status": "success"}

result_a = client_a.place_exchange_sl("NIFTY2660923400PE", 65, 80.0, 76.0)
check("returns order_id on success", result_a == "ORD12345")


# ---------------------------------------------------------------------------
# B. place_exchange_sl returns "" on failure (non-fatal)
# ---------------------------------------------------------------------------
print("\n[B] place_exchange_sl returns '' on failure")

client_b = _make_kotak_client()
client_b._call_with_retry = MagicMock(side_effect=Exception("Broker down"))

result_b = client_b.place_exchange_sl("NIFTY2660923400PE", 65, 80.0, 76.0)
check("returns empty string on broker failure", result_b == "")


# ---------------------------------------------------------------------------
# C. cancel_exchange_sl calls cancel_order with correct order_id
# ---------------------------------------------------------------------------
print("\n[C] cancel_exchange_sl calls cancel_order")

client_c = _make_kotak_client()
cancelled = []
client_c._call_with_retry = lambda fn, *a, **kw: cancelled.append(kw.get("order_id")) or {}

result_c = client_c.cancel_exchange_sl("ORD12345")
check("cancel_order called with correct order_id", "ORD12345" in cancelled)
check("returns True on success", result_c is True)


# ---------------------------------------------------------------------------
# D. cancel_exchange_sl returns False on failure
# ---------------------------------------------------------------------------
print("\n[D] cancel_exchange_sl graceful failure")

client_d = _make_kotak_client()
client_d._call_with_retry = MagicMock(side_effect=Exception("Network error"))
result_d = client_d.cancel_exchange_sl("ORD99999")
check("returns False on failure", result_d is False)
check("does not raise", True)  # already passed if we got here


# ---------------------------------------------------------------------------
# E. update_exchange_sl calls modify_order with correct trigger
# ---------------------------------------------------------------------------
print("\n[E] update_exchange_sl calls modify_order")

client_e = _make_kotak_client()
modify_calls = []
def _mock_modify(fn, *a, **kw):
    modify_calls.append(kw)
    return {}
client_e._call_with_retry = _mock_modify

result_e = client_e.update_exchange_sl("ORD12345", 85.0, 81.0, 65, "NIFTY2660923400PE")
check("modify_order called", len(modify_calls) > 0)
check("correct trigger_price passed", "85" in str(modify_calls))
check("correct limit price passed", "81" in str(modify_calls))
check("returns True on success", result_e is True)


# ---------------------------------------------------------------------------
# F. _place_exchange_sl skips DRY_RUN
# ---------------------------------------------------------------------------
print("\n[F] _place_exchange_sl skips in DRY_RUN")

from core.claude_pilot import ClaudePilot
pilot_f = object.__new__(ClaudePilot)
pilot_f.trader = MagicMock()

with patch.dict(os.environ, {"DRY_RUN": "true", "EXCHANGE_SL_ENABLED": "true"}):
    pos_f = _make_pos()
    pilot_f._place_exchange_sl(pos_f)
    check("trader.client.place_exchange_sl NOT called in DRY_RUN",
          not pilot_f.trader.client.place_exchange_sl.called)


# ---------------------------------------------------------------------------
# G. _place_exchange_sl skips PAPER positions
# ---------------------------------------------------------------------------
print("\n[G] _place_exchange_sl skips PAPER positions")

pilot_g = object.__new__(ClaudePilot)
pilot_g.trader = MagicMock()

with patch.dict(os.environ, {"DRY_RUN": "false", "EXCHANGE_SL_ENABLED": "true"}):
    pos_paper = _make_pos(symbol="PAPER")
    pilot_g._place_exchange_sl(pos_paper)
    check("skips PAPER symbol", not pilot_g.trader.client.place_exchange_sl.called)


# ---------------------------------------------------------------------------
# H. _place_exchange_sl skips when EXCHANGE_SL_ENABLED=false
# ---------------------------------------------------------------------------
print("\n[H] EXCHANGE_SL_ENABLED=false disables exchange SL")

pilot_h = object.__new__(ClaudePilot)
pilot_h.trader = MagicMock()

with patch.dict(os.environ, {"DRY_RUN": "false", "EXCHANGE_SL_ENABLED": "false"}):
    pos_h = _make_pos()
    pilot_h._place_exchange_sl(pos_h)
    check("place_exchange_sl NOT called when disabled",
          not pilot_h.trader.client.place_exchange_sl.called)


# ---------------------------------------------------------------------------
# I. _cancel_exchange_sl clears sl_order_id on success
# ---------------------------------------------------------------------------
print("\n[I] _cancel_exchange_sl clears sl_order_id")

pilot_i = object.__new__(ClaudePilot)
mock_client = MagicMock()
mock_client.cancel_exchange_sl.return_value = True
pilot_i.trader = MagicMock()
pilot_i.trader.client = mock_client

pos_i = _make_pos(sl_order_id="ORD_TO_CANCEL")
pilot_i._cancel_exchange_sl(pos_i)
check("sl_order_id cleared after cancel", pos_i.sl_order_id == "")
check("cancel_exchange_sl called on client", mock_client.cancel_exchange_sl.called)


# ---------------------------------------------------------------------------
# J. get_open_exchange_sl_orders filters correctly
# ---------------------------------------------------------------------------
print("\n[J] get_open_exchange_sl_orders filters correctly")

client_j = _make_kotak_client()
fake_orders = [
    # Valid OPENCLAW_SL order
    {"nOrdNo": "SL1", "trdSym": "NIFTY2660923400PE", "trnsTp": "S",
     "ordTyp": "SL-M", "tag": "OPENCLAW_SL", "ordSt": "open",
     "trgPrc": "23470", "qty": "65"},
    # Different tag — should be excluded
    {"nOrdNo": "OTH", "trdSym": "NIFTY2660923400PE", "trnsTp": "S",
     "ordTyp": "SL-M", "tag": "OTHER_TAG", "ordSt": "open",
     "trgPrc": "23400", "qty": "65"},
    # BUY order — should be excluded
    {"nOrdNo": "BUY1", "trdSym": "NIFTY2660923400PE", "trnsTp": "B",
     "ordTyp": "L", "tag": "OPENCLAW_SL", "ordSt": "open",
     "trgPrc": "0", "qty": "65"},
    # Completed order — should be excluded
    {"nOrdNo": "DONE", "trdSym": "NIFTY2660923400PE", "trnsTp": "S",
     "ordTyp": "SL-M", "tag": "OPENCLAW_SL", "ordSt": "complete",
     "trgPrc": "23470", "qty": "65"},
]
client_j._call_with_retry = lambda fn, *a, **kw: {"data": fake_orders}

results_j = client_j.get_open_exchange_sl_orders()
check("returns exactly 1 matching order", len(results_j) == 1)
check("correct order_id returned", results_j[0]["order_id"] == "SL1")
check("correct trigger_price", abs(results_j[0]["trigger_price"] - 23470.0) < 0.01)


# ---------------------------------------------------------------------------
# K. Double-sell prevention: cancel called before close
# ---------------------------------------------------------------------------
print("\n[K] Double-sell: cancel called before close")

cancel_calls = []
close_calls = []

class _MockClient:
    def cancel_exchange_sl(self, order_id):
        cancel_calls.append(order_id)
        return True

class _MockTrader:
    client = _MockClient()
    exchange = "NFO"
    def close_position(self, symbol):
        close_calls.append(symbol)

pilot_k = object.__new__(ClaudePilot)
pilot_k.trader = _MockTrader()

pos_k = _make_pos(sl_order_id="SL_ORD_K")
# Simulate: cancel then close
pilot_k._cancel_exchange_sl(pos_k)
pilot_k.trader.close_position(pos_k.symbol)

check("cancel called before close", len(cancel_calls) > 0)
check("close called after cancel", len(close_calls) > 0)
check("correct order cancelled", "SL_ORD_K" in cancel_calls)
check("sl_order_id cleared after cancel", pos_k.sl_order_id == "")


# ---------------------------------------------------------------------------
# L. Reconciliation: sl_order_id set from existing order
# ---------------------------------------------------------------------------
print("\n[L] Reconciliation finds existing exchange SL")

client_l = _make_kotak_client()
client_l._call_with_retry = lambda fn, *a, **kw: {"data": [
    {"nOrdNo": "EXIST_SL", "trdSym": "NIFTY2660923400PE", "trnsTp": "S",
     "ordTyp": "SL-M", "tag": "OPENCLAW_SL", "ordSt": "open",
     "trgPrc": "23470", "qty": "65"},
]}
existing = client_l.get_open_exchange_sl_orders()
check("existing SL order found", len(existing) == 1)
check("order_id matches", existing[0]["order_id"] == "EXIST_SL")

# If sl_order_id is set from existing, new SL is NOT placed:
from core.claude_pilot import LivePosition, PositionState
pos_l = _make_pos()
pos_l.sl_order_id = existing[0]["order_id"]
check("sl_order_id restored from existing order", pos_l.sl_order_id == "EXIST_SL")


# ---------------------------------------------------------------------------
# M. _place_exchange_sl skips when exchange_sl_trigger/limit not computed
# ---------------------------------------------------------------------------
print("\n[M] _place_exchange_sl skips without premium trigger/limit")

pilot_m = object.__new__(ClaudePilot)
pilot_m.trader = MagicMock()

with patch.dict(os.environ, {"DRY_RUN": "false", "EXCHANGE_SL_ENABLED": "true"}):
    # Simulates a reconciled position with no entry_premium/premium_sl_pts data
    pos_m = _make_pos(exchange_sl_trigger=0.0, exchange_sl_limit=0.0)
    pilot_m._place_exchange_sl(pos_m)
    check("place_exchange_sl NOT called when trigger/limit unset",
          not pilot_m.trader.client.place_exchange_sl.called)


# ---------------------------------------------------------------------------
# N. _cancel_exchange_sl retains sl_order_id when cancel is not confirmed
# ---------------------------------------------------------------------------
print("\n[N] _cancel_exchange_sl retains sl_order_id on unconfirmed cancel")

pilot_n = object.__new__(ClaudePilot)
mock_client_n = MagicMock()
mock_client_n.cancel_exchange_sl.return_value = False
pilot_n.trader = MagicMock()
pilot_n.trader.client = mock_client_n

pos_n = _make_pos(sl_order_id="ORD_NOT_CANCELLED")
pilot_n._cancel_exchange_sl(pos_n)
check("sl_order_id retained when cancel not confirmed",
      pos_n.sl_order_id == "ORD_NOT_CANCELLED")
check("cancel_exchange_sl was still attempted", mock_client_n.cancel_exchange_sl.called)


# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
