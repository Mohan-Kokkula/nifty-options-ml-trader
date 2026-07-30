"""
test_operational_safety.py — Operational-safety fixes 1-6 (2026-07-21).

Standalone script (repo convention), run with:
    python scripts/test_operational_safety.py

Covers every new failure path introduced by the 6 Critical-severity fixes
from the operational-safety audit:

  [1] core/kotak_neo_client.py  — BrokerCallTimeout on order-call timeout
  [2] core/claude_pilot.py      — _attempt_protected_close (close-then-
                                   confirm-then-cancel-SL ordering)
  [3] core/claude_pilot.py      — emergency_stop() 5-step sequence
  [4] core/claude_pilot.py      — monitor loop not gated on market hours
  [5] core/claude_pilot.py      — _reconcile_runtime() periodic reconciliation
  [6] security/__init__.py      — RiskManager fail-safe state persistence

As elsewhere in this repo, deeply-threaded/broker-coupled logic (the full
1s _position_monitor_loop, the background reconciliation thread itself) is
exercised through directly-callable methods/helpers rather than by spinning
up real daemon threads — matching the existing test_close_confirmation.py /
test_position_reconciliation.py / test_stop_management.py convention.
"""
import inspect
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ═══════════════════════════════════════════════════════════════════════
# [1] BrokerCallTimeout — bounded timeouts on order calls
# ═══════════════════════════════════════════════════════════════════════
print("\n[Fix 1] BrokerCallTimeout on order-call timeout")

import requests
from core.kotak_neo_client import KotakNeoClient, BrokerCallTimeout


def _make_client(notifier=None):
    c = object.__new__(KotakNeoClient)
    c._notifier = notifier
    c._consecutive_failures = 0
    c._unhealthy_alerted = False
    return c


# 1a. Order call that times out -> BrokerCallTimeout, called exactly once
#     (never blindly retried — duplicate-order risk).
notifier1 = MagicMock()
client1 = _make_client(notifier1)
calls = []


def _timeout_fn(**kw):
    calls.append(1)
    raise requests.exceptions.Timeout("connect timed out")


raised = None
try:
    client1._call_with_retry(_timeout_fn, is_order=True)
except Exception as e:
    raised = e

check("order-call timeout raises BrokerCallTimeout", isinstance(raised, BrokerCallTimeout))
check("order-call timeout NOT retried (called exactly once)", len(calls) == 1)
check("Telegram alert sent on timeout", notifier1.notify.called)
check("consecutive_failures incremented", client1._consecutive_failures == 1)

# 1b. Non-order call that times out -> falls through to the ORIGINAL
#     retry/backoff logic unchanged (regression guard: an earlier draft of
#     this fix used a separate except-clause that skipped this logic for
#     non-order timeouts entirely).
import core.kotak_neo_client as kc_mod

client2 = _make_client()
calls2 = []


def _timeout_then_ok(**kw):
    calls2.append(1)
    if len(calls2) < 2:
        raise requests.exceptions.Timeout("read timed out")
    return {"stat": "Ok"}


client2._check_session_error = lambda resp: False
orig_backoff = kc_mod.RETRY_BACKOFF
kc_mod.RETRY_BACKOFF = [0.0, 0.0, 0.0]
try:
    result = client2._call_with_retry(_timeout_then_ok, is_order=False)
finally:
    kc_mod.RETRY_BACKOFF = orig_backoff

check("non-order timeout retries (not a BrokerCallTimeout short-circuit)", len(calls2) == 2)
check("non-order timeout eventually succeeds via normal retry", result == {"stat": "Ok"})

# 1c. Order call, non-timeout exception -> original never-blindly-retry
#     behavior preserved (raises immediately, not wrapped as BrokerCallTimeout).
client3 = _make_client()
calls3 = []


def _value_error_fn(**kw):
    calls3.append(1)
    raise ValueError("bad request")


raised3 = None
try:
    client3._call_with_retry(_value_error_fn, is_order=True)
except Exception as e:
    raised3 = e

check("order-call non-timeout exception NOT wrapped as BrokerCallTimeout",
      not isinstance(raised3, BrokerCallTimeout))
check("order-call non-timeout exception still raised immediately (no retry)", len(calls3) == 1)


# ═══════════════════════════════════════════════════════════════════════
# [2] _attempt_protected_close — close-then-confirm-then-cancel-SL
# ═══════════════════════════════════════════════════════════════════════
print("\n[Fix 2] _attempt_protected_close ordering")

from core.claude_pilot import ClaudePilot, LivePosition, PositionState


def _make_pos(sl_order_id="SL123", symbol="NIFTY2660923200PE", state=PositionState.OPEN):
    """
    Default state=OPEN (changed from CLOSING in the round-3 ownership fix):
    _attempt_protected_close() itself never reads pos.state, so this never
    mattered for the Fix-2 tests that call it directly. But emergency_stop()
    and manual_close() now require OPEN to acquire close ownership via
    _try_acquire_close_ownership() first -- CLOSING would make every such
    test fail ownership acquisition before ever reaching the broker call.
    """
    return LivePosition(
        direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
        sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
        symbol=symbol, sl_order_id=sl_order_id, state=state,
    )


def _wire_finish_exit_attrs(pilot):
    """
    Attaches every attribute _finish_successful_exit() reads that
    object.__new__(ClaudePilot) skips (no __init__ ran) — shared by every
    pilot-construction helper below so this list isn't itself duplicated
    across them. Only fills in what isn't already set, so callers that
    deliberately pre-set something (e.g. a custom pilot.config) keep it.
    """
    if not hasattr(pilot, "config"):
        pilot.config = MagicMock()
    pilot.config.lot_size = 65
    pilot.config.loss_streak_halve_after = 3
    pilot.config.loss_streak_recovery_trades = 2
    pilot.config.whipsaw_cooldown_min = 30
    pilot._loss_streak = 0
    pilot._size_halved_remaining = 0
    pilot._day_halted_after_loss = False
    pilot._last_loss_time = 0.0
    pilot._last_loss_dir = ""
    if not hasattr(pilot, "_pos_peak_profit_pts"):
        pilot._pos_peak_profit_pts = 0.0
    if not hasattr(pilot, "_pos_max_drawdown_pts"):
        pilot._pos_max_drawdown_pts = 0.0
    if not hasattr(pilot, "_journal"):
        pilot._journal = MagicMock()
    if hasattr(pilot, "trader"):
        # trader.risk is an auto-vivified MagicMock attribute; give
        # daily_pnl a real float so _finish_successful_exit's log line
        # (f"...{self.trader.risk.daily_pnl:+.0f}") doesn't raise trying
        # to format a bare MagicMock -- that exception was harmless (just
        # a noisy caught-and-logged warning, record_trade_close itself
        # still gets called and tracked first), but this keeps test
        # output clean and matches a plausible real value.
        pilot.trader.risk.daily_pnl = 0.0
    return pilot


def _make_pilot_for_close():
    pilot = object.__new__(ClaudePilot)
    pilot._lock = threading.Lock()
    return pilot


call_order = []

# 2a. Close succeeds -> SL cancelled AFTER close, in that order.
pilot_a = _make_pilot_for_close()
pos_a = _make_pos()
trader_a = MagicMock()
trader_a.close_position.side_effect = lambda sym: (call_order.append("close"), {"status": "success"})[1]
pilot_a.trader = trader_a
pilot_a._cancel_exchange_sl = lambda pos: call_order.append("cancel_sl")

ok_a, exc_a = pilot_a._attempt_protected_close(pos_a, "SL")
check("close success -> returns True", ok_a is True and exc_a is None)
check("close success -> exchange SL was cancelled", call_order == ["close", "cancel_sl"])

# 2b. Close NOT confirmed (broker rejection) -> SL must NOT be cancelled.
call_order.clear()
pilot_b = _make_pilot_for_close()
pos_b = _make_pos()
trader_b = MagicMock()
trader_b.close_position.return_value = {"status": "failed", "error": "margin"}
pilot_b.trader = trader_b
pilot_b._cancel_exchange_sl = lambda pos: call_order.append("cancel_sl")

ok_b, exc_b = pilot_b._attempt_protected_close(pos_b, "SL")
check("close rejected -> returns False", ok_b is False and exc_b is None)
check("close rejected -> exchange SL NOT cancelled (still protected)", "cancel_sl" not in call_order)

# 2c. close_position raises -> False, exception captured, SL NOT cancelled.
call_order.clear()
pilot_c = _make_pilot_for_close()
pos_c = _make_pos()
trader_c = MagicMock()
trader_c.close_position.side_effect = RuntimeError("network down")
pilot_c.trader = trader_c
pilot_c._cancel_exchange_sl = lambda pos: call_order.append("cancel_sl")

ok_c, exc_c = pilot_c._attempt_protected_close(pos_c, "SL")
check("close raises -> returns False", ok_c is False)
check("close raises -> exception captured", isinstance(exc_c, RuntimeError))
check("close raises -> exchange SL NOT cancelled", "cancel_sl" not in call_order)

# 2d. PAPER position -> no broker call, treated as success.
pilot_d = _make_pilot_for_close()
pos_d = _make_pos(symbol="PAPER")
trader_d = MagicMock()
pilot_d.trader = trader_d
ok_d, exc_d = pilot_d._attempt_protected_close(pos_d, "TP")
check("PAPER position -> success without broker call", ok_d is True)
check("PAPER position -> close_position never called", not trader_d.close_position.called)

# 2e. Close succeeds but no sl_order_id -> cancel not attempted (nothing to cancel).
call_order.clear()
pilot_e = _make_pilot_for_close()
pos_e = _make_pos(sl_order_id="")
trader_e = MagicMock()
trader_e.close_position.return_value = {"status": "success"}
pilot_e.trader = trader_e
pilot_e._cancel_exchange_sl = lambda pos: call_order.append("cancel_sl")
ok_e, _ = pilot_e._attempt_protected_close(pos_e, "TP")
check("close success, no sl_order_id -> cancel not called", "cancel_sl" not in call_order)


# ═══════════════════════════════════════════════════════════════════════
# [3] emergency_stop() — 5-step sequence
# ═══════════════════════════════════════════════════════════════════════
print("\n[Fix 3] emergency_stop() sequence")


def _make_pilot_for_estop(live_position=None, close_confirms=True, remaining_positions=None):
    pilot = object.__new__(ClaudePilot)
    pilot._lock = threading.Lock()
    pilot._live_position = live_position
    pilot._running = True
    pilot._thread = None
    pilot._monitor_thread = None
    pilot._premium_poller_thread = None
    pilot._reconciliation_thread = None

    trader = MagicMock()
    trader.cancel_all.return_value = {"status": "success"}
    trader.close_position.return_value = (
        {"status": "success"} if close_confirms else {"status": "failed"}
    )
    trader.get_nifty_spot.return_value = 23350.0
    trader.default_qty = 65
    client = MagicMock()
    client.get_open_nifty_positions.return_value = (
        [] if remaining_positions is None else remaining_positions
    )
    trader.client = client
    pilot.trader = trader
    pilot.notifier = MagicMock()
    pilot._cancel_exchange_sl = lambda pos: None
    return _wire_finish_exit_attrs(pilot)


# 3a. Happy path: no open position -> entries blocked, orders cancelled,
#     zero verified immediately, threads stopped, all steps "ok".
pilot_3a = _make_pilot_for_estop(live_position=None)
report_3a = pilot_3a.emergency_stop(reason="test")
check("3a: entries blocked", pilot_3a._emergency_stopped is True)
check("3a: cancel_all called", pilot_3a.trader.cancel_all.called)
check("3a: no step reports failure/error",
      not any("FAILED" in str(v) or "error" in str(v) for v in report_3a["steps"].values()))
check("3a: _running set False (threads told to stop)", pilot_3a._running is False)

# 3b. Open position, close confirms on first attempt -> flattened, zero
#     verified, CLOSED state, exchange SL path exercised (via _attempt_protected_close).
pos_3b = _make_pos()
pilot_3b = _make_pilot_for_estop(live_position=pos_3b, close_confirms=True)
report_3b = pilot_3b.emergency_stop(reason="test")
check("3b: flatten_position ok", report_3b["steps"]["flatten_position"] == "ok")
check("3b: position ends CLOSED", pos_3b.state == PositionState.CLOSED)
check("3b: verify_zero_positions ok", report_3b["steps"]["verify_zero_positions"] == "ok")

# 3c. Open position, close NEVER confirms -> flatten reported FAILED, but
#     entries were still blocked immediately (step 1 unconditional) and
#     threads are STILL stopped (step 5 always runs, even on failure).
pos_3c = _make_pos()
pilot_3c = _make_pilot_for_estop(live_position=pos_3c, close_confirms=False)
pilot_3c._attempt_protected_close = lambda pos, reason: (False, None)  # force perpetual failure, skip real sleeps
# Speed up the retry loop (no real 1s sleeps in a unit test).
_orig_sleep = time.sleep
time.sleep = lambda s: None
try:
    report_3c = pilot_3c.emergency_stop(reason="test")
finally:
    time.sleep = _orig_sleep
check("3c: entries STILL blocked even though flatten failed", pilot_3c._emergency_stopped is True)
check("3c: flatten_position reports FAILED", "FAILED" in report_3c["steps"]["flatten_position"])
check("3c: threads STILL stopped despite flatten failure (step 5 always runs)",
      pilot_3c._running is False)
check("3c: INCOMPLETE alert sent", pilot_3c.notifier.notify_trade.called)

# 3d. Broker still reports an open position after flattening -> zero-check
#     reports FAILED (bounded retries, not infinite).
pos_3d = _make_pos()
pilot_3d = _make_pilot_for_estop(
    live_position=pos_3d, close_confirms=True,
    remaining_positions=[{"symbol": "NIFTY2660923200PE"}],
)
time.sleep = lambda s: None
try:
    report_3d = pilot_3d.emergency_stop(reason="test")
finally:
    time.sleep = _orig_sleep
check("3d: verify_zero_positions reports FAILED when broker still shows a position",
      "FAILED" in report_3d["steps"]["verify_zero_positions"])


# ═══════════════════════════════════════════════════════════════════════
# [4] Monitor loop not gated on market hours (source regression guard)
# ═══════════════════════════════════════════════════════════════════════
print("\n[Fix 4] Position monitor not gated on market hours")

_src = inspect.getsource(ClaudePilot._position_monitor_loop)
# The top-of-loop gate section: from the method start up to the SL/TP-hit
# spot-price fetch. This must contain the `pos is None` and
# `pos.state != PositionState.OPEN` skip-checks but must NOT re-introduce
# an *executed* `_is_market_hours()` gate ahead of them (that was the
# C-scenario bug: a square-off retry pushed past 15:15:00 would be
# silently abandoned). Comment lines are stripped first since the fix's
# own explanatory comment legitimately mentions `_is_market_hours()` by
# name when describing what it no longer does.
_gate_section = _src.split("Spot price (Phase-1")[0]
_gate_code_only = "\n".join(
    line for line in _gate_section.splitlines() if not line.strip().startswith("#")
)
check("monitor loop still skips when pos is None", "pos is None" in _gate_code_only)
check("monitor loop still skips when state != OPEN", "pos.state != PositionState.OPEN" in _gate_code_only)
check("monitor loop top gate does NOT execute _is_market_hours()",
      "_is_market_hours" not in _gate_code_only)


# ═══════════════════════════════════════════════════════════════════════
# [5] _reconcile_runtime — periodic broker reconciliation
# ═══════════════════════════════════════════════════════════════════════
print("\n[Fix 5] _reconcile_runtime periodic reconciliation")


def _make_pilot_for_reconcile(live_position, broker_positions, sl_orders=None, session_valid=True):
    pilot = object.__new__(ClaudePilot)
    pilot._lock = threading.Lock()
    pilot._live_position = live_position
    pilot.notifier = MagicMock()

    td = tempfile.mktemp(suffix=".jsonl")
    open(td, "w").close()
    from core.trade_journal import TradeJournal
    pilot._journal = TradeJournal(path=td)

    client = MagicMock()
    client._session_valid = session_valid
    client.get_open_nifty_positions.return_value = broker_positions
    client.get_open_exchange_sl_orders.return_value = sl_orders or []
    trader = MagicMock()
    trader.client = client
    trader.get_nifty_spot.return_value = 23300.0
    trader.default_qty = 65
    pilot.trader = trader
    pilot._current_atr = 35.0
    return _wire_finish_exit_attrs(pilot)


# 5a. Internal OPEN, broker FLAT -> repaired to CLOSED + alert sent.
pos_5a = LivePosition(
    direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
    sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
    symbol="NIFTY2660923200PE", state=PositionState.OPEN,
)
pilot_5a = _make_pilot_for_reconcile(pos_5a, broker_positions=[])
pilot_5a._reconcile_runtime()
check("5a: internal state corrected to CLOSED", pos_5a.state == PositionState.CLOSED)
check("5a: repair alert sent", pilot_5a.notifier.notify_trade.called)

# 5a-paper. Internal OPEN + PAPER symbol, broker FLAT -> NOT a mismatch
# (DRY_RUN positions never touch the broker by design, so broker-flat is
# expected, not a desync). Regression test for a live bug: prior to this
# fix, Case 1 had no PAPER guard and force-closed every paper position
# within one 15s cycle of entry (observed live: 23/23 occurrences across
# three days of production logs were paper trades).
pos_5a_paper = LivePosition(
    direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
    sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
    symbol="PAPER", state=PositionState.OPEN,
)
pilot_5a_paper = _make_pilot_for_reconcile(pos_5a_paper, broker_positions=[])
pilot_5a_paper._reconcile_runtime()
check("5a-paper: PAPER position left OPEN (not force-closed)",
      pos_5a_paper.state == PositionState.OPEN)
check("5a-paper: no repair alert sent for PAPER", not pilot_5a_paper.notifier.notify_trade.called)

# 5b. Internal FLAT, broker OPEN (orphan) -> adopted via _reconcile_broker_position.
bp_5b = [{"symbol": "NIFTY2660923400CE", "direction": "CALL", "net_qty": 65, "ltp": 80.0}]
pilot_5b = _make_pilot_for_reconcile(None, broker_positions=bp_5b)
pilot_5b._reconcile_runtime()
check("5b: orphan position adopted", pilot_5b._live_position is not None)
check("5b: adopted direction correct",
      pilot_5b._live_position is not None and pilot_5b._live_position.direction == "CALL")
check("5b: repair alert sent", pilot_5b.notifier.notify_trade.called)

# 5c. Both OPEN, exchange SL missing from broker order book -> alert only,
#     internal state untouched (still OPEN), no repair action taken.
pos_5c = LivePosition(
    direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
    sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
    symbol="NIFTY2660923200PE", sl_order_id="SL_GONE", state=PositionState.OPEN,
)
bp_5c = [{"symbol": "NIFTY2660923200PE", "direction": "PUT", "net_qty": 65, "ltp": 80.0}]
pilot_5c = _make_pilot_for_reconcile(pos_5c, broker_positions=bp_5c, sl_orders=[])
pilot_5c._reconcile_runtime()
check("5c: state left untouched (still OPEN, not auto-repaired)", pos_5c.state == PositionState.OPEN)
check("5c: sl_order_id left untouched (no auto re-placement)", pos_5c.sl_order_id == "SL_GONE")
check("5c: alert sent for missing exchange SL", pilot_5c.notifier.notify_trade.called)

# 5d. No mismatch (internal OPEN, broker OPEN, SL present) -> no alert.
pos_5d = LivePosition(
    direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
    sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
    symbol="NIFTY2660923200PE", sl_order_id="SL_OK", state=PositionState.OPEN,
)
bp_5d = [{"symbol": "NIFTY2660923200PE", "direction": "PUT", "net_qty": 65, "ltp": 80.0}]
sl_5d = [{"symbol": "NIFTY2660923200PE", "order_id": "SL_OK", "trigger_price": 23470.0}]
pilot_5d = _make_pilot_for_reconcile(pos_5d, broker_positions=bp_5d, sl_orders=sl_5d)
pilot_5d._reconcile_runtime()
check("5d: no mismatch -> no alert sent", not pilot_5d.notifier.notify_trade.called)
check("5d: state unchanged", pos_5d.state == PositionState.OPEN)

# 5e. Session not ready -> no-op, no broker call made.
pilot_5e = _make_pilot_for_reconcile(None, broker_positions=[], session_valid=False)
pilot_5e._reconcile_runtime()
check("5e: not ready -> no broker call made", not pilot_5e.trader.client.get_open_nifty_positions.called)

# 5f. Broker query raises -> fail-open, no exception propagates.
pilot_5f = _make_pilot_for_reconcile(None, broker_positions=[])
pilot_5f.trader.client.get_open_nifty_positions.side_effect = RuntimeError("broker down")
raised_5f = False
try:
    pilot_5f._reconcile_runtime()
except Exception:
    raised_5f = True
check("5f: exception does not propagate (fail-open)", not raised_5f)


# ═══════════════════════════════════════════════════════════════════════
# [6] RiskManager — atomic + fsync + fail-safe persistence
# ═══════════════════════════════════════════════════════════════════════
print("\n[Fix 6] RiskManager fail-safe persistence")

from security import RiskManager

# 6a. Normal save produces a valid, complete JSON file (atomic write worked).
tmpdir = tempfile.mkdtemp()
state_path = str(Path(tmpdir) / "risk_state.json")
rm = RiskManager(
    max_daily_loss=5000, max_loss_per_trade=1000, max_open_positions=1,
    state_file=state_path,
)
rm.record_trade_open()
check("6a: state file exists after save", Path(state_path).exists())
check("6a: no leftover .tmp file after atomic replace", not Path(state_path).with_suffix(".tmp").exists())
loaded = json.loads(Path(state_path).read_text())
check("6a: written state round-trips correctly", loaded["open_positions"] == 1)

# 6b. Persistent (3x consecutive) write failure -> lockout engages, entries
#     disabled, Telegram alert sent.
notifier_6b = MagicMock()
rm2 = RiskManager(
    max_daily_loss=5000, max_loss_per_trade=1000, max_open_positions=1,
    state_file=str(Path(tmpdir) / "risk_state_2.json"),
    notifier=notifier_6b,
)
with patch("builtins.open", side_effect=OSError("disk full")):
    rm2.record_trade_open()   # failure 1
    check("6b: 1 failure does NOT yet lock out (transient blip tolerated)",
          rm2._persistence_locked_out is False)
    rm2.record_trade_open()   # failure 2
    check("6b: 2 failures still not locked out", rm2._persistence_locked_out is False)
    rm2.record_trade_open()   # failure 3 -> threshold hit
    check("6b: 3rd consecutive failure trips the lockout", rm2._persistence_locked_out is True)

check("6b: Telegram alert sent on lockout", notifier_6b.notify.called)
allowed, msg = rm2.can_open_trade()
check("6b: can_open_trade() now blocks new entries", allowed is False)
check("6b: block message explains why", "persistence" in msg.lower())

# 6c. Lockout does NOT auto-clear on a later successful save (manual
#     recovery required, unlike the self-healing strategy-health pause).
rm2.record_trade_open()   # this call succeeds (open() no longer patched)
check("6c: failure streak resets on a real success", rm2._persist_failures == 0)
check("6c: lockout itself persists across a later successful save",
      rm2._persistence_locked_out is True)
allowed2, _ = rm2.can_open_trade()
check("6c: entries remain blocked after the successful save", allowed2 is False)


# ═══════════════════════════════════════════════════════════════════════
# [Independent-verification fixes, 2026-07-21] — regressions found by the
# verification-only audit of fixes 1-6 above, and their repairs:
#   [IV-1] _clear_live_position() canonical helper (critical)
#   [IV-2] cancel_all_orders() routed through _call_with_retry
#   [IV-3] cancel_exchange_sl/update_exchange_sl use is_order=True
#   [IV-4] manual /close routed through _attempt_protected_close
#   [IV-5] SIGTERM/SIGINT invokes emergency_stop() when a position is live
#   [IV-6] sticky TIME_EXIT survives an hour rollover
#   [IV-7] persistence lockout survives a restart
# ═══════════════════════════════════════════════════════════════════════
print("\n[IV-1] _clear_live_position — _live_position becomes None")

from core.claude_pilot import _sticky_time_exit

# 1a. emergency_stop() on a real open position -> _live_position is None
#     afterward (not just pos.state == CLOSED).
pos_iv1a = _make_pos()
pilot_iv1a = _make_pilot_for_estop(live_position=pos_iv1a, close_confirms=True)
pilot_iv1a.emergency_stop(reason="test")
check("IV-1a: _live_position is None after emergency_stop() flattens",
      pilot_iv1a._live_position is None)
check("IV-1a: the closed position object itself still reports CLOSED",
      pos_iv1a.state == PositionState.CLOSED)

# 1b. emergency_stop() with no open position -> still ends with
#     _live_position None (nothing to clear, trivially satisfied).
pilot_iv1b = _make_pilot_for_estop(live_position=None)
pilot_iv1b.emergency_stop(reason="test")
check("IV-1b: _live_position stays None when there was nothing to flatten",
      pilot_iv1b._live_position is None)

# 1c. _reconcile_runtime() Case 1 repair -> _live_position is None
#     afterward (not just pos.state == CLOSED).
pos_iv1c = LivePosition(
    direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
    sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
    symbol="NIFTY2660923200PE", state=PositionState.OPEN,
)
pilot_iv1c = _make_pilot_for_reconcile(pos_iv1c, broker_positions=[])
pilot_iv1c._reconcile_runtime()
check("IV-1c: _live_position is None after reconciliation Case-1 repair",
      pilot_iv1c._live_position is None)
check("IV-1c: the repaired position object itself still reports CLOSED",
      pos_iv1c.state == PositionState.CLOSED)

# 1d. The regression this bug caused: entry gate unblocked afterward. The
#     real gate is `self._live_position is not None` -- prove it now reads
#     False (unblocked) rather than True (permanently stuck) after 1a/1c.
check("IV-1d: entry gate reads unblocked after emergency_stop (1a)",
      pilot_iv1a._live_position is None)
check("IV-1d: entry gate reads unblocked after reconcile repair (1c)",
      pilot_iv1c._live_position is None)

# 1e. _clear_live_position does not clobber a DIFFERENT position that has
#     since replaced the one being closed (identity-guarded).
pos_iv1e_old = _make_pos(symbol="OLD")
pos_iv1e_new = _make_pos(symbol="NEW")
pilot_iv1e = _make_pilot_for_close()
pilot_iv1e._live_position = pos_iv1e_new   # a different position is now live
cleared = pilot_iv1e._clear_live_position(pos_iv1e_old)
check("IV-1e: clearing a stale pos object returns False (no-op on _live_position)",
      cleared is False)
check("IV-1e: the stale object itself is still marked CLOSED",
      pos_iv1e_old.state == PositionState.CLOSED)
check("IV-1e: the actually-current position is untouched",
      pilot_iv1e._live_position is pos_iv1e_new)


print("\n[IV-2] cancel_all_orders() routed through _call_with_retry")

from core.kotak_neo_client import KotakNeoClient as _KNC

client_iv2 = object.__new__(_KNC)
client_iv2._rate_gate = lambda *a, **kw: None
client_iv2._neo = MagicMock()
client_iv2._neo.order_report.return_value = {
    "data": [{"nOrdNo": "OID1", "ordSt": "open"}]
}
_retry_calls = []


def _fake_call_with_retry(fn, *args, **kwargs):
    _retry_calls.append((fn, kwargs))
    return {"stat": "Ok"}


client_iv2._call_with_retry = _fake_call_with_retry
result_iv2 = client_iv2.cancel_all_orders()
check("IV-2: cancel_all_orders routes the cancel through _call_with_retry",
      len(_retry_calls) == 1 and _retry_calls[0][0] == client_iv2._neo.cancel_order)
check("IV-2: the routed call uses is_order=True",
      _retry_calls[0][1].get("is_order") is True)
check("IV-2: order_id was forwarded correctly", _retry_calls[0][1].get("order_id") == "OID1")
check("IV-2: reports success when the retry-wrapped call succeeds",
      result_iv2["status"] == "success" and "OID1" in result_iv2["cancelled"])


print("\n[IV-3] cancel_exchange_sl / update_exchange_sl use is_order=True")

client_iv3 = object.__new__(_KNC)
client_iv3._rate_gate = lambda *a, **kw: None
client_iv3._neo = MagicMock()
_retry_calls_3 = []
client_iv3._call_with_retry = lambda fn, *a, **kw: (_retry_calls_3.append((fn, kw)), {"stat": "Ok"})[1]

client_iv3.cancel_exchange_sl("SL123")
check("IV-3: cancel_exchange_sl routes through _neo.cancel_order",
      _retry_calls_3[0][0] == client_iv3._neo.cancel_order)
check("IV-3: cancel_exchange_sl passes is_order=True",
      _retry_calls_3[0][1].get("is_order") is True)

_retry_calls_3.clear()
client_iv3.update_exchange_sl("SL123", new_trigger=100.0, new_limit=95.0, qty=65, symbol="NIFTY2660923200PE")
check("IV-3: update_exchange_sl routes through _neo.modify_order",
      _retry_calls_3[0][0] == client_iv3._neo.modify_order)
check("IV-3: update_exchange_sl passes is_order=True",
      _retry_calls_3[0][1].get("is_order") is True)

# A timeout on either must raise BrokerCallTimeout (bounded, not blindly
# retried) -- prove it end-to-end through the REAL _call_with_retry this
# time, not the stub above.
client_iv3b = object.__new__(_KNC)
client_iv3b._rate_gate = lambda *a, **kw: None
client_iv3b._notifier = None
client_iv3b._consecutive_failures = 0
client_iv3b._unhealthy_alerted = False
_sl_cancel_calls = []


def _sl_cancel_timeout(**kw):
    _sl_cancel_calls.append(1)
    raise requests.exceptions.Timeout("read timed out")


client_iv3b._neo = MagicMock()
client_iv3b._neo.cancel_order = _sl_cancel_timeout
ok_iv3b = client_iv3b.cancel_exchange_sl("SL999")
check("IV-3b: a real timeout is caught (returns False, never raises out of cancel_exchange_sl)",
      ok_iv3b is False)
check("IV-3b: NOT blindly retried -- called exactly once", len(_sl_cancel_calls) == 1)


print("\n[IV-4] Manual /close routed through _attempt_protected_close")


def _make_pilot_for_manual_close(live_position, close_confirms=True):
    pilot = object.__new__(ClaudePilot)
    pilot._lock = threading.Lock()
    pilot._live_position = live_position
    pilot._pos_peak_profit_pts = 0.0
    pilot._pos_max_drawdown_pts = 0.0
    pilot.config = MagicMock(lot_size=65)

    trader = MagicMock()
    trader.close_position.return_value = {"status": "success" if close_confirms else "failed"}
    trader.get_nifty_spot.return_value = 23350.0
    trader.default_qty = 65
    pilot.trader = trader
    pilot.notifier = MagicMock()
    pilot._journal = MagicMock()
    pilot._cancel_exchange_sl = MagicMock()
    return _wire_finish_exit_attrs(pilot)


# 4a. Happy path: broker confirms -> exchange SL cancelled, journal
#     written, RiskManager updated, _live_position cleared.
pos_iv4a = _make_pos(sl_order_id="SL_A")
pilot_iv4a = _make_pilot_for_manual_close(pos_iv4a, close_confirms=True)
result_iv4a = pilot_iv4a.manual_close()
check("IV-4a: reports success", result_iv4a["status"] == "success")
check("IV-4a: broker close_position was called (protected-close path, not a raw bypass)",
      pilot_iv4a.trader.close_position.called)
check("IV-4a: exchange SL was cancelled", pilot_iv4a._cancel_exchange_sl.called)
check("IV-4a: journal exit was recorded", pilot_iv4a._journal.record_exit.called)
check("IV-4a: RiskManager P&L was recorded", pilot_iv4a.trader.risk.record_trade_close.called)
check("IV-4a: _live_position cleared (bot left synchronized)",
      pilot_iv4a._live_position is None)

# 4b. No tracked position -> clean error, no broker call at all (must NOT
#     fall back to a raw untracked close -- that would reintroduce the bug).
pilot_iv4b = _make_pilot_for_manual_close(None)
result_iv4b = pilot_iv4b.manual_close(symbol="NIFTY2660923200PE")
check("IV-4b: no tracked position -> error, not a bypassed broker call",
      result_iv4b["status"] == "error")
check("IV-4b: broker close_position never called", not pilot_iv4b.trader.close_position.called)

# 4c. Symbol mismatch -> rejected before touching the broker.
pos_iv4c = _make_pos(symbol="NIFTY2660923200PE")
pilot_iv4c = _make_pilot_for_manual_close(pos_iv4c)
result_iv4c = pilot_iv4c.manual_close(symbol="NIFTY2660923400CE")
check("IV-4c: symbol mismatch -> error", result_iv4c["status"] == "error")
check("IV-4c: broker close_position never called on mismatch",
      not pilot_iv4c.trader.close_position.called)

# 4d. Broker never confirms -> exchange SL NOT cancelled (still protected),
#     position left OPEN (still monitored), _live_position NOT cleared.
pos_iv4d = _make_pos(sl_order_id="SL_D")
pilot_iv4d = _make_pilot_for_manual_close(pos_iv4d, close_confirms=False)
_orig_sleep_iv4 = time.sleep
time.sleep = lambda s: None
try:
    result_iv4d = pilot_iv4d.manual_close()
finally:
    time.sleep = _orig_sleep_iv4
check("IV-4d: unconfirmed close -> error result", result_iv4d["status"] == "error")
check("IV-4d: exchange SL NOT cancelled (still protected)", not pilot_iv4d._cancel_exchange_sl.called)
check("IV-4d: position left OPEN, not stuck in CLOSING", pos_iv4d.state == PositionState.OPEN)
check("IV-4d: _live_position NOT cleared (still tracked for retry/monitoring)",
      pilot_iv4d._live_position is pos_iv4d)


print("\n[IV-5] SIGTERM/SIGINT invokes emergency_stop() when a position is live")

_main_src = Path(__file__).parent.parent.joinpath("main.py").read_text(encoding="utf-8")
_shutdown_start = _main_src.index("def _graceful_shutdown():")
_shutdown_end = _main_src.index("def _signal_handler(")
_shutdown_src = _main_src[_shutdown_start:_shutdown_end]
check("IV-5: _graceful_shutdown checks _live_position before deciding",
      "_live_position" in _shutdown_src)
check("IV-5: _graceful_shutdown calls emergency_stop() somewhere in that branch",
      "emergency_stop(" in _shutdown_src)
check("IV-5: plain stop() is still reachable for the already-flat case",
      "_pilot.stop()" in _shutdown_src)


print("\n[IV-6] Sticky TIME_EXIT survives an hour rollover")

check("IV-6a: not yet triggered, not latched -> stays False",
      _sticky_time_exit(False, False) is False)
check("IV-6b: triggers this cycle (15:14:30 reached) -> True",
      _sticky_time_exit(True, False) is True)
check("IV-6c: hour rolled 15:59->16:00 (raw goes False again) but was "
      "already latched -> STAYS True",
      _sticky_time_exit(False, True) is True)
check("IV-6d: still latched and still raw-true -> True", _sticky_time_exit(True, True) is True)

# Simulate the exact multi-cycle sequence a real retry would go through.
_latched = False
for _raw in (False, False, True, True, False, False):   # last two = post-16:00 cycles
    _latched = _sticky_time_exit(_raw, _latched)
check("IV-6e: full cycle simulation ends latched=True after crossing the hour boundary",
      _latched is True)


print("\n[IV-7] Persistence lockout survives a restart")

tmpdir_iv7 = tempfile.mkdtemp()
state_path_iv7 = str(Path(tmpdir_iv7) / "risk_state_iv7.json")

rm_iv7 = RiskManager(
    max_daily_loss=5000, max_loss_per_trade=1000, max_open_positions=1,
    state_file=state_path_iv7,
)
with patch("builtins.open", side_effect=OSError("disk full")):
    rm_iv7.record_trade_open()
    rm_iv7.record_trade_open()
    rm_iv7.record_trade_open()
check("IV-7a: lockout tripped in the original instance", rm_iv7._persistence_locked_out is True)

# Force a successful save now so the lockout gets written to disk (the
# tripping save itself couldn't succeed -- disk was "full" at that moment).
rm_iv7._save_state()
loaded_iv7 = json.loads(Path(state_path_iv7).read_text())
check("IV-7b: persistence_locked_out was written to risk_state.json",
      loaded_iv7.get("persistence_locked_out") is True)

# Simulate a process restart: brand-new RiskManager instance pointed at the
# same state file.
rm_iv7_restarted = RiskManager(
    max_daily_loss=5000, max_loss_per_trade=1000, max_open_positions=1,
    state_file=state_path_iv7,
)
check("IV-7c: lockout is RESTORED on a fresh instance (restart does not clear it)",
      rm_iv7_restarted._persistence_locked_out is True)
allowed_iv7, msg_iv7 = rm_iv7_restarted.can_open_trade()
check("IV-7d: entries remain blocked immediately after restart",
      allowed_iv7 is False)


# ═══════════════════════════════════════════════════════════════════════
# [Round-3 CRITICAL fix] Atomic close-ownership — real concurrency tests
# ═══════════════════════════════════════════════════════════════════════
# These tests use real threading.Thread + threading.Barrier so the race is
# GENUINE (both sides actually run concurrently), not simulated by call
# ordering. The broker close itself sleeps briefly so both contenders are
# truly in flight before either can finish, closing the exact window the
# round-3 audit proved was exploitable.
#
# "monitor" is exercised via _simulate_monitor_close (below), which calls
# the identical production sequence the real _position_monitor_loop's
# CLOSING block calls -- _try_acquire_close_ownership, then
# _attempt_protected_close, then (on success) the same RiskManager/journal/
# _clear_live_position calls the real loop makes. Only the surrounding
# SL/TP-detection plumbing (spot fetch, threshold comparison) is omitted,
# since that's orthogonal to the ownership race being proven here and is
# already covered by the non-concurrency tests above.
print("\n[Round-3 CRITICAL] Atomic close ownership under real concurrency")


def _make_concurrent_pilot(close_delay=0.05):
    """A pilot wired for real multi-threaded close-ownership races."""
    pos = LivePosition(
        direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
        sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
        symbol="NIFTY2660923200PE", sl_order_id="SL_CONC", state=PositionState.OPEN,
    )
    pilot = object.__new__(ClaudePilot)
    pilot._lock = threading.Lock()
    pilot._live_position = pos
    pilot._running = True
    pilot._thread = None
    pilot._monitor_thread = None
    pilot._premium_poller_thread = None
    pilot._reconciliation_thread = None
    pilot._pos_peak_profit_pts = 0.0
    pilot._pos_max_drawdown_pts = 0.0
    pilot.config = MagicMock(lot_size=65)

    broker_close_calls = []
    broker_lock = threading.Lock()

    def _slow_close(symbol):
        # Real broker latency simulation -- both racing threads must
        # actually be in flight concurrently, not just serialized by
        # scheduling luck.
        time.sleep(close_delay)
        with broker_lock:
            broker_close_calls.append(symbol)
        return {"status": "success"}

    trader = MagicMock()
    trader.close_position.side_effect = _slow_close
    trader.cancel_all.return_value = {"status": "success"}
    trader.get_nifty_spot.return_value = 23350.0
    trader.default_qty = 65
    client = MagicMock()
    client.get_open_nifty_positions.return_value = []
    trader.client = client
    pilot.trader = trader

    pilot.notifier = MagicMock()
    pilot._journal = MagicMock()
    pilot._cancel_exchange_sl = MagicMock()
    _wire_finish_exit_attrs(pilot)

    return pilot, pos, broker_close_calls


def _simulate_monitor_close(pilot, pos, reason="SL"):
    """
    Mirrors EXACTLY the real _position_monitor_loop's CLOSING-decision
    sequence: _try_acquire_close_ownership -> _attempt_protected_close ->
    on success, _finish_successful_exit() (round-3 canonicalization fix —
    the SAME single helper production code calls); on failure/loss, no
    further action. Every call in this function is the real production
    method -- only the SL/TP threshold-detection plumbing that decides
    *whether* to call this is omitted (irrelevant to the ownership race
    itself).
    """
    if not pilot._try_acquire_close_ownership(pos, "monitor"):
        return False
    confirmed, _exc = pilot._attempt_protected_close(pos, reason)
    if not confirmed:
        pilot._release_close_ownership(pos, "monitor")
        return False
    pnl_pts = 5.0
    pilot._finish_successful_exit(pos, reason, 23350.0, pnl_pts)
    return True


def _run_concurrently(*fns):
    """Run each zero-arg callable on its own thread, released simultaneously
    via a Barrier so they genuinely race rather than run sequentially."""
    barrier = threading.Barrier(len(fns))
    results = [None] * len(fns)
    errors = [None] * len(fns)

    def _wrap(i, fn):
        try:
            barrier.wait(timeout=5)
            results[i] = fn()
        except Exception as e:
            errors[i] = e

    threads = [threading.Thread(target=_wrap, args=(i, fn)) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return results, errors


# ── 1. monitor vs manual_close ──────────────────────────────────────────
pilot1, pos1, calls1 = _make_concurrent_pilot()
results1, errors1 = _run_concurrently(
    lambda: _simulate_monitor_close(pilot1, pos1, "SL"),
    lambda: pilot1.manual_close(),
)
check("1: no exceptions raised by either contender", errors1 == [None, None])
check("1: exactly one broker close call happened", len(calls1) == 1)
check("1: exactly one RiskManager.record_trade_close call", pilot1.trader.risk.record_trade_close.call_count == 1)
check("1: exactly one journal.record_exit call", pilot1._journal.record_exit.call_count == 1)
check("1: _live_position cleared exactly once (ends None)", pilot1._live_position is None)
check("1: exactly one of the two contenders reports success",
      (results1[0] is True) != (results1[1] is not None and results1[1].get("status") == "success"))

# ── 2. monitor vs emergency_stop ────────────────────────────────────────
# emergency_stop() does not perform RiskManager/journal accounting on its
# own success path (a separately-tracked, already-disclosed gap, NOT part
# of this round's fix) -- so record_trade_close can legitimately be 0 (if
# emergency_stop wins) or 1 (if monitor wins), but NEVER more than 1. The
# decisive proof against the round-3 race is broker_close_calls == 1.
pilot2, pos2, calls2 = _make_concurrent_pilot()
results2, errors2 = _run_concurrently(
    lambda: _simulate_monitor_close(pilot2, pos2, "SL"),
    lambda: pilot2.emergency_stop(reason="race_test"),
)
check("2: no exceptions raised by either contender", errors2 == [None, None])
check("2: exactly one broker close call happened (no double execution)", len(calls2) == 1)
check("2: RiskManager.record_trade_close never double-counted (<=1)",
      pilot2.trader.risk.record_trade_close.call_count <= 1)
check("2: _live_position cleared (ends None)", pilot2._live_position is None)

# ── 3. manual_close vs emergency_stop ───────────────────────────────────
pilot3, pos3, calls3 = _make_concurrent_pilot()
results3, errors3 = _run_concurrently(
    lambda: pilot3.manual_close(),
    lambda: pilot3.emergency_stop(reason="race_test"),
)
check("3: no exceptions raised by either contender", errors3 == [None, None])
check("3: exactly one broker close call happened", len(calls3) == 1)
check("3: RiskManager.record_trade_close never double-counted (<=1)",
      pilot3.trader.risk.record_trade_close.call_count <= 1)
check("3: _live_position cleared (ends None)", pilot3._live_position is None)

# ── 4. multiple concurrent manual_close calls ───────────────────────────
pilot4, pos4, calls4 = _make_concurrent_pilot()
results4, errors4 = _run_concurrently(
    lambda: pilot4.manual_close(),
    lambda: pilot4.manual_close(),
)
check("4: no exceptions raised by either call", errors4 == [None, None])
check("4: exactly one broker close call happened", len(calls4) == 1)
check("4: exactly one RiskManager.record_trade_close call", pilot4.trader.risk.record_trade_close.call_count == 1)
check("4: exactly one journal.record_exit call", pilot4._journal.record_exit.call_count == 1)
check("4: at least one of the two calls reports an error (lost the race)",
      any(r is not None and r.get("status") == "error" for r in results4))
check("4: _live_position cleared (ends None)", pilot4._live_position is None)

# ── 5. multiple concurrent emergency_stop calls ─────────────────────────
pilot5, pos5, calls5 = _make_concurrent_pilot()
results5, errors5 = _run_concurrently(
    lambda: pilot5.emergency_stop(reason="race_a"),
    lambda: pilot5.emergency_stop(reason="race_b"),
)
check("5: no exceptions raised by either call", errors5 == [None, None])
check("5: exactly one broker close call happened", len(calls5) == 1)
check("5: both calls returned a report dict", all(isinstance(r, dict) for r in results5))
check("5: _live_position cleared (ends None)", pilot5._live_position is None)

# ── 6. ownership already acquired ───────────────────────────────────────
pilot6, pos6, calls6 = _make_concurrent_pilot()
won6 = pilot6._try_acquire_close_ownership(pos6, "first_owner")
lost6 = pilot6._try_acquire_close_ownership(pos6, "second_owner")
check("6: first acquisition succeeds", won6 is True)
check("6: second acquisition on an already-owned position fails", lost6 is False)
check("6: close_owner records the winner", pos6.close_owner == "first_owner")
check("6: state transitioned to CLOSING", pos6.state == PositionState.CLOSING)

# ── 7/8/9. losing thread performs ZERO accounting/broker-calls/cleanup ──
# Deterministic (non-racy) version: pre-acquire ownership in the main
# thread so the subsequent call is GUARANTEED to lose, then assert it did
# nothing whatsoever.
pilot789, pos789, calls789 = _make_concurrent_pilot()
pilot789._try_acquire_close_ownership(pos789, "pre_existing_owner")

result789 = pilot789.manual_close()
check("7: losing manual_close performs ZERO RiskManager calls",
      not pilot789.trader.risk.record_trade_close.called)
check("8: losing manual_close performs ZERO broker close calls",
      len(calls789) == 0)
check("9: losing manual_close performs ZERO cleanup (journal/SL-cancel/clear)",
      not pilot789._journal.record_exit.called
      and not pilot789._cancel_exchange_sl.called
      and pilot789._live_position is pos789)
check("7/8/9: losing manual_close reports an error", result789.get("status") == "error")

# Same guarantee for emergency_stop's flatten step when it loses ownership.
pilot789b, pos789b, calls789b = _make_concurrent_pilot()
pilot789b._try_acquire_close_ownership(pos789b, "pre_existing_owner")
report789b = pilot789b.emergency_stop(reason="should_not_own")
check("7b: losing emergency_stop performs ZERO broker close calls",
      len(calls789b) == 0)
check("8b: losing emergency_stop's flatten step reports it was skipped, not ok/failed",
      "skipped" in report789b["steps"]["flatten_position"])
check("9b: losing emergency_stop does not clear the pre-existing owner's position",
      pos789b.close_owner == "pre_existing_owner" and pos789b.state == PositionState.CLOSING)

# ── 10/11/12/13. exactly-one assertions (re-stated explicitly from the
#    race tests above, for direct traceability to the requirement list) ──
check("10: exactly one RiskManager.record_trade_close() across test 1", pilot1.trader.risk.record_trade_close.call_count == 1)
check("11: exactly one journal.record_exit() across test 1", pilot1._journal.record_exit.call_count == 1)
check("12: exactly one _clear_live_position() effect across test 1 (_live_position is None)", pilot1._live_position is None)
check("13: exactly one broker close() across test 1", len(calls1) == 1)

# ── 14. stress test with many racing threads ────────────────────────────
pilot14, pos14, calls14 = _make_concurrent_pilot(close_delay=0.02)
_N_RACERS = 20
_fns = []
for _i in range(_N_RACERS):
    if _i % 3 == 0:
        _fns.append(lambda: _simulate_monitor_close(pilot14, pos14, "SL"))
    elif _i % 3 == 1:
        _fns.append(lambda: pilot14.manual_close())
    else:
        _fns.append(lambda: pilot14.emergency_stop(reason="stress"))
results14, errors14 = _run_concurrently(*_fns)
check("14: no exceptions across 20 racing threads", all(e is None for e in errors14))
check("14: exactly one broker close call across all 20 contenders", len(calls14) == 1)
check("14: RiskManager.record_trade_close never double-counted under stress",
      pilot14.trader.risk.record_trade_close.call_count <= 1)
check("14: journal.record_exit never double-written under stress",
      pilot14._journal.record_exit.call_count <= 1)
check("14: _live_position cleared exactly once under stress (ends None)",
      pilot14._live_position is None)


# ═══════════════════════════════════════════════════════════════════════
# [Canonicalization] _finish_successful_exit() — ONE function owns every
# post-close side effect (accounting, journal, analytics, notification,
# cleanup) for ALL FOUR confirmed-exit sources.
# ═══════════════════════════════════════════════════════════════════════
print("\n[Canonicalization] _finish_successful_exit — single post-close authority")

# ── 1. Normal exit (via the monitor-loop-mirroring helper — see its
#      docstring: every call inside it is the real production method) ──
pilot_c1, pos_c1, calls_c1 = _make_concurrent_pilot()
won_c1 = _simulate_monitor_close(pilot_c1, pos_c1, "SL")
check("1 (normal exit): close succeeded", won_c1 is True)
check("1 (normal exit): RiskManager updated", pilot_c1.trader.risk.record_trade_close.called)
check("1 (normal exit): journal updated", pilot_c1._journal.record_exit.called)
check("1 (normal exit): notification sent with EXIT_SL action",
      pilot_c1.notifier.notify_trade.call_args.kwargs.get("action") == "EXIT_SL")
check("1 (normal exit): _live_position cleared", pilot_c1._live_position is None)

# ── 2. Manual close ──────────────────────────────────────────────────
pos_c2 = _make_pos(sl_order_id="SL_C2")
pilot_c2 = _make_pilot_for_manual_close(pos_c2, close_confirms=True)
result_c2 = pilot_c2.manual_close()
check("2 (manual close): reports success", result_c2["status"] == "success")
check("2 (manual close): RiskManager updated", pilot_c2.trader.risk.record_trade_close.called)
check("2 (manual close): journal updated", pilot_c2._journal.record_exit.called)
check("2 (manual close): notification uses the SAME unified EXIT_{reason} "
      "naming as every other exit source (round-3: no more separate "
      "'MANUAL_CLOSE' action tag duplicating this logic)",
      pilot_c2.notifier.notify_trade.call_args.kwargs.get("action") == "EXIT_MANUAL_CLOSE")
check("2 (manual close): _live_position cleared", pilot_c2._live_position is None)

# ── 3. Emergency stop (previously disclosed gap: emergency_stop() never
#      touched RiskManager/journal at all -- canonicalization fixes this
#      as a direct consequence of routing through the shared helper) ──
pos_c3 = _make_pos(sl_order_id="SL_C3")
pilot_c3 = _make_pilot_for_estop(live_position=pos_c3, close_confirms=True)
report_c3 = pilot_c3.emergency_stop(reason="test")
check("3 (emergency stop): flatten_position ok", report_c3["steps"]["flatten_position"] == "ok")
check("3 (emergency stop): RiskManager NOW updated (previously-disclosed gap fixed)",
      pilot_c3.trader.risk.record_trade_close.called)
check("3 (emergency stop): journal NOW updated (previously-disclosed gap fixed)",
      pilot_c3._journal.record_exit.called)
# emergency_stop() sends TWO notifications: the per-position exit
# notification (via _finish_successful_exit, action=EXIT_EMERGENCY_STOP)
# and its own overall operation-status summary afterward (action=
# EMERGENCY_STOP) -- these are different concerns (one position, one
# whole operation), not a duplicate of the same notification, so check
# the full call list rather than just the last call.
_c3_actions = [c.kwargs.get("action") for c in pilot_c3.notifier.notify_trade.call_args_list]
check("3 (emergency stop): per-position notification uses the unified EXIT_{reason} naming",
      "EXIT_EMERGENCY_STOP" in _c3_actions)
check("3 (emergency stop): _live_position cleared", pilot_c3._live_position is None)

# ── 4. Reconciliation repair (unknown-outcome mode — see design decision
#      A vs B in _finish_successful_exit's docstring) ──────────────────
pos_c4 = LivePosition(
    direction="PUT", entry_price=23400.0, entry_time=time.monotonic(),
    sl_price=23470.0, tp_price=23260.0, initial_sl=23470.0,
    symbol="NIFTY2660923200PE", state=PositionState.OPEN,
)
pilot_c4 = _make_pilot_for_reconcile(pos_c4, broker_positions=[])
pilot_c4._reconcile_runtime()
check("4 (reconciliation): RiskManager decremented with pnl=0.0 (open_positions "
      "fixed, daily_pnl NOT touched with a guess)",
      pilot_c4.trader.risk.record_trade_close.call_args.kwargs.get("pnl") == 0.0)
check("4 (reconciliation): _live_position cleared", pilot_c4._live_position is None)
check("4 (reconciliation): notification sent", pilot_c4.notifier.notify_trade.called)

# ── 5. Duplicate calls impossible (integration-level: two sequential
#      manual_close() calls on the same already-cleared position) ──────
pos_c5 = _make_pos(sl_order_id="SL_C5")
pilot_c5 = _make_pilot_for_manual_close(pos_c5, close_confirms=True)
result_c5a = pilot_c5.manual_close()
result_c5b = pilot_c5.manual_close()   # position already cleared by the first call
check("5: first manual_close succeeds", result_c5a["status"] == "success")
check("5: second manual_close on the same (now-cleared) position cannot "
      "duplicate anything -- no tracked position left to act on",
      result_c5b["status"] == "error")
check("5: RiskManager touched exactly once across both calls",
      pilot_c5.trader.risk.record_trade_close.call_count == 1)
check("5: journal touched exactly once across both calls",
      pilot_c5._journal.record_exit.call_count == 1)

# ── 6. Accounting executed exactly once (direct, deterministic) ────────
check("6: RiskManager.record_trade_close call_count == 1 (test 2)",
      pilot_c2.trader.risk.record_trade_close.call_count == 1)

# ── 7. Journal executed exactly once (direct, deterministic) ───────────
check("7: journal.record_exit call_count == 1 (test 2)",
      pilot_c2._journal.record_exit.call_count == 1)

# ── 8. Cleanup executed exactly once (direct, deterministic) ───────────
check("8: _live_position transitions non-None -> None exactly once (test 2)",
      pilot_c2._live_position is None and pos_c2.state == PositionState.CLOSED)

# ── 9. Notification executed exactly once (direct, deterministic) ──────
check("9: notifier.notify_trade call_count == 1 (test 2)",
      pilot_c2.notifier.notify_trade.call_count == 1)

# ── 10. RiskManager values correct ──────────────────────────────────────
pos_c10 = _make_pos(sl_order_id="SL_C10")
pilot_c10 = _make_pilot_for_manual_close(pos_c10, close_confirms=True)
pilot_c10.trader.get_nifty_spot.return_value = pos_c10.entry_price + 10.0   # PUT: +10 spot = -10pts
pilot_c10.manual_close()
_expected_pnl_pts = -10.0   # PUT direction, spot moved against it by 10pts
_expected_pnl_rupees = _expected_pnl_pts * 65
_actual_pnl = pilot_c10.trader.risk.record_trade_close.call_args.kwargs.get("pnl")
check(f"10: RiskManager receives the correctly-signed/scaled P&L "
      f"(expected {_expected_pnl_rupees}, got {_actual_pnl})",
      _actual_pnl is not None and abs(_actual_pnl - _expected_pnl_rupees) < 0.01)

# ── 11. No accounting on failed close ───────────────────────────────────
pos_c11 = _make_pos(sl_order_id="SL_C11")
pilot_c11 = _make_pilot_for_manual_close(pos_c11, close_confirms=False)
_orig_sleep_c11 = time.sleep
time.sleep = lambda s: None
try:
    result_c11 = pilot_c11.manual_close()
finally:
    time.sleep = _orig_sleep_c11
check("11: failed close reports error", result_c11["status"] == "error")
check("11: NO RiskManager call on a failed (unconfirmed) close",
      not pilot_c11.trader.risk.record_trade_close.called)
check("11: NO journal call on a failed (unconfirmed) close",
      not pilot_c11._journal.record_exit.called)
check("11: _live_position NOT cleared on a failed close",
      pilot_c11._live_position is pos_c11)

# ── 12. No accounting on ownership loss ─────────────────────────────────
pos_c12 = _make_pos(sl_order_id="SL_C12")
pilot_c12 = _make_pilot_for_manual_close(pos_c12, close_confirms=True)
pilot_c12._try_acquire_close_ownership(pos_c12, "someone_else")   # pre-steal ownership
result_c12 = pilot_c12.manual_close()
check("12: manual_close loses the race and reports error",
      result_c12["status"] == "error")
check("12: NO RiskManager call when ownership was never acquired",
      not pilot_c12.trader.risk.record_trade_close.called)
check("12: NO journal call when ownership was never acquired",
      not pilot_c12._journal.record_exit.called)
check("12: NO notification (exit-specific) when ownership was never acquired",
      not pilot_c12.notifier.notify_trade.called)
check("12: _live_position untouched when ownership was never acquired",
      pilot_c12._live_position is pos_c12)


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL else 0)
