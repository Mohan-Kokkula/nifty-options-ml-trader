"""
test_position_reconciliation.py — Tests for position reconciliation (Issue #5).

Run:
    python scripts/test_position_reconciliation.py

Tests:
  A. get_open_nifty_positions parser — various Kotak response shapes
  B. find_open_entry — finds today's un-exited live entry
  C. mark_entry_closed — writes synthetic EXIT record
  D. Reconcile: Outcome A — broker open + journal matched → LivePosition rebuilt
  E. Reconcile: Outcome B — broker open + no journal → conservative LivePosition
  F. Reconcile: Outcome C — broker flat + stale journal → synthetic EXIT written
  G. Reconcile: no position anywhere — no-op
  H. Reconcile: session not ready — no-op (normal 04:00 startup)
  I. Reconcile: fail-open — exception does not block start()
"""
import json
import sys
import tempfile
import time
from datetime import datetime
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_kotak_row(symbol, buy_qty, sell_qty, ltp=80.0):
    return {"trdSym": symbol, "flBuyQty": buy_qty, "flSellQty": sell_qty, "ltp": ltp}


def _journal_entry(direction="PUT", option_type="PE", entry_time=None,
                   sl_price=23300.0, tp_price=23150.0, entry_spot=23370.0,
                   is_dry_run=False):
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "trade_date":    today,
        "entry_time":    entry_time or datetime.now().strftime("%H:%M:%S"),
        "direction":     direction,
        "option_type":   option_type,
        "sl_price":      sl_price,
        "tp_price":      tp_price,
        "entry_spot":    entry_spot,
        "entry_premium": 95.0,
        "qty":           65,
        "result":        None,       # None = not yet closed
        "exit_reason":   None,
        "is_dry_run":    is_dry_run,
    }


# ---------------------------------------------------------------------------
# A. get_open_nifty_positions parser
# ---------------------------------------------------------------------------
print("\n[A] get_open_nifty_positions parser")

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.kotak_neo_client import KotakNeoClient

client = object.__new__(KotakNeoClient)

def _parse(rows):
    """Call the parser with mocked internals."""
    from unittest.mock import MagicMock
    client._neo = MagicMock()        # _neo.positions is accessed before _call_with_retry
    client._call_with_retry = lambda fn, **kw: {"data": rows}
    client._rate_gate = lambda *a, **kw: None
    return client.get_open_nifty_positions()

# Standard open PE
r = _parse([_fake_kotak_row("NIFTY2660923200PE", buy_qty=65, sell_qty=0)])
check("open PE detected",                len(r) == 1 and r[0]["direction"] == "PUT")
check("symbol correct",                   r[0]["symbol"] == "NIFTY2660923200PE")
check("net_qty correct",                  r[0]["net_qty"] == 65)

# Open CE
r = _parse([_fake_kotak_row("NIFTY2660924000CE", buy_qty=130, sell_qty=0)])
check("open CE detected as CALL",         len(r) == 1 and r[0]["direction"] == "CALL")

# Closed position (net_qty == 0)
r = _parse([_fake_kotak_row("NIFTY2660923200PE", buy_qty=65, sell_qty=65)])
check("flat position (net=0) excluded",   len(r) == 0)

# BANKNIFTY excluded
r = _parse([_fake_kotak_row("BANKNIFTY2660943000PE", buy_qty=25, sell_qty=0)])
check("BANKNIFTY excluded",               len(r) == 0)

# Empty response
r = _parse([])
check("empty response returns []",        r == [])

# Dict response format (Kotak API variant)
client._call_with_retry = lambda fn, **kw: [
    _fake_kotak_row("NIFTY2660923400PE", buy_qty=65, sell_qty=0)
]
r = client.get_open_nifty_positions()
check("list response format handled",     len(r) == 1)


# ---------------------------------------------------------------------------
# B. find_open_entry
# ---------------------------------------------------------------------------
print("\n[B] find_open_entry")

from core.trade_journal import TradeJournal

def _make_journal(lines):
    td = tempfile.mktemp(suffix=".jsonl")
    with open(td, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return TradeJournal(path=td)

# Single open entry — no exit record
jnl = _make_journal([_journal_entry()])
e = jnl.find_open_entry()
check("finds single open entry",          e is not None)
check("returns correct direction",        e["direction"] == "PUT")

# Entry + matching EXIT — should return None
entry = _journal_entry()
exit_rec = dict(entry)
exit_rec["exit_reason"] = "SL"
exit_rec["result"] = "LOSS"
exit_rec["event"] = "EXIT"
jnl2 = _make_journal([entry, exit_rec])
check("no open entry when EXIT exists",   jnl2.find_open_entry() is None)

# DRY_RUN entry excluded
dry_entry = _journal_entry(is_dry_run=True)
jnl3 = _make_journal([dry_entry])
check("dry-run entry not returned",       jnl3.find_open_entry() is None)

# Most recent of two open entries is returned
e1 = _journal_entry(entry_time="09:45:00")
e2 = _journal_entry(entry_time="11:30:00")
jnl4 = _make_journal([e1, e2])
found = jnl4.find_open_entry()
check("returns most recent of two",       found is not None and found["entry_time"] == "11:30:00")

# Empty journal
jnl5 = _make_journal([])
check("empty journal returns None",       jnl5.find_open_entry() is None)


# ---------------------------------------------------------------------------
# C. mark_entry_closed
# ---------------------------------------------------------------------------
print("\n[C] mark_entry_closed")

jnl_c = _make_journal([_journal_entry()])
entry_c = jnl_c.find_open_entry()
jnl_c.mark_entry_closed(entry_c, reason="TEST_CLOSE")
# After closing, find_open_entry should return None
check("after mark_closed, no open entry", jnl_c.find_open_entry() is None)
# Load raw to verify EXIT record was written
lines = [json.loads(l) for l in open(jnl_c.path).readlines() if l.strip()]
exit_lines = [l for l in lines if l.get("exit_reason") == "TEST_CLOSE"]
check("synthetic EXIT record written",    len(exit_lines) == 1)
check("EXIT event field set",             exit_lines[0].get("event") == "EXIT")


# ---------------------------------------------------------------------------
# D. Reconcile Outcome A — broker open + journal matched
# ---------------------------------------------------------------------------
print("\n[D] Reconcile: broker open + journal match -> LivePosition rebuilt")

from core.claude_pilot import ClaudePilot, LivePosition, PositionState
import threading

def _make_pilot(broker_positions, journal_lines, session_valid=True):
    """Build a minimal ClaudePilot wired to fake broker and journal."""
    trader  = MagicMock()
    trader.get_nifty_spot.return_value = 23250.0

    mock_client = MagicMock()
    mock_client._session_valid = session_valid
    mock_client.get_open_nifty_positions.return_value = broker_positions
    trader.client = mock_client

    pilot = object.__new__(ClaudePilot)
    pilot._lock              = threading.Lock()
    pilot._live_position     = None
    pilot._current_atr       = 35.0
    pilot.trader             = trader
    pilot.notifier           = MagicMock()

    # Wire journal
    td = tempfile.mktemp(suffix=".jsonl")
    with open(td, "w") as f:
        for line in journal_lines:
            f.write(json.dumps(line) + "\n")
    from core.trade_journal import TradeJournal
    pilot._journal = TradeJournal(path=td)

    return pilot

entry_d = _journal_entry(direction="PUT", sl_price=23320.0, tp_price=23150.0, entry_spot=23370.0)
bp_d    = [{"symbol": "NIFTY2660923200PE", "direction": "PUT", "net_qty": 65, "ltp": 85.0}]
pilot_d = _make_pilot(bp_d, [entry_d])
pilot_d._reconcile_broker_position()

check("Outcome A: LivePosition set",             pilot_d._live_position is not None)
check("Outcome A: direction correct",            pilot_d._live_position.direction == "PUT")
check("Outcome A: symbol correct",               pilot_d._live_position.symbol == "NIFTY2660923200PE")
check("Outcome A: SL from journal",              pilot_d._live_position.sl_price == 23320.0)
check("Outcome A: TP from journal",              pilot_d._live_position.tp_price == 23150.0)
check("Outcome A: entry_price from journal",     pilot_d._live_position.entry_price == 23370.0)
check("Outcome A: state=OPEN",                   pilot_d._live_position.state == PositionState.OPEN)
check("Outcome A: Telegram alert sent",          pilot_d.notifier.notify_trade.called)


# ---------------------------------------------------------------------------
# E. Reconcile Outcome B — broker open + no journal
# ---------------------------------------------------------------------------
print("\n[E] Reconcile: broker open + no journal -> conservative LivePosition")

bp_e    = [{"symbol": "NIFTY2660923200PE", "direction": "PUT", "net_qty": 65, "ltp": 80.0}]
pilot_e = _make_pilot(bp_e, [])   # empty journal
pilot_e._reconcile_broker_position()

check("Outcome B: LivePosition set",         pilot_e._live_position is not None)
check("Outcome B: direction correct",        pilot_e._live_position.direction == "PUT")
# Conservative SL should be ABOVE spot for PUT (spot + 1.5*ATR)
spot_e = 23250.0
atr_e  = 35.0
expected_sl = spot_e + atr_e * 1.5
check("Outcome B: SL is conservative wide",
      abs(pilot_e._live_position.sl_price - expected_sl) < 0.1)
check("Outcome B: Telegram alert sent",      pilot_e.notifier.notify_trade.called)


# ---------------------------------------------------------------------------
# F. Reconcile Outcome C — broker flat + stale journal
# ---------------------------------------------------------------------------
print("\n[F] Reconcile: broker flat + stale journal -> synthetic EXIT")

entry_f = _journal_entry()
pilot_f = _make_pilot(broker_positions=[], journal_lines=[entry_f])
pilot_f._reconcile_broker_position()

check("Outcome C: _live_position stays None",    pilot_f._live_position is None)
check("Outcome C: journal entry now closed",     pilot_f._journal.find_open_entry() is None)


# ---------------------------------------------------------------------------
# G. No position anywhere
# ---------------------------------------------------------------------------
print("\n[G] No position anywhere -> no-op")

pilot_g = _make_pilot(broker_positions=[], journal_lines=[])
pilot_g._reconcile_broker_position()
check("No position: _live_position stays None",  pilot_g._live_position is None)
check("No position: no alert sent",              not pilot_g.notifier.notify_trade.called)


# ---------------------------------------------------------------------------
# H. Session not ready (04:00 startup) -> no-op
# ---------------------------------------------------------------------------
print("\n[H] Session not ready -> no-op")

pilot_h = _make_pilot(broker_positions=[], journal_lines=[], session_valid=False)
entry_h = _journal_entry()
open(pilot_h._journal.path, "w").write(json.dumps(entry_h) + "\n")
pilot_h._reconcile_broker_position()

check("Not ready: no broker call made",
      not pilot_h.trader.client.get_open_nifty_positions.called)
check("Not ready: _live_position stays None",    pilot_h._live_position is None)


# ---------------------------------------------------------------------------
# I. Exception in reconcile -> fail-open (start() not blocked)
# ---------------------------------------------------------------------------
print("\n[I] Exception in reconcile -> fail-open")

pilot_i = _make_pilot(broker_positions=[], journal_lines=[])
pilot_i.trader.client.get_open_nifty_positions.side_effect = RuntimeError("broker exploded")
pilot_i.trader.client._session_valid = True  # session is valid so reconcile runs

raised = False
try:
    pilot_i._reconcile_broker_position()
except Exception:
    raised = True
check("exception does not propagate from reconcile",  not raised)
check("_live_position still None after error",        pilot_i._live_position is None)


# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
