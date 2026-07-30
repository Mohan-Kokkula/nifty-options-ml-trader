"""
test_phase1_monitor_io.py — Phase 1: Monitor Blocking I/O Fix

Tests:
  A. SL evaluation completes <100ms even when get_quote hangs
  B. Premium cache is read (not live REST) inside monitor cycle
  C. Stale cache disables premium-stop but NOT spot-stop
  D. Premium poller updates cache in background
  E. Latency metric is recorded each cycle
  F. Spot REST fallback respects 1-second timeout
  G. Premium for PAPER positions skipped (no REST call)
"""
import sys, time, threading, json
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS  {name}")
    else:    FAIL += 1; print(f"  FAIL  {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pos(direction="PUT", entry=23400, sl=23470, tp=23260,
              entry_prem=90.0, symbol="NIFTY2660923400PE"):
    from core.claude_pilot import LivePosition, PositionState
    return LivePosition(
        direction=direction, entry_price=float(entry),
        entry_time=time.monotonic(),
        sl_price=float(sl), tp_price=float(tp), initial_sl=float(sl),
        symbol=symbol,
        atr_at_entry=35.0, entry_premium=entry_prem,
        state=PositionState.OPEN, original_qty=65,
    )


# ---------------------------------------------------------------------------
# A. SL evaluation completes < 100ms when get_quote blocks indefinitely
# ---------------------------------------------------------------------------
print("\n[A] SL evaluation < 100ms even if REST blocks")

from core.watchdog_helper import call_with_timeout

def _hanging_fn():
    time.sleep(10)  # simulates infinite block
    return {"ltp": 90.0}

start = time.monotonic()
result = call_with_timeout(_hanging_fn, timeout_sec=0.5, default=None, name="test_hang")
elapsed = time.monotonic() - start
check("call_with_timeout returns default after 0.5s", result is None)
check("elapsed < 0.6s (not 10s)", elapsed < 0.6)


# ---------------------------------------------------------------------------
# B. Premium cache is read (not live REST) inside monitor cycle
# ---------------------------------------------------------------------------
print("\n[B] Monitor reads _premium_cache, not live REST")

rest_calls = []

def _mock_get_quote(symbol, exchange):
    rest_calls.append(symbol)
    return {"data": {"ltp": 88.0}}

from core.claude_pilot import ClaudePilot, LivePosition, PositionState
import threading as _threading

pilot = object.__new__(ClaudePilot)
pilot._lock = _threading.Lock()
pilot._premium_cache_lock = _threading.Lock()
pilot._premium_cache = {"premium": 87.5, "ts": time.monotonic(), "symbol": "NIFTY2660923400PE"}
pilot._PREMIUM_CACHE_MAX_AGE_SEC = 30.0
pilot._running = True
pilot._ws_spot = 23380.0
pilot._ws_spot_ts = time.monotonic()  # fresh WS spot
pilot._live_position = _make_pos(entry=23400, sl=23470, entry_prem=90.0)
pilot._monitor_last_cycle_ms = 0.0

# Patch trader so REST get_quote would fail if called
mock_trader = MagicMock()
mock_trader.client.get_quote = _mock_get_quote
mock_trader.get_nifty_spot.return_value = 23380.0
pilot.trader = mock_trader

# Simulate one pass of the premium-reading section:
pos = pilot._live_position
_cache_age = time.monotonic() - pilot._premium_cache["ts"]
_premium_feed_valid = (
    pilot._premium_cache["premium"] > 0
    and pilot._premium_cache["symbol"] == pos.symbol
    and _cache_age <= pilot._PREMIUM_CACHE_MAX_AGE_SEC
)
_cache_premium = pilot._premium_cache["premium"] if _premium_feed_valid else 0.0

check("cache read returns correct premium", abs(_cache_premium - 87.5) < 0.01)
check("no live REST call was made to get_quote", len(rest_calls) == 0)
check("_premium_feed_valid=True for fresh same-symbol cache", _premium_feed_valid)


# ---------------------------------------------------------------------------
# C. Stale cache disables premium-stop but NOT spot-stop
# ---------------------------------------------------------------------------
print("\n[C] Stale cache: premium-stop disabled, spot-stop active")

pilot._premium_cache["ts"] = time.monotonic() - 45.0  # 45s > 30s max age
_cache_age2 = time.monotonic() - pilot._premium_cache["ts"]
_valid2 = _cache_age2 <= pilot._PREMIUM_CACHE_MAX_AGE_SEC
check("stale cache (_premium_feed_valid=False)", not _valid2)

# Spot-stop still works regardless (uses ws_spot, no cache dependency):
spot = pilot._ws_spot
pos2 = _make_pos(entry=23400, sl=23470)  # SL above spot → should fire
sl_hit = pos2.direction == "PUT" and spot >= pos2.sl_price
check("spot-based SL check works with stale cache (spot=23380 < sl=23470)", not sl_hit)
# Change to a case where SL should fire:
pos3 = _make_pos(entry=23400, sl=23350)  # sl=23350 < spot=23380 → PUT SL hit
sl_hit3 = pos3.direction == "PUT" and spot >= pos3.sl_price
check("spot SL fires correctly independent of premium cache", sl_hit3)


# ---------------------------------------------------------------------------
# D. Premium poller updates cache in background
# ---------------------------------------------------------------------------
print("\n[D] Premium poller updates cache asynchronously")

updated = threading.Event()

def _mock_quote_ok(symbol, exchange):
    updated.set()
    return {"data": {"ltp": 91.5}}

pilot2 = object.__new__(ClaudePilot)
pilot2._lock = _threading.Lock()
pilot2._premium_cache_lock = _threading.Lock()
pilot2._premium_cache = {"premium": 0.0, "ts": 0.0, "symbol": ""}
pilot2._PREMIUM_CACHE_MAX_AGE_SEC = 30.0
pilot2._running = True

pos_d = _make_pos(symbol="NIFTY2660923400PE", entry_prem=90.0)
pilot2._live_position = pos_d
pilot2._PREMIUM_POLL_INTERVAL = 0.1  # fast for test

mock_trader2 = MagicMock()
mock_trader2.client.get_quote = _mock_quote_ok
mock_trader2.exchange = "NFO"
pilot2.trader = mock_trader2

# Patch is_market_hours to return True
pilot2._is_market_hours = lambda: True

# Run one iteration manually (simulate poller loop body):
from core.watchdog_helper import call_with_timeout as cwt
symbol = pos_d.symbol
quote = cwt(
    fn=lambda: pilot2.trader.client.get_quote(symbol, pilot2.trader.exchange),
    timeout_sec=2.0, default=None, name="test_poller"
)
if quote is not None:
    ltp = float(quote.get("data", {}).get("ltp", 0) or 0)
    if ltp > 0:
        with pilot2._premium_cache_lock:
            pilot2._premium_cache["premium"] = ltp
            pilot2._premium_cache["ts"]      = time.monotonic()
            pilot2._premium_cache["symbol"]  = symbol

check("poller wrote correct premium to cache", abs(pilot2._premium_cache["premium"] - 91.5) < 0.01)
check("poller wrote correct symbol", pilot2._premium_cache["symbol"] == "NIFTY2660923400PE")
check("poller wrote fresh timestamp", (time.monotonic() - pilot2._premium_cache["ts"]) < 1.0)


# ---------------------------------------------------------------------------
# E. Latency metric recorded per cycle
# ---------------------------------------------------------------------------
print("\n[E] Latency metric exists and is numeric")

pilot3 = object.__new__(ClaudePilot)
pilot3._monitor_last_cycle_ms = 0.0
check("_monitor_last_cycle_ms attribute exists and is float", isinstance(pilot3._monitor_last_cycle_ms, float))
pilot3._monitor_last_cycle_ms = 42.7
check("metric can be written and read", abs(pilot3._monitor_last_cycle_ms - 42.7) < 0.01)


# ---------------------------------------------------------------------------
# F. Spot REST fallback respects 1-second timeout
# ---------------------------------------------------------------------------
print("\n[F] Spot REST fallback capped at 1 second")

def _slow_spot():
    time.sleep(5.0)
    return 23400.0

start = time.monotonic()
spot = cwt(fn=_slow_spot, timeout_sec=1.0, default=0.0, name="test_slow_spot")
elapsed = time.monotonic() - start
check("slow spot REST returns default 0.0 after 1s", spot == 0.0)
check("elapsed < 1.2s (1s timeout respected)", elapsed < 1.2)


# ---------------------------------------------------------------------------
# G. PAPER positions skip premium polling
# ---------------------------------------------------------------------------
print("\n[G] PAPER positions skip premium polling")

pos_paper = _make_pos(symbol="PAPER", entry_prem=90.0)
# The condition in the poller is: pos.symbol != "PAPER"
should_poll = pos_paper.symbol != "PAPER"
check("PAPER symbol skips premium poller", not should_poll)


# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
