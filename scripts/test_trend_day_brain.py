"""Unit tests for core/trend_day_brain.py pure helpers.

Run standalone (repo convention):  python scripts/test_trend_day_brain.py

These cover the entry rule, the structural stop/target maths and the
one-trade-per-day / window gating. The daemon loop itself is not tested
(needs a live feed) -- extracting these as pure functions is what makes
the validated logic checkable at all.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from core.trend_day_brain import (  # noqa: E402
    TrendDayConfig,
    compute_stop_target,
    consecutive_closes,
    drop_auction_print,
    evaluate_entry,
    in_entry_window,
    session_vwap_proxy,
)

CFG = TrendDayConfig()
_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def bars(closes, opens=None):
    """Build a minimal intraday frame; H/L hug the close so the VWAP proxy
    ((H+L+C)/3) equals the close and the arithmetic stays checkable by hand."""
    closes = list(closes)
    opens = list(opens) if opens else closes
    return pd.DataFrame({
        "open": opens,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
    })


print("\n-- session_vwap_proxy --")
check("expanding mean of typical price", abs(session_vwap_proxy(bars([100, 200])) - 150.0) < 1e-9)
check("empty frame -> nan", not np.isfinite(session_vwap_proxy(bars([]))))

print("\n-- consecutive_closes --")
check("3 rising -> +3", consecutive_closes([10, 11, 12, 13], 3) == 3)
check("3 falling -> -3", consecutive_closes([13, 12, 11, 10], 3) == -3)
check("mixed -> 0", consecutive_closes([10, 12, 11, 13], 3) == 0)
check("flat bar breaks the run", consecutive_closes([10, 11, 11, 12], 3) == 0)
check("too few bars -> 0", consecutive_closes([10, 11], 3) == 0)
check("only the LAST n bars count", consecutive_closes([50, 10, 11, 12, 13], 3) == 3)

print("\n-- in_entry_window --")
check("11:30 is the inclusive open", in_entry_window(datetime(2026, 8, 5, 11, 30), CFG))
check("14:00 is the inclusive close", in_entry_window(datetime(2026, 8, 5, 14, 0), CFG))
check("09:45 rejected (open noise)", not in_entry_window(datetime(2026, 8, 5, 9, 45), CFG))
# 10:30 is now REJECTED: the trend-continuation edge is negative before
# 11:30 (morning mean-reverts), so a continuation rule must not fire there.
check("10:30 rejected (mean-reverting half)",
      not in_entry_window(datetime(2026, 8, 5, 10, 30), CFG))
check("11:25 rejected (just before the flip)",
      not in_entry_window(datetime(2026, 8, 5, 11, 25), CFG))
check("14:05 rejected (late chop)", not in_entry_window(datetime(2026, 8, 5, 14, 5), CFG))

print("\n-- compute_stop_target --")
# PUT: day high 24560, entry 24500 -> stop 24570 (buffer 10), risk 70 -> capped at 60
s, t, r = compute_stop_target("PUT", 24500.0, 24560.0, 24400.0, CFG)
check("PUT risk capped at max_stop_pts", abs(r - 60.0) < 1e-9)
check("PUT stop = entry + capped risk", abs(s - 24560.0) < 1e-9)
check("PUT target = entry - rr*risk", abs(t - (24500.0 - 120.0)) < 1e-9)

# PUT with a close day high: 24520 -> stop 24530, risk 30 (under the cap)
s, t, r = compute_stop_target("PUT", 24500.0, 24520.0, 24400.0, CFG)
check("PUT uses structural stop when inside cap", abs(r - 30.0) < 1e-9)
check("PUT structural stop = high + buffer", abs(s - 24530.0) < 1e-9)
check("PUT target scales with actual risk", abs(t - (24500.0 - 60.0)) < 1e-9)

# CALL mirror: day low 24480 -> stop 24470, risk 30
s, t, r = compute_stop_target("CALL", 24500.0, 24600.0, 24480.0, CFG)
check("CALL structural stop = low - buffer", abs(s - 24470.0) < 1e-9)
check("CALL risk from day low", abs(r - 30.0) < 1e-9)
check("CALL target = entry + rr*risk", abs(t - (24500.0 + 60.0)) < 1e-9)

s, t, r = compute_stop_target("CALL", 24500.0, 24600.0, 24300.0, CFG)
check("CALL risk capped too", abs(r - 60.0) < 1e-9)

# Degenerate guard. At the default 10pt buffer this branch is UNREACHABLE
# (day_high >= close always, so PUT risk >= 10), so it is exercised with a
# zero-buffer config -- it is a defensive guard, not live behaviour.
zero_buf = TrendDayConfig(stop_buffer_pts=0.0)
s, t, r = compute_stop_target("PUT", 24500.0, 24500.0, 24400.0, zero_buf)
check("degenerate PUT risk (<5pts) rejected", s is None and t is None and r == 0.0)
s, t, r = compute_stop_target("CALL", 24500.0, 24600.0, 24500.0, zero_buf)
check("degenerate CALL risk (<5pts) rejected", s is None and t is None and r == 0.0)
check("default buffer keeps the guard unreachable",
      compute_stop_target("PUT", 24500.0, 24500.0, 24400.0, CFG)[2] == 10.0)

print("\n-- evaluate_entry --")
# Falling sequence far below the VWAP proxy and below the day open -> PUT.
# opens[0]=24600 sets day_open; closes trend down to 24400.
down = bars([24600, 24560, 24520, 24480, 24440, 24400])
check("clean downtrend -> PUT", evaluate_entry(down, CFG) == "PUT")

up = bars([24400, 24440, 24480, 24520, 24560, 24600])
check("clean uptrend -> CALL", evaluate_entry(up, CFG) == "CALL")

# Falling closes but price still ABOVE the day open -> no trade (the rule
# requires agreement between the trend and the day's reference level).
noopen = bars([24000, 24800, 24760, 24720, 24680])
check("below-day-open condition blocks PUT", evaluate_entry(noopen, CFG) is None)

# Trending but the deviation from the VWAP proxy is under min_dev_pts.
tiny = bars([24500, 24495, 24490, 24485])
check("sub-25pt deviation rejected", evaluate_entry(tiny, CFG) is None)

# A single large spike down is not 3 sequential closes.
spike = bars([24600, 24600, 24600, 24400])
check("single spike is not a sequence", evaluate_entry(spike, CFG) is None)

check("empty frame -> None", evaluate_entry(bars([]), CFG) is None)
check("too few bars -> None", evaluate_entry(bars([24600, 24500]), CFG) is None)

print("\n-- drop_auction_print --")
# Real shape observed 2026-08-04: a phantom 15:25 bar 151pts above the
# 15:15 close. The backtest excluded post-15:15 bars; the live loop must too.
idx = pd.to_datetime(["2026-08-04 15:05", "2026-08-04 15:10",
                      "2026-08-04 15:15", "2026-08-04 15:25"])
day = bars([24452.30, 24462.95, 24463.45, 24614.90]).set_index(idx)
kept = drop_auction_print(day, CFG)
check("phantom 15:25 auction bar dropped", len(kept) == 3)
check("15:15 bar retained", kept.index[-1].strftime("%H:%M") == "15:15")
check("EOD fill uses the real close, not +151pts",
      abs(float(kept["close"].iloc[-1]) - 24463.45) < 1e-9)
check("empty frame survives", drop_auction_print(bars([]), CFG).empty)
intraday = bars([24500, 24490, 24480]).set_index(
    pd.to_datetime(["2026-08-04 11:00", "2026-08-04 11:05", "2026-08-04 11:10"]))
check("intraday bars untouched", len(drop_auction_print(intraday, CFG)) == 3)

print("\n-- config guards (the validated params) --")
check("rr is 2.0 (VAL-selected)", CFG.rr == 2.0)
check("max_stop_pts is 60 (the recency fix)", CFG.max_stop_pts == 60.0)
check("min_dev_pts is 25", CFG.min_dev_pts == 25.0)
check("seq_n is 3", CFG.seq_n == 3)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
