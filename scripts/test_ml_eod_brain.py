"""Unit tests for core/ml_eod_brain.py pure helpers.

Run standalone:  python scripts/test_ml_eod_brain.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_eod_brain import (  # noqa: E402
    MLEodConfig, compute_atr, compute_stop, drop_auction_print,
    in_entry_window, is_eod,
)

CFG = MLEodConfig()
_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"  PASS  {name}")
    else:
        _failed += 1; print(f"  FAIL  {name}")


def bars(closes, spread=10.0, index=None):
    closes = list(closes)
    df = pd.DataFrame({"open": closes,
                       "high": [c + spread/2 for c in closes],
                       "low":  [c - spread/2 for c in closes],
                       "close": closes})
    if index is not None:
        df.index = index
    return df


print("\n-- compute_atr --")
flat = bars([24500] * 30, spread=10.0)
check("constant 10pt range -> ATR ~10", abs(compute_atr(flat, 14) - 10.0) < 0.5)
check("insufficient history -> nan", not np.isfinite(compute_atr(bars([1, 2, 3]), 14)))

print("\n-- compute_stop  (risk = stop_R x atr_mult x ATR = 4 x ATR) --")
s, r = compute_stop("CALL", 24500.0, 15.0, CFG)
check("CALL risk = 4 x ATR", abs(r - 60.0) < 1e-9)
check("CALL stop below entry", abs(s - 24440.0) < 1e-9)
s, r = compute_stop("PUT", 24500.0, 15.0, CFG)
check("PUT stop above entry", abs(s - 24560.0) < 1e-9)
check("PUT risk identical", abs(r - 60.0) < 1e-9)
s, r = compute_stop("CALL", 24500.0, float("nan"), CFG)
check("nan ATR rejected", s is None and r == 0.0)
s, r = compute_stop("CALL", 24500.0, 0.0, CFG)
check("zero ATR rejected", s is None and r == 0.0)
s, r = compute_stop("CALL", 24500.0, 1.0, CFG)
check("sub-5pt risk rejected", s is None and r == 0.0)

print("\n-- in_entry_window (cutoff 15:00, matches claude_pilot) --")
check("09:20 allowed", in_entry_window(datetime(2026, 8, 7, 9, 20), CFG))
check("14:59 allowed", in_entry_window(datetime(2026, 8, 7, 14, 59), CFG))
check("15:00 REJECTED", not in_entry_window(datetime(2026, 8, 7, 15, 0), CFG))
check("15:05 rejected", not in_entry_window(datetime(2026, 8, 7, 15, 5), CFG))

print("\n-- is_eod (square-off 15:10) --")
check("15:09 not yet EOD", not is_eod(datetime(2026, 8, 7, 15, 9), CFG))
check("15:10 IS EOD", is_eod(datetime(2026, 8, 7, 15, 10), CFG))
check("15:25 is EOD", is_eod(datetime(2026, 8, 7, 15, 25), CFG))

print("\n-- drop_auction_print --")
idx = pd.to_datetime(["2026-08-07 15:05", "2026-08-07 15:15", "2026-08-07 15:25"])
d = bars([24450, 24463, 24614], index=idx)
kept = drop_auction_print(d, CFG)
check("phantom 15:25 bar dropped", len(kept) == 2)
check("real close retained", abs(float(kept['close'].iloc[-1]) - 24463) < 1e-9)
check("empty frame survives", drop_auction_print(bars([]), CFG).empty)

print("\n-- config guards (the validated structure) --")
check("stop_R is 2.0 (best of 5 exit variants)", CFG.stop_R == 2.0)
check("atr_mult is 2.0 (R definition)", CFG.atr_mult == 2.0)
check("entry cutoff 15:00 matches pilot", CFG.entry_cutoff == "15:00")
check("max 3 trades/day", CFG.max_trades_per_day == 3)
# There is deliberately NO take-profit: the whole point is holding to EOD.
check("no take-profit field exists", not hasattr(CFG, "target_R"))

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
