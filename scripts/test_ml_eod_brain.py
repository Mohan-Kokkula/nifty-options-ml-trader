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

print("\n-- compute_stop  (risk = stop_R x atr_mult x ATR = 3 x ATR) --")
s, r = compute_stop("CALL", 24500.0, 15.0, CFG)
check("CALL risk = 3 x ATR", abs(r - 45.0) < 1e-9)
check("CALL stop below entry", abs(s - 24455.0) < 1e-9)
s, r = compute_stop("PUT", 24500.0, 15.0, CFG)
check("PUT stop above entry", abs(s - 24545.0) < 1e-9)
check("PUT risk identical", abs(r - 45.0) < 1e-9)
s, r = compute_stop("CALL", 24500.0, float("nan"), CFG)
check("nan ATR rejected", s is None and r == 0.0)
s, r = compute_stop("CALL", 24500.0, 0.0, CFG)
check("zero ATR rejected", s is None and r == 0.0)
s, r = compute_stop("CALL", 24500.0, 1.0, CFG)
check("sub-5pt risk rejected", s is None and r == 0.0)

print("\n-- in_entry_window (11:30-14:00 after the OAT tune) --")
check("11:30 is the inclusive open", in_entry_window(datetime(2026, 8, 7, 11, 30), CFG))
check("13:59 allowed", in_entry_window(datetime(2026, 8, 7, 13, 59), CFG))
# Morning is now REJECTED: the trend-continuation edge is negative before
# 11:30, and the VAL sweep put the best window start there (1.196 vs 1.030).
check("09:20 REJECTED (morning mean-reverts)",
      not in_entry_window(datetime(2026, 8, 7, 9, 20), CFG))
check("11:25 rejected (just before the flip)",
      not in_entry_window(datetime(2026, 8, 7, 11, 25), CFG))
check("14:00 REJECTED (cutoff is exclusive)",
      not in_entry_window(datetime(2026, 8, 7, 14, 0), CFG))
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

print("\n-- config guards (the OAT-tuned structure) --")
check("stop = 3.0 x ATR", abs(CFG.stop_R * CFG.atr_mult - 3.0) < 1e-9)
check("window starts 11:30", CFG.entry_window_start == "11:30")
check("cutoff 14:00", CFG.entry_cutoff == "14:00")
check("max 2 trades/day", CFG.max_trades_per_day == 2)
# Floor deliberately DISABLED: VAL and TEST curves pointed opposite ways
# (VAL peak 0.425 falling to 0.807 by 0.60; TEST rising to 1.795 by 0.60).
check("confidence floor disabled", abs(CFG.min_confidence) < 1e-9)
# There is deliberately NO take-profit: the whole point is holding to EOD.
check("no take-profit field exists", not hasattr(CFG, "target_R"))

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
