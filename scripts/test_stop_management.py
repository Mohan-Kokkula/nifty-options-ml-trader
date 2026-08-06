#!/usr/bin/env python3
"""
Unit tests for the new advanced-stop pure helper functions in
core/claude_pilot.py: _atr_trail_sl_candidate, _premium_trail_stop_hit,
_max_hold_exceeded, _gap_protection_widen, _pcr_mood.

Standalone script (repo convention), run with:
    python scripts/test_stop_management.py

These are extracted as module-level pure functions specifically so they
can be tested without constructing a full ClaudePilot/broker/thread stack
— the monitor loop itself (_position_monitor_loop) stays untested here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claude_pilot import (
    _atr_trail_sl_candidate,
    _premium_trail_stop_hit,
    _max_hold_exceeded,
    _gap_protection_widen,
    _pcr_mood,
)


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True

    # ── _atr_trail_sl_candidate ──────────────────────────────────────
    sl = _atr_trail_sl_candidate("CALL", current_price=25000.0, atr=40.0, multiplier=1.5)
    all_ok &= check(f"CALL trail SL is below current price (got {sl})", sl == 25000.0 - 60.0)

    sl = _atr_trail_sl_candidate("PUT", current_price=25000.0, atr=40.0, multiplier=1.5)
    all_ok &= check(f"PUT trail SL is above current price (got {sl})", sl == 25000.0 + 60.0)

    # ── _premium_trail_stop_hit ──────────────────────────────────────
    hit = _premium_trail_stop_hit(peak_premium=100.0, current_premium=60.0, giveback_pct=0.35)
    all_ok &= check(f"40% retrace >= 35% giveback fires (got {hit})", hit is True)

    hit = _premium_trail_stop_hit(peak_premium=100.0, current_premium=80.0, giveback_pct=0.35)
    all_ok &= check(f"20% retrace < 35% giveback does not fire (got {hit})", hit is False)

    hit = _premium_trail_stop_hit(peak_premium=0.0, current_premium=60.0, giveback_pct=0.35)
    all_ok &= check(f"no peak yet -> never fires (got {hit})", hit is False)

    hit = _premium_trail_stop_hit(peak_premium=100.0, current_premium=60.0, giveback_pct=0.0)
    all_ok &= check(f"giveback_pct=0 (disabled) -> never fires (got {hit})", hit is False)

    # ── _max_hold_exceeded ───────────────────────────────────────────
    exceeded = _max_hold_exceeded(entry_time_monotonic=1000.0, max_hold_minutes=30, now_monotonic=1000.0 + 31*60)
    all_ok &= check(f"31min elapsed >= 30min max -> exceeded (got {exceeded})", exceeded is True)

    exceeded = _max_hold_exceeded(entry_time_monotonic=1000.0, max_hold_minutes=30, now_monotonic=1000.0 + 10*60)
    all_ok &= check(f"10min elapsed < 30min max -> not exceeded (got {exceeded})", exceeded is False)

    exceeded = _max_hold_exceeded(entry_time_monotonic=1000.0, max_hold_minutes=0, now_monotonic=1000.0 + 999*60)
    all_ok &= check(f"max_hold_minutes=0 (disabled) -> never exceeded (got {exceeded})", exceeded is False)

    # ── _gap_protection_widen ────────────────────────────────────────
    sl_pts, tp_pts = _gap_protection_widen(40.0, 100.0, minutes_since_open=2.0, window_min=5, widen_pct=0.20)
    all_ok &= check(
        f"within window widens both SL and TP by 20% (got sl={sl_pts}, tp={tp_pts})",
        sl_pts == 48.0 and tp_pts == 120.0,
    )

    sl_pts, tp_pts = _gap_protection_widen(40.0, 100.0, minutes_since_open=10.0, window_min=5, widen_pct=0.20)
    all_ok &= check(
        f"outside window leaves SL/TP unchanged (got sl={sl_pts}, tp={tp_pts})",
        sl_pts == 40.0 and tp_pts == 100.0,
    )

    sl_pts, tp_pts = _gap_protection_widen(40.0, 100.0, minutes_since_open=2.0, window_min=5, widen_pct=0.0)
    all_ok &= check(
        f"widen_pct=0 (disabled) leaves SL/TP unchanged (got sl={sl_pts}, tp={tp_pts})",
        sl_pts == 40.0 and tp_pts == 100.0,
    )

    sl_pts, tp_pts = _gap_protection_widen(40.0, 100.0, minutes_since_open=-5.0, window_min=5, widen_pct=0.20)
    all_ok &= check(
        f"before market open (negative minutes) leaves SL/TP unchanged (got sl={sl_pts}, tp={tp_pts})",
        sl_pts == 40.0 and tp_pts == 100.0,
    )

    # ── _pcr_mood ─────────────────────────────────────────────────────
    # pcr_score = (pcr-0.5)/0.8*100; >=60 -> CALL, <=35 -> PUT, else neutral
    mood = _pcr_mood(pcr=1.2)  # score = (1.2-0.5)/0.8*100 = 87.5 -> bullish
    all_ok &= check(f"high PCR (1.2) -> bullish mood=CALL (got {mood})", mood == "CALL")

    mood = _pcr_mood(pcr=0.5)  # score = 0.0 -> bearish
    all_ok &= check(f"low PCR (0.5) -> bearish mood=PUT (got {mood})", mood == "PUT")

    mood = _pcr_mood(pcr=0.9)  # score = 50.0 -> neutral (35 < 50 < 60)
    all_ok &= check(f"mid PCR (0.9) -> neutral mood=None (got {mood})", mood is None)

    mood = _pcr_mood(pcr=10.0)  # extreme value must clip, not error
    all_ok &= check(f"extreme high PCR clips to bullish, no crash (got {mood})", mood == "CALL")

    mood = _pcr_mood(pcr=0.0)  # extreme value must clip, not error
    all_ok &= check(f"extreme low PCR clips to bearish, no crash (got {mood})", mood == "PUT")

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
