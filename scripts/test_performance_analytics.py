#!/usr/bin/env python3
"""
Unit tests for the new analytics added to core/performance_tracker.py:
avg_r_multiple, max_consecutive_wins/losses, breakdown_by_strategy.

Standalone script (repo convention), run with:
    python scripts/test_performance_analytics.py

Writes synthetic trades to a TEMP journal file (never the production
logs/trade_journal.jsonl) and asserts computed metrics against known
values, then cleans up.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trade_journal import TradeJournal
from core.performance_tracker import PerformanceTracker


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def make_trade(trade_date, entry_time, result, pnl_pts, pnl_rupees, sl_pts, strategy_name="OpenClawNifty"):
    return {
        "trade_date": trade_date,
        "entry_time": entry_time,
        "is_dry_run": False,
        "result": result,
        "pnl_pts": pnl_pts,
        "pnl_rupees": pnl_rupees,
        "sl_pts": sl_pts,
        "strategy_name": strategy_name,
        "session": "MORNING",
        "direction": "CALL",
        "exit_reason": "TP" if result == "WIN" else "SL",
    }


def main():
    all_ok = True
    tmp_path = Path(__file__).parent.parent / "data" / "_test_perf_journal.jsonl"
    if tmp_path.exists():
        tmp_path.unlink()

    journal = TradeJournal(path=str(tmp_path))

    # Sequence (chronological): W, W, L, L, L, W  -> max_consec_wins=2, max_consec_losses=3
    # R-multiples: pnl_pts/sl_pts for each trade
    trades = [
        make_trade("2026-07-01", "09:20:00", "WIN",  60.0,  3900.0, 40.0),   # R = 1.5
        make_trade("2026-07-01", "09:40:00", "WIN",  40.0,  2600.0, 40.0),   # R = 1.0
        make_trade("2026-07-01", "10:00:00", "LOSS", -40.0, -2600.0, 40.0),  # R = -1.0
        make_trade("2026-07-01", "10:20:00", "LOSS", -20.0, -1300.0, 40.0),  # R = -0.5
        make_trade("2026-07-01", "10:40:00", "LOSS", -40.0, -2600.0, 40.0),  # R = -1.0
        make_trade("2026-07-01", "11:00:00", "WIN",  80.0,  5200.0, 40.0),   # R = 2.0
    ]
    for t in trades:
        journal._write_trade(t)

    tracker = PerformanceTracker(journal=journal)
    s = tracker.summary(days=365)

    all_ok &= check(f"trades counted correctly (got {s['trades']})", s["trades"] == 6)
    all_ok &= check(
        f"max_consecutive_wins == 2 (got {s['max_consecutive_wins']})",
        s["max_consecutive_wins"] == 2,
    )
    all_ok &= check(
        f"max_consecutive_losses == 3 (got {s['max_consecutive_losses']})",
        s["max_consecutive_losses"] == 3,
    )
    # avg R = (1.5 + 1.0 - 1.0 - 0.5 - 1.0 + 2.0) / 6 = 2.0/6 = 0.333...
    expected_avg_r = round((1.5 + 1.0 - 1.0 - 0.5 - 1.0 + 2.0) / 6, 2)
    all_ok &= check(
        f"avg_r_multiple == {expected_avg_r} (got {s['avg_r_multiple']})",
        s["avg_r_multiple"] == expected_avg_r,
    )

    # breakdown_by_strategy: add a second strategy's trade, confirm it's grouped separately
    journal._write_trade(make_trade("2026-07-01", "11:20:00", "WIN", 30.0, 1950.0, 40.0, strategy_name="AltStrategy"))
    by_strategy = tracker.breakdown_by_strategy(days=365)
    all_ok &= check(
        f"breakdown_by_strategy has 2 groups (got {list(by_strategy.keys())})",
        set(by_strategy.keys()) == {"OpenClawNifty", "AltStrategy"},
    )
    all_ok &= check(
        f"OpenClawNifty group has 6 trades (got {by_strategy['OpenClawNifty']['trades']})",
        by_strategy["OpenClawNifty"]["trades"] == 6,
    )
    all_ok &= check(
        f"AltStrategy group has 1 trade (got {by_strategy['AltStrategy']['trades']})",
        by_strategy["AltStrategy"]["trades"] == 1,
    )

    tmp_path.unlink()
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
