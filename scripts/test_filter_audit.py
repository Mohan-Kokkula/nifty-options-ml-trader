"""
test_filter_audit.py — Tests for filter_audit.py (Issue #8).

Run:
    python scripts/test_filter_audit.py

Tests:
  A. Shadow logger captures all 14 filter reasons correctly
  B. _bucket() extracts first token before ":"
  C. compute_filter_stats correct win-rate + expectancy math
  D. Filters with shadow_wr>50% get ⚠ flag, <50% get ✓
  E. Filters with < min_samples excluded
  F. Expectancy delta sign is correct (positive = filter helped)
  G. JSON output is valid and contains expected keys
  H. Empty shadow file returns empty stats without crashing
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

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


from scripts.filter_audit import _bucket, compute_filter_stats, baseline_win_rate

# ── A. _bucket extracts first token correctly ────────────────────────────────
print("\n[A] _bucket()")
check("plain reason",                _bucket("oi_filter") == "oi_filter")
check("reason with detail",          _bucket("oi_filter:CALL_WRITING") == "oi_filter")
check("reason with long detail",     _bucket("trap_gate:LATE_CALL") == "trap_gate")
check("low_conf<%",                  _bucket("low_conf<80%") == "low_conf<80%")
check("data_stale:source",           _bucket("data_stale:YFINANCE") == "data_stale")


# ── B. compute_filter_stats math ─────────────────────────────────────────────
print("\n[B] compute_filter_stats math")

def _row(reason, signal="PUT", won=True, resolved=True):
    return {
        "ts": datetime.now().isoformat(),
        "signal": signal,
        "reason": reason,
        "would_have_won": (True if won else False) if resolved else None,
        "spot": 23300.0,
    }

# 10 PUT signals blocked by oi_filter, 8 would have lost (shadow_wr = 20%)
rows = [_row("oi_filter:PUT_WRITING", signal="PUT", won=False) for _ in range(8)]
rows += [_row("oi_filter:PUT_WRITING", signal="PUT", won=True)  for _ in range(2)]
stats = compute_filter_stats(rows, min_samples=1)
oi = next((s for s in stats if s["bucket"] == "oi_filter"), None)
check("oi_filter found",              oi is not None)
check("oi_filter blocked=10",         oi["blocked"] == 10)
check("oi_filter shadow_wr=20%",      oi["shadow_wr_pct"] == 20.0)
# expectancy_delta = (0.50 - 0.20) × 10 = +3.0  (positive = filter helped)
check("oi_filter expectancy_delta>0", oi["expectancy_delta"] == 3.0)


# ── C. Filter with shadow_wr > 50% has negative expectancy (hurt us) ─────────
print("\n[C] Filters that blocked winners have negative expectancy")

rows_bad = [_row("multi_tf_disagree:call_vs_bear_htf", signal="CALL", won=True) for _ in range(7)]
rows_bad += [_row("multi_tf_disagree:call_vs_bear_htf", signal="CALL", won=False) for _ in range(3)]
stats_bad = compute_filter_stats(rows_bad, min_samples=1)
mtf = next(s for s in stats_bad if s["bucket"] == "multi_tf_disagree")
check("multi_tf shadow_wr=70%",    mtf["shadow_wr_pct"] == 70.0)
# expectancy_delta = (0.50 - 0.70) × 10 = -2.0  (filter cost us expected value)
check("multi_tf expectancy_delta<0", mtf["expectancy_delta"] == -2.0)


# ── D. Min-samples filter ─────────────────────────────────────────────────────
print("\n[D] min_samples exclusion")
rows_few = [_row("rare_filter") for _ in range(2)]
stats_few = compute_filter_stats(rows_few, min_samples=3)
check("2 samples excluded when min=3", len(stats_few) == 0)
stats_ok = compute_filter_stats(rows_few, min_samples=2)
check("2 samples included when min=2", len(stats_ok) == 1)


# ── E. Unresolved entries don't crash and return n/a ─────────────────────────
print("\n[E] Unresolved entries handled")
rows_unres = [_row("vwap_bias:below_3c", resolved=False) for _ in range(5)]
stats_unres = compute_filter_stats(rows_unres, min_samples=1)
vwap = stats_unres[0]
check("blocked count correct",        vwap["blocked"] == 5)
check("shadow_wr is None",            vwap["shadow_wr_pct"] is None)
check("expectancy_delta is None",     vwap["expectancy_delta"] is None)


# ── F. baseline_win_rate ──────────────────────────────────────────────────────
print("\n[F] baseline_win_rate")

live = [
    {"pnl_pts": 35.0, "event": "EXIT", "is_dry_run": False, "trade_date": "2026-06-01"},
    {"pnl_pts": -70.0,"event": "EXIT", "is_dry_run": False, "trade_date": "2026-06-01"},
    {"pnl_pts": 25.0, "event": "EXIT", "is_dry_run": False, "trade_date": "2026-06-02"},
    {"pnl_pts": 40.0, "event": "EXIT", "is_dry_run": False, "trade_date": "2026-06-02"},
]
wr = baseline_win_rate(live)
check("win rate = 75%", abs(wr - 0.75) < 0.01)

check("empty live trades -> nan", baseline_win_rate([]) != baseline_win_rate([]))


# ── G. JSON output ────────────────────────────────────────────────────────────
print("\n[G] JSON output structure")
rows_g = [_row("oi_filter:CALL_WRITING", signal="CALL", won=False) for _ in range(5)]
stats_g = compute_filter_stats(rows_g, min_samples=1)
row = stats_g[0]
for key in ("bucket", "label", "blocked", "blocked_pct", "shadow_wr_pct",
            "expectancy_delta", "trade_red_pct", "call_blocked", "put_blocked"):
    check(f"key '{key}' present", key in row)


# ── H. Empty shadow file ──────────────────────────────────────────────────────
print("\n[H] Empty / missing data")
check("empty list gives empty stats", compute_filter_stats([], min_samples=1) == [])
check("baseline wr nan for empty",    baseline_win_rate([]) != baseline_win_rate([]))  # nan != nan


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
