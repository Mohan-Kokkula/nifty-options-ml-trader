"""
filter_audit.py — Measure each filter's independent impact on trading.

Reads data/shadow_trades.jsonl (built up by the shadow logger as the bot runs)
and the trade journal (for actual live outcomes) to compute per-filter metrics.

Usage:
    python scripts/filter_audit.py                    # full report
    python scripts/filter_audit.py --days 30          # last 30 days only
    python scripts/filter_audit.py --min-samples 5    # skip filters with <5 samples
    python scripts/filter_audit.py --json             # machine-readable output

Output columns (per filter):
    blocked        number of signals the filter killed
    blocked_%      % of all directional signals the filter blocked
    shadow_wr%     if those signals HAD been taken, what % would have won
                   (based on 30-min outcome resolved in shadow_trades.jsonl)
    baseline_wr%   overall live-trade win rate (from trade_journal.jsonl)
    expectancy_Δ   (shadow_wr - 0.5) × blocked   — positive = filter saved money
                   0.50 is the random baseline; a filter that blocks bad trades
                   should have shadow_wr < 0.50 (i.e. the blocked trades would
                   have LOST more than they won)
    trade_red%     what % fewer trades per day if this filter didn't exist

Interpretation guide:
    shadow_wr < 50%  → filter was RIGHT to block (those would have lost)
    shadow_wr > 50%  → filter was WRONG to block (those would have won)
    shadow_wr ≈ 50%  → filter has no edge (random noise, could be removed)

    expectancy_Δ > 0 → filter added expected value (good filter)
    expectancy_Δ < 0 → filter destroyed expected value (hurting you)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


ROOT          = Path(__file__).parent.parent
SHADOW_FILE   = ROOT / "data"   / "shadow_trades.jsonl"
JOURNAL_FILE  = ROOT / "logs"   / "trade_journal.jsonl"


# ── Filter bucket mapping ──────────────────────────────────────────────────
# Maps the first token of the reason string (before ":") to a human-readable label.
# Filters that share a prefix are merged into the same row.
FILTER_LABELS: dict[str, str] = {
    "opening_threshold":   "Opening threshold (raw_p<0.60)",
    "rsi_filter":          "RSI multi-TF conflict",
    "reversal_smc":        "Reversal: no SMC zone (FVG/OB)",
    "oi_filter":           "OI HARD BLOCK (call/put writing)",
    "vix_trend_only":      "VIX regime trend-only filter",
    "vwap_bias":           "Persistent VWAP deficit",
    "smc_context":         "SMC 15m context conflict",
    "morning_hard_block":  "Morning hard block (first N min)",
    "morning_trap_guard":  "Morning trap guard (09:25-09:45)",
    "trap_gate":           "Trap gate (LATE_CALL/PUT chase)",
    "news_gate":           "News bias conflict",
    "vix_expansion":       "VIX expansion (2-way vol)",
    "multi_tf_disagree":   "Multi-TF disagree (5m vs 15m/60m)",
    "chop_gate":           "Chop-day gate (low ADX)",
    "pcr_momentum":        "PCR momentum shift",
    "strategy_k":          "Strategy-K (day-halt / call filter)",
    "low_conf":            "Low confidence (< min_conf%)",
    "data_stale":          "Data freshness gate",
    "dq_block":            "Data quality block (bar/feature)",
}


def _bucket(reason: str) -> str:
    """Extract the top-level bucket from a reason string."""
    return reason.split(":")[0].strip().lower()


def load_shadow(days: int) -> list[dict]:
    if not SHADOW_FILE.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = []
    with SHADOW_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("ts", "") < cutoff:
                continue
            if e.get("signal") not in ("CALL", "PUT"):
                continue   # SKIP signal — no directional bet, can't measure
            rows.append(e)
    return rows


def load_live_trades(days: int) -> list[dict]:
    if not JOURNAL_FILE.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    trades = []
    with JOURNAL_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            if t.get("event") != "EXIT":
                continue
            if t.get("is_dry_run"):
                continue
            if t.get("trade_date", "") < cutoff:
                continue
            trades.append(t)
    return trades


def baseline_win_rate(live_trades: list[dict]) -> float:
    """Win rate of actual executed trades (from trade journal)."""
    if not live_trades:
        return float("nan")
    wins = sum(1 for t in live_trades if (t.get("pnl_pts", 0) or 0) > 0)
    return wins / len(live_trades)


def compute_filter_stats(shadow_rows: list[dict], min_samples: int = 3) -> list[dict]:
    """
    Group shadow rows by filter bucket and compute per-filter metrics.
    Returns list of dicts sorted by absolute expectancy_delta (worst filters first).
    """
    buckets: dict[str, list] = defaultdict(list)
    for e in shadow_rows:
        b = _bucket(e.get("reason", "unknown"))
        buckets[b].append(e)

    # Total directional signals seen (denominator for trade_red%)
    total_signals = len(shadow_rows)

    results = []
    for bucket, rows in buckets.items():
        n = len(rows)
        if n < min_samples:
            continue

        # Only count rows where the 30-min outcome has been resolved
        resolved = [r for r in rows if r.get("would_have_won") is not None]
        n_resolved = len(resolved)

        shadow_wr = (
            sum(1 for r in resolved if r["would_have_won"]) / n_resolved
            if n_resolved > 0 else float("nan")
        )

        # Expectancy delta: (shadow_wr - 0.50) × n_resolved
        # Positive → filter saved expected value (blocked bad trades)
        # Negative → filter cost expected value (blocked good trades)
        if n_resolved > 0 and not (shadow_wr != shadow_wr):   # nan check
            expectancy_delta = (shadow_wr - 0.50) * n_resolved * -1
            # Note: we flip sign because "shadow_wr < 0.50" means the filter HELPED
            # (the trades it blocked would have lost), so the expectancy saved is positive.
            # Mathematically: saved_ev = (1 - shadow_wr) - shadow_wr = 1 - 2*shadow_wr
            # For consistency with "higher = better for us", use:
            expectancy_delta = (0.50 - shadow_wr) * n_resolved
        else:
            expectancy_delta = float("nan")

        trade_red_pct = n / max(total_signals, 1) * 100

        results.append({
            "bucket":          bucket,
            "label":           FILTER_LABELS.get(bucket, bucket),
            "blocked":         n,
            "blocked_pct":     round(trade_red_pct, 1),
            "n_resolved":      n_resolved,
            "shadow_wr_pct":   round(shadow_wr * 100, 1) if shadow_wr == shadow_wr else None,
            "expectancy_delta":round(expectancy_delta, 1) if expectancy_delta == expectancy_delta else None,
            "trade_red_pct":   round(trade_red_pct, 1),
            # Signals by direction
            "call_blocked":    sum(1 for r in rows if r.get("signal") == "CALL"),
            "put_blocked":     sum(1 for r in rows if r.get("signal") == "PUT"),
        })

    # Sort: highest absolute expectancy_delta first (most impactful filters at top)
    results.sort(
        key=lambda r: abs(r["expectancy_delta"] or 0),
        reverse=True,
    )
    return results


def _nan_str(v, fmt=".1f") -> str:
    if v is None or v != v:
        return "   n/a"
    return f"{v:{fmt}}"


def print_report(stats: list[dict], live_wr: float, days: int,
                 total_shadow: int, total_live: int) -> None:
    """Print the human-readable audit report."""
    print()
    print("=" * 80)
    print(f"  FILTER STACK AUDIT — last {days} days")
    print("=" * 80)
    print(
        f"  Shadow signals  : {total_shadow:,}  (directional skips logged by filters)"
    )
    print(
        f"  Live trades     : {total_live:,}  (actual executions in trade journal)"
    )
    if live_wr == live_wr:
        print(f"  Live win rate   : {live_wr*100:.1f}%  (baseline from real trades)")
    print()
    print(
        f"  {'Filter':<38} {'Blocked':>7} {'Blk%':>5} {'ShdwWR':>7} "
        f"{'ExpΔ':>7} {'TrdRed%':>7} {'C/P':>7}"
    )
    print("-" * 80)

    for r in stats:
        wr  = _nan_str(r["shadow_wr_pct"])
        exd = _nan_str(r["expectancy_delta"])
        # Highlight: if filter hurt (shadow_wr > 50%) add marker
        wr_flag = ""
        if r["shadow_wr_pct"] is not None:
            if r["shadow_wr_pct"] > 55:
                wr_flag = " ⚠"   # filter blocked good trades
            elif r["shadow_wr_pct"] < 45:
                wr_flag = " ✓"   # filter correctly blocked bad trades

        cp = f"{r['call_blocked']}C/{r['put_blocked']}P"
        print(
            f"  {r['label'][:38]:<38} "
            f"{r['blocked']:>7} "
            f"{r['blocked_pct']:>5.1f} "
            f"{wr:>6}{wr_flag:<2} "
            f"{exd:>7} "
            f"{r['trade_red_pct']:>7.1f} "
            f"{cp:>7}"
        )

    print()
    print("  Columns:")
    print("    Blocked   = # signals this filter killed")
    print("    Blk%      = % of ALL directional signals killed by this filter")
    print("    ShdwWR%   = if unblocked, what % would have won (shadow outcome)")
    print("    ExpΔ      = (0.50 - ShdwWR) × N_resolved  |  > 0 = filter helped")
    print("    TrdRed%   = trade volume reduction from this filter alone")
    print("    C/P       = CALL / PUT breakdown of blocked signals")
    print()
    print("  Interpretation:")
    print("    ShdwWR < 50%  ✓  filter was RIGHT (blocked bad trades)")
    print("    ShdwWR > 50%  ⚠  filter was WRONG (blocked good trades)")
    print("    ShdwWR ≈ 50%     no edge — filter adds no value (noise)")
    print("    ExpΔ > 0      filter added expected value")
    print("    ExpΔ < 0      filter destroyed expected value")
    print()
    print("  Note: shadow outcomes need 30-min price data. Unresolved entries")
    print("  show n/a. Run resolve_historical_outcomes() or wait for market data.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Filter stack audit")
    parser.add_argument("--days",        type=int, default=90,
                        help="Look-back window in days (default 90)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Min shadow entries to include a filter row (default 3)")
    parser.add_argument("--json",        action="store_true",
                        help="Output JSON instead of formatted table")
    args = parser.parse_args()

    shadow_rows = load_shadow(args.days)
    live_trades = load_live_trades(args.days)
    live_wr     = baseline_win_rate(live_trades)
    stats       = compute_filter_stats(shadow_rows, min_samples=args.min_samples)

    if args.json:
        print(json.dumps({
            "days":           args.days,
            "total_shadow":   len(shadow_rows),
            "total_live":     len(live_trades),
            "live_win_rate":  round(live_wr * 100, 1) if live_wr == live_wr else None,
            "filters":        stats,
        }, indent=2))
    else:
        print_report(stats, live_wr, args.days,
                     len(shadow_rows), len(live_trades))


if __name__ == "__main__":
    main()
