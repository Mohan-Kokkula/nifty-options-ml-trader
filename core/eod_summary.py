"""
eod_summary.py — Daily Telegram summary at 15:30 IST.

Reads the trade journal + risk manager state to produce a one-glance EOD
report. Sent to Telegram so you don't have to grep logs each evening.

Sample output:
    📊 EOD SUMMARY — 2026-05-05

    Trades:    3 (1W / 2L)
    Win rate:  33%
    P&L:       ₹-6,145
    Best:      +₹1,674 (PUT TRAIL_SL)
    Worst:     -₹3,916 (PUT SL)

    Risk cap:  ₹3,000 (BREACHED 2x)
    Max DD:    ₹6,145

    By regime:
      CHOP:        2 trades (50% WR)
      TREND_DOWN:  1 trade  (0% WR)

Triggered by daemon thread at 15:30 IST every weekday.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JOURNAL_FILE = Path("logs/trade_journal.jsonl")
DEFAULT_REPORT_TIME = "15:30"


def _read_today_trades() -> list[dict]:
    """Parse trade_journal.jsonl for today's REAL EXIT events (not paper)."""
    if not JOURNAL_FILE.exists():
        return []
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    try:
        with JOURNAL_FILE.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue

                # Skip paper/dry-run records
                if j.get("is_dry_run") or j.get("result") == "PAPER":
                    continue

                # Must be a real exit — check for event="EXIT" (new records)
                # OR result in WIN/LOSS/BREAKEVEN (legacy records before the fix)
                is_exit = (j.get("event") == "EXIT") or (
                    j.get("result") in ("WIN", "LOSS", "BREAKEVEN")
                    and j.get("exit_reason") is not None
                )
                if not is_exit:
                    continue

                # Date filter — use trade_date (most reliable) then timestamp
                td = j.get("trade_date") or j.get("timestamp", "")[:10]
                if td != today:
                    continue

                out.append(j)
    except Exception as e:
        logger.warning(f"Journal read failed: {e}")
    return out


def _read_today_paper_signals() -> list[dict]:
    """Parse trade_journal.jsonl for today's PAPER (dry-run) signals."""
    if not JOURNAL_FILE.exists():
        return []
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    try:
        with JOURNAL_FILE.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                # Paper signals have result="PAPER" or exit_reason="DRY_RUN"
                if j.get("result") != "PAPER" and j.get("exit_reason") != "DRY_RUN":
                    continue
                td = j.get("trade_date") or j.get("timestamp", "")[:10]
                if td != today:
                    continue
                out.append(j)
    except Exception as e:
        logger.warning(f"Journal paper-read failed: {e}")
    return out


def _format_summary(trades: list[dict], risk_cap: float = 3000.0) -> str:
    today = datetime.now().strftime("%d %b %Y")
    if not trades:
        # No REAL trades — but check if paper signals were fired
        papers = _read_today_paper_signals()
        if papers:
            n_paper = len(papers)
            calls = sum(1 for p in papers if (p.get("direction") or "").upper() == "CALL")
            puts  = sum(1 for p in papers if (p.get("direction") or "").upper() == "PUT")
            avg_conf = sum(p.get("confidence", 0) or 0 for p in papers) / max(n_paper, 1)
            return (
                f"📊 <b>EOD SUMMARY — {today}</b>\n\n"
                f"No REAL trades today (DRY_RUN mode).\n"
                f"Paper signals: {n_paper} (CE={calls} PE={puts})\n"
                f"Avg confidence: {avg_conf:.0f}%\n"
                f"Set DRY_RUN=false to enable live trading."
            )
        return (
            f"📊 <b>EOD SUMMARY — {today}</b>\n\n"
            f"No trades today.\n"
            f"Bot ran but never fired (gates filtered all signals)."
        )

    n = len(trades)
    wins = [t for t in trades if (t.get("pnl_pts", 0) or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_pts", 0) or 0) <= 0]
    total_pnl = sum((t.get("pnl_rupees", 0) or 0) for t in trades)
    pnl_pts = sum((t.get("pnl_pts", 0) or 0) for t in trades)

    best = max(trades, key=lambda t: t.get("pnl_rupees", 0) or 0)
    worst = min(trades, key=lambda t: t.get("pnl_rupees", 0) or 0)

    # Drawdown approximation: cumulative low
    running = 0
    dd = 0
    peak = 0
    for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
        running += t.get("pnl_rupees", 0) or 0
        peak = max(peak, running)
        dd = min(dd, running - peak)

    breached = sum(1 for r in [running, *[risk_cap]] if running < -risk_cap)
    breached = 1 if total_pnl < -risk_cap else 0

    by_regime = Counter()
    by_regime_wins = Counter()
    for t in trades:
        regime = t.get("regime") or t.get("entry_regime") or "UNKNOWN"
        by_regime[regime] += 1
        if (t.get("pnl_pts", 0) or 0) > 0:
            by_regime_wins[regime] += 1

    win_emoji = "🟢" if total_pnl > 0 else "🔴"
    pnl_sign = "+" if total_pnl >= 0 else ""

    lines = [
        f"📊 <b>EOD SUMMARY — {today}</b>",
        "",
        f"<b>Trades:</b> {n} ({len(wins)}W / {len(losses)}L)",
        f"<b>Win rate:</b> {len(wins)*100/n:.0f}%",
        f"<b>P&amp;L:</b> {win_emoji} {pnl_sign}₹{total_pnl:,.0f} ({pnl_pts:+.0f}pts)",
        "",
        f"<b>Best:</b>  +₹{best.get('pnl_rupees', 0):,.0f} "
        f"({best.get('direction', '?')} {best.get('reason', '?')})",
        f"<b>Worst:</b> ₹{worst.get('pnl_rupees', 0):,.0f} "
        f"({worst.get('direction', '?')} {worst.get('reason', '?')})",
    ]

    if abs(dd) > 0:
        lines.append(f"<b>Max DD:</b> ₹{dd:,.0f}")

    if breached:
        lines.append(f"")
        lines.append(f"⚠️ <b>RISK CAP BREACHED</b> (₹{risk_cap:.0f})")
        lines.append(f"   Loss exceeded cap by ₹{abs(total_pnl) - risk_cap:,.0f}")

    if by_regime:
        lines.append("")
        lines.append("<b>By regime:</b>")
        for regime, count in by_regime.most_common():
            wr = by_regime_wins[regime] * 100 / count
            lines.append(f"  {regime:<14} {count} trade(s)  {wr:.0f}% WR")

    # ── Slippage section (Issue #7) ───────────────────────────────────────────
    try:
        from core.trade_journal import TradeJournal
        _jnl = TradeJournal()
        slip = _jnl.get_daily_slippage_report()
        if slip.get("n_trades", 0) > 0:
            lines.append("")
            lines.append("<b>Slippage (execution cost):</b>")
            lines.append(
                f"  Trades with data : {slip['n_trades']}"
            )
            lines.append(
                f"  Avg market impact: {slip['avg_market_impact']:+.2f} pts/trade "
                f"(fill vs signal)"
            )
            lines.append(
                f"  Avg exec slippage: {slip['avg_exec_slippage']:+.2f} pts/trade "
                f"(fill vs IOC limit)"
            )
            lines.append(
                f"  Worst impact     : {slip['max_market_impact']:+.2f} pts"
            )
            lines.append(
                f"  Total slip cost  : Rs.{slip['total_slip_cost_rs']:+,.0f}"
            )
            if slip["slip_over_1pt"] > 0:
                lines.append(
                    f"  ⚠️ {slip['slip_over_1pt']} trade(s) with >1pt market impact "
                    f"— review IOC limit pricing"
                )
    except Exception as _slip_e:
        pass   # slippage section is optional — never break the EOD summary

    return "\n".join(lines)


def send_eod_summary(notifier=None, risk_cap: float = 3000.0) -> str:
    """Build and send EOD summary. Returns the rendered text."""
    trades = _read_today_trades()
    summary = _format_summary(trades, risk_cap=risk_cap)
    logger.warning(f"📊 EOD SUMMARY:\n{summary}")
    try:
        if notifier and hasattr(notifier, "telegram") and notifier.telegram and notifier.telegram.enabled:
            notifier.telegram.send_message(summary)
    except Exception as e:
        logger.warning(f"EOD telegram send failed: {e}")
    return summary


# ──────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────

def _seconds_until(target_h: int, target_m: int) -> float:
    now = datetime.now()
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _scheduler_loop(notifier, risk_cap: float, target_h: int, target_m: int):
    logger.info(f"📊 EOD summary scheduler started — fires at {target_h:02d}:{target_m:02d} IST weekdays")
    sent_today = ""
    while True:
        try:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            # Skip weekends
            if now.weekday() >= 5:
                time.sleep(3600)
                continue

            # If past target and not yet sent today
            target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
            if now >= target and sent_today != today:
                send_eod_summary(notifier, risk_cap=risk_cap)
                sent_today = today
                # Sleep ~12h to next morning before checking
                time.sleep(12 * 3600)
                continue

            # Sleep until target
            wait = (target - now).total_seconds()
            logger.info(f"📊 EOD summary sleeping {wait/3600:.1f}h until {target_h:02d}:{target_m:02d}")
            time.sleep(min(wait, 1800))
        except Exception as e:
            logger.error(f"EOD scheduler error: {e}")
            time.sleep(300)


def schedule_eod_summary(notifier=None, risk_cap: float = 3000.0,
                          target_time: Optional[str] = None) -> threading.Thread:
    """Start the EOD summary scheduler in a background thread."""
    target_time = target_time or os.getenv("EOD_REPORT_TIME", DEFAULT_REPORT_TIME)
    try:
        h, m = map(int, target_time.split(":"))
    except Exception:
        h, m = 15, 30

    t = threading.Thread(
        target=_scheduler_loop,
        args=(notifier, risk_cap, h, m),
        daemon=True,
        name="EODSummaryScheduler",
    )
    t.start()
    return t
