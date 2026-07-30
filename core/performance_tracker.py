"""
performance_tracker.py — Live Performance Metrics
==================================================
Computes win rate, P&L, Sharpe, max drawdown, expectancy from trade journal.
Provides breakdown by session, direction, VIX regime, day-of-week.

Usage:
    from core.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker()
    print(tracker.summary(days=30))
    print(tracker.breakdown_by_session())
"""

import logging
import math
from collections import defaultdict
from typing import Optional

from core.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, journal: Optional[TradeJournal] = None):
        self.journal = journal or TradeJournal()

    def _executed_trades(self, days: int = 30) -> list:
        """Only real (non-dry-run) trades with exit data."""
        all_trades = self.journal.load_trades(days=days)
        return [
            t for t in all_trades
            if not t.get("is_dry_run", False) and t.get("pnl_pts") is not None
        ]

    def _paper_signals(self, days: int = 30) -> list:
        """Only DRY_RUN signals."""
        all_trades = self.journal.load_trades(days=days)
        return [t for t in all_trades if t.get("is_dry_run", False)]

    # ── Core metrics ────────────────────────────────────────────

    def summary(self, days: int = 30) -> dict:
        trades = self._executed_trades(days)
        if not trades:
            return {
                "trades": 0,
                "message": f"No executed trades in last {days} days",
                "paper_signals": len(self._paper_signals(days)),
            }

        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        breakevens = [t for t in trades if t["result"] == "BREAKEVEN"]

        total_pnl = sum(t["pnl_rupees"] for t in trades)
        gross_win = sum(t["pnl_rupees"] for t in wins)
        gross_loss = sum(t["pnl_rupees"] for t in losses)

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = gross_win / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        profit_factor = abs(gross_win / gross_loss) if gross_loss else float("inf")

        # Expectancy = (win_rate × avg_win) - (loss_rate × |avg_loss|)
        loss_rate = len(losses) / len(trades) * 100
        expectancy = (win_rate / 100 * avg_win) - (loss_rate / 100 * abs(avg_loss))

        # Sharpe (daily returns)
        daily_pnl = defaultdict(float)
        for t in trades:
            daily_pnl[t["trade_date"]] += t["pnl_rupees"]
        daily_returns = list(daily_pnl.values())
        if len(daily_returns) > 1:
            mean_r = sum(daily_returns) / len(daily_returns)
            var_r = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
            std_r = math.sqrt(var_r)
            sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe = 0

        # Max drawdown (running peak vs trough) + consecutive win/loss streaks
        # (both computed over the same chronologically-sorted trade list)
        ordered = sorted(trades, key=lambda x: (x["trade_date"], x.get("entry_time", "")))
        cum_pnl = 0
        peak = 0
        max_dd = 0
        max_consec_wins = 0
        max_consec_losses = 0
        cur_streak_result = None
        cur_streak_len = 0
        for t in ordered:
            cum_pnl += t["pnl_rupees"]
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            max_dd = max(max_dd, dd)

            result = t.get("result")
            if result in ("WIN", "LOSS"):
                if result == cur_streak_result:
                    cur_streak_len += 1
                else:
                    cur_streak_result = result
                    cur_streak_len = 1
                if result == "WIN":
                    max_consec_wins = max(max_consec_wins, cur_streak_len)
                else:
                    max_consec_losses = max(max_consec_losses, cur_streak_len)

        # R-multiple = pnl_pts / sl_pts (risk-normalized return per trade),
        # using the sl_pts already stored on every entry — no new journal
        # fields needed. Skips trades with sl_pts<=0 (shouldn't happen, but
        # avoid a division error if it does).
        r_multiples = [
            t["pnl_pts"] / t["sl_pts"]
            for t in trades
            if t.get("sl_pts", 0) and t["sl_pts"] > 0 and t.get("pnl_pts") is not None
        ]
        avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

        return {
            "period_days": days,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "breakevens": len(breakevens),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl_rupees": round(total_pnl, 0),
            "avg_win_rupees": round(avg_win, 0),
            "avg_loss_rupees": round(avg_loss, 0),
            "profit_factor": round(profit_factor, 2),
            "expectancy_per_trade": round(expectancy, 0),
            "sharpe_annualized": round(sharpe, 2),
            "max_drawdown_rupees": round(max_dd, 0),
            "avg_r_multiple": round(avg_r_multiple, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "trading_days": len(daily_pnl),
            "paper_signals": len(self._paper_signals(days)),
        }

    # ── Breakdowns ──────────────────────────────────────────────

    def breakdown_by(self, key: str, days: int = 30) -> dict:
        """Generic breakdown by any field (session/direction/vix_regime/day_of_week)."""
        trades = self._executed_trades(days)
        groups = defaultdict(list)
        for t in trades:
            groups[t.get(key, "UNKNOWN")].append(t)

        result = {}
        for group_name, group_trades in groups.items():
            wins = [t for t in group_trades if t["result"] == "WIN"]
            pnl = sum(t["pnl_rupees"] for t in group_trades)
            result[group_name] = {
                "trades": len(group_trades),
                "wins": len(wins),
                "win_rate_pct": round(len(wins) / len(group_trades) * 100, 1),
                "total_pnl": round(pnl, 0),
                "avg_pnl": round(pnl / len(group_trades), 0),
            }
        return result

    def breakdown_by_session(self, days: int = 30) -> dict:
        return self.breakdown_by("session", days)

    def breakdown_by_direction(self, days: int = 30) -> dict:
        return self.breakdown_by("direction", days)

    def breakdown_by_vix_regime(self, days: int = 30) -> dict:
        return self.breakdown_by("vix_regime", days)

    def breakdown_by_exit_reason(self, days: int = 30) -> dict:
        return self.breakdown_by("exit_reason", days)

    def breakdown_by_strategy(self, days: int = 30) -> dict:
        return self.breakdown_by("strategy_name", days)

    # ── Reporting ────────────────────────────────────────────────

    def format_report(self, days: int = 30) -> str:
        """Human-readable performance report."""
        s = self.summary(days)
        if s.get("trades", 0) == 0:
            return f"📊 PERFORMANCE ({days}d): No executed trades. Paper signals: {s.get('paper_signals', 0)}"

        lines = [
            f"📊 PERFORMANCE ({days} days)",
            "=" * 50,
            f"Trades: {s['trades']} ({s['wins']}W / {s['losses']}L / {s['breakevens']}BE)",
            f"Win Rate: {s['win_rate_pct']}%",
            f"Total P&L: ₹{s['total_pnl_rupees']:+,.0f}",
            f"Avg Win: ₹{s['avg_win_rupees']:+,.0f} | Avg Loss: ₹{s['avg_loss_rupees']:+,.0f}",
            f"Profit Factor: {s['profit_factor']}",
            f"Expectancy/Trade: ₹{s['expectancy_per_trade']:+,.0f}",
            f"Sharpe (annualized): {s['sharpe_annualized']}",
            f"Max Drawdown: ₹{s['max_drawdown_rupees']:,.0f}",
            f"Avg R-Multiple: {s['avg_r_multiple']:+.2f}R",
            f"Max Consecutive Wins/Losses: {s['max_consecutive_wins']}/{s['max_consecutive_losses']}",
            f"Trading Days: {s['trading_days']}",
        ]

        # Session breakdown
        sb = self.breakdown_by_session(days)
        if sb:
            lines.append("\n--- BY SESSION ---")
            for sess, data in sorted(sb.items()):
                lines.append(
                    f"{sess:12s} {data['trades']:3d} trades  "
                    f"WR={data['win_rate_pct']:5.1f}%  "
                    f"P&L=₹{data['total_pnl']:+,.0f}"
                )

        # Direction breakdown
        db = self.breakdown_by_direction(days)
        if db:
            lines.append("\n--- BY DIRECTION ---")
            for dir_, data in sorted(db.items()):
                lines.append(
                    f"{dir_:6s} {data['trades']:3d} trades  "
                    f"WR={data['win_rate_pct']:5.1f}%  "
                    f"P&L=₹{data['total_pnl']:+,.0f}"
                )

        # Exit reason breakdown
        eb = self.breakdown_by_exit_reason(days)
        if eb:
            lines.append("\n--- BY EXIT REASON ---")
            for reason, data in sorted(eb.items()):
                lines.append(
                    f"{reason:14s} {data['trades']:3d} trades  "
                    f"WR={data['win_rate_pct']:5.1f}%  "
                    f"P&L=₹{data['total_pnl']:+,.0f}"
                )

        return "\n".join(lines)
