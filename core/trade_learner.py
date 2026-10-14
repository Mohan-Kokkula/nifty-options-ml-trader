"""
trade_learner.py — Self-Improving Trade Analysis Engine
=======================================================
Hermes-style learning loop: analyzes completed trades, discovers patterns,
generates actionable rules, and sends suggestions via Telegram.

Key principle: SUGGEST, never auto-modify. The user approves/rejects
every suggestion before it touches production parameters.

Runs two analysis modes:
  1. Post-trade: quick insight after each trade closes (streak alerts,
     strategy hot/cold detection)
  2. Daily EOD: full analysis across all strategies — time patterns,
     regime patterns, SL/TP optimization, strategy ranking

Storage: data/learned_insights.json (persistent insights + rules)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

INSIGHTS_PATH = Path("data/learned_insights.json")
MIN_TRADES_FOR_INSIGHT = 5


class TradeLearner:
    """Self-improving trade analysis that learns from the journal."""

    def __init__(self, journal_path: str = "logs/trade_journal.jsonl",
                 notifier=None):
        self.journal_path = Path(journal_path)
        self.notifier = notifier
        self.insights_path = INSIGHTS_PATH
        self.insights_path.parent.mkdir(parents=True, exist_ok=True)
        self._insights = self._load_insights()
        self._last_daily_analysis = None

    def _load_insights(self) -> dict:
        if self.insights_path.exists():
            try:
                with open(self.insights_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "rules": [],
            "strategy_scores": {},
            "last_analysis": None,
            "suggestions_sent": [],
        }

    def _save_insights(self):
        try:
            with open(self.insights_path, "w") as f:
                json.dump(self._insights, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"TradeLearner: save insights failed: {e}")

    def _load_trades(self, days: int = 30) -> list:
        """Load completed trades from journal JSONL."""
        if not self.journal_path.exists():
            return []
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        trades = []
        try:
            with open(self.journal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        if t.get("trade_date", "") >= cutoff and t.get("exit_reason"):
                            trades.append(t)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"TradeLearner: load trades failed: {e}")
        return trades

    def _send(self, text: str):
        if self.notifier and hasattr(self.notifier, "send_message"):
            self.notifier.send_message(text)
        logger.info(f"TradeLearner: {text[:200]}")

    # ------------------------------------------------------------------
    # Post-trade analysis (called after each trade closes)
    # ------------------------------------------------------------------

    def on_trade_closed(self, trade: dict):
        """Quick analysis after a trade closes. Sends alerts for streaks
        and strategy hot/cold patterns."""
        try:
            self._check_streak(trade)
            self._check_strategy_hot_cold(trade)
        except Exception as e:
            logger.warning(f"TradeLearner post-trade analysis failed: {e}")

    def _check_streak(self, latest_trade: dict):
        """Alert on winning/losing streaks."""
        trades = self._load_trades(days=7)
        if len(trades) < 3:
            return

        streak = 0
        streak_type = latest_trade.get("result", "")
        for t in reversed(trades):
            if t.get("result") == streak_type:
                streak += 1
            else:
                break

        if streak >= 3 and streak_type == "LOSS":
            strat = latest_trade.get("strategy_name", "unknown")
            self._send(
                f"<b>LEARNER ALERT: {streak} consecutive losses</b>\n"
                f"Strategy: {strat}\n"
                f"Consider: pause {strat} for today or reduce size\n"
                f"<i>Suggestion only — no auto-changes made</i>"
            )
        elif streak >= 4 and streak_type == "WIN":
            strat = latest_trade.get("strategy_name", "unknown")
            self._send(
                f"<b>LEARNER: {streak} consecutive wins!</b>\n"
                f"Strategy: {strat} is on fire\n"
                f"<i>Keep running — conditions favor this strategy</i>"
            )

    def _check_strategy_hot_cold(self, latest_trade: dict):
        """Check if a strategy is running hot or cold over recent trades."""
        strat = latest_trade.get("strategy_name", "")
        if not strat:
            return

        trades = self._load_trades(days=14)
        strat_trades = [t for t in trades if t.get("strategy_name") == strat
                        and t.get("result") in ("WIN", "LOSS")]
        if len(strat_trades) < MIN_TRADES_FOR_INSIGHT:
            return

        recent = strat_trades[-10:]
        wins = sum(1 for t in recent if t["result"] == "WIN")
        wr = wins / len(recent) * 100

        if wr <= 20 and len(recent) >= 5:
            total_loss = sum(t.get("pnl_pts", 0) for t in recent)
            self._send(
                f"<b>LEARNER: {strat} running COLD</b>\n"
                f"Last {len(recent)} trades: {wins}W/{len(recent)-wins}L "
                f"(WR={wr:.0f}%)\n"
                f"Recent PnL: {total_loss:+.0f} pts\n"
                f"<b>Suggestion:</b> Consider disabling {strat} temporarily\n"
                f"<i>No auto-changes made</i>"
            )

    # ------------------------------------------------------------------
    # Daily EOD analysis
    # ------------------------------------------------------------------

    def daily_analysis(self):
        """Full daily analysis. Call at market close or via EOD summary."""
        today = date.today().isoformat()
        if self._last_daily_analysis == today:
            return
        self._last_daily_analysis = today

        trades = self._load_trades(days=30)
        if len(trades) < MIN_TRADES_FOR_INSIGHT:
            logger.info("TradeLearner: not enough trades for daily analysis")
            return

        try:
            parts = ["<b>DAILY TRADE LEARNER REPORT</b>\n"]

            # Strategy ranking
            strat_analysis = self._analyze_by_strategy(trades)
            if strat_analysis:
                parts.append(self._format_strategy_ranking(strat_analysis))

            # Time-of-day patterns
            time_analysis = self._analyze_by_time(trades)
            if time_analysis:
                parts.append(self._format_time_patterns(time_analysis))

            # VIX regime patterns
            regime_analysis = self._analyze_by_regime(trades)
            if regime_analysis:
                parts.append(self._format_regime_patterns(regime_analysis))

            # SL/TP optimization
            sltp_suggestion = self._suggest_sl_tp(trades)
            if sltp_suggestion:
                parts.append(sltp_suggestion)

            # Generate and store rules
            rules = self._generate_rules(strat_analysis, time_analysis, regime_analysis)
            if rules:
                parts.append(self._format_rules(rules))
                self._insights["rules"] = rules

            self._insights["strategy_scores"] = strat_analysis
            self._insights["last_analysis"] = today
            self._save_insights()

            report = "\n".join(parts)
            self._send(report)
            logger.info("TradeLearner: daily analysis complete")

        except Exception as e:
            logger.error(f"TradeLearner daily analysis failed: {e}")

    def _analyze_by_strategy(self, trades: list) -> dict:
        """Per-strategy win rate, PF, avg PnL, trade count."""
        stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0,
                                      "win_pnl": 0.0, "loss_pnl": 0.0})
        for t in trades:
            strat = t.get("strategy_name", "unknown")
            if not strat:
                strat = "unknown"
            result = t.get("result", "")
            pnl = t.get("pnl_pts", 0) or 0

            if result == "WIN":
                stats[strat]["wins"] += 1
                stats[strat]["win_pnl"] += pnl
            elif result == "LOSS":
                stats[strat]["losses"] += 1
                stats[strat]["loss_pnl"] += abs(pnl)

            stats[strat]["pnl"] += pnl

        result = {}
        for strat, s in stats.items():
            total = s["wins"] + s["losses"]
            if total < 3:
                continue
            wr = s["wins"] / total * 100
            pf = s["win_pnl"] / s["loss_pnl"] if s["loss_pnl"] > 0 else float("inf")
            result[strat] = {
                "trades": total,
                "wins": s["wins"],
                "losses": s["losses"],
                "win_rate": round(wr, 1),
                "pf": round(pf, 2),
                "total_pnl": round(s["pnl"], 1),
                "avg_pnl": round(s["pnl"] / total, 1),
            }
        return result

    def _analyze_by_time(self, trades: list) -> dict:
        """Win rate by session (OPENING, MORNING, MIDDAY, AFTERNOON)."""
        stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
        for t in trades:
            session = t.get("session", "")
            if not session:
                continue
            result = t.get("result", "")
            if result not in ("WIN", "LOSS"):
                continue
            stats[session]["total"] += 1
            if result == "WIN":
                stats[session]["wins"] += 1
            stats[session]["pnl"] += t.get("pnl_pts", 0) or 0

        return {
            session: {
                "trades": s["total"],
                "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0,
                "total_pnl": round(s["pnl"], 1),
            }
            for session, s in stats.items()
            if s["total"] >= 3
        }

    def _analyze_by_regime(self, trades: list) -> dict:
        """Win rate by VIX regime."""
        stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
        for t in trades:
            regime = t.get("vix_regime", "")
            if not regime:
                continue
            result = t.get("result", "")
            if result not in ("WIN", "LOSS"):
                continue
            stats[regime]["total"] += 1
            if result == "WIN":
                stats[regime]["wins"] += 1
            stats[regime]["pnl"] += t.get("pnl_pts", 0) or 0

        return {
            regime: {
                "trades": s["total"],
                "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0,
                "total_pnl": round(s["pnl"], 1),
            }
            for regime, s in stats.items()
            if s["total"] >= 3
        }

    def _suggest_sl_tp(self, trades: list) -> Optional[str]:
        """Analyze actual trade outcomes to suggest SL/TP adjustments."""
        sl_hits = [t for t in trades if t.get("exit_reason") == "SL"]
        tp_hits = [t for t in trades if t.get("exit_reason") in ("TP", "TRAIL_SL")]

        if len(sl_hits) < 5 and len(tp_hits) < 5:
            return None

        parts = ["\n<b>SL/TP Analysis:</b>"]

        if sl_hits:
            avg_sl = sum(t.get("sl_pts", 0) for t in sl_hits) / len(sl_hits)
            quick_sl = [t for t in sl_hits if (t.get("duration_sec") or 999) < 120]
            parts.append(f"  SL hits: {len(sl_hits)} (avg SL={avg_sl:.0f}pts)")
            if len(quick_sl) > len(sl_hits) * 0.4:
                parts.append(
                    f"  {len(quick_sl)}/{len(sl_hits)} SL hits within 2min — "
                    f"SL may be too tight, consider widening"
                )

        if tp_hits:
            avg_tp = sum(t.get("tp_pts", 0) for t in tp_hits) / len(tp_hits)
            parts.append(f"  TP hits: {len(tp_hits)} (avg TP={avg_tp:.0f}pts)")

            trail_exits = [t for t in trades if t.get("exit_reason") == "TRAIL_SL"]
            if trail_exits:
                avg_trail_pnl = sum(t.get("pnl_pts", 0) for t in trail_exits) / len(trail_exits)
                parts.append(f"  Trail exits: {len(trail_exits)} (avg PnL={avg_trail_pnl:+.0f}pts)")

        # Check peak profit vs actual exit
        trades_with_peak = [t for t in trades if t.get("peak_profit_pts", 0) > 0]
        if len(trades_with_peak) >= 5:
            avg_peak = sum(t["peak_profit_pts"] for t in trades_with_peak) / len(trades_with_peak)
            avg_actual = sum(t.get("pnl_pts", 0) for t in trades_with_peak) / len(trades_with_peak)
            leak = avg_peak - avg_actual
            if leak > 10:
                parts.append(
                    f"  Profit leak: avg peak={avg_peak:+.0f} vs actual={avg_actual:+.0f} "
                    f"({leak:.0f}pts left on table)"
                )

        return "\n".join(parts) if len(parts) > 1 else None

    def _generate_rules(self, strat_analysis: dict, time_analysis: dict,
                         regime_analysis: dict) -> list:
        """Generate actionable rules from patterns."""
        rules = []
        today = date.today().isoformat()

        # Strategy rules
        for strat, s in strat_analysis.items():
            if s["trades"] >= 10 and s["win_rate"] < 30:
                rules.append({
                    "type": "strategy",
                    "rule": f"DISABLE {strat}",
                    "reason": f"WR={s['win_rate']}% over {s['trades']} trades "
                              f"(PnL={s['total_pnl']:+.0f}pts)",
                    "confidence": min(0.9, s["trades"] / 20),
                    "date": today,
                    "status": "pending",
                })
            elif s["trades"] >= 10 and s["win_rate"] > 55 and s["pf"] > 1.5:
                rules.append({
                    "type": "strategy",
                    "rule": f"PRIORITIZE {strat}",
                    "reason": f"WR={s['win_rate']}%, PF={s['pf']} over "
                              f"{s['trades']} trades (PnL={s['total_pnl']:+.0f}pts)",
                    "confidence": min(0.9, s["trades"] / 20),
                    "date": today,
                    "status": "pending",
                })

        # Time rules
        for session, s in time_analysis.items():
            if s["trades"] >= 8 and s["win_rate"] < 30:
                rules.append({
                    "type": "timing",
                    "rule": f"AVOID trading in {session} session",
                    "reason": f"WR={s['win_rate']}% over {s['trades']} trades "
                              f"(PnL={s['total_pnl']:+.0f}pts)",
                    "confidence": min(0.85, s["trades"] / 15),
                    "date": today,
                    "status": "pending",
                })

        # Regime rules
        for regime, s in regime_analysis.items():
            if s["trades"] >= 5 and s["win_rate"] < 25:
                rules.append({
                    "type": "regime",
                    "rule": f"REDUCE size in {regime} VIX regime",
                    "reason": f"WR={s['win_rate']}% over {s['trades']} trades "
                              f"(PnL={s['total_pnl']:+.0f}pts)",
                    "confidence": min(0.8, s["trades"] / 10),
                    "date": today,
                    "status": "pending",
                })

        return rules

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_strategy_ranking(self, analysis: dict) -> str:
        """Format strategy ranking for Telegram."""
        if not analysis:
            return ""
        sorted_strats = sorted(analysis.items(),
                                key=lambda x: x[1]["total_pnl"], reverse=True)
        lines = ["\n<b>Strategy Ranking (30d):</b>"]
        for strat, s in sorted_strats:
            icon = "+" if s["total_pnl"] > 0 else "-"
            lines.append(
                f"  {icon} {strat}: {s['trades']}T "
                f"WR={s['win_rate']}% PF={s['pf']} "
                f"PnL={s['total_pnl']:+.0f}pts"
            )
        return "\n".join(lines)

    def _format_time_patterns(self, analysis: dict) -> str:
        if not analysis:
            return ""
        lines = ["\n<b>Time Patterns:</b>"]
        for session in ["OPENING", "MORNING", "MIDDAY", "AFTERNOON"]:
            if session in analysis:
                s = analysis[session]
                lines.append(
                    f"  {session}: {s['trades']}T "
                    f"WR={s['win_rate']}% PnL={s['total_pnl']:+.0f}pts"
                )
        return "\n".join(lines)

    def _format_regime_patterns(self, analysis: dict) -> str:
        if not analysis:
            return ""
        lines = ["\n<b>VIX Regime Patterns:</b>"]
        for regime, s in sorted(analysis.items()):
            lines.append(
                f"  {regime}: {s['trades']}T "
                f"WR={s['win_rate']}% PnL={s['total_pnl']:+.0f}pts"
            )
        return "\n".join(lines)

    def _format_rules(self, rules: list) -> str:
        if not rules:
            return ""
        lines = ["\n<b>Suggested Rules:</b>"]
        for r in rules:
            lines.append(
                f"  [{r['type'].upper()}] {r['rule']}\n"
                f"    Why: {r['reason']}\n"
                f"    Confidence: {r['confidence']:.0%}"
            )
        lines.append("\n<i>These are suggestions only. No auto-changes made.</i>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Weekly deep analysis
    # ------------------------------------------------------------------

    def weekly_analysis(self) -> Optional[str]:
        """Deeper analysis over 30 days. Returns report string."""
        trades = self._load_trades(days=30)
        if len(trades) < 15:
            return None

        strat_analysis = self._analyze_by_strategy(trades)

        # Strategy correlation: which strategies fire together and win/lose
        strat_by_date = defaultdict(list)
        for t in trades:
            strat_by_date[t.get("trade_date", "")].append(t)

        # Days with multiple strategies
        multi_strat_days = {d: ts for d, ts in strat_by_date.items() if len(ts) >= 2}

        parts = ["<b>WEEKLY DEEP ANALYSIS (30d)</b>\n"]

        # Best/worst days
        day_pnl = {}
        for d, ts in strat_by_date.items():
            day_pnl[d] = sum(t.get("pnl_pts", 0) or 0 for t in ts)

        if day_pnl:
            best_day = max(day_pnl, key=day_pnl.get)
            worst_day = min(day_pnl, key=day_pnl.get)
            parts.append(
                f"Best day: {best_day} ({day_pnl[best_day]:+.0f}pts)\n"
                f"Worst day: {worst_day} ({day_pnl[worst_day]:+.0f}pts)"
            )

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in trades:
            exit_reasons[t.get("exit_reason", "unknown")] += 1
        parts.append("\n<b>Exit Reasons:</b>")
        for reason, count in sorted(exit_reasons.items(),
                                     key=lambda x: x[1], reverse=True):
            parts.append(f"  {reason}: {count}")

        # Strategy improvement suggestions
        parts.append("\n<b>Improvement Suggestions:</b>")
        for strat, s in strat_analysis.items():
            if s["pf"] < 1.0 and s["trades"] >= 10:
                parts.append(
                    f"  {strat}: PF={s['pf']} — losing strategy, "
                    f"review parameters or disable"
                )
            elif s["pf"] > 1.5 and s["trades"] >= 10:
                parts.append(
                    f"  {strat}: PF={s['pf']} — strong performer, "
                    f"consider increasing allocation"
                )

        parts.append("\n<i>Suggestions only — review and approve before changes</i>")
        return "\n".join(parts)
