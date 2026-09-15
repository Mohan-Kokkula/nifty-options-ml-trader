"""
learning_agent.py — Post-Session Learning Agent
================================================
Runs at end-of-day (or on demand) to analyze all trades from the journal,
identify patterns, and generate rules the next day's agent uses.

Patterns it detects:
  - Win rate by regime (VIX level), session (morning/mid/afternoon), direction
  - Losing setups: "CALL in sideways with VIX>20 loses 80% — avoid"
  - Winning setups: "PUT in TREND_DOWN morning session wins 75% — favor"
  - Confidence calibration: "trades at 65% conf win only 45% — raise threshold"
  - SL/TP effectiveness: "SL hit before TP in 70% of losses — SL too tight"

Output: data/learned_rules.json — loaded by debate engine and pilot next session.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RULES_PATH = Path("data/learned_rules.json")


@dataclass
class LearnedRule:
    rule_id: str = ""
    category: str = ""
    rule: str = ""
    evidence: str = ""
    confidence: float = 0.0
    sample_size: int = 0
    created: str = ""
    active: bool = True
    validated: bool = False


@dataclass
class LearningReport:
    date: str = ""
    trades_analyzed: int = 0
    rules_generated: int = 0
    rules_updated: int = 0
    key_insights: list = field(default_factory=list)
    performance_summary: dict = field(default_factory=dict)


class LearningAgent:
    """
    Analyzes trade history and generates actionable rules.
    """

    MIN_SAMPLE_SIZE = 5

    def __init__(self, journal=None, calibrator=None, path: Path = RULES_PATH):
        self.journal = journal
        self.calibrator = calibrator
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rules: list[LearnedRule] = []
        self._load_rules()

    def _load_rules(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self._rules = [LearnedRule(**r) for r in data]
                logger.info(f"Learned rules loaded: {len(self._rules)} rules")
            except Exception as e:
                logger.warning(f"Rules load failed: {e}")

    def _save_rules(self):
        try:
            with open(self.path, "w") as f:
                json.dump([asdict(r) for r in self._rules], f, indent=2)
        except Exception as e:
            logger.error(f"Rules save failed: {e}")

    def get_active_rules(self) -> list[LearnedRule]:
        """Only returns validated, active rules for production use."""
        return [r for r in self._rules if r.active and r.validated]

    def get_candidate_rules(self) -> list[LearnedRule]:
        """Returns rules pending validation — not used in production."""
        return [r for r in self._rules if r.active and not r.validated]

    def validate_rule(self, rule_id: str) -> bool:
        """Promote a candidate rule to production (validated)."""
        for r in self._rules:
            if r.rule_id == rule_id:
                r.validated = True
                self._save_rules()
                logger.info(f"Rule validated for production: {rule_id}")
                return True
        return False

    def reject_rule(self, rule_id: str) -> bool:
        """Reject a candidate rule — deactivate it."""
        for r in self._rules:
            if r.rule_id == rule_id:
                r.active = False
                r.validated = False
                self._save_rules()
                logger.info(f"Rule rejected: {rule_id}")
                return True
        return False

    def validate_all(self) -> int:
        """Promote all active candidates to production. Returns count."""
        count = 0
        for r in self._rules:
            if r.active and not r.validated:
                r.validated = True
                count += 1
        if count:
            self._save_rules()
            logger.info(f"Validated {count} candidate rules for production")
        return count

    def get_rules_for_context(self, direction: str = "", vix_regime: str = "",
                              session: str = "") -> list[str]:
        """Get validated rule strings relevant to the current trading context."""
        relevant = []
        for r in self._rules:
            if not r.active or not r.validated:
                continue
            text = r.rule.upper()
            if direction and direction.upper() in text:
                relevant.append(r.rule)
            elif vix_regime and vix_regime.upper() in text:
                relevant.append(r.rule)
            elif session and session.upper() in text:
                relevant.append(r.rule)
            elif r.category in ("general", "risk", "calibration"):
                relevant.append(r.rule)
        return relevant[:10]

    def get_rules_summary(self) -> str:
        """One-line summary of validated active rules for agent prompts."""
        active = self.get_active_rules()
        if not active:
            candidates = len(self.get_candidate_rules())
            if candidates:
                return f"No validated rules. {candidates} candidate(s) pending approval."
            return "No learned rules yet."
        parts = [r.rule for r in active[:8]]
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Main learning routine
    # ------------------------------------------------------------------

    def run_learning(self, days: int = 14) -> LearningReport:
        """
        Analyze recent trades and generate/update rules.
        Called at end-of-day or on demand.
        """
        if not self.journal:
            return LearningReport(key_insights=["No journal available"])

        report = LearningReport(
            date=date.today().isoformat(),
            performance_summary=self.journal.stats(days=days),
        )

        records = self.journal.get_recent(n=100)
        completed = [r for r in records if r.exit_reason]
        report.trades_analyzed = len(completed)

        if len(completed) < self.MIN_SAMPLE_SIZE:
            report.key_insights.append(
                f"Only {len(completed)} completed trades — need {self.MIN_SAMPLE_SIZE}+ for learning"
            )
            return report

        new_rules = []

        # 1. Regime analysis
        new_rules.extend(self._analyze_by_regime(completed))

        # 2. Session analysis
        new_rules.extend(self._analyze_by_session(completed))

        # 3. Direction analysis
        new_rules.extend(self._analyze_by_direction(completed))

        # 4. SL/TP effectiveness
        new_rules.extend(self._analyze_sl_tp(completed))

        # 5. Streak patterns
        new_rules.extend(self._analyze_streaks(completed))

        # 6. Confidence calibration rules
        new_rules.extend(self._analyze_confidence(completed))

        # 7. Combined pattern rules
        new_rules.extend(self._analyze_combined_patterns(completed))

        # Merge with existing rules
        for new_rule in new_rules:
            self._merge_rule(new_rule)
            report.rules_generated += 1

        # Prune stale rules (no supporting evidence in last 30 trades)
        self._prune_stale_rules(completed)

        self._save_rules()

        report.rules_updated = report.rules_generated
        report.key_insights = [r.rule for r in new_rules[:5]]

        logger.info(
            f"Learning complete: {report.trades_analyzed} trades analyzed, "
            f"{report.rules_generated} rules generated/updated"
        )
        return report

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _analyze_by_regime(self, trades: list) -> list[LearnedRule]:
        rules = []
        regimes = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

        for t in trades:
            key = t.vix_regime or "UNKNOWN"
            if t.is_win:
                regimes[key]["wins"] += 1
            else:
                regimes[key]["losses"] += 1
            regimes[key]["pnl"] += t.pnl_points

        for regime, data in regimes.items():
            total = data["wins"] + data["losses"]
            if total < self.MIN_SAMPLE_SIZE:
                continue
            wr = data["wins"] / total * 100

            if wr < 35:
                rules.append(LearnedRule(
                    rule_id=f"regime_{regime}_avoid",
                    category="regime",
                    rule=f"AVOID trading in {regime} regime — win rate only {wr:.0f}% over {total} trades",
                    evidence=f"W={data['wins']} L={data['losses']} PnL={data['pnl']:+.0f}pts",
                    confidence=min(0.9, total / 20),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))
            elif wr > 65:
                rules.append(LearnedRule(
                    rule_id=f"regime_{regime}_favor",
                    category="regime",
                    rule=f"FAVOR trading in {regime} regime — win rate {wr:.0f}% over {total} trades",
                    evidence=f"W={data['wins']} L={data['losses']} PnL={data['pnl']:+.0f}pts",
                    confidence=min(0.9, total / 20),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))

        return rules

    def _analyze_by_session(self, trades: list) -> list[LearnedRule]:
        rules = []
        sessions = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

        for t in trades:
            key = t.session or "UNKNOWN"
            if t.is_win:
                sessions[key]["wins"] += 1
            else:
                sessions[key]["losses"] += 1
            sessions[key]["pnl"] += t.pnl_points

        for session, data in sessions.items():
            total = data["wins"] + data["losses"]
            if total < self.MIN_SAMPLE_SIZE:
                continue
            wr = data["wins"] / total * 100

            if wr < 35:
                rules.append(LearnedRule(
                    rule_id=f"session_{session}_avoid",
                    category="session",
                    rule=f"AVOID trading in {session} session — win rate only {wr:.0f}%",
                    evidence=f"{total} trades, PnL={data['pnl']:+.0f}pts",
                    confidence=min(0.9, total / 15),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))
            elif wr > 65:
                rules.append(LearnedRule(
                    rule_id=f"session_{session}_favor",
                    category="session",
                    rule=f"FAVOR {session} session — win rate {wr:.0f}%, best performing time",
                    evidence=f"{total} trades, PnL={data['pnl']:+.0f}pts",
                    confidence=min(0.9, total / 15),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))

        return rules

    def _analyze_by_direction(self, trades: list) -> list[LearnedRule]:
        rules = []
        dirs = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

        for t in trades:
            key = t.direction or "UNKNOWN"
            if t.is_win:
                dirs[key]["wins"] += 1
            else:
                dirs[key]["losses"] += 1
            dirs[key]["pnl"] += t.pnl_points

        for direction, data in dirs.items():
            total = data["wins"] + data["losses"]
            if total < self.MIN_SAMPLE_SIZE:
                continue
            wr = data["wins"] / total * 100

            if wr < 40:
                rules.append(LearnedRule(
                    rule_id=f"dir_{direction}_weak",
                    category="direction",
                    rule=f"{direction} trades underperforming — win rate {wr:.0f}%, consider reducing",
                    evidence=f"{total} trades, PnL={data['pnl']:+.0f}pts",
                    confidence=min(0.85, total / 20),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))

        return rules

    def _analyze_sl_tp(self, trades: list) -> list[LearnedRule]:
        rules = []
        sl_hits = [t for t in trades if t.exit_reason == "SL"]
        tp_hits = [t for t in trades if t.exit_reason == "TP"]

        if len(sl_hits) >= self.MIN_SAMPLE_SIZE:
            quick_sl = [t for t in sl_hits if t.hold_duration_sec < 120]
            if len(quick_sl) > len(sl_hits) * 0.5:
                rules.append(LearnedRule(
                    rule_id="sl_too_tight",
                    category="risk",
                    rule=f"SL hit within 2min in {len(quick_sl)}/{len(sl_hits)} losses — consider wider SL or better entry timing",
                    evidence=f"Avg hold before SL: {sum(t.hold_duration_sec for t in quick_sl) / len(quick_sl):.0f}s",
                    confidence=0.7,
                    sample_size=len(sl_hits),
                    created=datetime.now().isoformat(),
                ))

            avg_sl_pts = sum(abs(t.pnl_points) for t in sl_hits) / len(sl_hits)
            if tp_hits:
                avg_tp_pts = sum(t.pnl_points for t in tp_hits) / len(tp_hits)
                rr = avg_tp_pts / max(avg_sl_pts, 1)
                if rr < 1.5:
                    rules.append(LearnedRule(
                        rule_id="rr_too_low",
                        category="risk",
                        rule=f"Actual R:R is only {rr:.1f}:1 — target wider TP or tighter SL to reach 2:1",
                        evidence=f"Avg SL loss={avg_sl_pts:.0f}pts, Avg TP win={avg_tp_pts:.0f}pts",
                        confidence=0.75,
                        sample_size=len(sl_hits) + len(tp_hits),
                        created=datetime.now().isoformat(),
                    ))

        trail_trades = [t for t in trades if t.trail_activated]
        if len(trail_trades) >= 3:
            trail_wr = sum(1 for t in trail_trades if t.is_win) / len(trail_trades) * 100
            if trail_wr > 70:
                rules.append(LearnedRule(
                    rule_id="trail_effective",
                    category="risk",
                    rule=f"Trailing stop highly effective — {trail_wr:.0f}% win rate when trail activates",
                    evidence=f"{len(trail_trades)} trades with trail activated",
                    confidence=0.8,
                    sample_size=len(trail_trades),
                    created=datetime.now().isoformat(),
                ))

        return rules

    def _analyze_streaks(self, trades: list) -> list[LearnedRule]:
        rules = []

        # Find max losing streak
        max_losing = 0
        current = 0
        for t in trades:
            if not t.is_win:
                current += 1
                max_losing = max(max_losing, current)
            else:
                current = 0

        if max_losing >= 4:
            rules.append(LearnedRule(
                rule_id="streak_warning",
                category="general",
                rule=f"Max losing streak was {max_losing} — consider stopping after 3 consecutive losses",
                evidence=f"Observed in last {len(trades)} trades",
                confidence=0.7,
                sample_size=len(trades),
                created=datetime.now().isoformat(),
            ))

        return rules

    def _analyze_confidence(self, trades: list) -> list[LearnedRule]:
        rules = []
        buckets = defaultdict(lambda: {"wins": 0, "total": 0})

        for t in trades:
            if t.claude_confidence <= 0:
                continue
            bucket = (t.claude_confidence // 10) * 10
            buckets[bucket]["total"] += 1
            if t.is_win:
                buckets[bucket]["wins"] += 1

        for bucket, data in sorted(buckets.items()):
            if data["total"] < 3:
                continue
            actual_wr = data["wins"] / data["total"] * 100
            expected_wr = bucket + 5

            if actual_wr < expected_wr - 15:
                rules.append(LearnedRule(
                    rule_id=f"conf_overconfident_{bucket}",
                    category="calibration",
                    rule=f"Trades at {bucket}-{bucket + 9}% confidence win only {actual_wr:.0f}% — system is overconfident at this level",
                    evidence=f"{data['total']} trades, {data['wins']} wins",
                    confidence=min(0.85, data["total"] / 10),
                    sample_size=data["total"],
                    created=datetime.now().isoformat(),
                ))

        return rules

    def _analyze_combined_patterns(self, trades: list) -> list[LearnedRule]:
        """Find specific combinations that win or lose disproportionately."""
        rules = []
        combos = defaultdict(lambda: {"wins": 0, "losses": 0})

        for t in trades:
            key = f"{t.direction}_{t.vix_regime}_{t.session}"
            if t.is_win:
                combos[key]["wins"] += 1
            else:
                combos[key]["losses"] += 1

        for combo, data in combos.items():
            total = data["wins"] + data["losses"]
            if total < self.MIN_SAMPLE_SIZE:
                continue

            wr = data["wins"] / total * 100
            parts = combo.split("_")
            if len(parts) < 3:
                continue

            direction, regime, session = parts[0], parts[1], "_".join(parts[2:])

            if wr < 30:
                rules.append(LearnedRule(
                    rule_id=f"combo_avoid_{combo}",
                    category="combined",
                    rule=f"AVOID {direction} in {regime} during {session} — only {wr:.0f}% win rate",
                    evidence=f"{total} trades: W={data['wins']} L={data['losses']}",
                    confidence=min(0.9, total / 10),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))
            elif wr > 70:
                rules.append(LearnedRule(
                    rule_id=f"combo_favor_{combo}",
                    category="combined",
                    rule=f"FAVOR {direction} in {regime} during {session} — {wr:.0f}% win rate, strong edge",
                    evidence=f"{total} trades: W={data['wins']} L={data['losses']}",
                    confidence=min(0.9, total / 10),
                    sample_size=total,
                    created=datetime.now().isoformat(),
                ))

        return rules

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def _merge_rule(self, new_rule: LearnedRule):
        """Update existing rule or add new one. New/updated rules start as candidates."""
        new_rule.validated = False
        for i, existing in enumerate(self._rules):
            if existing.rule_id == new_rule.rule_id:
                self._rules[i] = new_rule
                return
        self._rules.append(new_rule)

    def _prune_stale_rules(self, recent_trades: list):
        """Deactivate rules that no longer have supporting evidence."""
        for rule in self._rules:
            if not rule.active:
                continue
            if rule.sample_size < self.MIN_SAMPLE_SIZE and rule.confidence < 0.5:
                rule.active = False
                logger.info(f"Pruned stale rule: {rule.rule_id}")
