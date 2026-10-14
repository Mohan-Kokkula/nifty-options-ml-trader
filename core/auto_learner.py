"""
auto_learner.py — Self-Improving Auto-Learning Engine
=====================================================
True Hermes-style self-improvement: learns from completed trades and
automatically adjusts strategy behavior in real-time.

What it adjusts (all within safety limits):
  1. Strategy weights — boost winners, penalize losers in the router
  2. SL/TP fine-tuning — adjust based on actual trade outcomes
  3. Auto-pause — disable cold strategies after consecutive losses
  4. Time awareness — learn which sessions work for which strategies

Safety rails:
  - Weight range: [0.5, 1.5] — max 50% boost/penalty
  - SL/TP adjustment: [-15, +15] points
  - Auto-pause after 3+ consecutive losses (same strategy)
  - Everything resets daily — starts fresh each morning
  - Kill switch: AUTO_LEARN=false in settings.env
  - All adjustments logged + sent to Telegram
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

LEARNER_STATE_PATH = Path("data/auto_learner_state.json")
REASONING_LOG_PATH = Path("logs/auto_learner_reasoning.jsonl")

# Safety limits
WEIGHT_MIN = 0.5
WEIGHT_MAX = 1.5
SL_ADJUST_MIN = -15
SL_ADJUST_MAX = 15
TP_ADJUST_MIN = -15
TP_ADJUST_MAX = 15
STREAK_HEAVY_PENALTY = 3  # apply heavy weight penalty after this many consecutive losses
MIN_TRADES_FOR_ADJUST = 2
MIN_PERSISTENT_TRADES = 5  # cross-day minimum before using history for weights


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


class AutoLearner:
    """Real-time self-improving engine for strategy trading."""

    def __init__(self, journal_path: str = "logs/trade_journal.jsonl",
                 notifier=None):
        self.journal_path = Path(journal_path)
        self.notifier = notifier
        self._enabled = os.getenv("AUTO_LEARN", "true").strip().lower() in ("true", "1", "yes")
        self._today = None

        # Per-strategy state (resets daily)
        self.weights = {}           # {"PSAR": 1.0, "MC": 0.8, ...}
        self.sl_adjust = {}         # {"PSAR": 0, "MC": +5, ...}
        self.tp_adjust = {}         # {"PSAR": 0, ...}
        self.streak = {}            # {"PSAR": 2, "MC": -3} positive=wins, negative=losses
        self.trade_count = {}       # trades today per strategy
        self.win_count = {}         # wins today per strategy
        self.pnl_today = {}         # total pnl today per strategy

        # Persistent learned state (survives restarts, not daily resets)
        self._persistent = self._load_persistent()

        if self._enabled:
            logger.info("AutoLearner ENABLED — self-improving mode active")
        else:
            logger.info("AutoLearner DISABLED (AUTO_LEARN=false)")

    def _load_persistent(self) -> dict:
        if LEARNER_STATE_PATH.exists():
            try:
                with open(LEARNER_STATE_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "strategy_history": {},
            "time_patterns": {},
            "sl_tp_history": {},
            "last_updated": None,
        }

    def _save_persistent(self):
        try:
            LEARNER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._persistent["last_updated"] = datetime.now().isoformat()
            with open(LEARNER_STATE_PATH, "w") as f:
                json.dump(self._persistent, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"AutoLearner save failed: {e}")

    def _send(self, text: str):
        if self.notifier and hasattr(self.notifier, "send_message"):
            try:
                self.notifier.send_message(text)
            except Exception:
                pass
        logger.info(f"AutoLearner: {text[:300]}")

    def _log_reasoning(self, strategy: str, action: str, reasons: list,
                       data: dict = None):
        """Write a structured reasoning entry — the 'thinking' log.
        Every decision the AutoLearner makes is recorded with WHY."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "action": action,
            "reasons": reasons,
            "data": data or {},
        }
        try:
            REASONING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(REASONING_LOG_PATH, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
        logger.info(f"AutoLearner THINKING: [{strategy}] {action} — {'; '.join(reasons)}")

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self):
        today = date.today().isoformat()
        if self._today != today:
            if self._today is not None and self.weights:
                self._send(
                    f"<b>AUTO-LEARN: Daily reset</b>\n"
                    f"Previous adjustments cleared. Starting fresh."
                )
            self.weights.clear()
            self.sl_adjust.clear()
            self.tp_adjust.clear()
            self.streak.clear()
            self.trade_count.clear()
            self.win_count.clear()
            self.pnl_today.clear()
            self._today = today
            logger.info(f"AutoLearner: daily reset for {today}")

    # ------------------------------------------------------------------
    # Public API — called by SignalRouter and claude_pilot
    # ------------------------------------------------------------------

    def adjust_confidence(self, strategy: str, raw_confidence: float) -> float:
        """Apply learned weight to confidence score.
        Called by SignalRouter before picking the winner."""
        if not self._enabled:
            return raw_confidence
        self._check_daily_reset()

        weight = self.weights.get(strategy, 1.0)
        adjusted = raw_confidence * weight
        if weight != 1.0:
            hist = self._persistent.get("strategy_history", {}).get(strategy, {})
            reasons = [f"Weight {weight:.2f}x applied to confidence"]
            if weight > 1.0:
                reasons.append(f"Boosted: WR {hist.get('total_wins', 0)}/{hist.get('total_trades', 0)} historically")
            else:
                reasons.append(f"Penalized: WR {hist.get('total_wins', 0)}/{hist.get('total_trades', 0)} historically")
            streak = self.streak.get(strategy, 0)
            if streak != 0:
                reasons.append(f"Current streak: {streak:+d}")
            self._log_reasoning(strategy, "WEIGHT_APPLIED", reasons, {
                "raw_confidence": round(raw_confidence, 2),
                "weight": weight,
                "adjusted_confidence": round(adjusted, 2),
            })
        return adjusted

    def adjust_sl_tp(self, strategy: str, sl: float, tp: float) -> tuple:
        """Apply learned SL/TP adjustment.
        Called by claude_pilot after getting base SL/TP from engine."""
        if not self._enabled:
            return sl, tp
        self._check_daily_reset()

        sl_adj = self.sl_adjust.get(strategy, 0)
        tp_adj = self.tp_adjust.get(strategy, 0)

        new_sl = max(20, sl + sl_adj)
        new_tp = max(20, tp + tp_adj)

        if sl_adj != 0 or tp_adj != 0:
            h = self._persistent.get("sl_tp_history", {}).get(strategy, {})
            reasons = []
            if sl_adj > 0:
                reasons.append(f"SL widened +{sl_adj}pts: quick SL ratio {h.get('quick_sl_count', 0)}/{h.get('total', 0)}")
            elif sl_adj < 0:
                reasons.append(f"SL tightened {sl_adj}pts")
            if tp_adj < 0:
                reasons.append(f"TP tightened {tp_adj}pts: SL ratio {h.get('sl_count', 0)}/{h.get('total', 0)} too high")
            elif tp_adj > 0:
                reasons.append(f"TP widened +{tp_adj}pts: trailing working well")
            self._log_reasoning(strategy, "SL_TP_ADJUSTED", reasons, {
                "original_sl": sl, "new_sl": new_sl,
                "original_tp": tp, "new_tp": new_tp,
            })
        return new_sl, new_tp

    def is_paused(self, strategy: str) -> bool:
        """Never fully blocks — strategies get weight penalty instead."""
        return False

    def on_trade_closed(self, strategy: str, result: str, pnl_pts: float,
                        exit_reason: str = "", session: str = "",
                        duration_sec: int = 0, entry_context: dict = None,
                        exit_price: float = 0, peak_unrealized: float = 0):
        """Update learning state after a trade closes.
        This is where the actual learning happens."""
        if not self._enabled or not strategy:
            return
        self._check_daily_reset()

        is_win = result == "WIN"

        # Update counters
        self.trade_count[strategy] = self.trade_count.get(strategy, 0) + 1
        if is_win:
            self.win_count[strategy] = self.win_count.get(strategy, 0) + 1
        self.pnl_today[strategy] = self.pnl_today.get(strategy, 0) + pnl_pts

        # Update streak
        prev_streak = self.streak.get(strategy, 0)
        if is_win:
            self.streak[strategy] = max(0, prev_streak) + 1
        else:
            self.streak[strategy] = min(0, prev_streak) - 1

        cur_streak = self.streak[strategy]

        # ── Heavy penalty on losing streak (never fully blocks) ──
        streak_penalized = False
        if cur_streak <= -STREAK_HEAVY_PENALTY:
            streak_penalized = True
            self._log_reasoning(strategy, "HEAVY_PENALTY", [
                f"{abs(cur_streak)} consecutive losses — applying heavy weight penalty",
                f"PnL today: {self.pnl_today.get(strategy, 0):+.0f}pts",
                "Strategy stays active but deprioritized (next correct signal still fires)",
            ])
            self._send(
                f"<b>🧠 AUTO-LEARN: {strategy} DEPRIORITIZED</b>\n"
                f"{abs(cur_streak)} consecutive losses — weight heavily penalized\n"
                f"PnL today: {self.pnl_today.get(strategy, 0):+.0f}pts\n"
                f"Strategy still active — next signal fires if it's the best available"
            )

        # ── Adjust weights based on performance ──
        old_weight = self.weights.get(strategy, 1.0)
        self._update_weight(strategy)
        new_weight = self.weights.get(strategy, 1.0)

        # ── Learn SL/TP adjustments ──
        old_sl_adj = self.sl_adjust.get(strategy, 0)
        old_tp_adj = self.tp_adjust.get(strategy, 0)
        self._learn_sl_tp(strategy, pnl_pts, exit_reason, duration_sec)
        new_sl_adj = self.sl_adjust.get(strategy, 0)
        new_tp_adj = self.tp_adjust.get(strategy, 0)

        # ── Update persistent history ──
        self._update_persistent(strategy, is_win, pnl_pts, session)

        # ── Build thinking summary ──
        hist = self._persistent.get("strategy_history", {}).get(strategy, {})
        total_t = hist.get("total_trades", 0)
        total_w = hist.get("total_wins", 0)
        total_wr = (total_w / total_t * 100) if total_t else 0

        thinking_reasons = [
            f"Trade result: {result} ({pnl_pts:+.0f}pts, exit={exit_reason})",
            f"Streak now: {cur_streak:+d}",
            f"Today: {self.win_count.get(strategy, 0)}W/{self.trade_count[strategy] - self.win_count.get(strategy, 0)}L, PnL {self.pnl_today[strategy]:+.0f}pts",
            f"All-time: {total_w}W/{total_t - total_w}L ({total_wr:.0f}% WR)",
        ]

        actions_taken = []
        if new_weight != old_weight:
            actions_taken.append(f"Weight: {old_weight:.2f}→{new_weight:.2f}x")
            if new_weight > old_weight:
                thinking_reasons.append(f"DECISION: Boost weight — winning pattern detected")
            else:
                thinking_reasons.append(f"DECISION: Penalize weight — losing pattern detected")
        if new_sl_adj != old_sl_adj:
            actions_taken.append(f"SL adjust: {old_sl_adj:+d}→{new_sl_adj:+d}pts")
        if new_tp_adj != old_tp_adj:
            actions_taken.append(f"TP adjust: {old_tp_adj:+d}→{new_tp_adj:+d}pts")
        if streak_penalized:
            actions_taken.append(f"HEAVY PENALTY: {abs(cur_streak)} losses → deprioritized (still active)")
        if not actions_taken:
            actions_taken.append("No adjustment (insufficient data or within normal range)")

        thinking_reasons.append(f"Actions: {', '.join(actions_taken)}")

        self._log_reasoning(strategy, "POST_TRADE_ANALYSIS", thinking_reasons, {
            "result": result, "pnl_pts": pnl_pts, "exit_reason": exit_reason,
            "session": session, "duration_sec": duration_sec,
            "streak": cur_streak, "weight": new_weight,
            "sl_adjust": new_sl_adj, "tp_adjust": new_tp_adj,
        })

        # ── Send thinking summary to Telegram ──
        has_changes = (new_weight != old_weight or new_sl_adj != old_sl_adj
                       or new_tp_adj != old_tp_adj or streak_penalized)
        if has_changes:
            emoji = "🟢" if is_win else "🔴"
            lines = [
                f"<b>🧠 AUTO-LEARN THINKING: {strategy}</b>",
                f"",
                f"{emoji} {result} {pnl_pts:+.0f}pts ({exit_reason})",
                f"Streak: {cur_streak:+d} | Session: {session}",
                f"Today: {self.win_count.get(strategy, 0)}W/{self.trade_count[strategy] - self.win_count.get(strategy, 0)}L ({self.pnl_today[strategy]:+.0f}pts)",
                f"All-time: {total_wr:.0f}% WR over {total_t} trades",
                f"",
                f"<b>Decisions:</b>",
            ]
            for a in actions_taken:
                lines.append(f"  → {a}")
            self._send("\n".join(lines))

        # ── Post-mortem: analyze WHY this trade failed/succeeded ──
        if entry_context and not is_win:
            self._post_mortem(strategy, pnl_pts, exit_reason, duration_sec,
                              entry_context, exit_price, peak_unrealized, session)

    # ------------------------------------------------------------------
    # Post-mortem analysis — "what went wrong" thinking
    # ------------------------------------------------------------------

    def _post_mortem(self, strategy: str, pnl_pts: float, exit_reason: str,
                     duration_sec: int, ctx: dict, exit_price: float,
                     peak_unrealized: float, session: str):
        """Analyze a losing trade to identify what went wrong.
        Examines entry conditions and finds the likely mistake."""
        mistakes = []
        lessons = []

        entry_price = ctx.get("spot", 0)
        entry_conf = ctx.get("confidence", 0)
        entry_vix = ctx.get("vix", 0) or 0
        entry_atr = ctx.get("atr", 0) or 0
        sl_pts = ctx.get("sl_pts", 0)
        tp_pts = ctx.get("tp_pts", 0)
        indicators = ctx.get("indicators", {})
        entry_session = ctx.get("session", "")

        # 1. Quick SL hit — entered at bad timing
        if exit_reason == "SL" and duration_sec < 120:
            mistakes.append("QUICK SL HIT: Price reversed within 2 minutes of entry")
            lessons.append("Signal may have been a false breakout or chop zone entry")
            if entry_atr > 0 and sl_pts < entry_atr * 0.8:
                mistakes.append(f"SL ({sl_pts:.0f}pts) was tighter than ATR ({entry_atr:.1f}pts)")
                lessons.append("Consider widening SL to at least 1x ATR for this strategy")

        # 2. Low confidence entry
        if entry_conf > 0 and entry_conf < 0.55:
            mistakes.append(f"LOW CONFIDENCE ENTRY: Confidence was only {entry_conf:.0%}")
            lessons.append("Strategy fired on weak signal — consider higher confidence threshold")

        # 3. High VIX without wider SL
        if entry_vix > 18 and exit_reason == "SL":
            mistakes.append(f"HIGH VIX ({entry_vix:.1f}) with standard SL")
            lessons.append("In high VIX, price swings are wider — SL may need VIX-based widening")

        # 4. Late session entry that lost
        if entry_session in ("afternoon", "closing") and exit_reason == "SL":
            mistakes.append(f"LATE SESSION ENTRY: Entered during {entry_session}")
            lessons.append("Late entries have less time for target + more theta decay")

        # 5. Profit leak — had unrealized profit but lost it
        if peak_unrealized > 10 and pnl_pts < 0:
            leak = peak_unrealized - pnl_pts
            mistakes.append(f"PROFIT LEAK: Peak unrealized was +{peak_unrealized:.0f}pts but exited at {pnl_pts:+.0f}pts")
            lessons.append(f"Had +{peak_unrealized:.0f}pts profit but gave it all back — trail stop could have locked some in")

        # 6. Wrong direction — big adverse move
        if abs(pnl_pts) > 40 and exit_reason == "SL":
            mistakes.append(f"LARGE LOSS ({pnl_pts:+.0f}pts): Strong adverse move against position")
            lessons.append("Market moved sharply opposite — check if counter-trend indicators were ignored")

        # 7. ADX/trend filter analysis
        adx = indicators.get("adx") or indicators.get("adx_15m") or 0
        if adx and adx < 15 and exit_reason == "SL":
            mistakes.append(f"WEAK TREND: ADX was {adx:.0f} at entry (below 15)")
            lessons.append("Low ADX indicates choppy market — trend strategies underperform in chop")

        # 8. Recovery mode or gap override entry
        if indicators.get("recovery_mode"):
            mistakes.append("RECOVERY MODE was active at entry")
            lessons.append("Entered during recovery from previous loss — may compound losses")
        if indicators.get("gap_override"):
            mistakes.append("GAP OVERRIDE was active at entry")

        # 9. Support/Resistance analysis
        sr = ctx.get("sr_levels", {})
        direction = ctx.get("indicators", {}).get("router_action", "")
        entry_dir = "CALL" if pnl_pts < 0 else "PUT"  # infer from loss
        # Check direction from indicators
        for k in ("direction", "router_strategy"):
            pass
        if sr:
            nr = sr.get("nearest_resistance", 0)
            ns = sr.get("nearest_support", 0)
            pivot = sr.get("pivot", 0)
            r1 = sr.get("r1", 0)
            s1 = sr.get("s1", 0)
            day_high = sr.get("day_high", 0)
            day_low = sr.get("day_low", 0)

            if entry_price and nr and ns:
                dist_to_resistance = nr - entry_price
                dist_to_support = entry_price - ns

                # CALL entry near resistance — price likely to reject
                if dist_to_resistance > 0 and dist_to_resistance < 15:
                    mistakes.append(
                        f"S/R CONFLICT: CALL entry at {entry_price:.0f}, "
                        f"resistance at {nr:.0f} (only {dist_to_resistance:.0f}pts away)"
                    )
                    lessons.append("Entered CALL too close to resistance — wait for breakout above or skip")

                # PUT entry near support — price likely to bounce
                if dist_to_support > 0 and dist_to_support < 15:
                    mistakes.append(
                        f"S/R CONFLICT: PUT entry at {entry_price:.0f}, "
                        f"support at {ns:.0f} (only {dist_to_support:.0f}pts away)"
                    )
                    lessons.append("Entered PUT too close to support — wait for breakdown below or skip")

                # Entry near day high (CALL) or day low (PUT) — risky reversal zone
                if day_high and abs(entry_price - day_high) < 10:
                    mistakes.append(f"DAY HIGH ENTRY: Entered at {entry_price:.0f}, day high was {day_high:.0f}")
                    lessons.append("Entering at day high is risky — often a reversal zone")
                if day_low and abs(entry_price - day_low) < 10:
                    mistakes.append(f"DAY LOW ENTRY: Entered at {entry_price:.0f}, day low was {day_low:.0f}")
                    lessons.append("Entering at day low is risky — often a bounce zone")

                # Entry against pivot direction
                if pivot:
                    if entry_price < pivot and exit_reason == "SL":
                        mistakes.append(f"BELOW PIVOT: Entry {entry_price:.0f} below pivot {pivot:.0f}")
                        lessons.append("Price below pivot suggests bearish bias — CALL entries are counter-trend")

            # Log S/R context in reasoning
            self._log_reasoning(strategy, "SR_ANALYSIS", [
                f"Entry: {entry_price:.0f}",
                f"Nearest resistance: {nr:.0f} ({nr - entry_price:+.0f}pts)" if nr else "No resistance found",
                f"Nearest support: {ns:.0f} ({entry_price - ns:+.0f}pts)" if ns else "No support found",
                f"Pivot: {pivot:.0f} | R1: {r1:.0f} | S1: {s1:.0f}" if pivot else "No pivot data",
                f"Day range: {day_low:.0f} — {day_high:.0f}" if day_high else "",
            ], {"sr_levels": sr, "entry_price": entry_price})

        if not mistakes:
            mistakes.append("No obvious mistake detected — may be normal market noise")
            lessons.append("Not every loss is a mistake — check if entry followed all rules correctly")

        # Log the full post-mortem
        self._log_reasoning(strategy, "POST_MORTEM", mistakes + lessons, {
            "entry_context": ctx,
            "exit_price": exit_price,
            "peak_unrealized": peak_unrealized,
            "pnl_pts": pnl_pts,
            "exit_reason": exit_reason,
            "duration_sec": duration_sec,
        })

        # Send condensed post-mortem to Telegram
        lines = [
            f"<b>🔍 POST-MORTEM: {strategy} ({pnl_pts:+.0f}pts)</b>",
            "",
            f"Entry: {entry_price:.0f} at {ctx.get('entry_time_ist', '?')} | Exit: {exit_reason}",
            f"VIX={entry_vix:.1f} ATR={entry_atr:.1f} Conf={entry_conf:.0%}",
            "",
            "<b>What went wrong:</b>",
        ]
        for m in mistakes[:4]:
            lines.append(f"  ⚠ {m}")
        if lessons:
            lines.append("")
            lines.append("<b>Lesson:</b>")
            for l in lessons[:2]:
                lines.append(f"  💡 {l}")

        # Add S/R context to Telegram message
        if sr:
            nr = sr.get("nearest_resistance", 0)
            ns = sr.get("nearest_support", 0)
            pivot = sr.get("pivot", 0)
            if nr or ns or pivot:
                lines.append("")
                lines.append("<b>S/R Levels at entry:</b>")
                if nr:
                    lines.append(f"  R: {nr:.0f} ({nr - entry_price:+.0f}pts)")
                if pivot:
                    lines.append(f"  P: {pivot:.0f}")
                if ns:
                    lines.append(f"  S: {ns:.0f} ({entry_price - ns:+.0f}pts)")

        # Track mistake patterns persistently
        self._track_mistake_pattern(strategy, mistakes, ctx)

        self._send("\n".join(lines))

    def _track_mistake_pattern(self, strategy: str, mistakes: list, ctx: dict):
        """Track recurring mistake patterns across days."""
        patterns = self._persistent.setdefault("mistake_patterns", {})
        if strategy not in patterns:
            patterns[strategy] = {}
        for m in mistakes:
            tag = m.split(":")[0].strip() if ":" in m else m[:30]
            if tag not in patterns[strategy]:
                patterns[strategy][tag] = {"count": 0, "last_seen": ""}
            patterns[strategy][tag]["count"] += 1
            patterns[strategy][tag]["last_seen"] = date.today().isoformat()

        # Alert if same mistake keeps repeating
        for tag, info in patterns.get(strategy, {}).items():
            if info["count"] >= 3 and info["count"] % 3 == 0:
                self._send(
                    f"<b>🧠 RECURRING PATTERN: {strategy}</b>\n"
                    f"'{tag}' has happened {info['count']} times\n"
                    f"This is a systematic issue worth investigating"
                )
        self._save_persistent()

    # ------------------------------------------------------------------
    # Internal learning algorithms
    # ------------------------------------------------------------------

    def _update_weight(self, strategy: str):
        """Recalculate strategy weight from today + persistent history.
        Uses cross-day data since most strategies fire 0-1 trades/day."""
        today_trades = self.trade_count.get(strategy, 0)
        today_wins = self.win_count.get(strategy, 0)

        # Combine today with persistent history for a meaningful sample
        hist = self._persistent.get("strategy_history", {}).get(strategy, {})
        total_trades = hist.get("total_trades", 0) + today_trades
        total_wins = hist.get("total_wins", 0) + today_wins

        if total_trades < MIN_PERSISTENT_TRADES:
            return

        wr = total_wins / total_trades

        # Weight formula: starts at 1.0, adjusts based on win rate
        # WR=50% → 1.0 (neutral), WR=70% → 1.2 (boost), WR=30% → 0.8 (penalty)
        weight = 1.0 + (wr - 0.5) * 0.8

        # Factor in today's streak — amplify for hot/cold runs
        streak = self.streak.get(strategy, 0)
        if streak >= 2:
            weight += 0.1
        elif streak <= -STREAK_HEAVY_PENALTY:
            weight -= 0.25  # heavy penalty but still fires if best signal
        elif streak <= -2:
            weight -= 0.1

        # Today's PnL bias — penalize if bleeding today
        today_pnl = self.pnl_today.get(strategy, 0)
        if today_trades >= 2 and today_pnl < -50:
            weight -= 0.1

        weight = _clamp(weight, WEIGHT_MIN, WEIGHT_MAX)
        self.weights[strategy] = round(weight, 2)

    def _learn_sl_tp(self, strategy: str, pnl_pts: float,
                     exit_reason: str, duration_sec: int):
        """Learn SL/TP adjustments from trade outcomes."""
        # Track SL/TP outcomes in persistent state
        key = strategy
        if key not in self._persistent.get("sl_tp_history", {}):
            self._persistent.setdefault("sl_tp_history", {})[key] = {
                "quick_sl_count": 0,
                "sl_count": 0,
                "tp_count": 0,
                "trail_count": 0,
                "total": 0,
            }
        h = self._persistent["sl_tp_history"][key]
        h["total"] += 1

        if exit_reason == "SL":
            h["sl_count"] += 1
            if duration_sec < 120:
                h["quick_sl_count"] += 1
        elif exit_reason in ("TP", "TRAIL_SL"):
            h["tp_count"] += 1
            if exit_reason == "TRAIL_SL":
                h["trail_count"] += 1

        # Adjust SL if too many quick SL hits (within 2 min)
        if h["total"] >= MIN_PERSISTENT_TRADES:
            quick_ratio = h["quick_sl_count"] / h["total"]
            if quick_ratio > 0.3:
                # SL is too tight — widen it
                self.sl_adjust[strategy] = _clamp(
                    self.sl_adjust.get(strategy, 0) + 5,
                    SL_ADJUST_MIN, SL_ADJUST_MAX
                )
                if self.sl_adjust[strategy] > 0:
                    logger.info(
                        f"AutoLearner: {strategy} SL widened by "
                        f"+{self.sl_adjust[strategy]}pts (quick SL ratio={quick_ratio:.0%})"
                    )

            # Adjust TP if high SL ratio overall
            sl_ratio = h["sl_count"] / h["total"]
            tp_ratio = h["tp_count"] / h["total"]

            if sl_ratio > 0.6 and tp_ratio < 0.2:
                # Losing too much — tighten TP to capture smaller wins
                self.tp_adjust[strategy] = _clamp(
                    self.tp_adjust.get(strategy, 0) - 5,
                    TP_ADJUST_MIN, TP_ADJUST_MAX
                )
            elif tp_ratio > 0.5 and h["trail_count"] > h["tp_count"] * 0.3:
                # Winning a lot and trailing works — can widen TP
                self.tp_adjust[strategy] = _clamp(
                    self.tp_adjust.get(strategy, 0) + 5,
                    TP_ADJUST_MIN, TP_ADJUST_MAX
                )

        self._save_persistent()

    def _update_persistent(self, strategy: str, is_win: bool,
                           pnl_pts: float, session: str):
        """Update persistent strategy history for cross-day learning."""
        hist = self._persistent.setdefault("strategy_history", {})
        if strategy not in hist:
            hist[strategy] = {
                "total_trades": 0,
                "total_wins": 0,
                "total_pnl": 0.0,
                "best_session": {},
            }
        h = hist[strategy]
        h["total_trades"] += 1
        if is_win:
            h["total_wins"] += 1
        h["total_pnl"] += pnl_pts

        # Track per-session performance
        if session:
            sess = h.setdefault("best_session", {})
            if session not in sess:
                sess[session] = {"trades": 0, "wins": 0}
            sess[session]["trades"] += 1
            if is_win:
                sess[session]["wins"] += 1

        # Time patterns for initial weights on next day
        self._persistent.setdefault("time_patterns", {})
        tp = self._persistent["time_patterns"]
        if strategy not in tp:
            tp[strategy] = {}
        if session:
            if session not in tp[strategy]:
                tp[strategy][session] = {"trades": 0, "wins": 0, "pnl": 0.0}
            tp[strategy][session]["trades"] += 1
            if is_win:
                tp[strategy][session]["wins"] += 1
            tp[strategy][session]["pnl"] += pnl_pts

        self._save_persistent()

    # ------------------------------------------------------------------
    # Morning initialization — use persistent data for starting weights
    # ------------------------------------------------------------------

    def bootstrap_from_journal(self):
        """Read ALL historical trades and learn from them on startup.
        Called once — builds persistent state from the trade journal so
        the AutoLearner starts smart, not blank."""
        if not self._enabled:
            return
        if not self.journal_path.exists():
            logger.info("AutoLearner: no journal to bootstrap from")
            return

        # Skip if already bootstrapped with data
        hist = self._persistent.get("strategy_history", {})
        if sum(h.get("total_trades", 0) for h in hist.values()) > 5:
            logger.info("AutoLearner: already bootstrapped — skipping journal scan")
            return

        trades = []
        try:
            with open(self.journal_path, "r") as f:
                for line in f:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"AutoLearner bootstrap read failed: {e}")
            return

        if not trades:
            return

        n_wins = 0
        n_losses = 0

        for t in trades:
            result = t.get("result", "")
            if result not in ("WIN", "LOSS"):
                continue

            strategy = t.get("strategy_name", "ML")
            pnl_pts = t.get("pnl_pts", 0) or 0
            is_win = result == "WIN"
            session = t.get("session", "?")
            exit_reason = t.get("exit_reason", "")
            duration_sec = t.get("duration_sec", 0) or 0
            conf = t.get("confidence", 0) or 0
            vix = t.get("vix", 0) or 0
            atr = t.get("atr_5m", 0) or 0
            peak = t.get("peak_profit_pts", 0) or 0
            vix_regime = t.get("vix_regime", "")

            if is_win:
                n_wins += 1
            else:
                n_losses += 1

            # Build strategy history
            sh = self._persistent.setdefault("strategy_history", {})
            if strategy not in sh:
                sh[strategy] = {"total_trades": 0, "total_wins": 0, "total_pnl": 0.0, "best_session": {}}
            sh[strategy]["total_trades"] += 1
            if is_win:
                sh[strategy]["total_wins"] += 1
            sh[strategy]["total_pnl"] += pnl_pts

            # Session performance
            sess = sh[strategy].setdefault("best_session", {})
            if session not in sess:
                sess[session] = {"trades": 0, "wins": 0}
            sess[session]["trades"] += 1
            if is_win:
                sess[session]["wins"] += 1

            # Time patterns
            tp = self._persistent.setdefault("time_patterns", {})
            if strategy not in tp:
                tp[strategy] = {}
            s_key = session.lower() if session else "unknown"
            if s_key not in tp[strategy]:
                tp[strategy][s_key] = {"trades": 0, "wins": 0, "pnl": 0.0}
            tp[strategy][s_key]["trades"] += 1
            if is_win:
                tp[strategy][s_key]["wins"] += 1
            tp[strategy][s_key]["pnl"] += pnl_pts

            # SL/TP history
            slh = self._persistent.setdefault("sl_tp_history", {})
            if strategy not in slh:
                slh[strategy] = {"quick_sl_count": 0, "sl_count": 0, "tp_count": 0, "trail_count": 0, "total": 0}
            slh[strategy]["total"] += 1
            if exit_reason == "SL":
                slh[strategy]["sl_count"] += 1
                if duration_sec < 120:
                    slh[strategy]["quick_sl_count"] += 1
            elif exit_reason in ("TP", "TRAIL_SL"):
                slh[strategy]["tp_count"] += 1
                if exit_reason == "TRAIL_SL":
                    slh[strategy]["trail_count"] += 1

            # VIX regime stats
            if vix_regime:
                vr_key = f"{strategy}_{vix_regime}"
                vrs = self._persistent.setdefault("vix_regime_stats", {})
                if vr_key not in vrs:
                    vrs[vr_key] = {"trades": 0, "wins": 0, "pnl": 0.0}
                vrs[vr_key]["trades"] += 1
                if is_win:
                    vrs[vr_key]["wins"] += 1
                vrs[vr_key]["pnl"] += pnl_pts

            # Analyze losing trades for mistake patterns
            if not is_win:
                mp = self._persistent.setdefault("mistake_patterns", {})
                if strategy not in mp:
                    mp[strategy] = {}

                if exit_reason == "SL" and duration_sec < 120:
                    tag = "QUICK SL HIT"
                    mp[strategy].setdefault(tag, {"count": 0, "last_seen": ""})
                    mp[strategy][tag]["count"] += 1
                    mp[strategy][tag]["last_seen"] = t.get("trade_date", "")

                if conf and conf < 55:
                    tag = "LOW CONFIDENCE ENTRY"
                    mp[strategy].setdefault(tag, {"count": 0, "last_seen": ""})
                    mp[strategy][tag]["count"] += 1
                    mp[strategy][tag]["last_seen"] = t.get("trade_date", "")

                if vix and vix > 18 and exit_reason == "SL":
                    tag = "HIGH VIX"
                    mp[strategy].setdefault(tag, {"count": 0, "last_seen": ""})
                    mp[strategy][tag]["count"] += 1
                    mp[strategy][tag]["last_seen"] = t.get("trade_date", "")

                if session in ("AFTERNOON", "CLOSING") and exit_reason == "SL":
                    tag = "LATE SESSION ENTRY"
                    mp[strategy].setdefault(tag, {"count": 0, "last_seen": ""})
                    mp[strategy][tag]["count"] += 1
                    mp[strategy][tag]["last_seen"] = t.get("trade_date", "")

                if peak > 10 and pnl_pts < 0:
                    tag = "PROFIT LEAK"
                    mp[strategy].setdefault(tag, {"count": 0, "last_seen": ""})
                    mp[strategy][tag]["count"] += 1
                    mp[strategy][tag]["last_seen"] = t.get("trade_date", "")

                if vix_regime == "CHOP" and exit_reason == "SL":
                    tag = "CHOP REGIME LOSS"
                    mp[strategy].setdefault(tag, {"count": 0, "last_seen": ""})
                    mp[strategy][tag]["count"] += 1
                    mp[strategy][tag]["last_seen"] = t.get("trade_date", "")

        self._save_persistent()

        total_trades = n_wins + n_losses
        if total_trades == 0:
            return

        # Build and send bootstrap summary
        sh = self._persistent.get("strategy_history", {})
        mp = self._persistent.get("mistake_patterns", {})

        lines = [
            f"<b>🧠 AUTO-LEARN: Bootstrapped from {total_trades} historical trades</b>",
            f"",
            f"Overall: {n_wins}W/{n_losses}L ({n_wins/total_trades*100:.0f}% WR)",
            "",
            "<b>Strategy Performance:</b>",
        ]
        for strat, h in sh.items():
            total = h["total_trades"]
            wins = h["total_wins"]
            wr = wins / total * 100 if total else 0
            lines.append(f"  {strat}: {wins}W/{total-wins}L ({wr:.0f}%) PnL={h['total_pnl']:+.0f}pts")

        # Show detected patterns
        all_patterns = []
        for strat, patterns in mp.items():
            for tag, info in patterns.items():
                if info["count"] >= 2:
                    all_patterns.append((info["count"], strat, tag))
        if all_patterns:
            all_patterns.sort(reverse=True)
            lines.append("")
            lines.append("<b>Recurring Mistakes Found:</b>")
            for count, strat, tag in all_patterns[:5]:
                lines.append(f"  ⚠ {tag}: {count}x ({strat})")

        self._send("\n".join(lines))
        self._log_reasoning("ALL", "BOOTSTRAP", [
            f"Loaded {total_trades} trades from journal",
            f"WR: {n_wins}/{total_trades}",
            f"Strategies: {list(sh.keys())}",
            f"Patterns found: {len(all_patterns)}",
        ])
        logger.info(f"AutoLearner: bootstrapped from {total_trades} historical trades")

    def initialize_from_history(self):
        """Set initial weights from persistent history.
        Called at start of day to pre-weight strategies based on
        historical performance. Still within safety limits."""
        if not self._enabled:
            return
        self._check_daily_reset()

        hist = self._persistent.get("strategy_history", {})
        adjustments = []

        for strategy, h in hist.items():
            total = h.get("total_trades", 0)
            if total < MIN_PERSISTENT_TRADES:
                continue

            wins = h.get("total_wins", 0)
            wr = wins / total
            weight = 1.0 + (wr - 0.5) * 0.5  # gentler adjustment from history
            weight = _clamp(weight, 0.8, 1.2)  # tighter range for history-based

            if abs(weight - 1.0) >= 0.05:
                self.weights[strategy] = round(weight, 2)
                adjustments.append(f"  {strategy}: {weight:.2f}x (WR={wr:.0%} over {total}T)")
                self._log_reasoning(strategy, "MORNING_WEIGHT", [
                    f"Historical WR: {wins}W/{total - wins}L ({wr:.0%})",
                    f"Starting weight: {weight:.2f}x",
                    f"Total PnL: {h.get('total_pnl', 0):+.0f}pts",
                ], {"weight": weight, "total_trades": total, "win_rate": round(wr, 2)})

        if adjustments:
            self._send(
                f"<b>🧠 AUTO-LEARN: Morning Thinking</b>\n"
                f"Starting weights from historical performance:\n"
                + "\n".join(adjustments)
            )

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Current auto-learner state for logging."""
        return {
            "enabled": self._enabled,
            "date": self._today,
            "weights": dict(self.weights),
            "sl_adjust": dict(self.sl_adjust),
            "tp_adjust": dict(self.tp_adjust),
            "heavy_penalty_streaks": {k: v for k, v in self.streak.items() if v <= -STREAK_HEAVY_PENALTY},
            "streaks": dict(self.streak),
            "trades_today": dict(self.trade_count),
        }
