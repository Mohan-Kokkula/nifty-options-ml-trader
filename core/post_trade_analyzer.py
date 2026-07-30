"""
post_trade_analyzer.py — Async Claude Post-Trade Journal Analysis
=================================================================
WHY this exists:
  Claude API calls take 2-8 seconds. Calling Claude in the live 5-minute
  execution cycle ruins options pricing and R:R. This module removes Claude
  entirely from the hot path.

What it does:
  1. Runs in a BACKGROUND THREAD (daemon) started once at bot startup.
  2. Every N minutes (default: 30), reads new entries from trade_journal.jsonl
  3. Sends each unanalyzed trade to Claude with full market context
  4. Writes Claude's retrospective analysis to post_trade_analysis.jsonl
  5. End-of-day: sends a full session summary

This way Claude works as a COACH (end-of-day review) not a GATEKEEPER
(blocking live execution).

Usage:
  analyzer = PostTradeAnalyzer(api_key=ANTHROPIC_API_KEY)
  analyzer.start()           # launches background thread
  analyzer.stop()            # clean shutdown at EOD
  analyzer.trigger_eod()     # manual end-of-day report
"""

import json
import logging
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL   = 30 * 60          # 30 minutes between sweeps
_EOD_HOUR           = 15               # 15:30 IST triggers EOD report
_EOD_MINUTE         = 30
_MODEL              = "claude-sonnet-4-5"
_MAX_TOKENS         = 1024
_JOURNAL_PATH       = Path("logs/trade_journal.jsonl")
_ANALYSIS_PATH      = Path("logs/post_trade_analysis.jsonl")
_ANALYZED_IDS_PATH  = Path("logs/.analyzed_ids.json")


class PostTradeAnalyzer:
    """
    Background thread that reads completed trades and sends them to
    Claude for retrospective analysis — completely off the live path.
    """

    def __init__(
        self,
        api_key: str,
        journal_path: Path = _JOURNAL_PATH,
        output_path: Path  = _ANALYSIS_PATH,
        interval_sec: int  = _DEFAULT_INTERVAL,
        model: str         = _MODEL,
    ):
        self._api_key      = api_key
        self._journal      = Path(journal_path)
        self._output       = Path(output_path)
        self._interval     = interval_sec
        self._model        = model
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._analyzed_ids: set = self._load_analyzed_ids()
        self._eod_done_today = False

        self._output.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start background analysis thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="PostTradeAnalyzer"
        )
        self._thread.start()
        logger.info(
            f"PostTradeAnalyzer started — sweeping every {self._interval//60}min | "
            f"model={self._model}"
        )

    def stop(self):
        """Signal clean shutdown."""
        self._running = False
        logger.info("PostTradeAnalyzer stopping...")

    def trigger_eod(self):
        """Manually trigger end-of-day summary (e.g. called from main at 15:30)."""
        threading.Thread(
            target=self._run_eod_summary,
            daemon=True,
            name="PostTradeAnalyzer-EOD",
        ).start()

    def trigger_now(self):
        """Force an immediate analysis sweep (useful after a trade closes)."""
        threading.Thread(
            target=self._run_sweep,
            daemon=True,
            name="PostTradeAnalyzer-Sweep",
        ).start()

    # ─────────────────────────────────────────────────────────
    # Background loop
    # ─────────────────────────────────────────────────────────

    def _loop(self):
        """Main background loop — sweep + EOD check."""
        while self._running:
            try:
                self._run_sweep()
                now = datetime.now()
                if (now.hour == _EOD_HOUR and now.minute >= _EOD_MINUTE
                        and not self._eod_done_today):
                    self._run_eod_summary()
                    self._eod_done_today = True
                if now.hour < _EOD_HOUR:
                    self._eod_done_today = False   # reset for next day
            except Exception as e:
                logger.error(f"PostTradeAnalyzer loop error: {e}")
            time.sleep(self._interval)

    def _run_sweep(self):
        """Read new closed trades from journal and analyze each one."""
        trades = self._load_new_trades()
        if not trades:
            logger.debug("PostTradeAnalyzer: no new trades to analyze")
            return

        logger.info(f"PostTradeAnalyzer: analyzing {len(trades)} new trade(s)")
        for trade in trades:
            try:
                analysis = self._analyze_trade(trade)
                if analysis:
                    self._write_analysis(analysis)
                    self._mark_analyzed(trade.get("trade_id", trade.get("entry_time", "")))
            except Exception as e:
                logger.warning(f"PostTradeAnalyzer: trade analysis failed: {e}")

    def _run_eod_summary(self):
        """Send full session to Claude for end-of-day coaching report."""
        try:
            all_today = self._load_today_trades()
            if not all_today:
                logger.info("PostTradeAnalyzer EOD: no trades today")
                return

            logger.info(f"PostTradeAnalyzer EOD: summarizing {len(all_today)} trades")
            summary = self._call_claude_eod(all_today)
            if summary:
                record = {
                    "type":      "EOD_SUMMARY",
                    "date":      date.today().isoformat(),
                    "timestamp": datetime.now().isoformat(),
                    "trades":    len(all_today),
                    "analysis":  summary,
                }
                self._write_analysis(record)
                logger.info("PostTradeAnalyzer EOD: summary written")
        except Exception as e:
            logger.error(f"PostTradeAnalyzer EOD failed: {e}")

    # ─────────────────────────────────────────────────────────
    # Claude calls
    # ─────────────────────────────────────────────────────────

    def _analyze_trade(self, trade: dict) -> Optional[dict]:
        """
        Send one closed trade to Claude for retrospective analysis.
        Returns an analysis record dict, or None on failure.
        """
        prompt = _build_trade_prompt(trade)
        response = self._call_claude(prompt)
        if not response:
            return None

        return {
            "type":       "TRADE_ANALYSIS",
            "trade_id":   trade.get("trade_id", trade.get("entry_time", "unknown")),
            "timestamp":  datetime.now().isoformat(),
            "direction":  trade.get("direction", ""),
            "pnl_pts":    trade.get("pnl_pts", 0),
            "exit_reason":trade.get("exit_reason", ""),
            "analysis":   response,
        }

    def _call_claude(self, prompt: str) -> Optional[str]:
        """Make a single Claude API call. Returns text or None."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            msg = client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else None
        except Exception as e:
            logger.warning(f"PostTradeAnalyzer Claude call failed: {e}")
            return None

    def _call_claude_eod(self, trades: list) -> Optional[str]:
        """End-of-day bulk session analysis."""
        prompt = _build_eod_prompt(trades)
        return self._call_claude(prompt)

    # ─────────────────────────────────────────────────────────
    # Journal I/O
    # ─────────────────────────────────────────────────────────

    def _load_new_trades(self) -> list:
        """Load closed trades not yet analyzed."""
        all_trades = self._load_journal()
        return [
            t for t in all_trades
            if t.get("status") in ("CLOSED", "closed")
            and t.get("trade_id", t.get("entry_time", "")) not in self._analyzed_ids
        ]

    def _load_today_trades(self) -> list:
        """Load all of today's closed trades."""
        today = date.today().isoformat()
        return [
            t for t in self._load_journal()
            if t.get("status") in ("CLOSED", "closed")
            and str(t.get("entry_time", "")).startswith(today)
        ]

    def _load_journal(self) -> list:
        """Read trade_journal.jsonl safely."""
        if not self._journal.exists():
            return []
        trades = []
        try:
            with open(self._journal, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.debug(f"Journal read failed: {e}")
        return trades

    def _write_analysis(self, record: dict):
        """Append analysis record to post_trade_analysis.jsonl."""
        try:
            with open(self._output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"PostTradeAnalyzer write failed: {e}")

    def _mark_analyzed(self, trade_id: str):
        """Remember this trade_id so we don't re-analyze it."""
        self._analyzed_ids.add(str(trade_id))
        self._save_analyzed_ids()

    def _load_analyzed_ids(self) -> set:
        try:
            if _ANALYZED_IDS_PATH.exists():
                with open(_ANALYZED_IDS_PATH, encoding="utf-8") as f:
                    return set(json.load(f))
        except Exception:
            pass
        return set()

    def _save_analyzed_ids(self):
        try:
            _ANALYZED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_ANALYZED_IDS_PATH, "w", encoding="utf-8") as f:
                json.dump(list(self._analyzed_ids), f)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────

def _build_trade_prompt(trade: dict) -> str:
    direction   = trade.get("direction", "")
    strike_mode = trade.get("strike_mode", "ATM")
    entry_spot  = trade.get("entry_spot", 0)
    exit_spot   = trade.get("exit_spot", 0)
    pnl_pts     = trade.get("pnl_pts", 0)
    sl_pts      = trade.get("sl_pts", 0)
    tp_pts      = trade.get("tp_pts", 0)
    exit_reason = trade.get("exit_reason", "")
    confidence  = trade.get("confidence", 0)
    vix         = trade.get("vix", 0)
    vix_regime  = trade.get("vix_regime", "")
    atr         = trade.get("atr_5m", 0)
    ml_proba    = trade.get("ml_proba", [0, 0])
    breakeven   = trade.get("breakeven_hit", False)
    trail       = trade.get("trail_activated", False)
    smc_signal  = trade.get("smc_signal", "NEUTRAL")
    entry_time  = trade.get("entry_time", "")

    rr_planned  = round(tp_pts / sl_pts, 2) if sl_pts else 0
    rr_actual   = round(pnl_pts / sl_pts, 2) if sl_pts and pnl_pts > 0 else round(pnl_pts / sl_pts, 2) if sl_pts else 0
    outcome     = "WIN" if pnl_pts > 0 else "LOSS" if pnl_pts < 0 else "BE"

    return f"""You are a quantitative trading coach reviewing a Nifty 50 options scalp trade.
Analyze this trade objectively — identify what went right, what went wrong, and give 2-3 specific
actionable improvements for future similar setups.

=== TRADE SUMMARY ===
Time:        {entry_time}
Direction:   {direction} ({strike_mode})
Entry Spot:  {entry_spot}
Exit Spot:   {exit_spot}
P&L:         {pnl_pts:+.0f} pts  ({outcome})

=== PARAMETERS ===
Stop Loss:     {sl_pts:.0f} pts
Take Profit:   {tp_pts:.0f} pts
Planned R:R:   1:{rr_planned}
Actual R:R:    1:{rr_actual} (positive = win, negative = loss)
Exit Reason:   {exit_reason}
Breakeven hit: {breakeven}
Trail active:  {trail}

=== MARKET CONTEXT ===
VIX:        {vix:.1f} ({vix_regime})
ATR (5m):   {atr:.0f} pts
ML Proba:   CALL={ml_proba[0] if isinstance(ml_proba, list) and len(ml_proba) > 0 else 0:.1%}  PUT={ml_proba[1] if isinstance(ml_proba, list) and len(ml_proba) > 1 else 0:.1%}
Confidence: {confidence}%
SMC Signal: {smc_signal}

Please give:
1. ASSESSMENT: Was this a high-quality setup? Why/why not?
2. EXECUTION: Any issues with entry/exit timing?
3. IMPROVEMENTS: 2-3 specific, measurable rule changes for next time.
Keep response under 200 words. Be direct and quantitative."""


def _build_eod_prompt(trades: list) -> str:
    total     = len(trades)
    wins      = sum(1 for t in trades if t.get("pnl_pts", 0) > 0)
    losses    = sum(1 for t in trades if t.get("pnl_pts", 0) < 0)
    total_pnl = sum(t.get("pnl_pts", 0) for t in trades)
    avg_conf  = (sum(t.get("confidence", 0) for t in trades) / total) if total else 0

    exit_reasons = {}
    for t in trades:
        r = t.get("exit_reason", "UNKNOWN")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    trade_lines = "\n".join([
        f"  {i+1}. {t.get('direction','?')} {t.get('strike_mode','ATM')} | "
        f"P&L={t.get('pnl_pts',0):+.0f}pts | conf={t.get('confidence',0)}% | "
        f"exit={t.get('exit_reason','?')} | VIX={t.get('vix',0):.1f}"
        for i, t in enumerate(trades)
    ])

    return f"""You are a quantitative trading coach reviewing today's full Nifty options session.

=== SESSION SUMMARY ===
Total Trades:  {total}
Wins:          {wins}   Losses: {losses}
Win Rate:      {wins/total:.0%} (target >60%)
Total P&L:     {total_pnl:+.0f} pts
Avg Confidence:{avg_conf:.0f}%
Exit Breakdown:{json.dumps(exit_reasons)}

=== INDIVIDUAL TRADES ===
{trade_lines}

Please provide:
1. SESSION QUALITY: Was today's execution disciplined? Key observations.
2. PATTERNS: Any repeating mistakes or edge patterns across trades?
3. RULE CHANGES: 2-3 specific rule adjustments for tomorrow.
4. REGIME NOTE: Given today's VIX and structure, what conditions to watch for tomorrow.
Keep under 300 words. Be direct, quantitative, and actionable."""
