"""
trade_journal.py — Per-Trade Win/Loss Journal
==============================================
Logs every trade with full context: entry/exit, session, direction,
confidence, gates passed, exit reason, P&L. Enables breakdown analysis
like "Morning CALL trades: 8/10 wins".

Stores as JSON lines (one JSON object per line) for easy pandas loading.
File: logs/trade_journal.jsonl
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path(__file__).parent.parent / "logs"
JOURNAL_FILE = JOURNAL_DIR / "trade_journal.jsonl"


class TradeJournal:
    """Append-only per-trade journal with full context."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else JOURNAL_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._open_trade: Optional[dict] = None

    def record_entry(
        self,
        direction: str,       # "CALL" or "PUT"
        option_type: str,     # "CE" or "PE"
        strike_mode: str,     # "ATM", "OTM1", etc.
        entry_spot: float,
        entry_premium: float,
        confidence: int,
        sl_pts: float,
        tp_pts: float,
        sl_price: float,
        tp_price: float,
        lots: int,
        qty: int,
        cycle: int,
        symbol: str = "",
        atr_5m: float = 0.0,
        atr_15m: float = 0.0,
        vix: float = 0.0,
        vix_regime: str = "",
        adx_15m: float = 0.0,
        ml_proba: list = None,
        recovery_mode: bool = False,
        gap_override: bool = False,
        gamma_squeeze: bool = False,
        dte: int = -1,
        is_expiry: bool = False,
        pcr: float = 0.0,
        pcr_available: bool = False,   # Phase 3 fix: False = OI fetch failed, pcr is default 1.0
        oi_magnetic_strike: float = 0.0,
        is_dry_run: bool = False,
        # Slippage fields (Issue #7) — all optional, default to 0 so
        # existing call-sites that don't pass them don't break
        signal_premium:    float = 0.0,   # option ask at signal time (fair value)
        limit_price:       float = 0.0,   # IOC limit price submitted
        fill_price:        float = 0.0,   # actual execution price (same as entry_premium if available)
        market_impact_pts: float = 0.0,   # fill_price - signal_premium
        exec_slippage_pts: float = 0.0,   # fill_price - limit_price
        strategy_name:     str = "",      # for per-strategy analytics breakdown
    ):
        """Record trade entry with full context."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        # Classify session
        if hour == 9 and minute < 45:
            session = "OPENING"      # 09:15-09:45
        elif hour < 11 or (hour == 10 and minute <= 30):
            session = "MORNING"      # 09:45-10:30
        elif hour < 13 or (hour == 13 and minute <= 30):
            session = "MIDDAY"       # 10:30-13:30
        else:
            session = "AFTERNOON"    # 13:30-15:15

        trade = {
            "trade_date": now.strftime("%Y-%m-%d"),
            "entry_time": now.strftime("%H:%M:%S"),
            "session": session,
            "direction": direction,
            "option_type": option_type,
            "strike_mode": strike_mode,
            "symbol": symbol,
            "entry_spot": round(entry_spot, 1),
            "entry_premium": round(entry_premium, 2),
            "confidence": confidence,
            "sl_pts": round(sl_pts, 1),
            "tp_pts": round(tp_pts, 1),
            "sl_price": round(sl_price, 1),
            "tp_price": round(tp_price, 1),
            "rr_ratio": round(tp_pts / sl_pts, 2) if sl_pts > 0 else 0,
            "lots": lots,
            "qty": qty,
            "cycle": cycle,
            # Context
            "atr_5m": round(atr_5m, 1),
            "atr_15m": round(atr_15m, 1),
            "vix": round(vix, 1),
            "vix_regime": vix_regime,
            "adx_15m": round(adx_15m, 1),
            "ml_proba_call": round(ml_proba[0], 3) if ml_proba else 0,
            "ml_proba_put": round(ml_proba[1], 3) if ml_proba else 0,
            "ml_proba_skip": round(ml_proba[2], 3) if ml_proba else 0,
            "recovery_mode": recovery_mode,
            "gap_override": gap_override,
            "gamma_squeeze": gamma_squeeze,
            "dte": dte,
            "is_expiry": is_expiry,
            "pcr": round(pcr, 2),
            "pcr_available": pcr_available,   # False = OI fetch failed; pcr is sentinel 1.0
            "oi_magnetic_strike": round(oi_magnetic_strike, 1),
            "is_dry_run": is_dry_run,
            "strategy_name": strategy_name,
            # Slippage fields (Issue #7) — zero means not captured (DRY_RUN or pre-fix)
            "signal_premium":    round(signal_premium,    4),
            "limit_price":       round(limit_price,       4),
            "fill_price":        round(fill_price,        4),
            "market_impact_pts": round(market_impact_pts, 4),
            "exec_slippage_pts": round(exec_slippage_pts, 4),
            # Exit fields (filled on close)
            "exit_time": None,
            "exit_spot": None,
            "exit_reason": None,
            "pnl_pts": None,
            "pnl_rupees": None,
            "result": None,  # "WIN", "LOSS", "BREAKEVEN"
            "duration_sec": None,
            "peak_profit_pts": None,
            "max_drawdown_pts": None,
            "breakeven_hit": None,
            "trail_activated": None,
        }

        with self._lock:
            self._open_trade = trade

        logger.info(
            f"JOURNAL ENTRY: {direction} {option_type} ({strike_mode}) "
            f"spot={entry_spot:.0f} conf={confidence}% SL={sl_pts:.0f} TP={tp_pts:.0f} "
            f"session={session} {'[DRY_RUN]' if is_dry_run else ''}"
        )

    def record_exit(
        self,
        exit_spot: float,
        exit_reason: str,      # "TP", "SL", "TRAIL_SL", "BE_SL", "PREMIUM_STOP", "TIME_EXIT"
        pnl_pts: float,
        pnl_rupees: float,
        peak_profit_pts: float = 0.0,
        max_drawdown_pts: float = 0.0,
        breakeven_hit: bool = False,
        trail_activated: bool = False,
    ):
        """Record trade exit and write complete trade to journal."""
        with self._lock:
            trade = self._open_trade
            if trade is None:
                logger.warning("JOURNAL: No open trade to close")
                return
            self._open_trade = None

        now = datetime.now()
        trade["exit_time"] = now.strftime("%H:%M:%S")
        trade["exit_spot"] = round(exit_spot, 1)
        trade["exit_reason"] = exit_reason
        trade["pnl_pts"] = round(pnl_pts, 1)
        trade["pnl_rupees"] = round(pnl_rupees, 0)
        trade["peak_profit_pts"] = round(peak_profit_pts, 1)
        trade["max_drawdown_pts"] = round(max_drawdown_pts, 1)
        trade["breakeven_hit"] = breakeven_hit
        trade["trail_activated"] = trail_activated

        # Classify result
        if pnl_pts > 3:
            trade["result"] = "WIN"
        elif pnl_pts < -3:
            trade["result"] = "LOSS"
        else:
            trade["result"] = "BREAKEVEN"

        # Mark as EXIT event so EOD summary can find it
        # BUG FIX 2026-06-04: was missing → EOD summary returned "No trades today"
        # even when real trades happened (eod_summary._read_today_trades filters on event=="EXIT")
        trade["event"] = "EXIT"

        # Duration
        try:
            entry_t = datetime.strptime(
                f"{trade['trade_date']} {trade['entry_time']}", "%Y-%m-%d %H:%M:%S"
            )
            trade["duration_sec"] = int((now - entry_t).total_seconds())
        except Exception:
            trade["duration_sec"] = 0

        # Write to JSONL file
        self._write_trade(trade)

        logger.info(
            f"JOURNAL EXIT: {trade['result']} | {trade['direction']} {trade['option_type']} "
            f"P&L={pnl_pts:+.0f}pts (Rs.{pnl_rupees:+.0f}) | "
            f"reason={exit_reason} | duration={trade['duration_sec']}s"
        )

    def record_dry_run_signal(
        self,
        direction: str,
        option_type: str,
        strike_mode: str,
        spot: float,
        confidence: int,
        sl_pts: float,
        tp_pts: float,
        cycle: int,
        ml_proba: list = None,
        **kwargs,
    ):
        """Record a DRY_RUN signal (no execution). Write immediately."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        if hour == 9 and minute < 45:
            session = "OPENING"
        elif hour < 11 or (hour == 10 and minute <= 30):
            session = "MORNING"
        elif hour < 13 or (hour == 13 and minute <= 30):
            session = "MIDDAY"
        else:
            session = "AFTERNOON"

        trade = {
            "trade_date": now.strftime("%Y-%m-%d"),
            "entry_time": now.strftime("%H:%M:%S"),
            "session": session,
            "direction": direction,
            "option_type": option_type,
            "strike_mode": strike_mode,
            "entry_spot": round(spot, 1),
            "confidence": confidence,
            "sl_pts": round(sl_pts, 1),
            "tp_pts": round(tp_pts, 1),
            "sl_price": round(spot - sl_pts if direction == "CALL" else spot + sl_pts, 1),
            "tp_price": round(spot + tp_pts if direction == "CALL" else spot - tp_pts, 1),
            "rr_ratio": round(tp_pts / sl_pts, 2) if sl_pts > 0 else 0,
            "cycle": cycle,
            "ml_proba_call": round(ml_proba[0], 3) if ml_proba else 0,
            "ml_proba_put": round(ml_proba[1], 3) if ml_proba else 0,
            "ml_proba_skip": round(ml_proba[2], 3) if ml_proba else 0,
            "is_dry_run": True,
            # No exit data for dry run — track manually or check later
            "exit_time": None,
            "exit_spot": None,
            "exit_reason": "DRY_RUN",
            "pnl_pts": None,
            "pnl_rupees": None,
            "result": "PAPER",
            **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))},
        }

        self._write_trade(trade)
        logger.info(
            f"JOURNAL PAPER: {direction} {option_type} spot={spot:.0f} "
            f"conf={confidence}% SL={sl_pts:.0f} TP={tp_pts:.0f} session={session}"
        )

    def _write_trade(self, trade: dict):
        """Append a trade record as a JSON line."""
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trade, default=str) + "\n")
            except Exception as e:
                logger.error(f"JOURNAL write failed: {e}")

    def find_open_entry(self, today_only: bool = True) -> Optional[dict]:
        """
        Scan the journal for a live-trade ENTRY record that has no matching EXIT.

        Used by position-reconciliation on startup to reconstruct sl_price /
        tp_price / entry_price for a position that survived a bot restart.

        Rules:
          - Looks only at today's records (today_only=True) to avoid matching
            a stale entry from a previous day that was never closed in the journal.
          - Prefers the MOST RECENT un-exited live entry (is_dry_run=False).
          - Returns None if no such record exists.
        """
        if not self.path.exists():
            return None

        today_str = datetime.now().strftime("%Y-%m-%d")
        entries: list = []   # list of entry records for today
        exited: set  = set() # set of entry_time strings for which EXIT exists

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    td = t.get("trade_date", "")
                    if today_only and td != today_str:
                        continue

                    # Classify: entry vs exit
                    if t.get("event") == "EXIT" or t.get("exit_reason") is not None:
                        # Key exits by (trade_date, entry_time)
                        exited.add((td, t.get("entry_time", "")))
                    elif (t.get("exit_reason") is None
                          and t.get("entry_time") is not None
                          and not t.get("is_dry_run", False)
                          and t.get("result") is None):   # result=None means not yet closed
                        entries.append(t)
        except Exception as e:
            logger.warning(f"JOURNAL find_open_entry failed: {e}")
            return None

        # Filter to entries with no matching exit
        open_entries = [
            e for e in entries
            if (e.get("trade_date", ""), e.get("entry_time", "")) not in exited
        ]

        if not open_entries:
            return None

        # Return the most recent one
        return max(open_entries, key=lambda e: e.get("entry_time", ""))

    def mark_entry_closed(self, entry: dict, reason: str = "RECONCILE_CLOSED") -> None:
        """
        Write a synthetic EXIT record for an orphaned journal entry.

        Used when reconciliation finds the broker is FLAT but the journal
        still shows an open entry — i.e. the position was closed outside
        the bot (manual close, broker auto-square-off, or mid-restart).
        """
        if entry is None:
            return
        synthetic_exit = dict(entry)
        synthetic_exit["exit_time"]    = datetime.now().strftime("%H:%M:%S")
        synthetic_exit["exit_spot"]    = 0.0
        synthetic_exit["exit_reason"]  = reason
        synthetic_exit["pnl_pts"]      = 0.0
        synthetic_exit["pnl_rupees"]   = 0.0
        synthetic_exit["result"]       = "UNKNOWN"
        synthetic_exit["event"]        = "EXIT"
        synthetic_exit["duration_sec"] = 0
        self._write_trade(synthetic_exit)
        logger.warning(f"JOURNAL: synthetic EXIT written for orphaned entry "
                       f"({entry.get('direction')} {entry.get('option_type')} "
                       f"@ {entry.get('entry_time')}, reason={reason})")

    def get_daily_slippage_report(self, today: str = "") -> dict:
        """
        Compute slippage statistics from today's journal entries.

        Reads all REAL live trade entries (is_dry_run=False, signal_premium > 0)
        and returns a summary dict:
          {
            "n_trades":          int,    trades with slippage data
            "avg_market_impact": float,  mean(fill - signal)  in option pts
            "avg_exec_slippage": float,  mean(fill - limit)   in option pts
            "max_market_impact": float,  worst single trade
            "total_slip_cost_rs":float,  sum(market_impact × qty) in rupees
            "slip_over_1pt":     int,    # trades where market_impact > 1 pt
            "entries":           list,   raw per-trade rows (for detailed view)
          }
        Returns empty dict with n_trades=0 when no qualifying entries exist.
        """
        today_str = today or datetime.now().strftime("%Y-%m-%d")
        result = {
            "n_trades": 0, "avg_market_impact": 0.0, "avg_exec_slippage": 0.0,
            "max_market_impact": 0.0, "total_slip_cost_rs": 0.0,
            "slip_over_1pt": 0, "entries": [],
        }
        if not self.path.exists():
            return result

        rows = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if t.get("trade_date") != today_str:
                        continue
                    if t.get("is_dry_run"):
                        continue
                    if t.get("event") == "EXIT":
                        continue
                    sp = float(t.get("signal_premium", 0) or 0)
                    if sp <= 0:
                        continue   # pre-fix entry, no slippage data
                    rows.append(t)
        except Exception as e:
            logger.warning(f"slippage report read failed: {e}")
            return result

        if not rows:
            return result

        impacts = [float(r.get("market_impact_pts", 0) or 0) for r in rows]
        execs   = [float(r.get("exec_slippage_pts",  0) or 0) for r in rows]
        qtys    = [int(  r.get("qty", 0)              or 0) for r in rows]

        n = len(rows)
        result["n_trades"]          = n
        result["avg_market_impact"] = round(sum(impacts) / n, 4)
        result["avg_exec_slippage"] = round(sum(execs)   / n, 4)
        result["max_market_impact"] = round(max(impacts),     4)
        result["total_slip_cost_rs"]= round(
            sum(imp * qty for imp, qty in zip(impacts, qtys)), 2
        )
        result["slip_over_1pt"] = sum(1 for i in impacts if i > 1.0)
        result["entries"]       = rows
        return result

    def load_trades(self, days: int = 30) -> list:
        """Load recent trades from journal file."""
        if not self.path.exists():
            return []

        cutoff = date.today().isoformat()
        trades = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        trades.append(t)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"JOURNAL load failed: {e}")

        # Filter to recent N days
        if days > 0:
            from datetime import timedelta
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            trades = [t for t in trades if t.get("trade_date", "") >= cutoff]

        return trades
