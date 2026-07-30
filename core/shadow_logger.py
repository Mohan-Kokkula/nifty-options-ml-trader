"""
shadow_logger.py — Logs "would-have-traded" decisions to disk for analysis.

When a signal is REJECTED by a gate, we log it as a "shadow trade" with:
  - the gate that killed it
  - what the signal looked like
  - what the price did after (for post-hoc evaluation)

This builds your dataset of missed opportunities so you can:
  1. Validate gates aren't over-restrictive
  2. Backtest the gates' value (does removing them help or hurt?)
  3. Find systematic blind spots

Output: data/shadow_trades.jsonl (one JSON per line, append-only).

Read with:
    cat data/shadow_trades.jsonl | python -m json.tool

Outcome resolution strategy:
  - In-session: update_outcomes() is called each cycle with current spot;
    entries logged in this session resolve within 30 min and are written back.
  - Cross-session: resolve_historical_outcomes() is called at startup;
    it reads nifty_5min.csv and fills in all NULL outcomes retroactively.

Win condition (simplified):
  CALL wins  if Nifty high in next 30-min window rises >= WIN_THRESHOLD pts
             AND Nifty low in that window never drops >= SL_THRESHOLD pts first.
  PUT wins   if Nifty low drops >= WIN_THRESHOLD pts
             AND Nifty high never rises >= SL_THRESHOLD pts first.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)



# Resolved from this file's own location, not the process's working
# directory. A CWD-relative path here previously depended on how the bot
# was launched (systemd WorkingDirectory, shell cwd, etc.) rather than the
# repo layout, which is fragile: it can silently resolve to a different (or
# unwritable) location under a launch method it wasn't tested against.
SHADOW_FILE = Path(__file__).resolve().parent.parent / "data" / "shadow_trades.jsonl"

# Outcome thresholds (Nifty points)
WIN_THRESHOLD = 25    # target: price moved >= 25 pts in signal direction
SL_THRESHOLD  = 50    # stop-loss proxy: price moved >= 50 pts against signal

# OHLCV CSV used for retroactive resolution
OHLCV_CSV = Path("data/nifty_5min.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Retroactive outcome resolver (called once at startup)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_historical_outcomes(ohlcv_csv: str | Path = OHLCV_CSV) -> dict:
    """
    Read shadow_trades.jsonl, find all entries with would_have_won=None,
    resolve them from the OHLCV CSV, and rewrite the file.

    Returns a summary dict: {resolved: int, already_done: int, no_data: int}
    """
    ohlcv_csv = Path(ohlcv_csv)
    if not SHADOW_FILE.exists():
        logger.debug("shadow_trades.jsonl not found — nothing to resolve")
        return {"resolved": 0, "already_done": 0, "no_data": 0}

    # Load & parse entries
    raw_lines = SHADOW_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            pass

    pending = [e for e in entries if e.get("would_have_won") is None and e.get("signal") in ("CALL", "PUT")]
    already_done = len(entries) - len(pending)

    if not pending:
        logger.debug(f"shadow outcomes: {already_done} already resolved, nothing new")
        return {"resolved": 0, "already_done": already_done, "no_data": 0}

    # Load OHLCV (5-min) for fast lookup — parse datetime index
    if not ohlcv_csv.exists():
        logger.warning(f"OHLCV CSV not found at {ohlcv_csv} — cannot resolve shadow outcomes")
        return {"resolved": 0, "already_done": already_done, "no_data": len(pending)}

    try:
        df = pd.read_csv(ohlcv_csv, parse_dates=[0])
        df.columns = [c.lower().strip() for c in df.columns]
        # detect datetime column
        dt_col = next((c for c in df.columns if "date" in c or "time" in c), df.columns[0])
        df = df.rename(columns={dt_col: "dt"})
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt").sort_index()
        # ensure numeric OHLC
        for col in ("high", "low", "close", "open"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception as e:
        logger.warning(f"OHLCV load failed: {e}")
        return {"resolved": 0, "already_done": already_done, "no_data": len(pending)}

    resolved_count = 0
    no_data_count = 0

    for e in pending:
        try:
            ts = datetime.fromisoformat(e["ts"])
            signal = e["signal"]   # "CALL" or "PUT"
            spot = float(e["spot"])

            # Look-ahead window: ts to ts + 35 min (7 × 5-min bars)
            window_start = ts
            window_end   = ts + timedelta(minutes=35)

            future = df.loc[window_start:window_end]
            if future.empty or len(future) < 2:
                no_data_count += 1
                continue

            highs = future["high"].values
            lows  = future["low"].values
            bars_5  = min(1, len(future))
            bars_15 = min(3, len(future))
            bars_30 = min(6, len(future))

            # Compute outcome_5min / 15min / 30min = directional move from entry spot
            def directional_move(idx_end):
                """Max favourable move in bars 0..idx_end (vs entry spot)."""
                if signal == "CALL":
                    return float(max(highs[:idx_end+1])) - spot
                else:
                    return spot - float(min(lows[:idx_end+1]))

            e["outcome_5min"]  = round(directional_move(bars_5 - 1), 1)
            e["outcome_15min"] = round(directional_move(bars_15 - 1), 1)
            e["outcome_30min"] = round(directional_move(bars_30 - 1), 1)

            # would_have_won: WIN_THRESHOLD reached before SL_THRESHOLD hit
            if signal == "CALL":
                best  = max(highs[:bars_30]) - spot
                worst = spot - min(lows[:bars_30])
            else:
                best  = spot - min(lows[:bars_30])
                worst = max(highs[:bars_30]) - spot

            if best >= WIN_THRESHOLD and worst < SL_THRESHOLD:
                e["would_have_won"] = True
            elif worst >= SL_THRESHOLD:
                e["would_have_won"] = False
            else:
                # Small move either way — treated as near-miss loss
                e["would_have_won"] = False

            resolved_count += 1

        except Exception as ex:
            logger.debug(f"Resolution failed for entry ts={e.get('ts')}: {ex}")
            no_data_count += 1

    # Rewrite the file atomically
    tmp_path = SHADOW_FILE.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, default=str) + "\n")
        tmp_path.replace(SHADOW_FILE)
        logger.info(
            f"shadow outcomes resolved: {resolved_count} new | "
            f"{already_done} already done | {no_data_count} no data"
        )
    except Exception as e:
        logger.warning(f"Failed to write resolved shadow trades: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {"resolved": resolved_count, "already_done": already_done, "no_data": no_data_count}


# ──────────────────────────────────────────────────────────────────────────────
# In-session file write-back helper
# ──────────────────────────────────────────────────────────────────────────────

def _update_line_in_file(ts_key: str, update: dict) -> None:
    """
    Find the line in shadow_trades.jsonl matching ts_key and merge update fields.
    Rewrites the file in-place (atomic via tmp file).
    """
    if not SHADOW_FILE.exists():
        return
    try:
        lines = SHADOW_FILE.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("ts") == ts_key:
                    entry.update(update)
                new_lines.append(json.dumps(entry, default=str))
            except Exception:
                new_lines.append(line)

        tmp = SHADOW_FILE.with_suffix(".tmp")
        tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        tmp.replace(SHADOW_FILE)
    except Exception as e:
        logger.debug(f"shadow file write-back failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# ShadowLogger
# ──────────────────────────────────────────────────────────────────────────────

class ShadowLogger:
    """Singleton — logs missed signals + their realized outcome."""

    _instance: Optional["ShadowLogger"] = None
    _cls_lock = threading.Lock()

    @classmethod
    def get(cls) -> "ShadowLogger":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = ShadowLogger()
            return cls._instance

    def __init__(self):
        SHADOW_FILE.parent.mkdir(exist_ok=True, parents=True)
        self._lock = threading.Lock()
        # In-memory tracking for outcome enrichment (current session only)
        self._pending: list = []

    def log_skip(self, *, cycle: int, signal: str, conf_pct: float,
                 reason: str, spot: float, ml_proba: list,
                 regime: str = "", extra: dict | None = None) -> None:
        """Log a rejected signal."""
        try:
            entry = {
                "ts": datetime.now().isoformat(),
                "cycle": cycle,
                "signal": signal,            # CALL / PUT / SKIP
                "ml_proba": list(ml_proba),
                "conf_pct": round(conf_pct, 1),
                "reason": reason,
                "spot": round(spot, 2),
                "regime": regime,
                "extra": extra or {},
                "outcome_5min": None,        # filled later
                "outcome_15min": None,
                "outcome_30min": None,
                "would_have_won": None,
            }
            with self._lock:
                self._pending.append(entry)
                # Append to file immediately
                with SHADOW_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            # Never block trading on a logging failure -- but a silent
            # DEBUG-level swallow here previously let this fail for weeks
            # without anyone noticing (no new entries since 2026-06-05).
            # Loud enough to show up in the normal log, still fail-open.
            logger.error(
                f"shadow log_skip failed for {reason} ({signal}, conf={conf_pct}%) "
                f"-> {SHADOW_FILE}: {e}"
            )

    def update_outcomes(self, current_spot: float) -> None:
        """
        Evaluate pending entries: if 5/15/30 min has passed since they were
        logged, compute what the trade would have done.

        Call this on every pilot cycle. Outcomes are also written back to the
        JSONL file so they survive restarts.
        """
        try:
            with self._lock:
                still_pending = []
                for e in self._pending:
                    age_sec = (datetime.now() - datetime.fromisoformat(e["ts"])).total_seconds()
                    spot_at_skip = e["spot"]
                    move = current_spot - spot_at_skip
                    if e["signal"] == "PUT":
                        move = -move

                    updated_fields = {}

                    if e["outcome_5min"] is None and age_sec >= 300:
                        e["outcome_5min"] = round(move, 1)
                        updated_fields["outcome_5min"] = e["outcome_5min"]

                    if e["outcome_15min"] is None and age_sec >= 900:
                        e["outcome_15min"] = round(move, 1)
                        updated_fields["outcome_15min"] = e["outcome_15min"]

                    if e["outcome_30min"] is None and age_sec >= 1800:
                        e["outcome_30min"] = round(move, 1)
                        updated_fields["outcome_30min"] = e["outcome_30min"]
                        # Final verdict: WIN = moved >= WIN_THRESHOLD without SL hit
                        # For in-session, we use current_spot as best-effort;
                        # precise SL simulation is only done in retroactive resolver.
                        e["would_have_won"] = move >= WIN_THRESHOLD
                        updated_fields["would_have_won"] = e["would_have_won"]

                    if updated_fields:
                        # Write resolved fields back to JSONL (non-blocking)
                        ts_key = e["ts"]
                        fields = dict(updated_fields)
                        threading.Thread(
                            target=_update_line_in_file,
                            args=(ts_key, fields),
                            daemon=True,
                        ).start()

                    if age_sec < 1800:
                        still_pending.append(e)

                self._pending = still_pending

        except Exception as e:
            logger.debug(f"shadow update_outcomes failed: {e}")

    def stats(self) -> dict:
        """Read whole file and produce summary."""
        try:
            if not SHADOW_FILE.exists():
                return {"entries": 0}
            entries = []
            with SHADOW_FILE.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
            evaluated = [e for e in entries if e.get("would_have_won") is not None]
            wins = sum(1 for e in evaluated if e["would_have_won"])
            return {
                "entries_total": len(entries),
                "entries_evaluated": len(evaluated),
                "would_have_won": wins,
                "would_have_lost": len(evaluated) - wins,
                "missed_win_rate_pct": round(
                    100 * wins / max(1, len(evaluated)), 1
                ),
                "by_reason": dict(self._reason_counts(entries)),
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _reason_counts(entries: list) -> dict:
        from collections import Counter
        return Counter(
            e.get("reason", "").split(":")[0] for e in entries
        ).most_common(20)


def get_shadow_logger() -> ShadowLogger:
    return ShadowLogger.get()
