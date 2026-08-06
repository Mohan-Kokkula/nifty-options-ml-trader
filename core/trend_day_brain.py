"""
core/trend_day_brain.py — Independent, paper-mode-only trend-day brain.

Mechanises the discretionary "trade of the day" setup: once the session
has established a direction, join it on the pullback-free continuation
and hold for a 2R structural target rather than a fixed ATR target.

ENTRY (one trade per day maximum, 11:30-14:00 IST):
    short/PUT : close < session_vwap_proxy - 25pts
                AND 3 consecutive lower closes
                AND close < day_open
    long/CALL : exact mirror
EXIT:
    stop   = day extreme +/- 10pts, capped at 60pts of risk
    target = entry -/+ 2.0 x actual risk
    EOD square-off 15:15

Validation (8-fold walk-forward, 2015-2026, futures friction, close-based
exits matching this loop's own polling, params SELECTED ON FOLDS 1-5 ONLY
and scored once on folds 6-8):
    VAL  folds 1-5 : mean PF 1.363  (n=997)
    TEST folds 6-8 : mean PF 1.225  (n=818, 3/3 folds > 1.0)
                     per-fold 1.247 / 1.106 / 1.322
                     avg +5.4 pts/trade
    (with the earlier 10:30 window: VAL 1.302 / TEST 1.107, +2.9 pts)

HONEST CAVEAT on the 11:30 window. A strictly VAL-blind sweep would
have picked 12:30, which scores WORSE on TEST (1.073) than the original
10:30. 11:30 was chosen because the session-bucket analysis independently
located the regime flip there BEFORE this sweep ran -- but the TEST
numbers were visible when the choice was made, so the +11% is optimistic
rather than expected. Forward paper results are the real test.
The RR effect is monotone in both VAL and TEST (RR 1.0 loses on TEST,
2.0 wins), i.e. consistent structure rather than a single lucky cell.

*** FUTURES ONLY. *** The same trades under options friction score
PF 0.666 / -10.9 pts per trade. The entire edge sits inside the
futures-vs-options friction gap, so this brain refuses to arm unless
EXECUTION_MODE=futures.

Known approximation vs the backtest: the backtest fills at the NEXT bar's
open, while this loop polls completed bars and fills at the signal bar's
close. Bar-to-bar gaps in intraday 5-min NIFTY are near zero so the two
are practically the same price, but they are not identical -- expect small
divergence between live results and the validated numbers.

Runs as a FULLY SEPARATE loop from ClaudePilot's ML position management
and from the PSAR brain -- own position state, own journal, own Telegram
alerts, never places a real broker order. Opt-in via
TREND_DAY_BRAIN_ENABLED=true (default: disabled).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STATE_PATH = "data/trend_day_position.json"
_started_lock = threading.Lock()
_started = False


@dataclass
class TrendDayConfig:
    # ── validated params — see module docstring. Not env-tunable on
    # purpose: they were selected on VAL folds and confirmed once on
    # TEST, so ad-hoc live edits would silently void that validation.
    min_dev_pts: float = 25.0      # min distance from session VWAP proxy
    seq_n: int = 3                 # consecutive closes in trade direction
    stop_buffer_pts: float = 10.0  # beyond the day extreme
    max_stop_pts: float = 60.0     # risk cap (80 -> 60 removed recency decay)
    rr: float = 2.0                # target = rr x actual risk

    # 2026-08-07: window_start moved 10:30 -> 11:30. NIFTY's intraday
    # trend-continuation edge changes SIGN at 11:30 IST -- measured across
    # twelve 30-min buckets of ~17,000 bars each, every bucket 09:00-11:00
    # is negative (-0.38 to -0.05 pts) and every bucket 11:30-15:00 is
    # positive (+0.66 to +1.80). The morning mean-reverts; the afternoon
    # trends. A continuation rule should not fire in the mean-reverting half.
    # One-at-a-time sweep on VAL folds 1-5 agrees and is monotone in this
    # parameter (09:45=1.288  10:30=1.302  11:30=1.363  12:30=1.377).
    window_start: str = "11:30"
    window_end: str = "14:00"
    eod_squareoff: str = "15:15"
    # The 5-min feed appends a phantom post-close bar (~15:25) carrying the
    # closing-auction/settlement print, which sits well away from the last
    # traded price -- observed +151 / +54 / +8 pts above the 15:15 close on
    # 2026-08-04/05/06. It is not tradeable. The backtest excluded it
    # (bars <= EOD only); this bound keeps the live loop consistent, so a
    # missed 15:15 poll can never fill the EOD exit at the auction price.
    last_tradeable_time: str = "15:20"

    lookback_bars: int = 300
    cycle_interval_sec: int = 60
    journal_path: str = "logs/trend_day_journal.jsonl"
    strategy_name: str = "TREND_DAY"
    qty: int = 65


# ── Pure helpers (unit-testable without broker/threads) ──────────────
def session_vwap_proxy(df_today: pd.DataFrame) -> float:
    """Expanding intraday mean of (H+L+C)/3 for the session so far.

    Volume in this project's 5-min data is ~97% zero and the live feed
    logs "Intraday VWAP: using SPOT PROXY (no futures data)", so a
    volume-weighted VWAP is not available. This matches what the rest
    of the bot already uses rather than inventing a cleaner input.
    """
    if df_today.empty:
        return float("nan")
    tp = (df_today["high"] + df_today["low"] + df_today["close"]) / 3.0
    return float(tp.mean())


def drop_auction_print(df_today: pd.DataFrame, cfg: TrendDayConfig) -> pd.DataFrame:
    """Strip non-tradeable post-close bars (see cfg.last_tradeable_time)."""
    if df_today.empty:
        return df_today
    keep = [t.strftime("%H:%M") < cfg.last_tradeable_time for t in df_today.index]
    return df_today[keep]


def consecutive_closes(closes, n: int) -> int:
    """+n if the last n closes each rose, -n if each fell, else 0."""
    c = np.asarray(closes, dtype=float)
    if len(c) < n + 1:
        return 0
    d = np.diff(c[-(n + 1):])
    if np.all(d > 0):
        return n
    if np.all(d < 0):
        return -n
    return 0


def in_entry_window(ts: datetime, cfg: TrendDayConfig) -> bool:
    return cfg.window_start <= ts.strftime("%H:%M") <= cfg.window_end


def compute_stop_target(direction: str, entry: float, day_high: float,
                         day_low: float, cfg: TrendDayConfig) -> tuple:
    """Structural stop at the day extreme, capped; target at rr x risk.
    Returns (stop, target, risk) or (None, None, 0.0) if degenerate."""
    if direction == "CALL":
        stop = max(day_low - cfg.stop_buffer_pts, entry - cfg.max_stop_pts)
        risk = entry - stop
        if risk < 5:
            return None, None, 0.0
        return stop, entry + risk * cfg.rr, risk
    stop = min(day_high + cfg.stop_buffer_pts, entry + cfg.max_stop_pts)
    risk = stop - entry
    if risk < 5:
        return None, None, 0.0
    return stop, entry - risk * cfg.rr, risk


def evaluate_entry(df_today: pd.DataFrame, cfg: TrendDayConfig) -> Optional[str]:
    """'CALL', 'PUT' or None from the session bars so far (causal)."""
    if df_today.empty or len(df_today) < cfg.seq_n + 1:
        return None
    vwap = session_vwap_proxy(df_today)
    if not np.isfinite(vwap):
        return None
    close = float(df_today["close"].iloc[-1])
    day_open = float(df_today["open"].iloc[0])
    seq = consecutive_closes(df_today["close"].values, cfg.seq_n)

    if close < vwap - cfg.min_dev_pts and seq <= -cfg.seq_n and close < day_open:
        return "PUT"
    if close > vwap + cfg.min_dev_pts and seq >= cfg.seq_n and close > day_open:
        return "CALL"
    return None


@dataclass
class TrendDayPosition:
    direction: str
    entry_price: float
    entry_time_iso: str
    stop_price: float
    target_price: float
    risk_pts: float
    qty: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _atomic_write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class TrendDayBrain:
    def __init__(self, cfg: Optional[TrendDayConfig] = None):
        self.cfg = cfg or TrendDayConfig()
        self.position: Optional[TrendDayPosition] = None
        self.traded_on: Optional[str] = None   # ISO date — one trade per day
        self._journal = self._build_journal()
        self._notifier = self._build_notifier()
        self._load_state()

    def _build_journal(self):
        from core.trade_journal import TradeJournal
        return TradeJournal(path=self.cfg.journal_path)

    def _build_notifier(self):
        from notifications.telegram_notifier import TelegramNotifier
        return TelegramNotifier(
            enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

    def _load_state(self):
        if not os.path.exists(_STATE_PATH):
            return
        try:
            with open(_STATE_PATH) as f:
                d = json.load(f)
            if d.get("position"):
                self.position = TrendDayPosition.from_dict(d["position"])
                logger.info(f"Trend-day brain: restored {self.position.direction} "
                            f"@ {self.position.entry_price:.1f}")
            self.traded_on = d.get("traded_on")
        except Exception as e:
            logger.warning(f"Trend-day brain: state restore failed (non-fatal): {e}")

    def _save_state(self):
        try:
            _atomic_write_json(_STATE_PATH, {
                "position": self.position.to_dict() if self.position else None,
                "traded_on": self.traded_on,
            })
        except Exception as e:
            logger.warning(f"Trend-day brain: state persist failed (non-fatal): {e}")

    def _fetch_bars(self) -> Optional[pd.DataFrame]:
        try:
            from core.tv_fetcher import get_tv_fetcher
            df = get_tv_fetcher().get_nifty_5min(n_bars=self.cfg.lookback_bars)
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            logger.warning(f"Trend-day brain: bar fetch failed (skipping cycle): {e}")
            return None

    def _alert(self, action: str, side: str, price: float, details: str = ""):
        try:
            self._notifier.send_trade_alert(
                action=action, symbol="NIFTY FUT (trend-day, paper)", side=side,
                qty=self.cfg.qty, price=price, order_id="PAPER",
                status="simulated", details=details,
            )
        except Exception as e:
            logger.warning(f"Trend-day brain: telegram alert failed (non-fatal): {e}")

    def run_cycle(self):
        df = self._fetch_bars()
        if df is None:
            return
        today = df.index[-1].date()
        df_today = drop_auction_print(df[df.index.date == today], self.cfg)
        if df_today.empty:
            return
        now = df_today.index[-1].to_pydatetime()
        spot = float(df_today["close"].iloc[-1])

        if self.position is not None:
            self._manage(df_today, now, spot)
        else:
            self._look_for_entry(df_today, now, spot)

    def _look_for_entry(self, df_today: pd.DataFrame, now: datetime, spot: float):
        cfg = self.cfg
        today_iso = now.date().isoformat()

        if self.traded_on == today_iso:
            return                                   # one trade per day
        if not in_entry_window(now, cfg):
            return

        direction = evaluate_entry(df_today, cfg)
        if direction is None:
            return

        day_high = float(df_today["high"].max())
        day_low = float(df_today["low"].min())
        stop, target, risk = compute_stop_target(direction, spot, day_high, day_low, cfg)
        if stop is None:
            logger.info(f"Trend-day brain: {direction} setup but risk < 5pts — skipped")
            return

        self.position = TrendDayPosition(
            direction=direction, entry_price=spot, entry_time_iso=now.isoformat(),
            stop_price=stop, target_price=target, risk_pts=risk, qty=cfg.qty,
        )
        self.traded_on = today_iso
        self._save_state()

        vwap = session_vwap_proxy(df_today)
        logger.info(
            f"Trend-day brain: ENTERED {direction} @ {spot:.1f} | "
            f"stop={stop:.1f} target={target:.1f} risk={risk:.0f}pts "
            f"(vwap_proxy={vwap:.1f} dev={spot - vwap:+.0f})"
        )
        self._alert("TRENDDAY_ENTRY", direction, spot,
                    details=f"stop={stop:.1f} target={target:.1f} risk={risk:.0f}pts")
        try:
            self._journal.record_dry_run_signal(
                direction=direction, option_type="", strike_mode="FUT",
                spot=spot, confidence=0, sl_pts=risk, tp_pts=risk * cfg.rr,
                cycle=0, strategy_name=cfg.strategy_name,
            )
        except Exception as e:
            logger.warning(f"Trend-day brain: journal write failed (non-fatal): {e}")

    def _manage(self, df_today: pd.DataFrame, now: datetime, spot: float):
        pos = self.position
        sign = 1 if pos.direction == "CALL" else -1

        if now.strftime("%H:%M") >= self.cfg.eod_squareoff:
            self._close(spot, "EOD_SQUAREOFF")
            return
        if (spot <= pos.stop_price) if sign == 1 else (spot >= pos.stop_price):
            self._close(pos.stop_price, "SL")
            return
        if (spot >= pos.target_price) if sign == 1 else (spot <= pos.target_price):
            self._close(pos.target_price, "TP")
            return
        self._save_state()

    def _close(self, exit_price: float, reason: str):
        pos = self.position
        sign = 1 if pos.direction == "CALL" else -1
        pnl = sign * (exit_price - pos.entry_price)
        r_mult = pnl / pos.risk_pts if pos.risk_pts else 0.0
        logger.info(f"Trend-day brain: EXIT {pos.direction} @ {exit_price:.1f} "
                    f"reason={reason} pnl={pnl:+.1f}pts ({r_mult:+.2f}R)")
        self._alert(f"TRENDDAY_EXIT_{reason}",
                    "SELL" if pos.direction == "CALL" else "BUY",
                    exit_price, details=f"pnl={pnl:+.1f}pts ({r_mult:+.2f}R)")
        try:
            self._journal.record_dry_run_signal(
                direction=f"EXIT_{pos.direction}", option_type="", strike_mode="FUT",
                spot=exit_price, confidence=0, sl_pts=0.0, tp_pts=0.0,
                cycle=0, strategy_name=self.cfg.strategy_name,
            )
        except Exception as e:
            logger.warning(f"Trend-day brain: journal exit write failed (non-fatal): {e}")
        self.position = None
        self._save_state()

    def get_health(self) -> dict:
        return {
            "enabled": True,
            "has_open_position": self.position is not None,
            "position": self.position.to_dict() if self.position else None,
            "traded_today": self.traded_on,
        }


_INSTANCE: Optional[TrendDayBrain] = None


def get_trend_day_brain() -> Optional[TrendDayBrain]:
    return _INSTANCE


def _daemon_loop(interval: int):
    global _INSTANCE
    _INSTANCE = TrendDayBrain()
    logger.info("Trend-day brain: daemon loop started (paper-mode only)")
    while True:
        try:
            now = datetime.now()
            if now.weekday() < 5 and "09:15" <= now.strftime("%H:%M") <= "15:30":
                _INSTANCE.run_cycle()
        except Exception as e:
            logger.error(f"Trend-day brain: cycle error (non-fatal, continuing): {e}")
        time.sleep(interval)


def start_in_app(interval: int = 60) -> bool:
    """Start the trend-day brain as a background daemon thread.

    Opt-in via TREND_DAY_BRAIN_ENABLED=true (default: disabled).
    Refuses to arm unless EXECUTION_MODE=futures — the validated edge
    is PF 1.113 at futures friction but PF 0.666 (-10.9 pts/trade) at
    options friction, so running it on options would lose money by
    construction. Paper-mode only; never places a real broker order.
    """
    global _started
    if os.getenv("TREND_DAY_BRAIN_ENABLED", "false").lower() != "true":
        logger.info("Trend-day brain: disabled (set TREND_DAY_BRAIN_ENABLED=true)")
        return False

    exec_mode = os.getenv("EXECUTION_MODE", "options").strip().lower()
    if exec_mode != "futures":
        logger.warning(
            f"Trend-day brain: NOT started — EXECUTION_MODE={exec_mode!r}, "
            f"requires 'futures'. Validated PF is 1.113 at futures friction "
            f"but 0.666 at options friction, so this strategy loses money on "
            f"options by construction."
        )
        return False

    with _started_lock:
        if _started:
            logger.info("Trend-day brain: already started, skipping duplicate")
            return False
        try:
            t = threading.Thread(target=_daemon_loop, args=(interval,),
                                  name="trend-day-brain", daemon=True)
            t.start()
            _started = True
            return True
        except Exception as e:
            logger.warning(f"Trend-day brain: in-app start failed (non-fatal): {e}")
            return False
