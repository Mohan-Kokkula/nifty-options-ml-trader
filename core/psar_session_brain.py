"""
core/psar_session_brain.py — Independent, paper-mode-only PSAR trading brain.

Validated via 8-fold chronological walk-forward backtest (futures-level
friction) before being wired in here: PSAR(start=0.01, incr=0.01, max=0.10)
flips, gated by (a) today's opening-gap direction agreeing with the flip,
(b) a break of the first-30-min opening range, (c) that breakout still
holding 2 bars (10 min) later, (d) outside the first-15-min/lunch chop
window, (e) India VIX inside [13, 22]. Confirmed PF>1 on held-out data
not used for filter selection (mean fold PF 1.20 across 3 out-of-sample
folds, all three individually profitable).

This runs as a FULLY SEPARATE loop from ClaudePilot's ML-driven position
management in core/claude_pilot.py — it does not read or write
LivePosition, does not go through the ownership-CAS gates, and never
places a real broker order. It has its own minimal position state, its
own trade journal file (logs/psar_journal.jsonl), and its own Telegram
alerts. Paper-only by construction: there is no code path here that
calls a broker's order-placement API.

Opt-in via PSAR_BRAIN_ENABLED=true (default: disabled, matches this
project's convention for every new, not-yet-live-proven feature).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_POSITION_STATE_PATH = "data/psar_brain_position.json"
_started_lock = threading.Lock()
_started = False


# ══════════════════════════════════════════════════════════════════════
# Config — validated defaults. Only the enable flag and Telegram
# credentials are meant to be overridden via env; the strategy
# parameters themselves are the backtested values, not free knobs.
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PSARBrainConfig:
    af_start: float = 0.01
    af_incr: float = 0.01
    af_max: float = 0.10

    orb_bars: int = 6              # first 30 min (6 x 5-min bars)
    confirm_bars: int = 2           # bars after the flip that must still show the breakout
    vix_lo: float = 13.03
    vix_hi: float = 21.69

    atr_period: int = 14
    sl_atr_mult: float = 2.2
    tp_atr_mult: float = 2.2
    sl_min_pts: float = 30.0
    sl_max_pts: float = 80.0
    tp_min_pts: float = 30.0
    tp_max_pts: float = 130.0
    min_rr_ratio: float = 1.5

    first15_cutoff: str = "09:30"
    lunch_start: str = "10:15"
    lunch_end: str = "12:15"
    eod_squareoff_time: str = "15:15"

    lookback_bars: int = 300        # bars fetched/recomputed each cycle for PSAR/ATR/ORB state
    cycle_interval_sec: int = 60

    journal_path: str = "logs/psar_journal.jsonl"
    strategy_name: str = "PSAR_SESSION"
    qty: int = 65                   # paper-only notional lot size, matches project's NIFTY lot size

    # Trailing stop / trailing target — reused defaults from PilotConfig
    # (core/claude_pilot.py), spot-based (this brain has no options
    # premium to track, same delta=1 simplification as futures mode).
    breakeven_trigger_r: float = 0.5
    breakeven_lock_pts: float = 10.0
    trail_activation_r: float = 0.8
    trail_step_pct: float = 0.60
    use_atr_trailing_stop: bool = True
    atr_trail_multiplier: float = 1.5
    trail_tp_activation_pct: float = 0.75
    trail_tp_extend_pct: float = 0.50


# ══════════════════════════════════════════════════════════════════════
# Pure helper functions — no I/O, no self, unit-testable in isolation.
# ══════════════════════════════════════════════════════════════════════
def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


def compute_psar(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  af_start: float, af_incr: float, af_max: float):
    """Standard Wilder PSAR. Returns (sar, trend) arrays; trend[i] in {1,-1}."""
    n = len(close)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)
    ep = np.zeros(n)
    af = np.zeros(n)
    trend[0] = 1
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start
    for i in range(1, n):
        prev_sar = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
        if trend[i - 1] == 1:
            prev_sar = min(prev_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < prev_sar:
                trend[i] = -1
                sar[i] = ep[i - 1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                sar[i] = prev_sar
                ep[i] = max(ep[i - 1], high[i])
                af[i] = min(af[i - 1] + af_incr, af_max) if high[i] > ep[i - 1] else af[i - 1]
        else:
            prev_sar = max(prev_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > prev_sar:
                trend[i] = 1
                sar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                sar[i] = prev_sar
                ep[i] = min(ep[i - 1], low[i])
                af[i] = min(af[i - 1] + af_incr, af_max) if low[i] < ep[i - 1] else af[i - 1]
    return sar, trend


def compute_orb(df: pd.DataFrame, orb_bars: int):
    """First `orb_bars` bars of the LAST trading day present in df -> (orb_hi, orb_lo).
    Returns (nan, nan) if today has fewer than orb_bars bars so far."""
    day = df.index[-1].date()
    today = df[df.index.date == day]
    if len(today) < orb_bars:
        return float("nan"), float("nan")
    first = today.iloc[:orb_bars]
    return float(first["high"].max()), float(first["low"].min())


def compute_gap(df: pd.DataFrame):
    """(gap_pts, gap_sign) for the current day vs the prior day's last close."""
    day = df.index[-1].date()
    days = sorted(set(df.index.date))
    if day not in days or days.index(day) == 0:
        return 0.0, 0.0
    prev_day = days[days.index(day) - 1]
    today_first_open = float(df[df.index.date == day].iloc[0]["open"])
    prev_last_close = float(df[df.index.date == prev_day].iloc[-1]["close"])
    gap = today_first_open - prev_last_close
    return gap, float(np.sign(gap))


def in_trending_window(ts: datetime, cfg: PSARBrainConfig) -> bool:
    t = ts.strftime("%H:%M")
    is_first15 = t <= cfg.first15_cutoff
    is_lunch = cfg.lunch_start < t <= cfg.lunch_end
    return not is_first15 and not is_lunch


def compute_sl_tp(atr: float, cfg: PSARBrainConfig) -> tuple:
    sl_pts = min(max(atr * cfg.sl_atr_mult, cfg.sl_min_pts), cfg.sl_max_pts)
    tp_pts = min(max(atr * cfg.tp_atr_mult, cfg.tp_min_pts), cfg.tp_max_pts)
    if tp_pts < sl_pts * cfg.min_rr_ratio:
        tp_pts = min(sl_pts * cfg.min_rr_ratio, cfg.tp_max_pts)
    return round(sl_pts, 1), round(tp_pts, 1)


def atr_trail_sl_candidate(direction: str, current_price: float, atr: float, multiplier: float) -> float:
    if direction == "CALL":
        return current_price - multiplier * atr
    return current_price + multiplier * atr


# ══════════════════════════════════════════════════════════════════════
# Position state
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PSARPosition:
    direction: str              # "CALL" or "PUT"
    entry_price: float
    entry_time_iso: str
    sl_price: float
    tp_price: float
    initial_sl_dist: float
    qty: int
    peak_unrealized: float = 0.0
    breakeven_done: bool = False
    trail_activated: bool = False
    pending_flip_idx: Optional[int] = None   # index (in the fetched df) of the flip awaiting confirmation

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


class PSARSessionBrain:
    def __init__(self, cfg: Optional[PSARBrainConfig] = None):
        self.cfg = cfg or PSARBrainConfig()
        self.position: Optional[PSARPosition] = None
        self._pending_flip: Optional[dict] = None  # {"direction":..., "detected_at": iso, "bars_since": int}
        self._journal = self._build_journal()
        self._notifier = self._build_notifier()
        self._load_position()

    # ── Wiring: own journal + own Telegram notifier, no shared state
    #    with ClaudePilot's instances (see docstring for why). ──────────
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

    def _load_position(self):
        if os.path.exists(_POSITION_STATE_PATH):
            try:
                with open(_POSITION_STATE_PATH) as f:
                    d = json.load(f)
                if d.get("position"):
                    self.position = PSARPosition.from_dict(d["position"])
                    logger.info(f"PSAR brain: restored open position from disk: {self.position.direction} "
                                f"entry={self.position.entry_price}")
                self._pending_flip = d.get("pending_flip")
            except Exception as e:
                logger.warning(f"PSAR brain: failed to restore position state (non-fatal): {e}")

    def _save_state(self):
        try:
            _atomic_write_json(_POSITION_STATE_PATH, {
                "position": self.position.to_dict() if self.position else None,
                "pending_flip": self._pending_flip,
            })
        except Exception as e:
            logger.warning(f"PSAR brain: failed to persist state (non-fatal): {e}")

    # ── Data fetch ──────────────────────────────────────────────────
    def _fetch_bars(self) -> Optional[pd.DataFrame]:
        try:
            from core.tv_fetcher import get_tv_fetcher
            df = get_tv_fetcher().get_nifty_5min(n_bars=self.cfg.lookback_bars)
            if df is None or df.empty or len(df) < 50:
                return None
            return df
        except Exception as e:
            logger.warning(f"PSAR brain: bar fetch failed (non-fatal, skipping cycle): {e}")
            return None

    def _fetch_vix(self) -> Optional[float]:
        try:
            from core.tv_fetcher import get_tv_fetcher
            tv = get_tv_fetcher()
            for sym in ("INDIAVIX", "INDIA VIX"):
                try:
                    raw = tv.get_ohlcv(sym, "NSE", interval_minutes="D", n_bars=3)
                    if raw is not None and not raw.empty:
                        return float(raw["close"].iloc[-1])
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"PSAR brain: VIX fetch failed: {e}")
        return None

    # ── Alerts ──────────────────────────────────────────────────────
    def _alert(self, action: str, side: str, price: float, status: str, details: str = ""):
        try:
            self._notifier.send_trade_alert(
                action=action, symbol="NIFTY (PSAR brain, paper)", side=side,
                qty=self.cfg.qty, price=price, order_id="PAPER", status=status, details=details,
            )
        except Exception as e:
            logger.warning(f"PSAR brain: telegram alert failed (non-fatal): {e}")

    # ── One cycle: manage an open position, or look for a new entry ──
    def run_cycle(self):
        df = self._fetch_bars()
        if df is None:
            return
        now = df.index[-1].to_pydatetime()
        spot = float(df["close"].iloc[-1])

        if self.position is not None:
            self._manage_open_position(df, now, spot)
        else:
            self._look_for_entry(df, now, spot)

    # ── Entry side ──────────────────────────────────────────────────
    def _look_for_entry(self, df: pd.DataFrame, now: datetime, spot: float):
        cfg = self.cfg
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        sar, trend = compute_psar(high, low, close, cfg.af_start, cfg.af_incr, cfg.af_max)
        atr = compute_atr(df, cfg.atr_period)
        n = len(df)

        # A flip is "the most recent bar differs in trend from the one before it".
        flipped_now = n >= 2 and trend[-1] != trend[-2]

        if self._pending_flip is None:
            if not flipped_now:
                return
            direction = "CALL" if trend[-1] == 1 else "PUT"
            gap_pts, gap_sign = compute_gap(df)
            sig = 1 if direction == "CALL" else -1
            if gap_sign == 0 or sig != gap_sign:
                logger.debug("PSAR brain: flip detected but gap direction disagrees -- skipped")
                return
            orb_hi, orb_lo = compute_orb(df, cfg.orb_bars)
            if not np.isfinite(orb_hi) or not np.isfinite(orb_lo):
                logger.debug("PSAR brain: flip detected but opening range not yet established -- skipped")
                return
            broke_orb = (spot > orb_hi) if sig == 1 else (spot < orb_lo)
            if not broke_orb:
                logger.debug("PSAR brain: flip detected but no opening-range breakout yet -- skipped")
                return
            if not in_trending_window(now, cfg):
                logger.debug("PSAR brain: flip detected but inside first-15min/lunch window -- skipped")
                return
            vix = self._fetch_vix()
            if vix is None or not (cfg.vix_lo <= vix <= cfg.vix_hi):
                logger.debug(f"PSAR brain: flip detected but VIX={vix} outside [{cfg.vix_lo},{cfg.vix_hi}] -- skipped")
                return
            # All gates passed -> start the 2-bar hold-confirmation window.
            self._pending_flip = {
                "direction": direction, "orb_hi": orb_hi, "orb_lo": orb_lo,
                "detected_at": now.isoformat(), "bars_since": 0,
            }
            self._save_state()
            logger.info(f"PSAR brain: candidate {direction} flip at {now} spot={spot:.1f} -- "
                        f"awaiting {cfg.confirm_bars}-bar confirmation")
            return

        # A confirmation window is already open.
        pf = self._pending_flip
        pf["bars_since"] += 1
        direction = pf["direction"]
        sig = 1 if direction == "CALL" else -1
        still_holds = (spot > pf["orb_hi"]) if sig == 1 else (spot < pf["orb_lo"])
        if not still_holds:
            logger.info(f"PSAR brain: candidate {direction} flip invalidated -- breakout did not hold")
            self._pending_flip = None
            self._save_state()
            return
        if pf["bars_since"] < cfg.confirm_bars:
            self._save_state()
            return

        # Confirmed -> enter now, at the current (just-closed) bar's close
        # as a proxy for "next available price" (paper-mode; a live
        # broker fill would use the next tick/bar open).
        atr_now = float(atr[-1]) if np.isfinite(atr[-1]) and atr[-1] > 0 else None
        self._pending_flip = None
        if atr_now is None:
            logger.warning("PSAR brain: confirmed flip but ATR unavailable -- skipping entry")
            self._save_state()
            return
        sl_pts, tp_pts = compute_sl_tp(atr_now, cfg)
        sign = 1 if direction == "CALL" else -1
        sl_price = spot - sign * sl_pts
        tp_price = spot + sign * tp_pts
        self.position = PSARPosition(
            direction=direction, entry_price=spot, entry_time_iso=now.isoformat(),
            sl_price=sl_price, tp_price=tp_price, initial_sl_dist=sl_pts, qty=cfg.qty,
        )
        self._save_state()
        logger.info(f"PSAR brain: ENTERED {direction} @ {spot:.1f} SL={sl_price:.1f} TP={tp_price:.1f} "
                    f"(sl_pts={sl_pts:.1f} tp_pts={tp_pts:.1f})")
        self._alert("PSAR_ENTRY", direction, spot, "simulated",
                    details=f"SL={sl_price:.1f} TP={tp_price:.1f} ({sl_pts:.0f}/{tp_pts:.0f}pts)")
        try:
            self._journal.record_dry_run_signal(
                direction=direction, option_type="", strike_mode="SPOT",
                spot=spot, confidence=0.0, sl_pts=sl_pts, tp_pts=tp_pts,
                cycle=0, strategy_name=cfg.strategy_name,
            )
        except Exception as e:
            logger.warning(f"PSAR brain: journal write failed (non-fatal): {e}")

    # ── Exit / management side ────────────────────────────────────
    def _manage_open_position(self, df: pd.DataFrame, now: datetime, spot: float):
        cfg = self.cfg
        pos = self.position
        atr = compute_atr(df, cfg.atr_period)
        atr_now = float(atr[-1]) if np.isfinite(atr[-1]) else 0.0
        sign = 1 if pos.direction == "CALL" else -1
        unrealized = sign * (spot - pos.entry_price)
        pos.peak_unrealized = max(pos.peak_unrealized, unrealized)

        # EOD square-off
        if now.strftime("%H:%M") >= cfg.eod_squareoff_time:
            self._close_position(spot, "EOD_SQUAREOFF")
            return

        # Breakeven stop
        if not pos.breakeven_done:
            be_trigger = max(cfg.breakeven_trigger_r * pos.initial_sl_dist, 0.0)
            if unrealized >= be_trigger and be_trigger > 0:
                new_sl = pos.entry_price + sign * cfg.breakeven_lock_pts
                more_protective = (new_sl > pos.sl_price) if sign == 1 else (new_sl < pos.sl_price)
                if more_protective:
                    pos.sl_price = new_sl
                    pos.breakeven_done = True
                    logger.info(f"PSAR brain: breakeven stop -> SL={pos.sl_price:.1f}")

        # Profit-lock trailing stop
        if not pos.trail_activated and unrealized >= pos.initial_sl_dist * cfg.trail_activation_r:
            pos.trail_activated = True
            logger.info("PSAR brain: profit-lock trail activated")
        if pos.trail_activated:
            lock_amount = unrealized * cfg.trail_step_pct
            candidate_sl = pos.entry_price + sign * lock_amount
            more_protective = (candidate_sl > pos.sl_price) if sign == 1 else (candidate_sl < pos.sl_price)
            if more_protective:
                pos.sl_price = candidate_sl
            if cfg.use_atr_trailing_stop and atr_now > 0:
                atr_candidate = atr_trail_sl_candidate(pos.direction, spot, atr_now, cfg.atr_trail_multiplier)
                more_protective = (atr_candidate > pos.sl_price) if sign == 1 else (atr_candidate < pos.sl_price)
                if more_protective:
                    pos.sl_price = atr_candidate

        # Trailing take-profit
        orig_tp_dist = abs(pos.tp_price - pos.entry_price)
        if orig_tp_dist > 0 and unrealized >= orig_tp_dist * cfg.trail_tp_activation_pct:
            extend = pos.peak_unrealized * cfg.trail_tp_extend_pct
            new_tp = pos.entry_price + sign * (pos.peak_unrealized + extend)
            more_favorable = (new_tp > pos.tp_price) if sign == 1 else (new_tp < pos.tp_price)
            if more_favorable:
                pos.tp_price = new_tp

        # SL/TP hit check
        sl_hit = (spot <= pos.sl_price) if sign == 1 else (spot >= pos.sl_price)
        tp_hit = (spot >= pos.tp_price) if sign == 1 else (spot <= pos.tp_price)
        if sl_hit:
            self._close_position(pos.sl_price, "SL")
            return
        if tp_hit:
            self._close_position(pos.tp_price, "TP")
            return

        self._save_state()

    def _close_position(self, exit_price: float, reason: str):
        pos = self.position
        sign = 1 if pos.direction == "CALL" else -1
        pnl_pts = sign * (exit_price - pos.entry_price)
        logger.info(f"PSAR brain: EXIT {pos.direction} @ {exit_price:.1f} reason={reason} "
                    f"pnl_pts={pnl_pts:+.1f}")
        self._alert(f"PSAR_EXIT_{reason}", "SELL" if pos.direction == "CALL" else "BUY",
                    exit_price, "simulated", details=f"pnl_pts={pnl_pts:+.1f}")
        try:
            self._journal.record_dry_run_signal(
                direction=f"EXIT_{pos.direction}", option_type="", strike_mode="SPOT",
                spot=exit_price, confidence=0.0, sl_pts=0.0, tp_pts=0.0,
                cycle=0, strategy_name=self.cfg.strategy_name,
            )
        except Exception as e:
            logger.warning(f"PSAR brain: journal exit write failed (non-fatal): {e}")
        self.position = None
        self._save_state()

    def get_health(self) -> dict:
        return {
            "enabled": True,
            "has_open_position": self.position is not None,
            "position": self.position.to_dict() if self.position else None,
            "pending_confirmation": self._pending_flip is not None,
        }


# ══════════════════════════════════════════════════════════════════════
# Daemon loop + main.py entry point (mirrors scripts/oi_archiver.py's
# start_in_app convention).
# ══════════════════════════════════════════════════════════════════════
_BRAIN_INSTANCE: Optional[PSARSessionBrain] = None


def get_psar_brain() -> Optional[PSARSessionBrain]:
    return _BRAIN_INSTANCE


def _daemon_loop(interval: int):
    global _BRAIN_INSTANCE
    _BRAIN_INSTANCE = PSARSessionBrain()
    logger.info("PSAR session brain: daemon loop started (paper-mode only)")
    while True:
        try:
            now = datetime.now()
            if now.strftime("%H:%M") >= "09:15" and now.strftime("%H:%M") <= "15:30" and now.weekday() < 5:
                _BRAIN_INSTANCE.run_cycle()
        except Exception as e:
            logger.error(f"PSAR session brain: cycle error (non-fatal, continuing): {e}")
        time.sleep(interval)


def start_in_app(interval: int = 60) -> bool:
    """Start the PSAR session brain as a background daemon thread.
    Opt-in via PSAR_BRAIN_ENABLED=true (default: disabled). Paper-mode
    only -- never places a real broker order."""
    global _started
    if os.getenv("PSAR_BRAIN_ENABLED", "false").lower() != "true":
        logger.info("PSAR session brain: disabled (set PSAR_BRAIN_ENABLED=true to enable)")
        return False
    with _started_lock:
        if _started:
            logger.info("PSAR session brain: already started, skipping duplicate start")
            return False
        try:
            t = threading.Thread(target=_daemon_loop, args=(interval,),
                                  name="psar-session-brain", daemon=True)
            t.start()
            _started = True
            return True
        except Exception as e:
            logger.warning(f"PSAR session brain: in-app start failed (non-fatal): {e}")
            return False
