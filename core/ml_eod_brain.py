"""
core/ml_eod_brain.py — Independent, paper-mode-only ML brain with EOD exits.

Takes the SAME model signal the live pilot uses (core.ml_engine.predict)
but replaces the exit structure. The pilot exits on ATR-scaled targets
with trailing -- a short-hold design. This brain holds to the close.

WHY
---
2026-08-07 finding: the model predicts DESTINATION well and ROUTE badly.

    directional accuracy (3-bar label)   65%
    path predictability  (triple-barrier AUC)  0.52

A short-hold exit is a bet on route. Holding to EOD collects the
destination instead. Measured on the frozen TEST set, replayed under the
live constraints (ONE position at a time, max 3/day, no entry after
15:00, EOD square-off):

    variant                          n     PF     avg   win%   maxDD
    1R stop + 2R target (the pilot) 393  0.943   -2.0  41.5%   1636
    1R stop + EOD                   317  1.022   +0.8  38.8%   1607
 -> 2R stop + EOD  (THIS BRAIN)     238  1.053   +2.3  51.7%   1461
    3R stop + EOD                   216  0.996   -0.2  53.7%   1876
    no stop  + EOD                  194  0.981   -0.9  53.6%   1847

R = 2 x ATR14, so the 2R stop is ~4 x ATR (~63 pts on recent data).

HONEST CAVEATS -- read before enabling
--------------------------------------
1. The edge is MARGINAL. PF 1.053 and +2.3 pts/trade. The trend-day
   brain is PF 1.225 and +5.4 pts on the same friction. If you only run
   one thing, run that one.
2. Max drawdown 1,461 pts ~= Rs 95,000 at 65 qty, to earn ~Rs 35,000
   over the same window. The drawdown-to-return ratio is poor.
3. The 2R variant was chosen by comparing FIVE exit structures on TEST,
   so some of its margin over 1R (1.053 vs 1.022) is selection luck.
   The robust part is the DIRECTION: every EOD variant beat the pilot's
   short-hold structure, which was outright negative at 0.943.
4. Removing the stop entirely is worse AND fattens the tail -- worst
   single trade goes -166 -> -522 pts. A stop earns its place by
   REACTING to adverse moves, not by predicting them.

*** FUTURES ONLY. *** The same replay at options friction (29 pts) is
negative in all ten cells, best 0.622. start_in_app refuses to arm
unless EXECUTION_MODE=futures.

Runs as a FULLY SEPARATE loop -- own position state, own journal, own
Telegram. Never touches LivePosition, never places a real broker order.
Opt-in via ML_EOD_BRAIN_ENABLED=true (default: disabled).
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

_STATE_PATH = "data/ml_eod_position.json"
_started_lock = threading.Lock()
_started = False


@dataclass
class MLEodConfig:
    # ── 2026-08-07: ONE change from the original, after re-validation ──
    # An OAT sweep initially altered five parameters at once. Re-checking
    # each FROM ITS OWN context showed the stack was worse than its best
    # single component, and that every original value was still optimal
    # once the window moved:
    #   candidate            VALpf   TESTpf   TESTdd
    #   ORIGINAL             1.030    1.053    1461
    #   + window only        1.196    1.255     838   <- adopted
    #   + window + cutoff    1.152    1.225     868
    #   + window + stop      1.042    1.258     635
    #   + all five           1.037    1.269     627
    # Re-swept from the adopted context: stop 4.0 is the VAL peak
    # (2.0=1.087 3.0=1.062 4.0=1.196 5.0=0.994), cutoff 15:00 is the peak
    # (monotone 13:00=1.082 -> 15:00=1.196), maxday 3 ties the peak.
    # The OAT had picked 3.0/14:00/2 only because it scored them against
    # the OLD 09:15 window -- classic one-at-a-time interaction blindness.
    # So: keep the window, revert everything else.
    atr_period: int = 14
    atr_mult: float = 2.0
    stop_R: float = 2.0             # stop = stop_R x atr_mult x ATR = 4.0 ATR

    # The ONLY change from the original config. Independently supported:
    # the session-bucket analysis put the trend-continuation sign flip at
    # 11:30 across twelve 30-min buckets of ~17,000 bars each, and the VAL
    # curve is elevated either side of it (11:00=1.102, 12:00=1.060) vs
    # 09:15=1.030 and 10:30=0.962.
    entry_window_start: str = "11:30"
    entry_cutoff: str = "15:00"     # matches claude_pilot.entry_cutoff_hour
    # DISABLED after a fine sweep. The coarse OAT put this at 0.45, but a
    # 13-point grid showed VAL and TEST moving in OPPOSITE directions:
    #        minconf   VAL_pf   TEST_pf
    #          0.000    1.037    1.269
    #          0.425    1.198    1.298   <- VAL peak
    #          0.450    1.094    1.384
    #          0.600    0.807    1.795   <- TEST peak
    # VAL declines above 0.425 and goes sub-1.0 from 0.475; TEST rises the
    # whole way. A real threshold points the same way in both periods.
    # Also exposed an interaction: 0.45 scored 1.168 on VAL against the OLD
    # stop/window/cap and only 1.094 against the new ones -- the OAT sweep
    # never re-validated the stack jointly. Set >0 only with fresh evidence.
    min_confidence: float = 0.0
    eod_squareoff: str = "15:10"
    last_tradeable_time: str = "15:20"   # strip the post-close auction print
    max_trades_per_day: int = 3

    lookback_bars: int = 300
    cycle_interval_sec: int = 60
    journal_path: str = "logs/ml_eod_journal.jsonl"
    strategy_name: str = "ML_EOD"
    qty: int = 65


# ── Pure helpers (unit-testable) ─────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR of the last bar. NaN if insufficient history."""
    if len(df) < period + 1:
        return float("nan")
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])


def drop_auction_print(df_today: pd.DataFrame, cfg: MLEodConfig) -> pd.DataFrame:
    """Strip non-tradeable post-close bars (the ~15:25 auction print sits
    up to 151 pts away from the real close -- see trend_day_brain)."""
    if df_today.empty:
        return df_today
    keep = [t.strftime("%H:%M") < cfg.last_tradeable_time for t in df_today.index]
    return df_today[keep]


def compute_stop(direction: str, entry: float, atr: float,
                 cfg: MLEodConfig) -> tuple:
    """Returns (stop_price, risk_pts) or (None, 0.0) if ATR is unusable."""
    if not np.isfinite(atr) or atr <= 0:
        return None, 0.0
    risk = cfg.stop_R * cfg.atr_mult * atr
    if risk < 5:
        return None, 0.0
    stop = entry - risk if direction == "CALL" else entry + risk
    return stop, risk


def in_entry_window(ts: datetime, cfg: MLEodConfig) -> bool:
    return cfg.entry_window_start <= ts.strftime("%H:%M") < cfg.entry_cutoff


def is_eod(ts: datetime, cfg: MLEodConfig) -> bool:
    return ts.strftime("%H:%M") >= cfg.eod_squareoff


@dataclass
class MLEodPosition:
    direction: str
    entry_price: float
    entry_time_iso: str
    stop_price: float
    risk_pts: float
    confidence: float
    qty: int

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d): return cls(**d)


def _atomic_write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class MLEodBrain:
    def __init__(self, cfg: Optional[MLEodConfig] = None):
        self.cfg = cfg or MLEodConfig()
        self.position: Optional[MLEodPosition] = None
        self.trades_on: Optional[str] = None
        self.trades_today: int = 0
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
                self.position = MLEodPosition.from_dict(d["position"])
                logger.info(f"ML-EOD brain: restored {self.position.direction} "
                            f"@ {self.position.entry_price:.1f}")
            self.trades_on = d.get("trades_on")
            self.trades_today = int(d.get("trades_today", 0) or 0)
        except Exception as e:
            logger.warning(f"ML-EOD brain: state restore failed (non-fatal): {e}")

    def _save_state(self):
        try:
            _atomic_write_json(_STATE_PATH, {
                "position": self.position.to_dict() if self.position else None,
                "trades_on": self.trades_on,
                "trades_today": self.trades_today,
            })
        except Exception as e:
            logger.warning(f"ML-EOD brain: state persist failed (non-fatal): {e}")

    def _fetch_bars(self) -> Optional[pd.DataFrame]:
        try:
            from core.tv_fetcher import get_tv_fetcher
            df = get_tv_fetcher().get_nifty_5min(n_bars=self.cfg.lookback_bars)
            return df if df is not None and not df.empty else None
        except Exception as e:
            logger.warning(f"ML-EOD brain: bar fetch failed (skipping cycle): {e}")
            return None

    def _alert(self, action: str, side: str, price: float, details: str = ""):
        try:
            self._notifier.send_trade_alert(
                action=action, symbol="NIFTY FUT (ml-eod, paper)", side=side,
                qty=self.cfg.qty, price=price, order_id="PAPER",
                status="simulated", details=details)
        except Exception as e:
            logger.warning(f"ML-EOD brain: telegram alert failed (non-fatal): {e}")

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

        today_iso = now.date().isoformat()
        if self.trades_on != today_iso:
            self.trades_on, self.trades_today = today_iso, 0

        if self.position is not None:
            self._manage(now, spot)
        else:
            self._look_for_entry(df, df_today, now, spot)

    def _look_for_entry(self, df_all, df_today, now, spot):
        cfg = self.cfg
        if not in_entry_window(now, cfg):
            return
        if self.trades_today >= cfg.max_trades_per_day:
            return
        try:
            from core import ml_engine
            if not ml_engine.is_ready():
                return
            signal, proba, conf, _ = ml_engine.predict(df_all)
        except Exception as e:
            logger.warning(f"ML-EOD brain: predict failed (skipping): {e}")
            return
        if signal == 2:
            return
        if conf < cfg.min_confidence:
            logger.debug(f"ML-EOD brain: signal below confidence floor "
                         f"({conf:.2f} < {cfg.min_confidence})")
            return
        direction = "CALL" if signal == 0 else "PUT"

        atr = compute_atr(df_today, cfg.atr_period)
        stop, risk = compute_stop(direction, spot, atr, cfg)
        if stop is None:
            logger.info(f"ML-EOD brain: {direction} signal but ATR unusable — skipped")
            return

        self.position = MLEodPosition(
            direction=direction, entry_price=spot, entry_time_iso=now.isoformat(),
            stop_price=stop, risk_pts=risk, confidence=float(conf), qty=cfg.qty)
        self.trades_today += 1
        self._save_state()

        logger.info(f"ML-EOD brain: ENTERED {direction} @ {spot:.1f} | "
                    f"stop={stop:.1f} ({risk:.0f}pts) conf={conf:.2f} "
                    f"| hold to EOD | trade {self.trades_today}/"
                    f"{cfg.max_trades_per_day} today")
        self._alert("MLEOD_ENTRY", direction, spot,
                    details=f"stop={stop:.1f} risk={risk:.0f}pts conf={conf:.2f}")
        try:
            self._journal.record_dry_run_signal(
                direction=direction, option_type="", strike_mode="FUT",
                spot=spot, confidence=int(round(conf * 100)), sl_pts=risk,
                tp_pts=0.0, cycle=0, strategy_name=cfg.strategy_name)
        except Exception as e:
            logger.warning(f"ML-EOD brain: journal write failed (non-fatal): {e}")

    def _manage(self, now, spot):
        pos = self.position
        sign = 1 if pos.direction == "CALL" else -1
        if is_eod(now, self.cfg):
            self._close(spot, "EOD_SQUAREOFF"); return
        if (spot <= pos.stop_price) if sign == 1 else (spot >= pos.stop_price):
            self._close(pos.stop_price, "SL"); return
        self._save_state()

    def _close(self, exit_price: float, reason: str):
        pos = self.position
        sign = 1 if pos.direction == "CALL" else -1
        pnl = sign * (exit_price - pos.entry_price)
        r = pnl / pos.risk_pts if pos.risk_pts else 0.0
        logger.info(f"ML-EOD brain: EXIT {pos.direction} @ {exit_price:.1f} "
                    f"reason={reason} pnl={pnl:+.1f}pts ({r:+.2f}R)")
        self._alert(f"MLEOD_EXIT_{reason}",
                    "SELL" if pos.direction == "CALL" else "BUY",
                    exit_price, details=f"pnl={pnl:+.1f}pts ({r:+.2f}R)")
        try:
            self._journal.record_dry_run_signal(
                direction=f"EXIT_{pos.direction}", option_type="", strike_mode="FUT",
                spot=exit_price, confidence=0, sl_pts=0.0, tp_pts=0.0,
                cycle=0, strategy_name=self.cfg.strategy_name)
        except Exception as e:
            logger.warning(f"ML-EOD brain: journal exit write failed (non-fatal): {e}")
        self.position = None
        self._save_state()

    def get_health(self) -> dict:
        return {"enabled": True,
                "has_open_position": self.position is not None,
                "position": self.position.to_dict() if self.position else None,
                "trades_today": self.trades_today,
                "trades_on": self.trades_on}


_INSTANCE: Optional[MLEodBrain] = None


def get_ml_eod_brain() -> Optional[MLEodBrain]:
    return _INSTANCE


def _daemon_loop(interval: int):
    global _INSTANCE
    _INSTANCE = MLEodBrain()
    logger.info("ML-EOD brain: daemon loop started (paper-mode only)")
    while True:
        try:
            now = datetime.now()
            if now.weekday() < 5 and "09:15" <= now.strftime("%H:%M") <= "15:30":
                _INSTANCE.run_cycle()
        except Exception as e:
            logger.error(f"ML-EOD brain: cycle error (non-fatal, continuing): {e}")
        time.sleep(interval)


def start_in_app(interval: int = 60) -> bool:
    """Start the ML-EOD brain as a background daemon thread.

    Opt-in via ML_EOD_BRAIN_ENABLED=true (default: disabled). Refuses to
    arm unless EXECUTION_MODE=futures -- the same replay at options
    friction is negative in all ten exit variants (best PF 0.622).
    Paper-mode only; never places a real broker order.
    """
    global _started
    if os.getenv("ML_EOD_BRAIN_ENABLED", "false").lower() != "true":
        logger.info("ML-EOD brain: disabled (set ML_EOD_BRAIN_ENABLED=true)")
        return False

    exec_mode = os.getenv("EXECUTION_MODE", "options").strip().lower()
    if exec_mode != "futures":
        logger.warning(
            f"ML-EOD brain: NOT started — EXECUTION_MODE={exec_mode!r}, "
            f"requires 'futures'. At options friction this structure is "
            f"negative in every exit variant tested (best PF 0.622).")
        return False

    with _started_lock:
        if _started:
            logger.info("ML-EOD brain: already started, skipping duplicate")
            return False
        try:
            t = threading.Thread(target=_daemon_loop, args=(interval,),
                                 name="ml-eod-brain", daemon=True)
            t.start()
            _started = True
            return True
        except Exception as e:
            logger.warning(f"ML-EOD brain: in-app start failed (non-fatal): {e}")
            return False
