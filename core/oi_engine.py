"""
oi_engine.py — OI/PCR Flow Signal Engine (Strategy 3)
=====================================================
Primary signal: OI buildup/unwinding patterns + PCR shift
Confirmation:   PSAR direction
Context:        VIX regime

Signal logic:
  CALL: PCR rising (put writers adding = bullish)
        + OI buildup on put side at support strikes
        + PSAR 15m bullish
  PUT:  PCR falling (call writers adding = bearish)
        + OI buildup on call side at resistance strikes
        + PSAR 15m bearish
  SKIP: no clear OI pattern, or PSAR disagrees

NOTE: This engine collects OI snapshots for future analysis.
      Live trading signals are generated only after sufficient
      data has been collected (min 20 trading days).
"""

import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

from core.strategy_base import StrategyEngine
from core.psar_engine import compute_psar

logger = logging.getLogger(__name__)

OI_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OI_SNAPSHOT_FILE = os.path.join(OI_DATA_DIR, "oi_snapshots.json")
MIN_SNAPSHOTS_FOR_SIGNAL = 6  # need at least 6 snapshots today to detect shift
MIN_DAYS_FOR_LIVE = 20  # need 20 days of data before generating live signals


def _psar_direction(df: pd.DataFrame, af_start=0.03, af_step=0.02,
                    af_max=0.2) -> int:
    if df is None or df.empty or len(df) < 5:
        return 0
    psar_val, psar_dir, _ = compute_psar(df, af_start, af_step, af_max)
    idx = -2 if len(df) > 5 else -1
    return int(psar_dir.iloc[idx])


class OIEngine(StrategyEngine):
    """OI/PCR flow-based signal engine.

    Phase 1 (current): Collect OI snapshots every cycle, store to disk.
    Phase 2 (after 20+ days): Generate signals from PCR shift + OI patterns.
    """

    BASE_SL = 50
    BASE_TP = 100
    PCR_BULL_THRESHOLD = 0.05   # PCR increase of 0.05+ = bullish
    PCR_BEAR_THRESHOLD = -0.05  # PCR decrease of 0.05+ = bearish

    def __init__(self):
        super().__init__("OI")
        self._snapshots_today: list = []
        self._today_date: str = ""
        self._last_vix_mult = 1.0
        self._days_collected = self._count_collected_days()
        self._live_mode = self._days_collected >= MIN_DAYS_FOR_LIVE
        logger.info(
            f"OIEngine: {self._days_collected} days collected, "
            f"live_mode={'ON' if self._live_mode else 'OFF (collecting data)'}"
        )

    def _count_collected_days(self) -> int:
        """Count how many days of OI data we have."""
        if not os.path.exists(OI_SNAPSHOT_FILE):
            return 0
        try:
            with open(OI_SNAPSHOT_FILE, "r") as f:
                data = json.load(f)
            return len(data.get("days", {}))
        except Exception:
            return 0

    def collect_snapshot(self, chain_data: dict, timestamp: str = ""):
        """Store an OI snapshot from Kotak Neo option chain.

        Call this every cycle with the latest option chain data.
        """
        if not chain_data or not chain_data.get("chain"):
            return

        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today_date:
            self._today_date = today
            self._snapshots_today = []

        ts = timestamp or datetime.now().strftime("%H:%M:%S")
        snapshot = {
            "time": ts,
            "pcr": round(chain_data.get("pcr", 1.0), 4),
            "total_ce_oi": chain_data.get("total_ce_oi", 0),
            "total_pe_oi": chain_data.get("total_pe_oi", 0),
            "spot": chain_data.get("spot", 0),
            "atm": chain_data.get("atm_strike", 0),
        }

        # Store top 5 strikes by OI for CE and PE
        chain = chain_data.get("chain", [])
        ce_by_oi = sorted(
            [(s["strike"], s.get("ce_oi", 0)) for s in chain if s.get("ce_oi", 0) > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        pe_by_oi = sorted(
            [(s["strike"], s.get("pe_oi", 0)) for s in chain if s.get("pe_oi", 0) > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        snapshot["top_ce_oi"] = ce_by_oi
        snapshot["top_pe_oi"] = pe_by_oi

        self._snapshots_today.append(snapshot)

        # Persist to disk
        self._save_snapshot(today, snapshot)

    def _save_snapshot(self, date: str, snapshot: dict):
        """Append snapshot to daily collection on disk."""
        os.makedirs(OI_DATA_DIR, exist_ok=True)
        data = {"days": {}}
        if os.path.exists(OI_SNAPSHOT_FILE):
            try:
                with open(OI_SNAPSHOT_FILE, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"days": {}}

        if date not in data["days"]:
            data["days"][date] = []
        data["days"][date].append(snapshot)

        try:
            with open(OI_SNAPSHOT_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"OI snapshot save failed: {e}")

    def predict(self, df5: pd.DataFrame, df15: pd.DataFrame, df30: pd.DataFrame,
                vix: float = 15.0, current_hm: int = 1000,
                chain_data: dict = None) -> tuple:
        self._ready = True
        indicators = {
            "strategy": "OI",
            "vix": round(vix, 2),
            "days_collected": self._days_collected,
            "live_mode": self._live_mode,
            "snapshots_today": len(self._snapshots_today),
        }

        # Collect snapshot if chain data provided
        if chain_data:
            self.collect_snapshot(chain_data)

        # Time filter
        skip = self._time_filter(current_hm)
        if skip:
            return self._skip(skip, indicators)

        # Phase 1: data collection mode — don't generate signals yet
        if not self._live_mode:
            return self._skip(
                f"collecting_data ({self._days_collected}/{MIN_DAYS_FOR_LIVE} days)",
                indicators
            )

        # Phase 2: need enough snapshots today to detect PCR shift
        if len(self._snapshots_today) < MIN_SNAPSHOTS_FOR_SIGNAL:
            return self._skip(
                f"insufficient_snapshots ({len(self._snapshots_today)}/{MIN_SNAPSHOTS_FOR_SIGNAL})",
                indicators
            )

        # ── Primary: PCR shift detection ──
        recent = self._snapshots_today[-3:]  # last 3 snapshots
        earlier = self._snapshots_today[:3]  # first 3 snapshots

        pcr_now = np.mean([s["pcr"] for s in recent])
        pcr_earlier = np.mean([s["pcr"] for s in earlier])
        pcr_shift = pcr_now - pcr_earlier

        indicators["pcr_now"] = round(pcr_now, 4)
        indicators["pcr_earlier"] = round(pcr_earlier, 4)
        indicators["pcr_shift"] = round(pcr_shift, 4)

        if abs(pcr_shift) < abs(self.PCR_BULL_THRESHOLD):
            return self._skip(f"pcr_shift_small ({pcr_shift:+.4f})", indicators)

        # ── Confirmation: PSAR direction on 15m ──
        psar_dir_15m = _psar_direction(df15)
        indicators["psar_15m_dir"] = psar_dir_15m

        vix_mult = self._vix_multiplier(vix)
        self._last_vix_mult = vix_mult

        # PCR rising = put writers adding = bullish
        if pcr_shift >= self.PCR_BULL_THRESHOLD and psar_dir_15m == 1:
            confidence = self._calc_confidence(pcr_shift, vix)
            p_call = confidence
            p_put = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 0, np.array([p_call, p_put, p_skip]), confidence, indicators

        # PCR falling = call writers adding = bearish
        if pcr_shift <= self.PCR_BEAR_THRESHOLD and psar_dir_15m == -1:
            confidence = self._calc_confidence(abs(pcr_shift), vix)
            p_put = confidence
            p_call = (1.0 - confidence) * 0.2
            p_skip = 1.0 - p_call - p_put
            return 1, np.array([p_call, p_put, p_skip]), confidence, indicators

        return self._skip("psar_disagrees", indicators)

    SL_ATR_MULT = 1.7
    TP_ATR_MULT = 3.3

    def get_sl_tp(self, vix: float = 15.0, max_loss_budget: float = 0,
                  lot_size: int = 0, atr: float = 0.0) -> tuple:
        vix_mult = self._vix_multiplier(vix)

        if atr > 0:
            sl = max(self.SL_ATR_MULT * atr * vix_mult, self.BASE_SL)
            tp = max(self.TP_ATR_MULT * atr * vix_mult, self.BASE_TP)
        else:
            sl = self.BASE_SL * vix_mult
            tp = self.BASE_TP * vix_mult

        if max_loss_budget > 0 and lot_size > 0:
            max_sl = (max_loss_budget / lot_size) * 0.995
            if max_sl < sl:
                ratio = max_sl / sl
                sl = max_sl
                tp = tp * ratio

        return round(sl, 1), round(tp, 1)

    def _calc_confidence(self, pcr_shift_abs: float, vix: float) -> float:
        base = 0.50
        if pcr_shift_abs > 0.15:
            base += 0.10
        elif pcr_shift_abs > 0.10:
            base += 0.06
        elif pcr_shift_abs > 0.05:
            base += 0.03

        if vix >= 28:
            base -= 0.05
        elif vix >= 22:
            base -= 0.03

        return min(0.85, max(0.30, round(base, 3)))
