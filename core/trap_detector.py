"""
Trap Detector — rejects late-move / exhausted / low-momentum entries.

Root cause from 2026-04-24 analysis:
  - Nifty crashed -300pts between 09:35 and 11:15.
  - Bot fired 41 PUT signals, but the majority arrived AFTER 11:30 when the
    move was already exhausted — classic "chase the crash" trap.
  - Bot direction was RIGHT. Timing was LATE.

This module provides advisory gates that reject a signal when:
  1. LATE_MOVE:      Spot is within X% of day-extreme on the signal side
                     AND the extreme was made > N minutes ago.
  2. RANGE_CONTRACT: Last K 5m bars have a range < Y% of ATR — market is
                     consolidating, not trending. Directional bets die here.
  3. EXHAUSTION:     RSI extreme (<20 for PUT chase, >80 for CALL chase)
                     combined with price already far from VWAP on that side.

All checks are SOFT gates. Failure → reject the signal. Missing data → pass.
Fail-safe: any exception = pass (do not block live trading if this module dies).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrapVerdict:
    is_trap: bool
    reason: str        # short tag for logs
    detail: str        # human-readable explanation

    @classmethod
    def pass_(cls) -> "TrapVerdict":
        return cls(False, "ok", "no trap detected")

    @classmethod
    def block(cls, reason: str, detail: str) -> "TrapVerdict":
        return cls(True, reason, detail)


class TrapDetector:
    """
    Stateless-ish detector. Keeps rolling day high/low/extreme_time updated
    via `update_tick(spot)` — call once per cycle.
    """

    def __init__(
        self,
        late_move_pct: float = 0.25,     # within 0.25% of day extreme = "at extreme"
        late_move_min_age_min: int = 20, # extreme made >20 min ago = stale
        range_contract_bars: int = 3,
        range_contract_pct_of_atr: float = 0.40,
        rsi_exhaustion_put: float = 22.0,
        rsi_exhaustion_call: float = 78.0,
        vwap_far_pts: float = 80.0,
    ):
        self.late_move_pct = late_move_pct
        self.late_move_min_age_min = late_move_min_age_min
        self.range_contract_bars = range_contract_bars
        self.range_contract_pct_of_atr = range_contract_pct_of_atr
        self.rsi_exhaustion_put = rsi_exhaustion_put
        self.rsi_exhaustion_call = rsi_exhaustion_call
        self.vwap_far_pts = vwap_far_pts

        self._day_high: float = 0.0
        self._day_low: float = 0.0
        self._day_high_ts: Optional[datetime] = None
        self._day_low_ts: Optional[datetime] = None
        self._session_date: Optional[str] = None

    def _reset_if_new_session(self, now: datetime) -> None:
        today = now.strftime("%Y-%m-%d")
        if self._session_date != today:
            self._session_date = today
            self._day_high = 0.0
            self._day_low = 0.0
            self._day_high_ts = None
            self._day_low_ts = None

    def update_tick(self, spot: float, ts: Optional[datetime] = None) -> None:
        try:
            ts = ts or datetime.now()
            self._reset_if_new_session(ts)
            if spot <= 0:
                return
            if self._day_high == 0.0 or spot > self._day_high:
                self._day_high = spot
                self._day_high_ts = ts
            if self._day_low == 0.0 or spot < self._day_low:
                self._day_low = spot
                self._day_low_ts = ts
        except Exception as e:
            logger.debug(f"TrapDetector.update_tick failed: {e}")

    def is_trap(
        self,
        *,
        option_type: str,        # "CE" or "PE"
        spot: float,
        ml_indicators: dict,
        now: Optional[datetime] = None,
    ) -> TrapVerdict:
        """
        Return TrapVerdict. `is_trap=True` → pilot should SKIP the signal.
        """
        try:
            now = now or datetime.now()
            self.update_tick(spot, now)

            ot = (option_type or "").upper()
            # ── Gate 1: LATE MOVE ────────────────────────────────────
            if ot == "PE" and self._day_low > 0 and self._day_low_ts:
                pct_above_low = ((spot - self._day_low) / self._day_low) * 100.0
                age_min = (now - self._day_low_ts).total_seconds() / 60.0
                if pct_above_low <= self.late_move_pct and age_min >= self.late_move_min_age_min:
                    return TrapVerdict.block(
                        "LATE_PUT",
                        f"spot {spot:.0f} only {pct_above_low:.2f}% above day-low "
                        f"{self._day_low:.0f} (made {age_min:.0f}m ago) — chase trap"
                    )
            if ot == "CE" and self._day_high > 0 and self._day_high_ts:
                pct_below_high = ((self._day_high - spot) / self._day_high) * 100.0
                age_min = (now - self._day_high_ts).total_seconds() / 60.0
                if pct_below_high <= self.late_move_pct and age_min >= self.late_move_min_age_min:
                    return TrapVerdict.block(
                        "LATE_CALL",
                        f"spot {spot:.0f} only {pct_below_high:.2f}% below day-high "
                        f"{self._day_high:.0f} (made {age_min:.0f}m ago) — chase trap"
                    )

            # ── Gate 2: RANGE CONTRACTION ────────────────────────────
            try:
                atr = float(ml_indicators.get("atr_14", 0) or ml_indicators.get("atr", 0) or 0.0)
                recent_range = float(ml_indicators.get("recent_bar_range", 0) or 0.0)
                if atr > 0 and recent_range > 0:
                    ratio = recent_range / atr
                    if ratio < self.range_contract_pct_of_atr:
                        return TrapVerdict.block(
                            "RANGE_CONTRACT",
                            f"last-bar range {recent_range:.1f} / ATR {atr:.1f} = "
                            f"{ratio:.2f} < {self.range_contract_pct_of_atr:.2f} — chop"
                        )
            except Exception:
                pass

            # ── Gate 3: EXHAUSTION (RSI extreme + far from VWAP) ─────
            try:
                rsi = float(ml_indicators.get("rsi_14", ml_indicators.get("rsi", 50)) or 50)
                vwap_dist = float(ml_indicators.get("pa_vwap_distance", 0) or 0)
                if ot == "PE" and rsi <= self.rsi_exhaustion_put and vwap_dist <= -self.vwap_far_pts:
                    return TrapVerdict.block(
                        "PUT_EXHAUSTION",
                        f"RSI {rsi:.1f} <= {self.rsi_exhaustion_put} AND "
                        f"VWAP_dist {vwap_dist:+.0f} <= -{self.vwap_far_pts:.0f} — oversold bounce risk"
                    )
                if ot == "CE" and rsi >= self.rsi_exhaustion_call and vwap_dist >= self.vwap_far_pts:
                    return TrapVerdict.block(
                        "CALL_EXHAUSTION",
                        f"RSI {rsi:.1f} >= {self.rsi_exhaustion_call} AND "
                        f"VWAP_dist {vwap_dist:+.0f} >= {self.vwap_far_pts:.0f} — overbought"
                    )
            except Exception:
                pass

            return TrapVerdict.pass_()

        except Exception as e:
            logger.debug(f"TrapDetector.is_trap failed — fail-open: {e}")
            return TrapVerdict.pass_()


# ─── Singleton ──────────────────────────────────────────────────────────────
_INSTANCE: Optional[TrapDetector] = None


def get_trap_detector() -> TrapDetector:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = TrapDetector()
    return _INSTANCE
