"""
regime_engine.py — Real-time market regime classifier.

Inspired by:
- Robert Carver, "Systematic Trading" (regime allocation)
- Andreas Clenow, "Following the Trend" (trend strength via R²)
- Larry Williams, "Long-Term Secrets to Short-Term Trading" (cycle phases)
- López de Prado, "Advances in Financial ML" Ch. 17 (Microstructural features)

CORE INSIGHT:
The same ML signal has different expected value in different regimes.
A CALL signal in TREND_UP regime: ~65% historical win rate.
A CALL signal in TREND_DOWN regime: ~35% historical win rate.
A CALL signal in REVERSAL regime: ~58% historical win rate.

The bot should adjust its confidence threshold based on regime — not
apply blanket rules. This module provides that classification.

Regime types:
    TREND_UP       — directional uptrend, follow it
    TREND_DOWN     — directional downtrend, follow it
    REVERSAL_UP    — recent low + bouncing (bottom-fishing window)
    REVERSAL_DOWN  — recent high + falling (top-fade window)
    CHOP           — sideways, both sides losing
    OPENING        — first 15 min, special rules
    UNKNOWN        — not enough data
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Regime:
    name: str               # "TREND_UP", "TREND_DOWN", etc.
    confidence: float       # 0..1 — how confidently we classified
    trend_strength: float   # 0..1 — R² of linear fit on last 20 bars
    range_pct: float        # last-20-bar range as % of session range
    reversal_potential: float  # 0..1 — likelihood of imminent reversal
    favored_direction: str  # "CALL", "PUT", "BOTH", "NEITHER"
    confidence_floor: int   # min confidence ML must produce to act in this regime
    notes: str

    @classmethod
    def unknown(cls) -> "Regime":
        return cls("UNKNOWN", 0.0, 0.0, 0.0, 0.0, "NEITHER", 65, "insufficient data")


class RegimeEngine:
    """
    Stateless classifier — call classify() each cycle with current bars.
    """

    def __init__(self):
        self._last_classify_ts: float = 0.0
        self._last_regime: Optional[Regime] = None
        self._cache_ttl_sec: int = 60   # only re-classify once per minute

    def classify(self, df_5m, spot: float, day_high: float, day_low: float,
                 ref_time: Optional[datetime] = None) -> Regime:
        """
        df_5m: pandas DataFrame with columns [open, high, low, close]
               last ~30-60 bars of 5-minute data
        spot:  current Nifty spot
        day_high/day_low: session extremes
        ref_time: timestamp to use for "is opening/lunch/closing" checks.
                  Defaults to datetime.now(). Pass bar timestamp in backtest.
        """
        # Use ref_time as cache key — in backtest this varies per bar, so
        # the cache automatically disables itself. In live, ref_time=None
        # → falls back to wall-clock cache.
        if ref_time is None:
            now_ts = time.time()
            cache_key = int(now_ts // self._cache_ttl_sec)
        else:
            cache_key = ref_time.isoformat()
            now_ts = ref_time.timestamp()

        if (self._last_regime is not None and
            getattr(self, "_cache_key", None) == cache_key):
            return self._last_regime

        try:
            r = self._do_classify(df_5m, spot, day_high, day_low, ref_time=ref_time)
        except Exception as e:
            logger.debug(f"RegimeEngine classify failed: {e}")
            r = Regime.unknown()

        self._last_regime = r
        self._last_classify_ts = now_ts
        self._cache_key = cache_key
        return r

    def _do_classify(self, df_5m, spot: float, day_high: float, day_low: float,
                     ref_time: Optional[datetime] = None) -> Regime:
        import numpy as np

        if df_5m is None or df_5m.empty or len(df_5m) < 20:
            return Regime.unknown()

        closes = df_5m["close"].astype(float).values[-30:]
        highs  = df_5m["high"].astype(float).values[-30:]
        lows   = df_5m["low"].astype(float).values[-30:]
        n = len(closes)

        # ── Feature 1: Trend strength (R² of linear regression) ──────
        x = np.arange(n)
        slope, intercept = np.polyfit(x, closes, 1)
        fit = slope * x + intercept
        ss_res = np.sum((closes - fit) ** 2)
        ss_tot = np.sum((closes - closes.mean()) ** 2)
        r_squared = max(0.0, min(1.0, 1 - ss_res / max(ss_tot, 1e-9)))
        slope_pct = slope / closes.mean() * 100  # % per bar

        # ── Feature 2: Range as % of session ─────────────────────────
        recent_range = float(highs[-20:].max() - lows[-20:].min())
        session_range = max(day_high - day_low, 1.0)
        range_ratio = recent_range / session_range

        # ── Feature 3: Position within session range ─────────────────
        # 0 = at day low, 1 = at day high
        if day_high > day_low:
            pos_in_range = (spot - day_low) / (day_high - day_low)
        else:
            pos_in_range = 0.5

        # ── Feature 4: Reversal potential ────────────────────────────
        # High when at extreme + recent bars showing reversal candle pattern
        last_3_closes = closes[-3:]
        last_3_lows = lows[-3:]
        last_3_highs = highs[-3:]
        # Higher lows from recent bottom = bullish reversal forming
        higher_lows = (last_3_lows[-1] > last_3_lows[-2] > last_3_lows[-3])
        # Lower highs from recent top = bearish reversal forming
        lower_highs = (last_3_highs[-1] < last_3_highs[-2] < last_3_highs[-3])

        reversal_potential = 0.0
        if pos_in_range <= 0.15 and higher_lows:
            reversal_potential = 0.85   # at day low + higher lows = strong bounce
        elif pos_in_range <= 0.25 and higher_lows:
            reversal_potential = 0.6
        elif pos_in_range >= 0.85 and lower_highs:
            reversal_potential = 0.85   # at day high + lower highs = strong fade
        elif pos_in_range >= 0.75 and lower_highs:
            reversal_potential = 0.6

        # ── Feature 5: Time of day awareness ─────────────────────────
        # Use ref_time if provided (for backtest) — wall clock breaks backtest
        now = ref_time if ref_time is not None else datetime.now()
        minutes_into_session = (now.hour - 9) * 60 + now.minute - 15
        is_opening = 0 <= minutes_into_session <= 15
        is_closing = minutes_into_session >= 350  # after 15:05
        is_lunch = 60 <= minutes_into_session <= 180  # 10:15-12:15

        # ── Decision tree ────────────────────────────────────────────
        if is_opening:
            return Regime(
                name="OPENING",
                confidence=1.0,
                trend_strength=r_squared,
                range_pct=range_ratio,
                reversal_potential=reversal_potential,
                favored_direction="NEITHER",
                confidence_floor=80,    # very high bar — opening is noise
                notes=f"opening 15min (slope={slope_pct:+.2f}%/bar)",
            )

        # Reversal regime — at extreme + reversal candles forming
        if reversal_potential >= 0.6:
            if pos_in_range <= 0.25 and higher_lows:
                return Regime(
                    name="REVERSAL_UP",
                    confidence=reversal_potential,
                    trend_strength=r_squared,
                    range_pct=range_ratio,
                    reversal_potential=reversal_potential,
                    favored_direction="CALL",
                    # 2026-06-10 FIX: was 50 ("high-quality setup" assumption,
                    # never validated). Shadow trade evidence (data/shadow_trades.jsonl,
                    # n=10 REVERSAL_UP CALL signals) shows 10% would_have_won rate —
                    # the WORST of any regime — yet this had the LOWEST floor of any
                    # regime. Raised to 75 until re-validated with real outcome data.
                    confidence_floor=75,
                    notes=(
                        f"at day-low ({pos_in_range:.0%} of range), "
                        f"higher lows confirmed, R²={r_squared:.2f}"
                    ),
                )
            if pos_in_range >= 0.75 and lower_highs:
                return Regime(
                    name="REVERSAL_DOWN",
                    confidence=reversal_potential,
                    trend_strength=r_squared,
                    range_pct=range_ratio,
                    reversal_potential=reversal_potential,
                    favored_direction="PUT",
                    confidence_floor=50,
                    notes=(
                        f"at day-high ({pos_in_range:.0%} of range), "
                        f"lower highs confirmed, R²={r_squared:.2f}"
                    ),
                )

        # 2026-05-07 FIX: thresholds were too strict — May 6 had 80% CHOP
        # classifications with floor=70% blocking most signals on a clearly
        # directional day. Lowered R² gates to capture moderate trends.

        # Strong trend — R² >= 0.55 + meaningful slope (was 0.70)
        # Real intraday Nifty rarely sustains R²>0.7 for 30 bars due to noise.
        if r_squared >= 0.55 and abs(slope_pct) >= 0.03:
            if slope > 0:
                return Regime(
                    name="TREND_UP",
                    confidence=r_squared,
                    trend_strength=r_squared,
                    range_pct=range_ratio,
                    reversal_potential=reversal_potential,
                    favored_direction="CALL",
                    confidence_floor=55,
                    notes=f"uptrend R²={r_squared:.2f} slope={slope_pct:+.3f}%/bar",
                )
            else:
                return Regime(
                    name="TREND_DOWN",
                    confidence=r_squared,
                    trend_strength=r_squared,
                    range_pct=range_ratio,
                    reversal_potential=reversal_potential,
                    favored_direction="PUT",
                    # 2026-06-11: 55→50. PUT WR tracks backtest (vs CALL 22%);
                    # Jun 10 PM downtrend produced zero trades. CALL twin stays 55.
                    confidence_floor=50,
                    notes=f"downtrend R²={r_squared:.2f} slope={slope_pct:+.3f}%/bar",
                )

        # Mild trend (R² 0.35-0.55) — leaning but not strong, lower confidence
        if r_squared >= 0.35 and abs(slope_pct) >= 0.02:
            if slope > 0:
                return Regime(
                    name="TREND_UP",
                    confidence=r_squared,
                    trend_strength=r_squared,
                    range_pct=range_ratio,
                    reversal_potential=reversal_potential,
                    favored_direction="CALL",
                    confidence_floor=60,
                    notes=f"mild uptrend R²={r_squared:.2f} slope={slope_pct:+.3f}%/bar",
                )
            else:
                return Regime(
                    name="TREND_DOWN",
                    confidence=r_squared,
                    trend_strength=r_squared,
                    range_pct=range_ratio,
                    reversal_potential=reversal_potential,
                    favored_direction="PUT",
                    # 2026-06-11: 60→55 — same PUT-side relaxation as strong trend.
                    confidence_floor=55,
                    notes=f"mild downtrend R²={r_squared:.2f} slope={slope_pct:+.3f}%/bar",
                )

        # Lunch hour chop — 10:15-12:15 with very low directional R²
        if is_lunch and r_squared < 0.25:
            return Regime(
                name="CHOP",
                confidence=1.0 - r_squared,
                trend_strength=r_squared,
                range_pct=range_ratio,
                reversal_potential=reversal_potential,
                favored_direction="NEITHER",
                confidence_floor=75,    # was 85 (still strict but less brutal)
                notes=f"lunch chop R²={r_squared:.2f}, range={range_ratio:.0%}",
            )

        # Default — sideways, moderate restrictions (was floor=70 → 65)
        return Regime(
            name="CHOP",
            confidence=0.5,
            trend_strength=r_squared,
            range_pct=range_ratio,
            reversal_potential=reversal_potential,
            favored_direction="NEITHER" if r_squared < 0.20 else
                              ("CALL" if slope > 0 else "PUT"),
            confidence_floor=65,
            notes=f"weak structure R²={r_squared:.2f} slope={slope_pct:+.3f}%/bar",
        )


# ─── Singleton ──────────────────────────────────────────────────────────────
_INSTANCE: Optional[RegimeEngine] = None


def get_regime_engine() -> RegimeEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RegimeEngine()
    return _INSTANCE
