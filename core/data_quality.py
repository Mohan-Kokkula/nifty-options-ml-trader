"""
data_quality.py — OHLCV and Feature Matrix Quality Checks
==========================================================
Called from _run_ml_prediction() to detect data problems before
the ML model sees them. Problems that reach the model silently
produce wrong probabilities (e.g. 0-RSI, 0-ATR, zeroed ADX).

Two check functions:

  check_bar_data(df5, df15, df30, df60)
      Checks raw OHLCV bars BEFORE feature engineering:
        1. Minimum bar count
        2. Last-bar age (session-aware, IST-localised)
        3. Intraday gap detection (missing 5-min slots)
        4. Flat-close sequence (frozen feed)
        5. Duplicate bar timestamps

  check_feature_matrix(df_feat)
      Checks the feature DataFrame AFTER ffill() but BEFORE fillna(0),
      so genuine NaN from data gaps are still visible:
        1. NaN rate in last row (the row sent to the model)
        2. Infinite values
        3. Flat-close confirmation

Severity levels:
  "ok"    — no issues
  "warn"  — degraded data, log and continue
  "block" — bad enough to skip the trade cycle

All thresholds are overridable via environment variables so you can
tighten/loosen them without a code deploy.

Fail-open policy:
  Both check functions wrap internals in try/except.  Any exception
  inside a check returns DataQualityReport(ok=True) so a bug here
  can never silently halt all trading.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds (env-overridable) ─────────────────────────────────────────────
_MIN_5MIN_BARS    = int(  os.getenv("DQ_MIN_5MIN_BARS",      "55"))
_MAX_BAR_AGE_SEC  = int(  os.getenv("DQ_MAX_BAR_AGE_SEC",   "900"))   # 15 min
_MAX_GAP_FRACTION = float(os.getenv("DQ_MAX_GAP_FRACTION",  "0.05"))  # 5%
_MAX_NAN_FRACTION = float(os.getenv("DQ_MAX_NAN_FRACTION",  "0.15"))  # 15%
_FLAT_BARS_CHECK  = int(  os.getenv("DQ_FLAT_BARS_CHECK",     "5"))   # consecutive flat bars
_INTRADAY_GAP_MULTIPLIER = 1.5   # flag gap if > 1.5 × expected interval

# ── CRITICAL features — Phase 2 fix (2026-06-09) ─────────────────────────────
# Features that the model cannot function without.  Zeroing a NaN here silently
# corrupts every derived feature that depends on it (ATR-based SL/TP, RSI
# divergence, VWAP distance, etc.) and produces misleading ML probabilities.
# If any of these is NaN after ffill(), the data feed has genuinely failed for
# a foundational indicator — inference MUST be blocked, not just warned.
#
# All names match the column names produced by train_model_v9.py feature
# engineering functions.  Case-sensitive.
_CRITICAL_FEATURES = frozenset({
    # Raw price (5-min bar)
    "close", "open", "high", "low",
    # Volatility
    "atr_14", "atr_7",
    # Momentum
    "rsi_14", "rsi_7",
    # Trend
    "adx_14", "ema_9", "ema_21",
    # Volume / VWAP
    "volume", "vwap",
})


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class DataQualityReport:
    """
    Result of a quality check.  Consumers should check `.severity`:
      "ok"    → clean, trade normally
      "warn"  → log but continue
      "block" → return SKIP, don't trade
    """
    ok: bool = True
    severity: str = "ok"
    issues: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add_issue(self, sev: str, msg: str) -> None:
        self.issues.append(f"[{sev.upper()}] {msg}")
        if sev == "block":
            self.ok = False
            self.severity = "block"
        elif sev == "warn" and self.severity == "ok":
            self.severity = "warn"

    def summary(self) -> str:
        if not self.issues:
            return "ok"
        return "; ".join(self.issues)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bar_age_seconds(df: pd.DataFrame) -> float:
    """
    Return age of the last bar in seconds, IST-localised.

    TV bars are timezone-naive IST.  We localise them to Asia/Kolkata
    before comparing so the result is timezone-independent.
    Returns 1e9 if the index is unparseable.
    """
    try:
        last = pd.Timestamp(df.index[-1])
        if last.tzinfo is None:
            last = last.tz_localize("Asia/Kolkata")
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        return float((now - last).total_seconds())
    except Exception:
        return 1e9


def _count_intraday_gaps(df: pd.DataFrame,
                          expected_freq_min: int = 5) -> tuple[int, int]:
    """
    Count intraday bar gaps (missing time slots within a session).

    A gap = two consecutive bars spaced > (expected_freq_min × 1.5)
    minutes apart, but NOT a cross-session/overnight gap (> 2 hours).

    Cross-session gaps (e.g. 15:30 → 09:15 next day) are expected and
    are excluded so we don't flag normal day boundaries.

    Returns (gap_count, total_consecutive_pairs).
    """
    if df is None or len(df) < 2:
        return 0, 0
    try:
        idx = pd.to_datetime(df.index)
        diffs = idx[1:] - idx[:-1]
        threshold  = timedelta(minutes=expected_freq_min * _INTRADAY_GAP_MULTIPLIER)
        overnight  = timedelta(hours=2)
        intraday_gaps = int(sum(1 for d in diffs if threshold < d < overnight))
        return intraday_gaps, len(diffs)
    except Exception:
        return 0, 0


# ── Public API ────────────────────────────────────────────────────────────────

def check_bar_data(
    df5:  pd.DataFrame,
    df15: Optional[pd.DataFrame] = None,
    df30: Optional[pd.DataFrame] = None,
    df60: Optional[pd.DataFrame] = None,
) -> DataQualityReport:
    """
    Check OHLCV bar quality before feature engineering.

    Checks (in order):
      1. df5 must exist and have >= DQ_MIN_5MIN_BARS rows (default 55)
      2. Last 5-min bar must be < DQ_MAX_BAR_AGE_SEC old (default 900s=15min)
      3. Intraday gap fraction must be < DQ_MAX_GAP_FRACTION (default 5%)
      4. Last DQ_FLAT_BARS_CHECK bars must not all have identical close
      5. Duplicate bar timestamps logged as warning
      6. HTF bar counts checked (warn only; HTF failures have fallbacks)
    """
    rpt = DataQualityReport()
    try:
        # ── 1. Minimum bar count ──────────────────────────────────────────
        if df5 is None or df5.empty:
            rpt.add_issue("block", "df5 empty — no 5-min bars fetched")
            return rpt

        n5 = len(df5)
        rpt.metrics["df5_n"] = n5
        if n5 < _MIN_5MIN_BARS:
            rpt.add_issue("block",
                f"df5 has {n5} bars < min {_MIN_5MIN_BARS} (insufficient warmup for indicators)")
            return rpt  # no point checking further

        # ── 2. Last-bar age ───────────────────────────────────────────────
        age = _bar_age_seconds(df5)
        rpt.metrics["df5_bar_age_sec"] = round(age, 1)

        if age > _MAX_BAR_AGE_SEC:
            rpt.add_issue("block",
                f"df5 last bar is {age/60:.1f}min old "
                f"(limit {_MAX_BAR_AGE_SEC/60:.0f}min) — feed stale")
        elif age > _MAX_BAR_AGE_SEC * 0.6:
            # 60% of limit — warning zone
            rpt.add_issue("warn",
                f"df5 last bar is {age/60:.1f}min old (approaching {_MAX_BAR_AGE_SEC/60:.0f}min limit)")

        # ── 3. Intraday gap detection ─────────────────────────────────────
        gaps, total = _count_intraday_gaps(df5, expected_freq_min=5)
        rpt.metrics["df5_gaps"]  = gaps
        rpt.metrics["df5_pairs"] = total

        if total > 0:
            gap_frac = gaps / total
            rpt.metrics["df5_gap_frac"] = round(gap_frac, 4)
            if gap_frac > _MAX_GAP_FRACTION:
                rpt.add_issue("block",
                    f"df5 has {gaps}/{total} intraday gaps "
                    f"({gap_frac:.0%} > {_MAX_GAP_FRACTION:.0%}) — "
                    "rolling indicators (RSI/ATR/ADX) computed across gap are unreliable")
            elif gaps > 0:
                rpt.add_issue("warn",
                    f"df5 has {gaps} intraday gap(s) in {total} pairs "
                    f"({gap_frac:.1%}) — minor indicator distortion possible")

        # ── 4. Flat-close detection (frozen feed) ─────────────────────────
        if "close" in df5.columns and len(df5) >= _FLAT_BARS_CHECK:
            last_closes = df5["close"].iloc[-_FLAT_BARS_CHECK:]
            if last_closes.nunique() == 1 and float(last_closes.iloc[-1]) > 0:
                rpt.add_issue("block",
                    f"df5 close flat for last {_FLAT_BARS_CHECK} bars "
                    f"at {last_closes.iloc[-1]:.2f} — feed appears frozen")

        # ── 5. Duplicate bar timestamps ───────────────────────────────────
        dup_count = int(df5.index.duplicated().sum())
        rpt.metrics["df5_dups"] = dup_count
        if dup_count > 0:
            rpt.add_issue("warn",
                f"df5 has {dup_count} duplicate index entries — "
                "features may have slight bias toward duplicated bars")

        # ── 6. HTF bar counts (warn only) ─────────────────────────────────
        for name, dfx, min_bars in [
            ("df15", df15, 10),
            ("df30", df30,  6),
            ("df60", df60,  4),
        ]:
            if dfx is not None and not dfx.empty:
                rpt.metrics[f"{name}_n"] = len(dfx)
                if len(dfx) < min_bars:
                    rpt.add_issue("warn",
                        f"{name} has only {len(dfx)} bars < {min_bars} — "
                        "HTF features may be unreliable")

    except Exception as e:
        logger.warning(f"DataQuality: check_bar_data raised unexpectedly ({e}) — fail-open")

    return rpt


def check_feature_matrix(df_feat: pd.DataFrame) -> DataQualityReport:
    """
    Check the feature matrix AFTER ffill() but BEFORE fillna(0).

    Called at this specific point so that genuine NaN from data gaps are
    still visible. fillna(0) after this converts NaN to 0 silently — a
    zero RSI, zero ATR, or zero ADX fed to a tree model is harmful because
    the model was trained on realistic values, not zeros.

    Checks:
      1. NaN rate in last row > DQ_MAX_NAN_FRACTION (default 15%) → block
      2. Infinite values in last row → block
      3. Flat-close confirmation (cross-check with bar check)
    """
    rpt = DataQualityReport()
    try:
        if df_feat is None or df_feat.empty:
            rpt.add_issue("block", "feature matrix is empty after engineering")
            return rpt

        last = df_feat.iloc[-1]
        n_total = len(last)
        rpt.metrics["n_features"] = n_total

        # ── 1. NaN rate in prediction row ────────────────────────────────
        nan_n    = int(last.isna().sum())
        nan_frac = nan_n / max(n_total, 1)
        rpt.metrics["nan_n"]    = nan_n
        rpt.metrics["nan_frac"] = round(nan_frac, 4)

        if nan_frac > _MAX_NAN_FRACTION:
            rpt.add_issue("block",
                f"{nan_frac:.0%} of features are NaN in prediction row "
                f"({nan_n}/{n_total}) — exceeds {_MAX_NAN_FRACTION:.0%} threshold. "
                "Data gaps or insufficient warmup bars likely cause.")
        elif nan_n > 0:
            # Phase 2 fix (2026-06-09): check whether any CRITICAL feature is NaN.
            # Zeroing a foundational indicator (close, ATR, RSI…) silently corrupts
            # every derived feature that depends on it and produces wrong probabilities.
            # Block inference if critical; warn-only for optional/derived features.
            try:
                crit_nan = [
                    c for c in _CRITICAL_FEATURES
                    if c in last.index and pd.isna(last[c])
                ]
                if crit_nan:
                    rpt.add_issue("block",
                        f"CRITICAL feature(s) NaN — zeroing would corrupt inference: "
                        f"{crit_nan}. Check data feed for foundational indicators.")
                    rpt.metrics["critical_nan"] = crit_nan
                else:
                    rpt.add_issue("warn",
                        f"{nan_n}/{n_total} features are NaN in prediction row "
                        f"({nan_frac:.1%}) — all OPTIONAL; will be zeroed before inference")
                    rpt.metrics["optional_nan_n"] = nan_n
            except Exception:
                rpt.add_issue("warn",
                    f"{nan_n}/{n_total} features are NaN in prediction row "
                    f"({nan_frac:.1%}) — will be zeroed before inference")

        # ── 2. Infinite values ────────────────────────────────────────────
        try:
            numeric = df_feat.select_dtypes(include=[np.floating, np.integer])
            inf_n   = int(np.isinf(numeric.iloc[-1]).sum())
            rpt.metrics["inf_n"] = inf_n
            if inf_n > 0:
                rpt.add_issue("block",
                    f"{inf_n} infinite value(s) in feature matrix last row "
                    "— likely a division-by-zero in an indicator")
        except Exception:
            pass

        # ── 3. Flat-close confirmation ────────────────────────────────────
        if "close" in df_feat.columns and len(df_feat) >= _FLAT_BARS_CHECK:
            lc = df_feat["close"].iloc[-_FLAT_BARS_CHECK:]
            if lc.nunique() == 1 and float(lc.iloc[-1]) > 0:
                rpt.add_issue("block",
                    f"feature matrix close flat at {lc.iloc[-1]:.2f} "
                    f"for last {_FLAT_BARS_CHECK} rows — stale feed confirmed")

    except Exception as e:
        logger.warning(f"DataQuality: check_feature_matrix raised unexpectedly ({e}) — fail-open")

    return rpt
