"""
expiry_utils.py — Expiry Date & DTE Utilities
===============================================
Auto-calculates the next Nifty weekly expiry (Tuesday) when
NIFTY_EXPIRY is not set or stale. Manual override via env var
still works for holiday-shifted expiries.

Usage:
    from core.expiry_utils import get_dte, is_expiry_day, get_expiry_date
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_cached_expiry_date: Optional[date] = None
_EXPIRY_WEEKDAY = 1  # Tuesday = 1 (Monday=0 ... Sunday=6)


def _parse_expiry_str(s: str) -> Optional[date]:
    """Parse 'DDMMMYY' (e.g. '07APR26') into a date object."""
    for fmt in ("%d%b%y", "%d%b%Y", "%d-%b-%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s.strip().upper(), fmt).date()
        except ValueError:
            continue
    return None


def _next_tuesday(from_date: date = None) -> date:
    """Return the next Tuesday on or after from_date."""
    if from_date is None:
        from_date = date.today()
    days_ahead = (_EXPIRY_WEEKDAY - from_date.weekday()) % 7
    if days_ahead == 0:
        return from_date
    return from_date + timedelta(days=days_ahead)


def get_expiry_date() -> date:
    """
    Get the current weekly expiry date.
    Priority:
      1. NIFTY_EXPIRY env var (if set and not in the past)
      2. Cached from API
      3. Auto-calculate next Tuesday
    """
    global _cached_expiry_date

    env_val = os.getenv("NIFTY_EXPIRY", "").strip().upper()
    if env_val:
        parsed = _parse_expiry_str(env_val)
        if parsed:
            if parsed >= date.today():
                _cached_expiry_date = parsed
                return parsed
            else:
                auto = _next_tuesday()
                logger.warning(
                    f"NIFTY_EXPIRY={env_val} is in the past — "
                    f"auto-using next Tuesday {auto.strftime('%d%b%y').upper()}"
                )
                _cached_expiry_date = auto
                return auto
        else:
            logger.warning(f"Cannot parse NIFTY_EXPIRY='{env_val}' — expected DDMMMYY")

    if _cached_expiry_date and _cached_expiry_date >= date.today():
        return _cached_expiry_date

    auto = _next_tuesday()
    logger.info(f"No NIFTY_EXPIRY set — auto-calculated next Tuesday: {auto.strftime('%d%b%y').upper()}")
    _cached_expiry_date = auto
    return auto


def set_expiry_from_api(expiry_str: str):
    """Set expiry from OpenAlgo API response (called by strike_selector)."""
    global _cached_expiry_date
    parsed = _parse_expiry_str(expiry_str)
    if parsed:
        _cached_expiry_date = parsed


def get_dte() -> int:
    """
    Days to expiry from TODAY. Returns 0 on expiry day.
    Auto-calculates next Tuesday if env var is stale/missing.
    """
    exp = get_expiry_date()
    today = date.today()
    dte = (exp - today).days
    return max(0, dte)


def is_expiry_day() -> bool:
    """True if today is the expiry day."""
    return get_dte() == 0


def is_pre_expiry() -> bool:
    """True if tomorrow is expiry (DTE=1)."""
    return get_dte() == 1


def get_dte_norm() -> float:
    """Normalized DTE: 0.0 = expiry day, 1.0 = far from expiry (6+ days)."""
    return min(1.0, get_dte() / 6.0)


def get_expiry_context() -> dict:
    """
    Full expiry context for trading decisions.
    Returns dict with all expiry-related info.
    """
    dte = get_dte()
    exp = get_expiry_date()

    if dte == 0:
        label = "EXPIRY_DAY"
    elif dte == 1:
        label = "PRE_EXPIRY"
    else:
        label = f"DTE={dte}"

    return {
        "dte": dte,
        "dte_norm": get_dte_norm(),
        "is_expiry": dte == 0,
        "is_pre_expiry": dte == 1,
        "label": label,
        "expiry_date": exp.isoformat(),
        "expiry_day_name": exp.strftime("%A"),
    }
