"""
expiry_utils.py — Expiry Date & DTE Utilities
===============================================
Derives DTE from the actual NIFTY_EXPIRY env var (or OpenAlgo API),
NOT hardcoded to any day of week. Handles holiday-shifted expiries.

Usage:
    from core.expiry_utils import get_dte, is_expiry_day, get_expiry_date
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_cached_expiry_date: Optional[date] = None


def _parse_expiry_str(s: str) -> Optional[date]:
    """Parse 'DDMMMYY' (e.g. '07APR26') into a date object."""
    for fmt in ("%d%b%y", "%d%b%Y", "%d-%b-%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s.strip().upper(), fmt).date()
        except ValueError:
            continue
    return None


def get_expiry_date() -> Optional[date]:
    """
    Get the current weekly expiry date.
    Priority: NIFTY_EXPIRY env var → cached → None.
    """
    global _cached_expiry_date

    env_val = os.getenv("NIFTY_EXPIRY", "").strip().upper()
    if env_val:
        parsed = _parse_expiry_str(env_val)
        if parsed:
            _cached_expiry_date = parsed
            return parsed
        else:
            logger.warning(f"Cannot parse NIFTY_EXPIRY='{env_val}' — expected DDMMMYY")

    return _cached_expiry_date


def set_expiry_from_api(expiry_str: str):
    """Set expiry from OpenAlgo API response (called by strike_selector)."""
    global _cached_expiry_date
    parsed = _parse_expiry_str(expiry_str)
    if parsed:
        _cached_expiry_date = parsed


def get_dte() -> int:
    """
    Days to expiry from TODAY. Returns 0 on expiry day.
    Uses actual NIFTY_EXPIRY date — handles holiday shifts correctly.
    Falls back to 3 if no expiry date is available.
    """
    exp = get_expiry_date()
    if not exp:
        return 3  # safe middle-of-week default

    today = date.today()
    dte = (exp - today).days

    # If expiry is past (forgot to update env var), assume next week
    if dte < 0:
        logger.warning(
            f"NIFTY_EXPIRY={exp.strftime('%d%b%y')} is in the past "
            f"({dte} days ago) — update settings.env!"
        )
        return 3  # don't assume

    return dte


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
        "expiry_date": exp.isoformat() if exp else "UNKNOWN",
        "expiry_day_name": exp.strftime("%A") if exp else "UNKNOWN",
    }
