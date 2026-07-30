#!/usr/bin/env python3
"""
Regression test for C7: get_next_expiry() must not use a stale
NIFTY_EXPIRY env var for symbol construction.

Standalone script (repo convention), run with:
    python scripts/test_expiry_staleness.py

Covers core/strike_selector.py::StrikeSelector.get_next_expiry().

Bug being regression-tested: previously, Priority 1 (the NIFTY_EXPIRY env
var) was returned unconditionally whenever set, even if the date was
already in the past — unlike core/expiry_utils.py::get_dte(), which
already had an equivalent staleness check for a different consumer.
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strike_selector import StrikeSelector


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _make_selector():
    selector = object.__new__(StrikeSelector)
    selector._cached_expiry = None
    selector._cached_expiry_list = None
    selector.client = None
    return selector


def main():
    all_ok = True

    # ── Stale NIFTY_EXPIRY: must NOT be returned, must fall through ────
    stale = (date.today() - timedelta(days=3)).strftime("%d%b%y").upper()
    selector = _make_selector()
    with patch.dict(os.environ, {"NIFTY_EXPIRY": stale}):
        # Priority 2 (cache) is empty and Priority 3 (client) is None (raises),
        # so this exercises Priority 4's fallback -- confirming the stale
        # value was NOT returned directly by Priority 1.
        result = selector.get_next_expiry()
    all_ok &= check(
        f"stale NIFTY_EXPIRY ({stale}) is NOT returned as-is",
        result != stale,
    )
    # Priority 4 re-derives from the identical env var, so it must also be
    # skipped -- otherwise it would silently resurface the same stale date.
    days_ahead = (1 - date.today().weekday()) % 7 or 7
    expected_last_resort = (date.today() + timedelta(days=days_ahead)).strftime("%d%b%y").upper()
    all_ok &= check(
        "falls all the way through to the calculated last-resort date",
        result == expected_last_resort,
    )

    # ── Future NIFTY_EXPIRY: must still be returned exactly as before ──
    future = (date.today() + timedelta(days=5)).strftime("%d%b%y").upper()
    selector2 = _make_selector()
    with patch.dict(os.environ, {"NIFTY_EXPIRY": future}):
        result2 = selector2.get_next_expiry()
    all_ok &= check(
        f"future NIFTY_EXPIRY ({future}) is still returned as-is",
        result2 == future,
    )

    # ── Today's date (expiry day itself): must still be honored ────────
    today_str = date.today().strftime("%d%b%y").upper()
    selector3 = _make_selector()
    with patch.dict(os.environ, {"NIFTY_EXPIRY": today_str}):
        result3 = selector3.get_next_expiry()
    all_ok &= check(
        f"today's-date NIFTY_EXPIRY ({today_str}) is still honored (expiry day)",
        result3 == today_str,
    )

    # ── Unparseable NIFTY_EXPIRY: unchanged pre-existing behavior ───────
    selector4 = _make_selector()
    with patch.dict(os.environ, {"NIFTY_EXPIRY": "NOTADATE"}):
        result4 = selector4.get_next_expiry()
    all_ok &= check(
        "unparseable NIFTY_EXPIRY is still returned as-is (pre-existing behavior, out of scope)",
        result4 == "NOTADATE",
    )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
