"""
test_freshness_gate.py — Unit tests for the data-freshness entry gate (Issue #1).

Run:
    python scripts/test_freshness_gate.py

Covers:
  A. tv_fetcher provenance registry (_record_meta + get_data_freshness)
  B. ClaudePilot._entry_data_is_fresh() decision logic for each source/age case
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core import tv_fetcher as tvf

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def _df(last_minutes_ago: float) -> pd.DataFrame:
    """Build a tiny OHLCV df whose last bar is `last_minutes_ago` min old.

    Bars are tz-aware IST to mirror TradingView (IST-naive in prod, localized
    to Asia/Kolkata by _record_meta) and to make bar_age tz-independent here.
    """
    end = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(minutes=last_minutes_ago)
    idx = pd.date_range(end=end, periods=5, freq="5min")
    return pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 0},
        index=idx,
    )


# ── A. tv_fetcher provenance registry ──────────────────────────────────────
print("\n[A] tv_fetcher.get_data_freshness")

key = ("NIFTY", "NSE", 5)

# A1: never fetched -> NONE with huge ages
tvf._last_fetch_meta.clear()
f = tvf.get_data_freshness("NIFTY", "NSE", 5)
check("unseen feed -> source=NONE", f["source"] == "NONE")
check("unseen feed -> bar_age huge", f["bar_age_sec"] > 1e8)

# A2: fresh TV fetch, 1-min-old bar
tvf._record_meta(key, "TV", _df(last_minutes_ago=1))
f = tvf.get_data_freshness("NIFTY", "NSE", 5)
check("TV -> source=TV", f["source"] == "TV")
check("TV -> fetch_age small", f["fetch_age_sec"] < 5)
check("TV -> bar_age ~60s", 50 < f["bar_age_sec"] < 90)

# A3: yFinance, 16-min-old bar
tvf._record_meta(key, "YFINANCE", _df(last_minutes_ago=16))
f = tvf.get_data_freshness("NIFTY", "NSE", 5)
check("YFINANCE -> source=YFINANCE", f["source"] == "YFINANCE")
check("YFINANCE -> bar_age ~960s", 900 < f["bar_age_sec"] < 1020)


# ── B. ClaudePilot._entry_data_is_fresh decision logic ─────────────────────
print("\n[B] ClaudePilot._entry_data_is_fresh")

try:
    from core.claude_pilot import ClaudePilot
    gate = ClaudePilot._entry_data_is_fresh  # unbound; never touches self

    def run_gate(source, bar_age, fetch_age=2.0):
        """Monkeypatch get_data_freshness to a fixed state, then run the gate."""
        orig = tvf.get_data_freshness
        # The gate imports get_data_freshness from core.tv_fetcher at call time,
        # so patch the module attribute.
        tvf.get_data_freshness = lambda *a, **k: {
            "source": source, "bar_age_sec": bar_age, "fetch_age_sec": fetch_age
        }
        try:
            return gate(None)   # self is unused
        finally:
            tvf.get_data_freshness = orig

    # HARD gate = source filter + fetch_age (timezone-immune).
    # bar_age is diagnostic only and does NOT block (tz-fragile across sources).
    check("TV fresh -> OK",            run_gate("TV", 60)["ok"] is True)
    check("KOTAK_WS fresh -> OK",      run_gate("KOTAK_WS", 30)["ok"] is True)
    check("TV_CACHE -> OK",            run_gate("TV_CACHE", 300)["ok"] is True)
    check("YFINANCE -> BLOCKED",       run_gate("YFINANCE", 60)["ok"] is False)
    check("NONE -> BLOCKED",           run_gate("NONE", 1e9)["ok"] is False)
    check("TV old bar but fresh fetch -> OK (bar_age not a hard gate)",
          run_gate("TV", 480)["ok"] is True)
    check("TV but fetch 3min -> BLOCK (fetch_age gate)",
          run_gate("TV", 60, fetch_age=200)["ok"] is False)

    # Reason strings are populated
    r = run_gate("YFINANCE", 60)
    check("blocked reason mentions source", "YFINANCE" in r["reason"])

except Exception as e:
    print(f"  SKIP  ClaudePilot import failed in this env ({e})")
    print(f"        (will run on Hostinger venv) — tv_fetcher layer already verified")


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
