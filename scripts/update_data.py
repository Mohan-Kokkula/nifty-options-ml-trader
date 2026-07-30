"""
update_data.py — Merge fresh NIFTY bars into all CSV data files
==============================================================
Fetches 5-min, 15-min, 30-min, 60-min and daily bars from TradingView
and merges them into each CSV (fills internal gaps too, not just the
tail; no duplicates).

Usage:
    python scripts/update_data.py            # append today
    python scripts/update_data.py --days 5   # backfill last 5 days
    python scripts/update_data.py --verify   # just check CSV status
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which can't encode the checkmark /
# warning / box-drawing characters below and crashes verify() mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_path = Path(__file__).parent.parent / "config" / "settings.env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DATA_DIR = Path(__file__).parent.parent / "data"

FEEDS = [
    {"name": "5-min",  "file": "nifty_5min.csv",  "interval": 5,   "n_bars": 100},
    {"name": "15-min", "file": "nifty_15min.csv", "interval": 15,  "n_bars": 45},
    {"name": "30-min", "file": "nifty_30min.csv", "interval": 30,  "n_bars": 25},
    {"name": "60-min", "file": "nifty_60min.csv", "interval": 60,  "n_bars": 12},
    {"name": "Daily",  "file": "nifty_day.csv",   "interval": "D", "n_bars": 5},
]

VIX_FILE = "india_vix.csv"


def get_tv():
    try:
        from tvDatafeed import TvDatafeed
    except ImportError:
        print("ERROR: Run: pip install tradingview-datafeed")
        sys.exit(1)
    user = os.getenv("TV_USERNAME", "")
    pwd  = os.getenv("TV_PASSWORD", "")
    if user and pwd:
        print(f"  Logging in as {user} ...")
        return TvDatafeed(username=user, password=pwd)
    print("  WARNING: No TV credentials — anonymous session")
    return TvDatafeed()


def fetch(tv, interval, n_bars):
    from tvDatafeed import Interval
    imap = {
        5: Interval.in_5_minute, 15: Interval.in_15_minute,
        30: Interval.in_30_minute, 60: Interval.in_1_hour,
        "D": Interval.in_daily,
    }
    raw = tv.get_hist("NIFTY", "NSE", interval=imap[interval],
                      n_bars=n_bars + 20)
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw.columns = raw.columns.str.lower()
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"
    if interval != "D":
        raw = raw.between_time("09:15", "15:30")
    raw = raw[["open","high","low","close","volume"]].astype(float).dropna()
    return raw.sort_index()


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    dt = next((c for c in df.columns if "date" in c or "time" in c), None)
    if not dt:
        return pd.DataFrame()
    df[dt] = pd.to_datetime(df[dt], format="mixed")
    df = df.set_index(dt)
    df.index.name = "date"
    return df.sort_index()


def append_new(existing, fresh):
    """Merge fresh bars into existing, filling internal gaps as well as
    extending the tail. Keeping only rows strictly after the current max
    (the old behavior) is blind to holes earlier in the series -- once the
    tail has ever been synced, a run can never backfill a gap that opened
    up behind it. Merge-and-dedupe (freshest wins) fixes that."""
    if existing.empty:
        return fresh, len(fresh)
    before = len(existing)
    merged = pd.concat([existing, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    added = len(merged) - before
    return merged, added


def save(df, path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, date_format="%Y-%m-%d %H:%M:%S")


def fetch_vix(n_days=10):
    """
    Fetch India VIX daily data from TradingView.
    VIX is available as INDIAVIX on NSE.
    """
    try:
        tv = get_tv()
        from tvDatafeed import Interval
        raw = tv.get_hist("INDIAVIX", "NSE", interval=Interval.in_daily, n_bars=n_days + 5)
        if raw is None or raw.empty:
            # Try alternative symbol
            raw = tv.get_hist("INDIA VIX", "NSE", interval=Interval.in_daily, n_bars=n_days + 5)
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw.columns = raw.columns.str.lower()
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"
        raw = raw[["open", "high", "low", "close"]].astype(float).dropna()
        raw.columns = ["vix_open", "vix_high", "vix_low", "vix"]
        return raw[["vix"]].sort_index()
    except Exception as e:
        print(f"  ⚠ VIX fetch failed: {e}")
        return pd.DataFrame()


def verify():
    print("\n  CSV summary")
    print("  " + "─"*50)
    today = date.today()
    for feed in FEEDS:
        path = DATA_DIR / feed["file"]
        df = load_csv(path)
        if df.empty:
            print(f"  {feed['name']:8s}  MISSING — {path.name}")
            continue
        last  = df.index.max().date()
        gap   = (today - last).days
        rows  = len(df)
        status = "✅" if gap <= 1 else f"⚠ {gap}d behind"
        print(f"  {feed['name']:8s}  {rows:>8,} rows | last={last} | {status}")
    # VIX status
    vix_path = DATA_DIR / VIX_FILE
    df_vix = load_csv(vix_path)
    if df_vix.empty:
        print(f"  {'VIX':8s}  MISSING — run update to fetch")
    else:
        last = df_vix.index.max().date()
        gap  = (today - last).days
        status = "✅" if gap <= 1 else f"⚠ {gap}d behind"
        print(f"  {'VIX':8s}  {len(df_vix):>8,} rows | last={last} | {status}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    print("\n=== OpenClaw Data Update ===")

    if args.verify:
        verify()
        return

    tv = get_tv()

    for feed in FEEDS:
        path = DATA_DIR / feed["file"]
        n    = feed["n_bars"] * args.days + 20
        print(f"\n  Fetching {feed['name']} ...")
        fresh = fetch(tv, feed["interval"], n)
        if fresh.empty:
            print(f"  ⚠ No data returned for {feed['name']}")
            continue
        existing = load_csv(path)
        combined, added = append_new(existing, fresh)
        if added == 0:
            print(f"  ✅ {feed['name']} already up to date")
        else:
            save(combined, path)
            print(f"  ✅ {feed['name']}: appended {added} new bars → {len(combined):,} total")

    # ── Fetch India VIX ──────────────────────────────────────────
    print(f"\n  Fetching India VIX ...")
    vix_path = DATA_DIR / VIX_FILE
    fresh_vix = fetch_vix(n_days=args.days * 2 + 5)
    if not fresh_vix.empty:
        existing_vix = load_csv(vix_path)
        combined_vix, added_vix = append_new(existing_vix, fresh_vix)
        if added_vix == 0:
            print(f"  ✅ VIX already up to date")
        else:
            save(combined_vix, vix_path)
            print(f"  ✅ VIX: appended {added_vix} new rows → {len(combined_vix):,} total")
    else:
        print(f"  ⚠ VIX fetch failed — will retry tomorrow")

    verify()


if __name__ == "__main__":
    main()
