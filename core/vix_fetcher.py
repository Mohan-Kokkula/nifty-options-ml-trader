"""
vix_fetcher.py — India VIX Daily Fetcher from NSE
==================================================
Fetches India VIX EOD data directly from NSE's public API.
Appends today's VIX row to data/india_vix.csv every trading day.

NSE endpoint:
    https://www.nseindia.com/api/historical/vixhistory
    ?startdate=18-Apr-2026&enddate=18-Apr-2026

NSE requires:
  - Browser-like headers (User-Agent, Referer)
  - A session cookie obtained by first hitting the NSE homepage

CSV format (what train_model_v8.py expects):
    date, open, high, low, close

Usage:
    from core.vix_fetcher import update_vix_csv
    update_vix_csv()   # fetches today and appends to data/india_vix.csv
"""

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_VIX_CSV     = Path("data/india_vix.csv")
_NSE_HOME    = "https://www.nseindia.com"
_NSE_VIX_API = "https://www.nseindia.com/api/historical/vixhistory"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/market-data/india-vix",
}


def _get_nse_session() -> requests.Session:
    """
    Create a requests.Session with NSE cookies.
    NSE blocks direct API calls without a prior homepage visit.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    try:
        # Hit homepage first to get cookies
        session.get(_NSE_HOME, timeout=10)
        time.sleep(1.0)   # brief pause — NSE rate limits aggressive scrapers
        # Also hit the VIX page to get specific cookies
        session.get(
            "https://www.nseindia.com/market-data/india-vix",
            timeout=10
        )
        time.sleep(0.5)
    except Exception as e:
        logger.debug(f"VixFetcher: session init warning: {e}")
    return session


def fetch_vix_for_date(target_date: date) -> Optional[dict]:
    """
    Fetch VIX data for a specific date from NSE.
    Returns dict with keys: date, open, high, low, close
    Returns None if data unavailable (holiday / weekend / market closed).
    """
    date_str = target_date.strftime("%d-%b-%Y")   # e.g. 18-Apr-2026

    session = _get_nse_session()

    params = {
        "startdate": date_str,
        "enddate":   date_str,
    }

    try:
        resp = session.get(_NSE_VIX_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("data", [])
        if not rows:
            logger.debug(f"VixFetcher: no VIX data for {date_str} (holiday/weekend?)")
            return None

        row = rows[0]
        vix = {
            "date":  target_date.isoformat(),
            "open":  float(row.get("EOD_VIX_OPEN",  0)),
            "high":  float(row.get("EOD_VIX_HIGH",  0)),
            "low":   float(row.get("EOD_VIX_LOW",   0)),
            "close": float(row.get("EOD_VIX_CLOSE", 0)),
        }

        if vix["close"] <= 0:
            logger.debug(f"VixFetcher: zero VIX close for {date_str}")
            return None

        logger.info(
            f"VixFetcher: {date_str} — "
            f"O={vix['open']} H={vix['high']} L={vix['low']} C={vix['close']}"
        )
        return vix

    except Exception as e:
        logger.warning(f"VixFetcher: fetch failed for {date_str}: {e}")
        return None


def update_vix_csv(target_date: Optional[date] = None) -> bool:
    """
    Fetch VIX for target_date (default: today) and append to data/india_vix.csv.
    Skips if today's row already exists.
    Returns True if data was added, False otherwise.
    """
    if target_date is None:
        target_date = date.today()

    Path("data").mkdir(parents=True, exist_ok=True)

    # Check if today's data already exists
    if _VIX_CSV.exists():
        try:
            existing = pd.read_csv(_VIX_CSV, index_col=0, parse_dates=True)
            existing.index = pd.to_datetime(existing.index).date
            if target_date in existing.index:
                logger.debug(
                    f"VixFetcher: {target_date} already in india_vix.csv — skipping"
                )
                return False
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    # Fetch from NSE
    vix = fetch_vix_for_date(target_date)
    if vix is None:
        return False

    # Append row
    new_row = pd.DataFrame([{
        "open":  vix["open"],
        "high":  vix["high"],
        "low":   vix["low"],
        "close": vix["close"],
    }], index=pd.DatetimeIndex([vix["date"]]))

    if not existing.empty:
        combined = pd.concat([existing, new_row])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = new_row

    combined.to_csv(_VIX_CSV)
    logger.info(
        f"VixFetcher: india_vix.csv updated — "
        f"{len(combined)} total rows | latest close={vix['close']}"
    )
    return True


def backfill_vix(days: int = 30) -> int:
    """
    Backfill VIX data for the past N calendar days.
    Useful on first run to populate india_vix.csv with history.
    Returns number of rows successfully added.
    """
    added = 0
    today = date.today()

    logger.info(f"VixFetcher: backfilling last {days} calendar days...")

    for i in range(days, 0, -1):
        target = today - timedelta(days=i)
        # Skip weekends
        if target.weekday() >= 5:
            continue
        try:
            if update_vix_csv(target):
                added += 1
            time.sleep(1.5)   # be polite to NSE — avoid rate limiting
        except Exception as e:
            logger.debug(f"VixFetcher: backfill skip {target}: {e}")

    logger.info(f"VixFetcher: backfill complete — {added} trading days added")
    return added


if __name__ == "__main__":
    # Standalone test / manual backfill
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [VIX] %(message)s"
    )

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Backfilling VIX for last {days} days...")
    added = backfill_vix(days)
    print(f"Done — {added} rows added to data/india_vix.csv")

    # Show last 5 rows
    if _VIX_CSV.exists():
        df = pd.read_csv(_VIX_CSV, index_col=0)
        print(df.tail(5).to_string())
