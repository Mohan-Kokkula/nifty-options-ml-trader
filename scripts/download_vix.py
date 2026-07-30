"""
download_vix.py — Download historical India VIX data from NSE
=============================================================
NSE provides free VIX data going back to 2007.

Usage:
    uv run scripts/download_vix.py              # full history
    uv run scripts/download_vix.py --days 30   # last 30 days only

Output: data/india_vix.csv
"""

import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_FILE = DATA_DIR / "india_vix.csv"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def get_nse_session():
    """Get NSE session cookies (required for API access)."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        session.get("https://www.nseindia.com/reports-indices-historical-vix", timeout=10)
        time.sleep(1)
    except Exception as e:
        print(f"Warning: Session setup failed: {e}")
    return session


def fetch_vix_nse(from_date: str, to_date: str, session) -> pd.DataFrame:
    """
    Fetch VIX data from NSE historical API.
    Date format: DD-Mon-YYYY (e.g. 01-Jan-2024)
    """
    url = "https://www.nseindia.com/api/historical/vixhistory"
    params = {"from": from_date, "to": to_date}
    try:
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if "data" not in data:
            return pd.DataFrame()
        rows = data["data"]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # NSE columns: EOD_TIMESTAMP, EOD_OPEN_INDEX_VAL, EOD_HIGH_INDEX_VAL,
        #              EOD_LOW_INDEX_VAL, EOD_CLOSING_INDEX_VAL
        col_map = {
            "EOD_TIMESTAMP":        "date",
            "EOD_OPEN_INDEX_VAL":   "vix_open",
            "EOD_HIGH_INDEX_VAL":   "vix_high",
            "EOD_LOW_INDEX_VAL":    "vix_low",
            "EOD_CLOSING_INDEX_VAL":"vix",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()
        keep = [c for c in ["vix", "vix_open", "vix_high", "vix_low"] if c in df.columns]
        df = df[keep].astype(float)
        return df
    except Exception as e:
        print(f"  NSE fetch error: {e}")
        return pd.DataFrame()


def fetch_vix_tv() -> pd.DataFrame:
    """Fetch VIX from TradingView via core.tv_fetcher (same session as live bot)."""
    try:
        from core.tv_fetcher import get_tv_fetcher
        tv = get_tv_fetcher()
        for sym in ["INDIAVIX", "INDIA VIX", "INDIA_VIX"]:
            raw = tv.get_ohlcv(sym, "NSE", interval_minutes="D", n_bars=5000)
            if not raw.empty:
                df = raw[["close"]].rename(columns={"close": "vix"}).astype(float).dropna()
                return df.sort_index()
        return pd.DataFrame()
    except Exception as e:
        print(f"  TradingView VIX fetch failed: {e}")
        return pd.DataFrame()


def download_full_history():
    """Download full VIX history — TradingView first, NSE as fallback."""
    print("\n=== Downloading India VIX History ===\n")

    # Try TradingView first (faster, more reliable)
    print("  Trying TradingView ...", flush=True)
    df = fetch_vix_tv()
    if not df.empty:
        print(f"  TradingView: {len(df)} rows | VIX range {df['vix'].min():.1f}-{df['vix'].max():.1f}")
        return df

    # Fallback: NSE API (year-by-year chunks)
    print("  TradingView failed — trying NSE API ...\n")
    session = get_nse_session()
    all_dfs = []

    start_year = 2007
    end_year   = date.today().year

    for year in range(start_year, end_year + 1):
        from_dt = date(year, 1, 1)
        to_dt   = min(date(year, 12, 31), date.today())
        from_str = from_dt.strftime("%d-%b-%Y")
        to_str   = to_dt.strftime("%d-%b-%Y")

        print(f"  Fetching {year}: {from_str} -> {to_str} ... ", end="", flush=True)
        df_chunk = fetch_vix_nse(from_str, to_str, session)

        if df_chunk.empty:
            print("no data")
        else:
            all_dfs.append(df_chunk)
            print(f"{len(df_chunk)} rows | VIX range {df_chunk['vix'].min():.1f}-{df_chunk['vix'].max():.1f}")

        time.sleep(0.8)

    if not all_dfs:
        print("ERROR: Could not fetch VIX from any source")
        sys.exit(1)

    df = pd.concat(all_dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def download_recent(days=30):
    """Download only recent VIX data and append to existing CSV."""
    session = get_nse_session()
    to_dt   = date.today()
    from_dt = to_dt - timedelta(days=days)
    from_str = from_dt.strftime("%d-%b-%Y")
    to_str   = to_dt.strftime("%d-%b-%Y")
    print(f"  Fetching recent VIX: {from_str} -> {to_str} ...")
    df = fetch_vix_nse(from_str, to_str, session)
    if df.empty:
        print("  NSE failed — trying TradingView...")
        df = fetch_vix_tv()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=0,
                        help="Days to fetch (0=full history)")
    args = parser.parse_args()

    # Load env
    env_path = Path(__file__).parent.parent / "config" / "settings.env"
    if env_path.exists():
        import os
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.days > 0:
        # Append mode
        new_df = download_recent(args.days)
        if new_df.empty:
            print("No new VIX data fetched")
            return
        if OUT_FILE.exists():
            existing = pd.read_csv(OUT_FILE)
            existing["date"] = pd.to_datetime(existing["date"], format="mixed")
            existing = existing.set_index("date").sort_index()
            last = existing.index.max()
            new_rows = new_df[new_df.index > last]
            if new_rows.empty:
                print("  VIX already up to date")
                return
            combined = pd.concat([existing, new_rows]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df
        combined.to_csv(OUT_FILE, date_format="%Y-%m-%d")
        print(f"  OK VIX updated: {len(combined):,} rows -> {OUT_FILE}")
    else:
        # Full history download
        df = download_full_history()
        df.to_csv(OUT_FILE, date_format="%Y-%m-%d")
        print(f"\nOK VIX history saved: {len(df):,} rows -> {OUT_FILE}")
        print(f"   Date range: {df.index[0].date()} -> {df.index[-1].date()}")
        print(f"   VIX range:  {df['vix'].min():.1f} - {df['vix'].max():.1f}")
        print(f"\nNow retrain the model:")
        print(f"   uv run scripts/train_model_v8.py")


if __name__ == "__main__":
    main()
