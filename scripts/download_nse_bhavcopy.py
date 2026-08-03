"""
download_nse_bhavcopy.py — Download FREE historical NSE F&O Bhavcopy data.

Handles BOTH NSE URL formats:
  - NEW (Jul 2024+): BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip
  - OLD (pre Jul 2024): fo{DDMMMYYYY}bhav.csv.zip
  And BOTH CSV column formats automatically.

DEFAULT: downloads last 2 years.
Customize via CLI:

    # 2 years (default):
    python scripts/download_nse_bhavcopy.py

    # Specific range:
    python scripts/download_nse_bhavcopy.py --start 2024-01-01 --end 2026-05-19

    # Just check what's already downloaded:
    python scripts/download_nse_bhavcopy.py --progress

OUTPUT:
    data/nse_bhavcopy/raw/        ← individual daily CSVs (cached)
    data/nse_bhavcopy/nifty_options_merged.csv   ← all NIFTY options merged
    data/nse_bhavcopy/nifty_pcr_daily.csv        ← daily PCR summary
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "data" / "nse_bhavcopy" / "raw"
OUT_DIR   = BASE_DIR / "data" / "nse_bhavcopy"

# ── default date range: last 2 years from today ──────────────────────────────
TODAY            = date.today()
DEFAULT_START    = date(TODAY.year - 2, TODAY.month, TODAY.day)
DEFAULT_END      = TODAY

# ── NSE URL patterns ─────────────────────────────────────────────────────────
URL_NEW = ("https://nsearchives.nseindia.com/content/fo/"
           "BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip")
URL_OLD = ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
           "{YYYY}/{MMM}/fo{DDMMMYYYY}bhav.csv.zip")

# Date when NSE switched format (approximate)
FORMAT_CUTOVER = date(2024, 7, 8)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

SLEEP_SEC = 1.0

# ── NSE holidays (2024-2026) — script also handles 404s gracefully ───────────
NSE_HOLIDAYS = {
    # 2024
    date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25),
    date(2024, 3, 29), date(2024, 4, 11), date(2024, 4, 17), date(2024, 5, 1),
    date(2024, 5, 20), date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
    date(2024, 10, 2), date(2024, 11, 15), date(2024, 12, 25),
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31), date(2025, 4, 10),
    date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1), date(2025, 6, 7),
    date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 22),
    date(2025, 11, 5), date(2025, 12, 25),
    # 2026
    date(2026, 1, 26), date(2026, 2, 26), date(2026, 3, 17), date(2026, 4, 2),
    date(2026, 4, 10), date(2026, 4, 14), date(2026, 5, 1),
}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def trading_days(start: date, end: date) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


# ── download one day (tries both URL formats) ────────────────────────────────

def download_bhavcopy(d: date, session: requests.Session) -> tuple[bytes | None, str]:
    """
    Download zipped Bhavcopy. Returns (zip_bytes, format) where
    format is 'new' or 'old' (used to pick the right CSV parser).
    """
    # Pick primary URL based on date, but try the other as fallback
    primary_new = d >= FORMAT_CUTOVER

    urls_to_try = []
    if primary_new:
        urls_to_try.append(("new", URL_NEW.format(date=d.strftime("%Y%m%d"))))
        urls_to_try.append(("old", URL_OLD.format(
            YYYY=d.year,
            MMM=d.strftime("%b").upper(),
            DDMMMYYYY=d.strftime("%d%b%Y").upper(),
        )))
    else:
        urls_to_try.append(("old", URL_OLD.format(
            YYYY=d.year,
            MMM=d.strftime("%b").upper(),
            DDMMMYYYY=d.strftime("%d%b%Y").upper(),
        )))
        urls_to_try.append(("new", URL_NEW.format(date=d.strftime("%Y%m%d"))))

    for fmt, url in urls_to_try:
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 500:
                return resp.content, fmt
        except requests.RequestException as e:
            log.debug(f"  {d} {fmt}: {e}")
            continue

    return None, ""


# ── parse NEW format (Jul 2024+) ──────────────────────────────────────────────

def parse_new_format(content: str, d: date) -> list[dict]:
    """NSE new format: TckrSymb=NIFTY, FinInstrmTp=IDO."""
    rows = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        sym      = (row.get("TckrSymb") or "").strip().upper()
        fin_type = (row.get("FinInstrmTp") or "").strip().upper()
        opt_type = (row.get("OptnTp") or "").strip().upper()
        if sym != "NIFTY" or fin_type != "IDO" or opt_type not in ("CE", "PE"):
            continue
        try:
            strike     = int(float(row.get("StrkPric") or 0))
            settle     = float(row.get("SttlmPric") or row.get("ClsPric") or 0)
            if strike <= 0 or settle <= 0:
                continue
            rows.append({
                "date":       d.isoformat(),
                "expiry":     (row.get("XpryDt") or "").strip(),
                "strike":     strike,
                "opt_type":   opt_type,
                "open":       float(row.get("OpnPric") or 0),
                "high":       float(row.get("HghPric") or 0),
                "low":        float(row.get("LwPric")  or 0),
                "close":      float(row.get("ClsPric") or 0),
                "settle":     settle,
                "volume":     int(float(row.get("TtlTradgVol") or 0)),
                "oi":         int(float(row.get("OpnIntrst") or 0)),
                "oi_change":  int(float(row.get("ChngInOpnIntrst") or 0)),
                "underlying": float(row.get("UndrlygPric") or 0),
            })
        except (ValueError, TypeError):
            continue
    return rows


# ── parse OLD format (pre Jul 2024) ──────────────────────────────────────────

def parse_old_format(content: str, d: date) -> list[dict]:
    """
    NSE old format columns:
      INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
      OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
      OPEN_INT, CHG_IN_OI, TIMESTAMP

    Filter: INSTRUMENT=OPTIDX, SYMBOL=NIFTY.
    """
    rows = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        instr = (row.get("INSTRUMENT") or "").strip().upper()
        sym   = (row.get("SYMBOL") or "").strip().upper()
        opt   = (row.get("OPTION_TYP") or "").strip().upper()
        if instr != "OPTIDX" or sym != "NIFTY" or opt not in ("CE", "PE"):
            continue
        try:
            strike = int(float(row.get("STRIKE_PR") or 0))
            settle = float(row.get("SETTLE_PR") or row.get("CLOSE") or 0)
            if strike <= 0 or settle <= 0:
                continue
            # Old format expiry: '15-MAY-2024' → convert to ISO
            exp_raw = (row.get("EXPIRY_DT") or "").strip()
            try:
                exp_iso = datetime.strptime(exp_raw, "%d-%b-%Y").strftime("%Y-%m-%d")
            except ValueError:
                exp_iso = exp_raw
            rows.append({
                "date":       d.isoformat(),
                "expiry":     exp_iso,
                "strike":     strike,
                "opt_type":   opt,
                "open":       float(row.get("OPEN")  or 0),
                "high":       float(row.get("HIGH")  or 0),
                "low":        float(row.get("LOW")   or 0),
                "close":      float(row.get("CLOSE") or 0),
                "settle":     settle,
                "volume":     int(float(row.get("CONTRACTS") or 0)),
                "oi":         int(float(row.get("OPEN_INT")  or 0)),
                "oi_change":  int(float(row.get("CHG_IN_OI") or 0)),
                "underlying": 0.0,   # not in old format — will be backfilled from NIFTY spot
            })
        except (ValueError, TypeError):
            continue
    return rows


def extract_nifty_options(zip_bytes: bytes, fmt: str, d: date) -> list[dict]:
    """Extract NIFTY option rows. Picks parser based on detected format."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return []
            with zf.open(csv_names[0]) as f:
                content = f.read().decode("utf-8", errors="replace")

        if fmt == "new":
            return parse_new_format(content, d)
        else:
            return parse_old_format(content, d)
    except Exception as e:
        log.warning(f"Failed to parse bhavcopy for {d}: {e}")
        return []


# ── main download loop ────────────────────────────────────────────────────────

def download_all(start_date: date, end_date: date):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    days = trading_days(start_date, end_date)
    log.info(f"Date range: {start_date} → {end_date}")
    log.info(f"Trading days: {len(days)}")
    log.info(f"Estimated time: ~{len(days) * SLEEP_SEC / 60:.0f} min")

    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
        time.sleep(1)
    except Exception:
        pass

    all_rows  = []
    success   = skipped = failed = 0

    import pandas as pd

    for i, d in enumerate(days):
        raw_path = RAW_DIR / f"bhavcopy_{d.isoformat()}.csv"

        if raw_path.exists() and raw_path.stat().st_size > 500:
            log.info(f"  [{i+1:4d}/{len(days)}] SKIP {d} (cached)")
            try:
                df = pd.read_csv(raw_path)
                all_rows.extend(df.to_dict("records"))
            except Exception:
                pass
            skipped += 1
            continue

        zip_bytes, fmt = download_bhavcopy(d, session)
        if not zip_bytes:
            log.warning(f"  [{i+1:4d}/{len(days)}] {d}: no data (holiday or NSE error)")
            failed += 1
            time.sleep(SLEEP_SEC)
            continue

        rows = extract_nifty_options(zip_bytes, fmt, d)
        if rows:
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            all_rows.extend(rows)
            log.info(f"  [{i+1:4d}/{len(days)}] {d} ({fmt}): {len(rows)} rows")
            success += 1
        else:
            log.warning(f"  [{i+1:4d}/{len(days)}] {d} ({fmt}): parsed 0 rows")
            failed += 1

        time.sleep(SLEEP_SEC)

    # ── merge ALL raw files (not just this run's date range) ─────────────────
    # Also folds in the EXISTING merged CSV (if present) so a local raw/
    # cache with gaps (e.g. only ever backfilled partially on this machine)
    # can never regress the merged file backward — previously-merged rows
    # for dates missing from raw/ are preserved; fresh raw rows win on
    # overlap (keep="last" after existing-then-raw concat order).
    log.info("\nMerging ALL cached raw files...")
    all_raw_rows = []
    raw_files = sorted(RAW_DIR.glob("bhavcopy_*.csv"))
    for rf in raw_files:
        try:
            df = pd.read_csv(rf)
            if not df.empty:
                all_raw_rows.append(df)
        except Exception as e:
            log.warning(f"  Could not read {rf.name}: {e}")

    merged_path = OUT_DIR / "nifty_options_merged.csv"
    existing_rows = []
    if merged_path.exists():
        try:
            existing = pd.read_csv(merged_path)
            if not existing.empty:
                existing_rows.append(existing)
                log.info(f"  Folding in existing merged file: {len(existing):,} rows")
        except Exception as e:
            log.warning(f"  Could not read existing {merged_path.name}: {e}")

    if all_raw_rows or existing_rows:
        merged = pd.concat(existing_rows + all_raw_rows, ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values(["date", "expiry", "strike", "opt_type"])
        merged = merged.drop_duplicates(
            subset=["date", "expiry", "strike", "opt_type"], keep="last"
        )

        merged.to_csv(merged_path, index=False)
        log.info(f"Merged: {len(merged):,} rows, {merged['date'].dt.date.nunique()} days → {merged_path}")

        # PCR
        pcr_df = _compute_pcr(merged)
        pcr_path = OUT_DIR / "nifty_pcr_daily.csv"
        pcr_df.to_csv(pcr_path, index=False)
        log.info(f"PCR daily: {len(pcr_df)} rows → {pcr_path}")

    # ── summary ──────────────────────────────────────────────────────────────
    total_raw = len(list(RAW_DIR.glob("bhavcopy_*.csv")))
    print("\n" + "=" * 60)
    print("NSE BHAVCOPY DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Date range : {start_date} to {end_date}")
    print(f"  Success    : {success} days (newly downloaded)")
    print(f"  Skipped    : {skipped} days (already cached)")
    print(f"  Failed     : {failed} days (holiday or NSE error)")
    print(f"  Total cached days : {total_raw}")
    print(f"  Total merged rows : {len(all_raw_rows) and sum(len(d) for d in all_raw_rows):,}")
    print(f"  Output dir : {OUT_DIR}")
    print()
    print("Next step:")
    print("  python scripts/build_training_dataset.py")


def _compute_pcr(merged) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for (d, exp), grp in merged.groupby(["date", "expiry"]):
        ce_oi = grp[grp["opt_type"] == "CE"]["oi"].sum()
        pe_oi = grp[grp["opt_type"] == "PE"]["oi"].sum()
        rows.append({
            "date":      d,
            "expiry":    exp,
            "ce_oi":     int(ce_oi),
            "pe_oi":     int(pe_oi),
            "pcr_oi":    round(pe_oi / ce_oi, 4) if ce_oi > 0 else None,
            "total_oi":  int(ce_oi + pe_oi),
        })
    return pd.DataFrame(rows)


def print_progress(start_date: date, end_date: date):
    days = trading_days(start_date, end_date)
    downloaded = sum(
        1 for d in days
        if (RAW_DIR / f"bhavcopy_{d.isoformat()}.csv").exists()
    )
    print(f"\nProgress: {downloaded}/{len(days)} trading days downloaded")
    print(f"Range checked: {start_date} → {end_date}")

    merged = OUT_DIR / "nifty_options_merged.csv"
    if merged.exists():
        import pandas as pd
        df = pd.read_csv(merged)
        print(f"Merged file: {len(df):,} rows, {df['date'].nunique()} unique dates")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START} — 2 yrs back)")
    parser.add_argument("--end",   type=str, default=None,
                        help=f"End date YYYY-MM-DD (default: today {DEFAULT_END})")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress only, no download")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else DEFAULT_START
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end   else DEFAULT_END

    if args.progress:
        print_progress(start, end)
    else:
        download_all(start, end)
