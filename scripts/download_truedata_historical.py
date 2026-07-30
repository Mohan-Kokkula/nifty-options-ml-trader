"""
download_truedata_historical.py — Download historical NIFTY option chain data
from Truedata's historical REST API for V10 model training.

WHY THIS SCRIPT
───────────────
Truedata Velocity (real-time) CANNOT give you data for expired option contracts.
For training V10 we need 5-min OHLCV + OI for every expiry from Jan-May 2026.
This script downloads it via Truedata's historical API, which DOES support
expired contracts (they have full archive going back years).

PREREQUISITES
─────────────
1. pip install truedata-ws requests
2. Add to config/settings.env:
       TRUEDATA_USERNAME=your_truedata_username
       TRUEDATA_PASSWORD=your_truedata_password

USAGE
─────
    # Download just current-expiry's data (quick test):
    python scripts/download_truedata_historical.py --mode test

    # Download all historical expired option data (full training set):
    python scripts/download_truedata_historical.py --mode full

    # Download specific expiry:
    python scripts/download_truedata_historical.py --expiry 260519

OUTPUT
──────
    data/truedata_archive/options/NIFTY26051925000CE.csv  (one file per contract)
    data/truedata_archive/index/NIFTY_I.csv              (continuous futures)
    data/truedata_archive/index/NIFTY.csv                (spot index)
    data/truedata_archive/index/INDIAVIX.csv             (VIX)

    After download, run:  python scripts/build_training_dataset.py
    to merge into V10 feature matrix.
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── config ───────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "truedata_archive"
OPT_DIR    = DATA_DIR / "options"
IDX_DIR    = DATA_DIR / "index"

# Strike range for download (wide — covers entire Jan-May 2026 NIFTY range)
SPOT_MIN   = 22000
SPOT_MAX   = 26000
STEP       = 50

# Truedata bar size for 5-minute OHLCV
BAR_SIZE   = 5   # minutes

# Rate limit: Truedata free trial allows ~1 req/sec
SLEEP_BETWEEN_CALLS = 1.2  # seconds

# All historical weekly expiries Jan–May 2026
HISTORICAL_EXPIRIES: list[dict] = [
    # expiry_code, start_date (Monday of that week), end_date (expiry Tuesday)
    {"code": "260106", "start": "2026-01-02", "end": "2026-01-06"},
    {"code": "260113", "start": "2026-01-07", "end": "2026-01-13"},
    {"code": "260120", "start": "2026-01-14", "end": "2026-01-20"},
    {"code": "260127", "start": "2026-01-21", "end": "2026-01-27"},
    {"code": "260203", "start": "2026-01-28", "end": "2026-02-03"},
    {"code": "260210", "start": "2026-02-04", "end": "2026-02-10"},
    {"code": "260217", "start": "2026-02-11", "end": "2026-02-17"},
    {"code": "260224", "start": "2026-02-18", "end": "2026-02-24"},
    {"code": "260303", "start": "2026-02-25", "end": "2026-03-03"},
    {"code": "260310", "start": "2026-03-04", "end": "2026-03-10"},
    {"code": "260317", "start": "2026-03-11", "end": "2026-03-17"},
    {"code": "260324", "start": "2026-03-18", "end": "2026-03-24"},
    {"code": "260331", "start": "2026-03-25", "end": "2026-03-31"},
    {"code": "260407", "start": "2026-04-01", "end": "2026-04-07"},
    {"code": "260414", "start": "2026-04-08", "end": "2026-04-14"},
    {"code": "260421", "start": "2026-04-15", "end": "2026-04-21"},
    {"code": "260428", "start": "2026-04-22", "end": "2026-04-28"},
    {"code": "260505", "start": "2026-04-29", "end": "2026-05-05"},
    {"code": "260512", "start": "2026-05-06", "end": "2026-05-12"},
    {"code": "260519", "start": "2026-05-13", "end": "2026-05-19"},
]

# Index/futures: pull full range Jan-May 2026
INDEX_SYMBOLS = ["NIFTY-I", "NIFTY", "INDIAVIX"]
INDEX_DATE_FROM = "2026-01-01"
INDEX_DATE_TO   = "2026-05-19"


# ── credentials ──────────────────────────────────────────────────────────────

def _load_env() -> dict:
    """Load settings.env into os.environ (simple key=value parser)."""
    env_path = BASE_DIR / "config" / "settings.env"
    cfg = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    return cfg


# ── Truedata client wrapper ───────────────────────────────────────────────────

class TruedataClient:
    """
    Thin wrapper around Truedata's historical API.
    Supports both:
      A) truedata-ws Python SDK  (pip install truedata-ws)
      B) Truedata REST API      (requests fallback)

    The SDK is preferred; REST is the fallback if SDK isn't installed.
    """

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._sdk = None
        self._session = None
        self._mode: str = "none"

    def connect(self) -> bool:
        """Try SDK first, fall back to REST."""
        if self._try_sdk():
            return True
        if self._try_rest():
            return True
        log.error("❌ Could not connect to Truedata. Check credentials.")
        return False

    def _try_sdk(self) -> bool:
        try:
            from truedata_ws.websocket.TD import TD  # type: ignore
            log.info("Using truedata-ws SDK...")
            td = TD(self.username, self.password, live_port=8082, historical_api=True)
            self._sdk = td
            self._mode = "sdk"
            log.info("✅ Truedata SDK connected")
            return True
        except ImportError:
            log.info("truedata-ws not installed — falling back to REST API")
            return False
        except Exception as e:
            log.warning(f"SDK connect failed: {e} — falling back to REST")
            return False

    def _try_rest(self) -> bool:
        try:
            import requests
            # Truedata REST historical endpoint
            # Docs: https://truedata.in/api-doc/historical
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})
            self._mode = "rest"
            log.info("✅ Using Truedata REST API")
            return True
        except ImportError:
            log.error("requests not installed: pip install requests")
            return False

    def get_historical(
        self,
        symbol: str,
        from_date: str,   # YYYY-MM-DD
        to_date: str,     # YYYY-MM-DD
        bar_size: int = 5,
    ) -> list[dict]:
        """
        Fetch historical OHLCV+OI bars.
        Returns list of dicts: {date, time, open, high, low, close, volume, oi}
        """
        if self._mode == "sdk":
            return self._get_sdk(symbol, from_date, to_date, bar_size)
        elif self._mode == "rest":
            return self._get_rest(symbol, from_date, to_date, bar_size)
        return []

    def _get_sdk(self, symbol, from_date, to_date, bar_size) -> list[dict]:
        try:
            # Convert YYYY-MM-DD to DD-MM-YYYY for SDK
            fd = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d-%m-%Y")
            td_date = datetime.strptime(to_date,   "%Y-%m-%d").strftime("%d-%m-%Y")

            df = self._sdk.get_historic_data(
                scrip=symbol,
                bar_size=f"{bar_size} mins",
                from_date=fd,
                to_date=td_date,
            )
            if df is None or len(df) == 0:
                return []

            # SDK returns a DataFrame
            rows = []
            for _, row in df.iterrows():
                dt = row.get("Date") or row.get("date") or row.get("Datetime")
                if dt is None:
                    continue
                if isinstance(dt, str):
                    dt = datetime.strptime(dt, "%d-%m-%Y %H:%M:%S")
                rows.append({
                    "date":   dt.strftime("%d-%m-%Y"),
                    "time":   dt.strftime("%H:%M:%S"),
                    "open":   float(row.get("Open",  row.get("open",  0))),
                    "high":   float(row.get("High",  row.get("high",  0))),
                    "low":    float(row.get("Low",   row.get("low",   0))),
                    "close":  float(row.get("Close", row.get("close", 0))),
                    "volume": int(row.get("Volume", row.get("volume", 0))),
                    "oi":     int(row.get("OI",     row.get("oi",     0))),
                })
            return rows
        except Exception as e:
            log.warning(f"SDK get_historic_data failed for {symbol}: {e}")
            return []

    def _get_rest(self, symbol, from_date, to_date, bar_size) -> list[dict]:
        """
        Truedata REST historical endpoint.
        Endpoint (as per Truedata v2 API docs):
          GET https://history.truedata.in/getbars
          ?symbol=NIFTY26051925000CE
          &from=2026-05-13
          &to=2026-05-19
          &duration=5 mins
          &datatype=json
          &user=USERNAME
          &password=PASSWORD
        """
        import requests

        url = "https://history.truedata.in/getbars"
        params = {
            "symbol":   symbol,
            "from":     from_date,
            "to":       to_date,
            "duration": f"{bar_size} mins",
            "datatype": "json",
            "user":     self.username,
            "password": self.password,
        }
        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Handle various response formats
                if isinstance(data, list):
                    bars = data
                elif isinstance(data, dict):
                    bars = data.get("data", data.get("Records", []))
                else:
                    return []

                rows = []
                for bar in bars:
                    if isinstance(bar, (list, tuple)) and len(bar) >= 6:
                        # Format: [datetime_str, open, high, low, close, volume, oi, ...]
                        dt_str = bar[0]
                        try:
                            dt = _parse_datetime(dt_str)
                        except Exception:
                            continue
                        rows.append({
                            "date":   dt.strftime("%d-%m-%Y"),
                            "time":   dt.strftime("%H:%M:%S"),
                            "open":   float(bar[1]),
                            "high":   float(bar[2]),
                            "low":    float(bar[3]),
                            "close":  float(bar[4]),
                            "volume": int(bar[5]) if len(bar) > 5 else 0,
                            "oi":     int(bar[6]) if len(bar) > 6 else 0,
                        })
                    elif isinstance(bar, dict):
                        dt_str = bar.get("time") or bar.get("datetime") or bar.get("Date")
                        try:
                            dt = _parse_datetime(str(dt_str))
                        except Exception:
                            continue
                        rows.append({
                            "date":   dt.strftime("%d-%m-%Y"),
                            "time":   dt.strftime("%H:%M:%S"),
                            "open":   float(bar.get("open",  bar.get("Open",  0))),
                            "high":   float(bar.get("high",  bar.get("High",  0))),
                            "low":    float(bar.get("low",   bar.get("Low",   0))),
                            "close":  float(bar.get("close", bar.get("Close", 0))),
                            "volume": int(bar.get("volume", bar.get("Volume", 0))),
                            "oi":     int(bar.get("oi",     bar.get("OI",     0))),
                        })
                return rows
            else:
                log.warning(f"REST API {resp.status_code} for {symbol}: {resp.text[:200]}")
                return []
        except Exception as e:
            log.warning(f"REST get failed for {symbol}: {e}")
            return []

    def disconnect(self):
        if self._sdk:
            try:
                self._sdk.disconnect()
            except Exception:
                pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_datetime(s: str) -> datetime:
    """Parse datetime string in various formats."""
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def _save_csv(rows: list[dict], path: Path):
    """Save rows to CSV in Truedata's native format for compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # No header (matches Truedata Velocity export format)
        for r in rows:
            writer.writerow([
                r["date"], r["time"],
                r["open"], r["high"], r["low"], r["close"],
                r["volume"], r["oi"], "0.00", "0.00",
            ])


def _already_downloaded(path: Path) -> bool:
    """Skip if file already exists and has data (resume support)."""
    return path.exists() and path.stat().st_size > 100


def _expiry_str_to_date(code: str) -> date:
    """Convert '260526' → date(2026, 5, 26)."""
    yy = int(code[0:2])
    mm = int(code[2:4])
    dd = int(code[4:6])
    return date(2000 + yy, mm, dd)


# ── download functions ────────────────────────────────────────────────────────

def download_index(client: TruedataClient):
    """Download NIFTY futures, spot, VIX — full Jan-May 2026 range."""
    log.info("=" * 60)
    log.info("Downloading index/futures data...")
    for sym in INDEX_SYMBOLS:
        out_path = IDX_DIR / f"{sym.replace('-', '_')}.csv"
        if _already_downloaded(out_path):
            log.info(f"  SKIP {sym} (already exists: {out_path.name})")
            continue
        log.info(f"  Fetching {sym} ({INDEX_DATE_FROM} → {INDEX_DATE_TO})...")
        rows = client.get_historical(sym, INDEX_DATE_FROM, INDEX_DATE_TO, BAR_SIZE)
        if rows:
            _save_csv(rows, out_path)
            log.info(f"  ✅ {sym}: {len(rows)} bars → {out_path.name}")
        else:
            log.warning(f"  ⚠️  {sym}: no data returned")
        time.sleep(SLEEP_BETWEEN_CALLS)


def download_options_for_expiry(
    client: TruedataClient,
    expiry_info: dict,
    spot_min: int,
    spot_max: int,
    step: int,
    only_atm: bool = False,
    atm_centre: int = 24000,
    atm_band: int = 20,
):
    """Download all option contracts for one weekly expiry."""
    code      = expiry_info["code"]
    from_date = expiry_info["start"]
    to_date   = expiry_info["end"]

    log.info(f"  Expiry {code} ({from_date} → {to_date})")

    if only_atm:
        s_min = atm_centre - atm_band * step
        s_max = atm_centre + atm_band * step
    else:
        s_min, s_max = spot_min, spot_max

    strikes = list(range(s_min, s_max + 1, step))
    total   = len(strikes) * 2  # CE + PE
    done    = 0
    skipped = 0

    for strike in strikes:
        for opt_type in ("CE", "PE"):
            symbol   = f"NIFTY{code}{strike}{opt_type}"
            out_path = OPT_DIR / code / f"{symbol}.csv"

            if _already_downloaded(out_path):
                skipped += 1
                done    += 1
                continue

            rows = client.get_historical(symbol, from_date, to_date, BAR_SIZE)
            done += 1

            if rows:
                _save_csv(rows, out_path)
                log.info(
                    f"    [{done:4d}/{total}] ✅ {symbol}: {len(rows)} bars"
                )
            else:
                # Empty file to mark "tried but no data" (illiquid strike)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("")
                log.debug(f"    [{done:4d}/{total}] — {symbol}: no data (illiquid)")

            time.sleep(SLEEP_BETWEEN_CALLS)

    log.info(f"  Done expiry {code}: {done-skipped} downloaded, {skipped} skipped")


def download_full(client: TruedataClient):
    """Download everything: index + all 20 expired expiries."""
    download_index(client)

    log.info("=" * 60)
    log.info(f"Downloading {len(HISTORICAL_EXPIRIES)} historical expiries...")
    log.info(f"Strike range: {SPOT_MIN}–{SPOT_MAX} step {STEP}")
    log.info(f"({(SPOT_MAX - SPOT_MIN) // STEP + 1} strikes × 2 types = "
             f"{(SPOT_MAX - SPOT_MIN) // STEP + 1 * 2} symbols/expiry)")
    log.info(f"Estimated time: ~{len(HISTORICAL_EXPIRIES) * (SPOT_MAX - SPOT_MIN) // STEP * 2 * SLEEP_BETWEEN_CALLS / 60:.0f} min")

    for exp_info in HISTORICAL_EXPIRIES:
        log.info("-" * 60)
        download_options_for_expiry(
            client, exp_info,
            spot_min=SPOT_MIN, spot_max=SPOT_MAX, step=STEP,
        )


def download_test(client: TruedataClient):
    """Quick test: download just the last expired expiry, ±5 strikes around 24000."""
    log.info("TEST MODE: downloading last expiry (260519), ±5 strikes around 24000")
    download_index(client)
    download_options_for_expiry(
        client,
        expiry_info=HISTORICAL_EXPIRIES[-1],  # 260519
        spot_min=23750,
        spot_max=24250,
        step=50,
    )


def download_expiry(client: TruedataClient, expiry_code: str):
    """Download a specific expiry (all strikes)."""
    match = next((e for e in HISTORICAL_EXPIRIES if e["code"] == expiry_code), None)
    if not match:
        log.error(f"Expiry {expiry_code} not found in HISTORICAL_EXPIRIES list")
        sys.exit(1)
    log.info(f"Single expiry mode: {expiry_code}")
    download_options_for_expiry(client, match, SPOT_MIN, SPOT_MAX, STEP)


# ── progress report ───────────────────────────────────────────────────────────

def print_progress():
    """Show how many files are already downloaded."""
    print("\n📊 Download Progress Report")
    print("=" * 50)

    # Index
    for sym in INDEX_SYMBOLS:
        p = IDX_DIR / f"{sym.replace('-', '_')}.csv"
        status = f"✅ {p.stat().st_size // 1024}KB" if p.exists() and p.stat().st_size > 100 else "❌ missing"
        print(f"  {sym:20s}: {status}")

    print()
    total_symbols = 0
    downloaded    = 0
    for exp_info in HISTORICAL_EXPIRIES:
        code    = exp_info["code"]
        exp_dir = OPT_DIR / code
        n_files = len(list(exp_dir.glob("*.csv"))) if exp_dir.exists() else 0
        n_with_data = sum(
            1 for f in (exp_dir.glob("*.csv") if exp_dir.exists() else [])
            if f.stat().st_size > 100
        )
        expected = (SPOT_MAX - SPOT_MIN) // STEP + 1
        expected *= 2  # CE + PE
        total_symbols += expected
        downloaded    += n_with_data
        print(f"  Expiry {code}: {n_with_data:4d}/{expected:4d} with data")

    print()
    pct = downloaded / total_symbols * 100 if total_symbols else 0
    print(f"  TOTAL: {downloaded}/{total_symbols} ({pct:.1f}%) symbols downloaded")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download historical Truedata option chain data for V10 training"
    )
    parser.add_argument(
        "--mode", choices=["test", "full", "expiry", "progress"],
        default="test",
        help=(
            "test=quick smoke test (1 expiry ±5 strikes), "
            "full=download everything, "
            "expiry=specific expiry (use --expiry), "
            "progress=show what's already downloaded"
        ),
    )
    parser.add_argument("--expiry", default="260519",
                        help="6-digit expiry code e.g. 260519 (used with --mode expiry)")
    args = parser.parse_args()

    if args.mode == "progress":
        print_progress()
        return

    # Load credentials
    cfg = _load_env()
    username = cfg.get("TRUEDATA_USERNAME") or os.environ.get("TRUEDATA_USERNAME")
    password = cfg.get("TRUEDATA_PASSWORD") or os.environ.get("TRUEDATA_PASSWORD")

    if not username or not password:
        print("❌ TRUEDATA credentials not found!")
        print()
        print("Add to config/settings.env:")
        print("  TRUEDATA_USERNAME=your_username")
        print("  TRUEDATA_PASSWORD=your_password")
        print()
        print("(These are the same credentials you use to log in to Truedata Velocity)")
        sys.exit(1)

    log.info(f"Truedata user: {username}")

    client = TruedataClient(username, password)
    if not client.connect():
        sys.exit(1)

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    IDX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "test":
            download_test(client)
        elif args.mode == "full":
            download_full(client)
        elif args.mode == "expiry":
            download_expiry(client, args.expiry)
    finally:
        client.disconnect()

    print()
    print_progress()
    print()
    print("Next step: python scripts/build_training_dataset.py")


if __name__ == "__main__":
    main()
