"""
oi_archiver.py — PART C: standalone 1-minute OpenAlgo OI archive daemon.

Independent of the bot's 5-min cycle (the in-client per-strike hook in
openalgo_client.py remains; this daemon adds the compact 1-min ATM series).

Schema per row:
  timestamp, spot, atm_strike, call_oi, put_oi, call_iv, put_iv, pcr

Storage: data/openalgo_oi/oi_YYYY-MM-DD.parquet when a parquet engine is
available, else .csv (same schema; convert later). Daily rollover automatic.
Recovery: append-mode + resume detection. Gap logging: any inter-snapshot
gap > 90s during market hours is recorded in oi_YYYY-MM-DD_gaps.csv.

Usage:
  python scripts/oi_archiver.py --once     # single snapshot (smoke test)
  python scripts/oi_archiver.py --daemon   # poll every 60s during 09:15-15:30
"""
from __future__ import annotations

import argparse
import atexit
import csv
import errno
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "data" / "openalgo_oi"
OUTDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("oi_archiver")

COLS = ["timestamp", "spot", "atm_strike", "call_oi", "put_oi",
        "call_iv", "put_iv", "pcr",
        "call_wall_1", "call_wall_2", "call_wall_3",
        "put_wall_1", "put_wall_2", "put_wall_3"]
# NOTE: ATM OI *change* and PCR *velocity* are intentionally NOT stored —
# they are derived losslessly offline by scripts/oi_features.py, so formula
# revisions never invalidate collected data. Walls ARE stored because the
# full chain is discarded after each snapshot (not derivable later).


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def _paths(day: datetime):
    ext = "parquet" if _parquet_ok() else "csv"
    return (OUTDIR / f"oi_{day:%Y-%m-%d}.{ext}",
            OUTDIR / f"oi_{day:%Y-%m-%d}_gaps.csv")


# ── Deployment-reliability primitives (2026-06-17) ──────────────────────────
_FREE_MIN_BYTES = 100 * 1024 * 1024     # 100 MB minimum free space
_LOCKFILE = OUTDIR / ".oi_archiver.lock"

# Reliability/observability state, read via get_health() for monitoring.
# Never consulted by any trading/signal-generation code path.
_health = {
    "last_success_ts": None, "last_error": None, "last_error_ts": None,
    "consecutive_failures": 0, "total_snapshots_written": 0,
}
_FAILURE_LOG_EVERY = 10   # rate-limit repeated-failure warnings to avoid log spam


def get_health() -> dict:
    """Read-only snapshot of this archiver's health (last success, error
    streak). Monitoring only — never read by any trading decision."""
    return dict(_health)


def _disk_low(path: Path = OUTDIR) -> bool:
    """True if free disk space under `path` is below the configured floor."""
    try:
        free = shutil.disk_usage(path).free
        return free < _FREE_MIN_BYTES
    except Exception as e:
        log.warning(f"disk_usage check failed for {path}: {e}")
        return False


def _atomic_parquet_write(df, path: Path) -> None:
    """Atomic parquet write: tmp file + os.replace (POSIX & Windows same-vol)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_parquet_or_quarantine(path: Path):
    """Read parquet; if corrupt, quarantine and return None (no raise)."""
    import pandas as pd
    try:
        return pd.read_parquet(path)
    except Exception as e:
        quarantine = path.with_name(
            f"{path.stem}_corrupt_{int(time.time())}.parquet")
        try:
            path.rename(quarantine)
            log.error(f"corrupt parquet quarantined -> {quarantine.name}: {e}")
        except Exception as e2:
            log.error(f"failed to quarantine corrupt parquet {path.name}: {e2}")
        return None


def _claim_lock() -> bool:
    """Single-writer PID lockfile. Reclaims stale locks. Returns False if a
    live process already owns the lock."""
    try:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        if _LOCKFILE.exists():
            try:
                pid = int(_LOCKFILE.read_text().strip() or "0")
                if pid == os.getpid():
                    # OUR OWN pid. Under Docker the app is always PID 1, and
                    # atexit (hence _release_lock) does NOT run on SIGKILL --
                    # so any unclean restart leaves "1" behind. The next
                    # container start then reads pid 1, confirms it is alive
                    # (it is: it is US), and refuses forever. Observed in
                    # production 2026-08: OI archive stopped writing Aug 5.
                    # Reclaim instead of deadlocking.
                    log.info(
                        f"OI archiver: lockfile holds our own pid ({pid}) "
                        f"— stale from a previous incarnation, reclaiming "
                        f"{_LOCKFILE.name}")
                elif pid > 0:
                    try:
                        os.kill(pid, 0)             # ProcessLookupError if dead
                    except PermissionError:
                        # Process EXISTS but we can't signal it (different uid)
                        # — treat as a live owner; do NOT reclaim the lock.
                        log.warning(
                            f"OI archiver: lockfile owned by live process "
                            f"pid {pid} (different uid); refusing")
                        return False
                    log.warning(
                        f"OI archiver: another instance is running "
                        f"(pid {pid}); refusing to start a second writer")
                    return False
            except (ProcessLookupError, ValueError):
                log.info(f"OI archiver: stale lockfile — reclaiming "
                         f"{_LOCKFILE.name}")
            except OSError:
                log.info(f"OI archiver: stale lockfile — reclaiming "
                         f"{_LOCKFILE.name}")
        _LOCKFILE.write_text(str(os.getpid()))
        atexit.register(_release_lock)
        return True
    except Exception as e:
        log.warning(f"OI archiver: lock claim failed (continuing): {e}")
        return True


def _release_lock() -> None:
    try:
        if _LOCKFILE.exists() and _LOCKFILE.read_text().strip() == str(os.getpid()):
            _LOCKFILE.unlink()
    except Exception:
        pass


_expiry_cache = {"date": None, "expiry": ""}


def _nearest_expiry(client) -> str:
    """Resolve nearest NIFTY expiry once per day (OpenAlgo requires it)."""
    today = datetime.now().date()
    if _expiry_cache["date"] == today and _expiry_cache["expiry"]:
        return _expiry_cache["expiry"]
    expiry = ""
    try:
        dates = client.get_expiry_dates("NIFTY", "NFO")
        if dates:
            expiry = dates[0]                  # nearest first
    except Exception as e:
        log.warning(f"expiry list fetch failed: {e}")
    if not expiry:
        try:
            from core.expiry_utils import get_expiry_date
            d = get_expiry_date()
            if d is not None:
                expiry = d.strftime("%d-%b-%y").upper()
        except Exception as e:
            log.warning(f"expiry fallback failed: {e}")
    if expiry:
        _expiry_cache.update(date=today, expiry=expiry)
        log.info(f"Using expiry: {expiry}")
    return expiry


def snapshot(client) -> dict | None:
    """One compact ATM-focused snapshot from the chain."""
    expiry = _nearest_expiry(client)
    if not expiry:
        log.warning("No expiry resolvable — snapshot skipped")
        return None
    chain = client.get_option_chain(exchange="NFO", expiry_date=expiry)
    if not chain or not chain.get("chain"):
        return None
    atm = chain.get("atm_strike", 0)
    row = next((r for r in chain["chain"]
                if int(r.get("strike", 0)) == int(atm)), None)
    if row is None:
        return None
    # Top-3 OI walls (strikes ranked by OI, per side) — captured here because
    # the full chain is discarded after this snapshot
    by_ce = sorted(chain["chain"], key=lambda r: -r["ce"].get("oi", 0))[:3]
    by_pe = sorted(chain["chain"], key=lambda r: -r["pe"].get("oi", 0))[:3]
    walls = {}
    for i in range(3):
        walls[f"call_wall_{i+1}"] = int(by_ce[i]["strike"]) if i < len(by_ce) else 0
        walls[f"put_wall_{i+1}"] = int(by_pe[i]["strike"]) if i < len(by_pe) else 0
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spot": round(float(chain.get("underlying_ltp", 0)), 2),
        "atm_strike": int(atm),
        "call_oi": int(row["ce"].get("oi", 0)),
        "put_oi": int(row["pe"].get("oi", 0)),
        "call_iv": float(row["ce"].get("iv", 0)),
        "put_iv": float(row["pe"].get("iv", 0)),
        "pcr": round(float(chain.get("pcr", 0)), 4),
        **walls,
    }


# ── Priority-3: full-chain / strike-level OI + real IV capture ──────────────
# The compact ATM snapshot above is insufficient for dealer-gamma (GEX), which
# needs per-strike OI AND real IV. OpenAlgo returns IV=0, so for GEX this must
# be fed a client whose get_option_chain exposes IV (Kotak). Stored long-format
# in a separate file so the existing ATM series is untouched.
CHAIN_COLS = ["timestamp", "atm_strike", "spot", "strike",
              "ce_oi", "pe_oi", "ce_iv", "pe_iv"]


def _leg(r: dict, side: str, field: str):
    """Read a chain leg field across both schemas:
       OpenAlgo nested {ce:{oi,iv}} OR Kotak flat {ce_oi, ce_iv}."""
    nested = r.get(side)
    if isinstance(nested, dict) and field in nested:
        return nested.get(field, 0) or 0
    return r.get(f"{side}_{field}", 0) or 0


def parse_chain_rows(chain: dict, n_strikes: int = 11) -> list[dict]:
    """Pure: chain dict -> per-strike rows for ATM +/- n_strikes. Unit-testable.
    Handles OpenAlgo (nested ce/pe, underlying_ltp) and Kotak (flat ce_*, spot)."""
    rows = chain.get("chain") or []
    if not rows:
        return []
    atm = int(chain.get("atm_strike", 0) or 0)
    spot = round(float(chain.get("underlying_ltp", chain.get("spot", 0)) or 0), 2)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # keep the n_strikes closest to ATM on each side
    sel = sorted(rows, key=lambda r: abs(int(r.get("strike", 0)) - atm))[:n_strikes * 2 + 1]
    out = []
    for r in sorted(sel, key=lambda r: int(r.get("strike", 0))):
        out.append({
            "timestamp": ts, "atm_strike": atm, "spot": spot,
            "strike": int(r.get("strike", 0)),
            "ce_oi": int(float(_leg(r, "ce", "oi"))),
            "pe_oi": int(float(_leg(r, "pe", "oi"))),
            "ce_iv": float(_leg(r, "ce", "iv")),
            "pe_iv": float(_leg(r, "pe", "iv")),
        })
    return out


def snapshot_full_chain(client, n_strikes: int = 11) -> list[dict]:
    """Full per-strike OI+IV snapshot (best-effort). Real IV requires a Kotak
    client whose get_option_chain returns iv; OpenAlgo yields ce_iv/pe_iv=0."""
    expiry = _nearest_expiry(client)
    if not expiry:
        return []
    chain = client.get_option_chain(exchange="NFO", expiry_date=expiry)
    if not chain or not chain.get("chain"):
        return []
    return parse_chain_rows(chain, n_strikes)


def append_chain(rows: list[dict]):
    if not rows:
        return
    day = datetime.now()
    ext = "parquet" if _parquet_ok() else "csv"
    path = OUTDIR / f"oichain_{day:%Y-%m-%d}.{ext}"
    if ext == "parquet":
        import pandas as pd
        df = pd.DataFrame(rows, columns=CHAIN_COLS)
        if path.exists():
            old = _read_parquet_or_quarantine(path)
            if old is not None:
                df = pd.concat([old, df], ignore_index=True)
        _atomic_parquet_write(df, path)
    else:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CHAIN_COLS)
            if new:
                w.writeheader()
            w.writerows(rows)


def append(row: dict):
    path, _ = _paths(datetime.now())
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.DataFrame([row])
        if path.exists():
            old = _read_parquet_or_quarantine(path)
            if old is not None:
                df = pd.concat([old, df], ignore_index=True)
        _atomic_parquet_write(df, path)
    else:
        # schema-evolution safety: if an existing file has a different header
        # (e.g. pre-walls schema), rotate it aside instead of corrupting it
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                head = fh.readline().strip()
            if head and head != ",".join(COLS):
                rot = path.with_name(path.stem + "_oldschema.csv")
                path.rename(rot)
                log.info(f"Schema changed — rotated {path.name} -> {rot.name}")
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def log_gap(prev_ts: str, now_ts: str, seconds: float, reason: str):
    _, gaps = _paths(datetime.now())
    new = not gaps.exists()
    with open(gaps, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["gap_start", "gap_end", "seconds", "reason"])
        w.writerow([prev_ts, now_ts, int(seconds), reason])
    log.warning(f"GAP {int(seconds)}s ({reason}): {prev_ts} -> {now_ts}")


def _resume_ts() -> str | None:
    """Recovery: find last archived timestamp for today (restart-safe)."""
    path, _ = _paths(datetime.now())
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            df = _read_parquet_or_quarantine(path)
            if df is None or df.empty:
                return None
            return str(df["timestamp"].iloc[-1])
        with open(path, encoding="utf-8") as fh:
            last = None
            for line in fh:
                last = line
        return last.split(",")[0] if last and not last.startswith("timestamp") else None
    except Exception:
        return None


def market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= hm <= (15 * 60 + 30)


def _daemon_loop(interval: int = 60, kotak_client=None):
    """Poll loop used by both the CLI daemon and the in-app thread.
    kotak_client (optional): if provided, the FULL-CHAIN snapshot uses it so
    IV is real (OpenAlgo returns IV=0). The compact ATM series stays on
    OpenAlgo. Passing the bot's existing client avoids a second Kotak login."""
    from core.openalgo_client import create_openalgo_client
    client = None
    prev = _resume_ts()
    while True:
        try:
            now = datetime.now()
            if market_open(now):
                if _disk_low():
                    log.error(
                        f"DISK LOW: free space under {OUTDIR} below "
                        f"{_FREE_MIN_BYTES//(1024*1024)} MB — skipping this tick")
                    time.sleep(interval)
                    continue
                if client is None:
                    client = create_openalgo_client()
                    if client is None:
                        log.warning("OI archiver: OpenAlgo unreachable — "
                                    "retrying in 5 min")
                        time.sleep(300)
                        continue
                row = snapshot(client)
                ts = now.strftime("%Y-%m-%d %H:%M:%S")
                if row:
                    append(row)
                    _health["last_success_ts"] = ts
                    _health["consecutive_failures"] = 0
                    _health["total_snapshots_written"] += 1
                    # Priority-3: also persist the full per-strike chain (GEX).
                    import os as _os
                    if _os.getenv("OI_FULLCHAIN_ENABLED", "true").lower() \
                            not in ("0", "false", "no"):
                        try:
                            # Kotak client (real IV) if supplied, else OpenAlgo (IV=0)
                            src = kotak_client if kotak_client is not None else client
                            append_chain(snapshot_full_chain(src))
                        except Exception as _e:
                            _health["consecutive_chain_failures"] = \
                                _health.get("consecutive_chain_failures", 0) + 1
                            if _health["consecutive_chain_failures"] % _FAILURE_LOG_EVERY == 1:
                                log.warning(
                                    f"full-chain snapshot failed "
                                    f"({_health['consecutive_chain_failures']} "
                                    f"consecutive): {_e}")
                            else:
                                log.debug(f"full-chain snapshot failed (ignored): {_e}")
                        else:
                            _health["consecutive_chain_failures"] = 0
                    if prev:
                        gap_s = (now - datetime.strptime(
                            prev, "%Y-%m-%d %H:%M:%S")).total_seconds()
                        if gap_s > 90:
                            log_gap(prev, ts, gap_s, "missed_polls_or_api_fail")
                    prev = ts
                else:
                    _health["consecutive_failures"] += 1
                    if _health["consecutive_failures"] % _FAILURE_LOG_EVERY == 1:
                        log.warning(
                            f"OI archiver: no snapshot for "
                            f"{_health['consecutive_failures']} consecutive ticks")
        except OSError as e:
            if getattr(e, "errno", None) == errno.ENOSPC:
                log.error(f"ENOSPC during write — disk full: {e}")
            else:
                log.error(f"OS error during write (possible ENOSPC): {e}")
        except Exception as e:                  # never die inside the bot
            _health["last_error"] = str(e)
            _health["last_error_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _health["consecutive_failures"] += 1
            if _health["consecutive_failures"] % _FAILURE_LOG_EVERY == 1:
                log.warning(
                    f"OI archiver tick failed ({_health['consecutive_failures']} "
                    f"consecutive): {e}")
            else:
                log.debug(f"OI archiver tick failed (ignored): {e}")
        time.sleep(interval)


def start_in_app(interval: int = 60, kotak_client=None) -> bool:
    """
    Launch the archiver as a daemon THREAD inside the trading bot.
    Call once at startup (main.py). Fail-safe: returns False instead of
    raising; the thread swallows all errors; disable with
    OI_ARCHIVER_ENABLED=false.
    kotak_client: pass the bot's existing Kotak client so the full-chain
    snapshot captures REAL IV without a second login.
    """
    import os as _os
    import threading
    if _os.getenv("OI_ARCHIVER_ENABLED", "true").lower() in ("0", "false", "no"):
        log.info("OI archiver: disabled via OI_ARCHIVER_ENABLED")
        return False
    if not _claim_lock():
        return False
    try:
        t = threading.Thread(target=_daemon_loop, args=(interval, kotak_client),
                             name="oi-archiver", daemon=True)
        t.start()
        log.info(f"OI archiver: in-app thread started ({interval}s cadence, "
                 f"-> {OUTDIR})")
        return True
    except Exception as e:
        log.warning(f"OI archiver: in-app start failed (non-fatal): {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / "settings.env")
    if not _claim_lock():
        sys.exit("OI archiver: another instance is running — aborting.")
    from core.openalgo_client import create_openalgo_client
    client = create_openalgo_client()
    if client is None:
        sys.exit("OpenAlgo not reachable — check OPENALGO_URL.")

    last = _resume_ts()
    if last:
        log.info(f"Resuming today's archive (last snapshot: {last})")
        gap_s = (datetime.now()
                 - datetime.strptime(last, "%Y-%m-%d %H:%M:%S")).total_seconds()
        if gap_s > 90 and market_open(datetime.now()):
            log_gap(last, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    gap_s, "restart_recovery")

    if args.once:
        row = snapshot(client)
        if row:
            append(row)
            log.info(f"Snapshot archived: {row}")
        else:
            log.error("Snapshot failed (empty chain)")
        return

    log.info(f"Daemon: polling every {args.interval}s during market hours")
    _daemon_loop(args.interval)


if __name__ == "__main__":
    main()
