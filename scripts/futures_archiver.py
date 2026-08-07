"""
futures_archiver.py — Priority 2: 1-minute NIFTY futures + depth archive daemon
================================================================================
Built 2026-06-16. Forward-only collector for the futures-microstructure sources
(#3 real futures VWAP, #4 volume imbalance, #7 depth imbalance) that have NO
historical data. Mirrors oi_archiver.py: market-hours gate, append + resume,
gap logging, in-app daemon thread.

Schema per row:
  timestamp, fut_symbol, fut_ltp, fut_volume,
  bid_px_1..5, bid_qty_1..5, ask_px_1..5, ask_qty_1..5,
  obi, best_bid, best_ask, microprice, basis_live

Source: Kotak get_quote (LTP+volume) + get_market_depth (Level-5) on the active
near-month NIFTY future. REAL data only — if the broker is unreachable the
daemon idles; it never fabricates rows. Nothing here is wired into the model;
features are derived/validated offline once >=8-12 weeks have accumulated.

Usage:
  python scripts/futures_archiver.py --once
  python scripts/futures_archiver.py --daemon
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

OUTDIR = ROOT / "data" / "futures_archive"
OUTDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("futures_archiver")

_LVL = 5
COLS = (["timestamp", "fut_symbol", "fut_ltp", "fut_volume"]
        + [f"bid_px_{i}" for i in range(1, _LVL + 1)]
        + [f"bid_qty_{i}" for i in range(1, _LVL + 1)]
        + [f"ask_px_{i}" for i in range(1, _LVL + 1)]
        + [f"ask_qty_{i}" for i in range(1, _LVL + 1)]
        + ["obi", "best_bid", "best_ask", "microprice", "basis_live"])


# ──────────────────────────────────────────── pure, unit-testable helpers ──
def normalize_depth(raw: dict) -> dict:
    """Coerce broker depth payloads to {'bids':[(px,qty)], 'asks':[(px,qty)]}.
    Handles list-style ({'bids':[{price,quantity}]}) and Kotak flat bp1/bq1."""
    if not raw:
        return {"bids": [], "asks": []}
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if isinstance(data, list) and data:
        data = data[0]

    def _from_list(key_opts):
        for k in key_opts:
            seq = data.get(k) if isinstance(data, dict) else None
            if isinstance(seq, list) and seq:
                out = []
                for lvl in seq:
                    px = float(lvl.get("price", lvl.get("px", 0)) or 0)
                    qty = float(lvl.get("quantity", lvl.get("qty", 0)) or 0)
                    out.append((px, qty))
                return out
        return []

    bids = _from_list(["bids", "buy", "bidValues"])
    asks = _from_list(["asks", "sell", "askValues"])
    if not bids and isinstance(data, dict):                 # Kotak flat bp1/bq1
        bids = [(float(data.get(f"bp{i}", 0) or 0), float(data.get(f"bq{i}", 0) or 0))
                for i in range(1, _LVL + 1) if data.get(f"bp{i}") is not None]
        asks = [(float(data.get(f"sp{i}", 0) or 0), float(data.get(f"sq{i}", 0) or 0))
                for i in range(1, _LVL + 1) if data.get(f"sp{i}") is not None]
    return {"bids": bids[:_LVL], "asks": asks[:_LVL]}


def compute_depth_features(depth: dict) -> dict:
    """Pure: normalized depth -> obi, best bid/ask, microprice. Unit-testable."""
    bids, asks = depth.get("bids", []), depth.get("asks", [])
    bid_qty = sum(q for _, q in bids)
    ask_qty = sum(q for _, q in asks)
    tot = bid_qty + ask_qty
    obi = (bid_qty - ask_qty) / tot if tot > 0 else 0.0
    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    bq1 = bids[0][1] if bids else 0.0
    aq1 = asks[0][1] if asks else 0.0
    if best_bid > 0 and best_ask > 0 and (bq1 + aq1) > 0:
        microprice = (best_bid * aq1 + best_ask * bq1) / (bq1 + aq1)
    else:
        microprice = best_bid or best_ask
    return {"obi": round(obi, 6), "best_bid": best_bid, "best_ask": best_ask,
            "microprice": round(microprice, 2)}


# ──────────────────────────────────────────────────────── live collection ──
_contract = {"date": None, "symbol": ""}


def resolve_active_future(client) -> str:
    """Resolve the active near-month NIFTY future symbol (cached per day)."""
    today = datetime.now().date()
    if _contract["date"] == today and _contract["symbol"]:
        return _contract["symbol"]
    sym = ""
    try:
        res = client._call_with_retry(
            client._neo.search_scrip, exchange_segment="nse_fo", symbol="nifty")
        futs = [r for r in (res or [])
                if str(r.get("pInstType", "")).upper() in ("FUTIDX", "IF")
                or "FUT" in str(r.get("pTrdSymbol", "")).upper()]
        if futs:
            futs.sort(key=lambda r: str(r.get("pExpiryDate", r.get("expiry", ""))))
            sym = str(futs[0].get("pTrdSymbol", futs[0].get("symbol", "")))
    except Exception as e:
        log.debug(f"future symbol resolve failed: {e}")
    if sym:
        _contract.update(date=today, symbol=sym)
        log.info(f"Active future: {sym}")
    return sym


# Kotak quote payloads vary by endpoint/version: the price appears as
# "ltp", "last_price", "lp", "lastPrice" or "ltP", and the body is
# sometimes {"data": {...}} and sometimes {"data": [{...}]}. Mirrors the
# tolerance normalize_depth() already has.
_LTP_KEYS = ("ltp", "last_price", "lp", "lastPrice", "ltP", "last_traded_price")
_VOL_KEYS = ("volume", "v", "vol", "tradedQty", "volume_traded", "vtt", "vTrdQty")
_shape_logged = False


def _unwrap(payload):
    """Coerce a broker payload to a single dict of fields."""
    d = payload
    if isinstance(d, dict) and "data" in d:
        d = d["data"]
    if isinstance(d, list):
        d = d[0] if d else {}
    return d if isinstance(d, dict) else {}


def _first_num(d: dict, keys) -> float:
    for k in keys:
        if k in d:
            try:
                v = float(d[k] or 0)
                if v:
                    return v
            except (TypeError, ValueError):
                continue
    return 0.0


def _extract_ltp_volume(q) -> tuple:
    """(ltp, volume) from a broker quote payload, tolerant of shape/keys."""
    d = _unwrap(q)
    return _first_num(d, _LTP_KEYS), _first_num(d, _VOL_KEYS)


def _log_quote_shape(q) -> None:
    """Log the observed payload keys ONCE so an unknown schema is
    diagnosable without shipping another build to find out."""
    global _shape_logged
    if _shape_logged:
        return
    _shape_logged = True
    try:
        d = _unwrap(q)
        log.warning(
            f"futures archiver: quote had no usable LTP. "
            f"outer_type={type(q).__name__} "
            f"inner_keys={sorted(d.keys())[:25] if d else 'EMPTY'} "
            f"sample={ {k: d[k] for k in list(d)[:6]} if d else q}")
    except Exception as e:
        log.warning(f"futures archiver: quote shape log failed: {e}")


def snapshot(client) -> dict | None:
    sym = resolve_active_future(client)
    if not sym:
        return None
    q = client.get_quote(sym, "NFO")
    ltp, vol = _extract_ltp_volume(q)
    if ltp <= 0:
        # 2026-08-07: this guard silently discarded EVERY tick in production
        # ("no snapshot for N consecutive ticks") even though the symbol
        # resolved fine. The old code read only data["ltp"] on a dict, while
        # normalize_depth() right below already handled list-shaped payloads
        # and alternate key spellings -- the quote path never got the same
        # treatment. _extract_ltp_volume() now mirrors it. If it STILL fails,
        # log the payload shape (rate-limited) so the next failure is
        # diagnosable instead of silent.
        _log_quote_shape(q)
        return None
    depth = normalize_depth(client.get_market_depth(sym, "NFO"))
    feats = compute_depth_features(depth)
    try:
        spot = float(client._get_spot_price() or 0)
    except Exception:
        spot = 0.0
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "fut_symbol": sym, "fut_ltp": round(ltp, 2), "fut_volume": vol,
           "obi": feats["obi"], "best_bid": feats["best_bid"],
           "best_ask": feats["best_ask"], "microprice": feats["microprice"],
           "basis_live": round(ltp - spot, 2) if spot > 0 else 0.0}
    for i in range(_LVL):
        b = depth["bids"][i] if i < len(depth["bids"]) else (0.0, 0.0)
        a = depth["asks"][i] if i < len(depth["asks"]) else (0.0, 0.0)
        row[f"bid_px_{i+1}"], row[f"bid_qty_{i+1}"] = b
        row[f"ask_px_{i+1}"], row[f"ask_qty_{i+1}"] = a
    return row


def _paths(day: datetime):
    """Daily data path + gap-log path (storage rotation = daily rollover)."""
    ext = "parquet" if _parquet_ok() else "csv"
    return (OUTDIR / f"fut_{day:%Y-%m-%d}.{ext}",
            OUTDIR / f"fut_{day:%Y-%m-%d}_gaps.csv")


# ── Deployment-reliability primitives (2026-06-17) ──────────────────────────
_FREE_MIN_BYTES = 100 * 1024 * 1024     # 100 MB minimum free space
_LOCKFILE = OUTDIR / ".archiver.lock"

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
        return False                        # don't block writes on a probe failure


def _atomic_parquet_write(df, path: Path) -> None:
    """Atomic parquet write: tmp file + os.replace (POSIX & Windows same-vol)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_parquet_or_quarantine(path: Path):
    """Read parquet; if corrupt, rename to *_corrupt.parquet and return None.
    The caller starts a fresh DataFrame in that case. Never raises."""
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
    """Single-writer PID lockfile. Reclaims stale locks (dead PIDs). Returns
    True on success, False if another live process owns the lock."""
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
                    # production 2026-08: both archivers dead since Aug 5,
                    # futures_archive empty. Reclaim instead of deadlocking.
                    log.info(
                        f"futures archiver: lockfile holds our own pid "
                        f"({pid}) — stale from a previous incarnation, "
                        f"reclaiming {_LOCKFILE.name}")
                elif pid > 0:
                    try:
                        os.kill(pid, 0)             # ProcessLookupError if dead
                    except PermissionError:
                        # Process EXISTS but we can't signal it (different uid)
                        # — treat as a live owner; do NOT reclaim the lock.
                        log.warning(
                            f"futures archiver: lockfile owned by live "
                            f"process pid {pid} (different uid); refusing")
                        return False
                    log.warning(
                        f"futures archiver: another instance is running "
                        f"(pid {pid}); refusing to start a second writer")
                    return False
            except (ProcessLookupError, ValueError):
                log.info(f"futures archiver: stale lockfile (pid unreachable) "
                         f"— reclaiming {_LOCKFILE.name}")
            except OSError:
                log.info(f"futures archiver: stale lockfile — reclaiming "
                         f"{_LOCKFILE.name}")
        _LOCKFILE.write_text(str(os.getpid()))
        atexit.register(_release_lock)
        return True
    except Exception as e:
        log.warning(f"futures archiver: lock claim failed (continuing): {e}")
        return True                         # don't refuse to run on lock failure


def _release_lock() -> None:
    try:
        if _LOCKFILE.exists() and _LOCKFILE.read_text().strip() == str(os.getpid()):
            _LOCKFILE.unlink()
    except Exception:
        pass


def append(row: dict):
    path, _ = _paths(datetime.now())
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.DataFrame([row], columns=COLS)
        if path.exists():
            old = _read_parquet_or_quarantine(path)
            if old is not None:
                df = pd.concat([old, df], ignore_index=True)
        _atomic_parquet_write(df, path)
    else:
        # schema-evolution safety: if an existing file has a different header,
        # rotate it aside instead of corrupting it (mirrors oi_archiver).
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
            w.writerow({k: row.get(k, "") for k in COLS})


def log_gap(prev_ts: str, now_ts: str, seconds: float, reason: str):
    """Record an inter-snapshot gap > threshold to fut_<day>_gaps.csv."""
    _, gaps = _paths(datetime.now())
    new = not gaps.exists()
    with open(gaps, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["gap_start", "gap_end", "seconds", "reason"])
        w.writerow([prev_ts, now_ts, int(seconds), reason])
    log.warning(f"GAP {int(seconds)}s ({reason}): {prev_ts} -> {now_ts}")


def _resume_ts() -> str | None:
    """Restart safety: last archived timestamp for today (None if no file).
    A corrupt parquet is quarantined and treated as no-resume."""
    path, _ = _paths(datetime.now())
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            df = _read_parquet_or_quarantine(path)
            if df is None or df.empty:
                return None
            return str(df["timestamp"].iloc[-1])
        last = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                last = line
        return last.split(",")[0] if last and not last.startswith("timestamp") else None
    except Exception:
        return None


def _validate_storage() -> bool:
    """Storage validation: OUTDIR exists, is writable, serializer available."""
    try:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        probe = OUTDIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        log.info(f"storage OK ({'parquet' if _parquet_ok() else 'csv'}) -> {OUTDIR}")
        return True
    except Exception as e:
        log.error(f"storage validation FAILED for {OUTDIR}: {e}")
        return False


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


def market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= hm <= (15 * 60 + 30)


def _make_kotak_client():
    """Build a Kotak client from settings.env (same args as production main.py)."""
    from security import RateLimiter
    from core.kotak_neo_client import KotakNeoClient
    return KotakNeoClient(
        consumer_key=os.getenv("KOTAK_CONSUMER_KEY", ""),
        rate_limiter=RateLimiter(),
        environment=os.getenv("KOTAK_ENVIRONMENT", "prod"),
        neo_fin_key=os.getenv("KOTAK_NEO_FIN_KEY", "") or None,
        mobile=os.getenv("KOTAK_MOBILE", ""),
        password=os.getenv("KOTAK_PASSWORD", ""),
        totp_secret=os.getenv("KOTAK_TOTP_SECRET", ""),
        ucc=os.getenv("KOTAK_UCC", ""),
        mpin=os.getenv("KOTAK_MPIN", ""),
    )


def _daemon_loop(interval: int = 60, kotak_client=None):
    # Reuse the bot's Kotak client when run in-app (avoids a 2nd login that
    # would conflict with Kotak's single-session limit). Standalone CLI builds
    # its own via _make_kotak_client().
    client = kotak_client
    prev = _resume_ts()                       # restart safety: resume today's file
    if prev and market_open(datetime.now()):
        try:
            gap_s = (datetime.now()
                     - datetime.strptime(prev, "%Y-%m-%d %H:%M:%S")).total_seconds()
            if gap_s > 90:
                log_gap(prev, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        gap_s, "restart_recovery")
        except Exception:
            pass
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
                    client = _make_kotak_client()
                row = snapshot(client)
                if row:
                    append(row)
                    ts = now.strftime("%Y-%m-%d %H:%M:%S")
                    _health["last_success_ts"] = ts
                    _health["consecutive_failures"] = 0
                    _health["total_snapshots_written"] += 1
                    if prev:
                        try:
                            gap_s = (now - datetime.strptime(
                                prev, "%Y-%m-%d %H:%M:%S")).total_seconds()
                            if gap_s > 90:
                                log_gap(prev, ts, gap_s, "missed_polls_or_api_fail")
                        except Exception:
                            pass
                    prev = ts
                else:
                    # snapshot() returned None (no quote/depth) -- this is the
                    # tick-level equivalent of a failure for health purposes,
                    # even though it isn't an exception.
                    _health["consecutive_failures"] += 1
                    if _health["consecutive_failures"] % _FAILURE_LOG_EVERY == 1:
                        log.warning(
                            f"futures archiver: no snapshot for "
                            f"{_health['consecutive_failures']} consecutive ticks "
                            f"(no quote/depth returned)")
        except OSError as e:
            if getattr(e, "errno", None) == errno.ENOSPC:
                log.error(f"ENOSPC during write — disk full: {e}")
            else:
                log.error(f"OS error during write (possible ENOSPC): {e}")
        except Exception as e:
            _health["last_error"] = str(e)
            _health["last_error_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _health["consecutive_failures"] += 1
            if _health["consecutive_failures"] % _FAILURE_LOG_EVERY == 1:
                log.warning(
                    f"futures archiver tick failed ({_health['consecutive_failures']} "
                    f"consecutive): {e}")
            else:
                log.debug(f"futures archiver tick failed (ignored): {e}")
        time.sleep(interval)


def start_in_app(interval: int = 60, kotak_client=None) -> bool:
    import threading
    if os.getenv("FUTURES_ARCHIVER_ENABLED", "true").lower() in ("0", "false", "no"):
        log.info("futures archiver: disabled via FUTURES_ARCHIVER_ENABLED")
        return False
    if not _validate_storage():
        log.warning("futures archiver: storage validation failed — not starting")
        return False
    if not _claim_lock():
        return False
    try:
        t = threading.Thread(target=_daemon_loop, args=(interval, kotak_client),
                             name="futures-archiver", daemon=True)
        t.start()
        log.info(f"futures archiver: in-app thread started ({interval}s -> {OUTDIR})")
        return True
    except Exception as e:
        log.warning(f"futures archiver: in-app start failed (non-fatal): {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / "settings.env")
    if not _validate_storage():
        sys.exit("futures archiver: storage validation failed — aborting.")
    if not _claim_lock():
        sys.exit("futures archiver: another instance is running — aborting.")
    client = _make_kotak_client()
    if args.once:
        row = snapshot(client)
        if row:
            append(row); log.info(f"Snapshot archived: {row['fut_symbol']} "
                                  f"ltp={row['fut_ltp']} obi={row['obi']}")
        else:
            log.error("Snapshot failed (no quote/depth)")
        return
    log.info(f"Daemon: polling every {args.interval}s during market hours")
    _daemon_loop(args.interval)


if __name__ == "__main__":
    main()
