"""
archive_health_check.py — Phase 3: daily health report for the forward archives.
=================================================================================
Built 2026-06-16. Read-only audit of the futures + OI archives. Reports row
counts, timestamp span, duplicate timestamps, missing intra-session intervals,
stale-data warnings, IV completeness, strike coverage, and disk usage with a
12-week projection. Uses ACTUAL files only — never estimates or fabricates.

Usage:
  python scripts/archive_health_check.py            # latest day each stream
  python scripts/archive_health_check.py 2026-06-17 # a specific day
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FUT_DIR = ROOT / "data" / "futures_archive"
OI_DIR = ROOT / "data" / "openalgo_oi"

SESSION_START = (9, 15)
SESSION_END = (15, 30)
EXPECTED_ROWS = (15 * 60 + 30) - (9 * 60 + 15)   # 375 1-min snapshots/session
STALE_SECS = 180                                  # >3 min since last row = stale


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _latest(d: Path, stem: str) -> Path | None:
    files = sorted(list(d.glob(f"{stem}_2*.parquet")) + list(d.glob(f"{stem}_2*.csv")))
    files = [f for f in files if "gaps" not in f.name and "oldschema" not in f.name]
    return files[-1] if files else None


def _ts_diag(df: pd.DataFrame) -> dict:
    if df.empty or "timestamp" not in df.columns:
        return {"rows": len(df), "first": None, "last": None, "dups": 0,
                "missing_min": None, "stale": None}
    ts = df["timestamp"].dropna()
    uniq = ts.drop_duplicates()
    first, last = ts.min(), ts.max()
    # missing intra-session minutes (gaps in the per-minute grid within span)
    grid = pd.date_range(first.floor("min"), last.floor("min"), freq="1min")
    present = set(ts.dt.floor("min"))
    in_session = [g for g in grid if SESSION_START <= (g.hour, g.minute) <= SESSION_END]
    missing = sum(1 for g in in_session if g not in present)
    # stale: only meaningful for today's file during/after session
    stale = None
    if last.date() == datetime.now().date():
        stale = (datetime.now() - last).total_seconds() > STALE_SECS
    return {"rows": len(df), "unique_ts": len(uniq), "first": first, "last": last,
            "dups": int(len(ts) - len(uniq)), "missing_min": missing, "stale": stale}


def _disk(d: Path) -> int:
    return sum(f.stat().st_size for f in d.glob("*") if f.is_file()) if d.exists() else 0


def _fmt(v):
    return "—" if v is None else (v.strftime("%Y-%m-%d %H:%M:%S")
                                  if isinstance(v, (pd.Timestamp, datetime)) else v)


def report(day: str | None):
    lines = []
    lines.append("=" * 68)
    lines.append(f"  ARCHIVE HEALTH REPORT — {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 68)

    # ---- Futures archive ----
    fp = (FUT_DIR / f"fut_{day}.csv") if day else _latest(FUT_DIR, "fut")
    if fp and not fp.exists():
        fp = (FUT_DIR / f"fut_{day}.parquet") if day else fp
    fdf = _read(fp) if fp else pd.DataFrame()
    fd = _ts_diag(fdf)
    lines.append("\n## FUTURES ARCHIVE  (data/futures_archive/)")
    lines.append(f"  file:           {fp.name if fp else 'NONE'}")
    lines.append(f"  rows captured:  {fd['rows']}")
    lines.append(f"  first ts:       {_fmt(fd['first'])}")
    lines.append(f"  latest ts:      {_fmt(fd['last'])}")
    lines.append(f"  duplicate ts:   {fd['dups']}")
    lines.append(f"  missing min:    {fd['missing_min']} (of {EXPECTED_ROWS} session min)")
    lines.append(f"  stale (>3m):    {fd['stale']}")
    if not fdf.empty:
        vol_ok = (fdf.get("fut_volume", pd.Series([0])).fillna(0) > 0).mean() * 100
        depth_ok = (fdf.get("bid_qty_1", pd.Series([0])).fillna(0) > 0).mean() * 100
        ltp_ok = (fdf.get("fut_ltp", pd.Series([0])).fillna(0) > 0).mean() * 100
        lines.append(f"  LTP non-zero:   {ltp_ok:.0f}%   volume non-zero: {vol_ok:.0f}%"
                     f"   depth(bid_qty_1) non-zero: {depth_ok:.0f}%")

    # ---- OI full-chain archive ----
    cp = (OI_DIR / f"oichain_{day}.csv") if day else _latest(OI_DIR, "oichain")
    cdf = _read(cp) if cp else pd.DataFrame()
    cd = _ts_diag(cdf)
    lines.append("\n## OI FULL-CHAIN ARCHIVE  (data/openalgo_oi/oichain_*)")
    lines.append(f"  file:           {cp.name if cp else 'NONE'}")
    lines.append(f"  rows captured:  {cd['rows']}")
    lines.append(f"  first ts:       {_fmt(cd['first'])}")
    lines.append(f"  latest ts:      {_fmt(cd['last'])}")
    lines.append(f"  duplicate ts:   {cd['dups']}")
    if not cdf.empty:
        strikes = cdf.groupby("timestamp")["strike"].nunique().median() if "strike" in cdf else 0
        iv_cols = [c for c in ("ce_iv", "pe_iv") if c in cdf.columns]
        iv_ok = (cdf[iv_cols] > 0).any(axis=1).mean() * 100 if iv_cols else 0
        snaps = cdf["timestamp"].dt.floor("min").nunique()
        missing = sum(1 for g in pd.date_range(cd["first"].floor("min"),
                      cd["last"].floor("min"), freq="1min")
                      if SESSION_START <= (g.hour, g.minute) <= SESSION_END
                      and g not in set(cdf["timestamp"].dt.floor("min")))
        lines.append(f"  snapshots:      {snaps} (of {EXPECTED_ROWS} session min)")
        lines.append(f"  missing min:    {missing}")
        lines.append(f"  strikes/snap:   {strikes:.0f} (median)")
        lines.append(f"  IV completeness: {iv_ok:.0f}% rows with non-zero CE/PE IV")
        if iv_ok == 0:
            lines.append("  [WARN]  IV ALL ZERO — full-chain is on OpenAlgo; wire a "
                         "Kotak client (start_in_app(kotak_client=...)) for real IV.")

    # ---- Disk usage + projection ----
    fut_b, oi_b = _disk(FUT_DIR), _disk(OI_DIR)
    fut_days = max(len(list(FUT_DIR.glob("fut_2*"))), 1) if FUT_DIR.exists() else 1
    oi_days = max(len(list(OI_DIR.glob("oichain_2*"))), 1) if OI_DIR.exists() else 1
    fut_per_day = fut_b / fut_days
    oi_per_day = oi_b / oi_days
    proj = (fut_per_day + oi_per_day) * 60   # ~12 weeks of trading days
    lines.append("\n## DISK USAGE")
    lines.append(f"  futures archive: {fut_b/1e6:.2f} MB  (~{fut_per_day/1e3:.0f} KB/day)")
    lines.append(f"  oi-chain archive:{oi_b/1e6:.2f} MB  (~{oi_per_day/1e3:.0f} KB/day)")
    lines.append(f"  projected 12-wk (60 sessions): {proj/1e6:.1f} MB")

    # ---- Verdict ----
    lines.append("\n## STATUS")
    if fd["rows"] == 0 and cd["rows"] == 0:
        lines.append("  [FAIL] NO DATA YET — archivers not deployed/run, or no session "
                     "has completed. Forward-collection clock has NOT started.")
    else:
        lines.append(f"  futures rows={fd['rows']}  oichain rows={cd['rows']}  "
                     f"(collection in progress)")
    lines.append("=" * 68)
    print("\n".join(lines))


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else None
    report(day)


if __name__ == "__main__":
    main()
