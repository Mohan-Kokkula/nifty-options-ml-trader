"""
check_oi_archive.py — daily QA for the OpenAlgo OI snapshot archive.

Reports per day: snapshot count, strikes per snapshot, session coverage,
and any intra-session gaps > 10 minutes. Run any time; read-only.

Usage: python scripts/check_oi_archive.py [YYYY-MM-DD]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ARCHIVE = Path(__file__).resolve().parent.parent / "data" / "oi_archive"


def check(path: Path) -> None:
    df = pd.read_csv(path, parse_dates=["snapshot_ts"])
    snaps = df.groupby("snapshot_ts")
    n_snaps = snaps.ngroups
    strikes = snaps.size()
    ts = pd.Series(sorted(df["snapshot_ts"].unique()))
    gaps = ts.diff().dt.total_seconds().div(60).fillna(0)
    big = ts[gaps > 10]
    first, last = ts.iloc[0], ts.iloc[-1]
    print(f"{path.name}: snapshots={n_snaps} "
          f"({first:%H:%M}→{last:%H:%M}) | strikes/snap "
          f"min={strikes.min()} max={strikes.max()} | "
          f"zero-spot rows={int((df['spot'] <= 0).sum())} | "
          f"gaps>10min={len(big)}"
          + (f" at {[f'{t:%H:%M}' for t in big[:5]]}" if len(big) else ""))


def main():
    if not ARCHIVE.exists():
        print(f"No archive dir yet ({ARCHIVE}) — archiver activates on next "
              f"bot restart.")
        return
    files = sorted(ARCHIVE.glob("oi_*.csv"))
    if len(sys.argv) > 1:
        files = [f for f in files if sys.argv[1] in f.name]
    if not files:
        print("No archive files found.")
        return
    for f in files[-10:]:
        check(f)


if __name__ == "__main__":
    main()
