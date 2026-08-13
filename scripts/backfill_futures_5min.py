"""
backfill_futures_5min.py — build data/nifty_fut_5min.csv (REAL volume).

WHY THIS FILE MATTERS
    train_model_v9.py already expects it:

        CSV_FUT = "data/nifty_fut_5min.csv"   # Fix 1: real futures for VWAP/CMF

    features_futures() computes genuine VWAP = Sum(TP x Vol)/Sum(Vol) and
    CMF = Sum(MFV,20)/Sum(Volume,20) from it, and add_intraday_context()
    switches off the spot proxy the moment it exists. Nothing else has to
    change -- the training log's "Futures CSV not found — using spot-proxy
    VWAP/CMF" becomes "VWAP/CMF source: REAL FUTURES" on the next run.

    NIFTY spot volume is 0 for 2015-2025, so today every volume-named
    feature in V9 is a fake: cmf_proxy collapses to a close-position
    indicator, and every vwap feature uses an equal-weight average. This
    script is how that stops being true.

THE TRAP THIS GUARDS AGAINST
    core/tv_fetcher.py:114 maps ("NIFTY1!","NSE") -> "^NSEI" in its yfinance
    fallback -- the SPOT index, which has no volume. Backfilling through
    that path would silently produce another zero-volume file that looks
    fine. This script talks to TvDatafeed directly and REFUSES to write a
    file whose volume is empty, constant, or mostly zero.

WHAT YOU CAN REALISTICALLY GET
    TradingView caps intraday history. A logged-in session typically
    returns a few thousand 5-minute bars -- months, not years. That is not
    enough to retrain V11 (which uses ~212,000 bars). So:

      * run this repeatedly over time; it MERGES, never overwrites, so
        coverage grows with each run
      * your futures archiver is already accumulating forward data
      * for real depth (2+ years) an intraday vendor such as TrueData or
        Global Datafeeds is the only practical route, and this script will
        merge a vendor CSV in via --merge-csv

USAGE
    python scripts/backfill_futures_5min.py
    python scripts/backfill_futures_5min.py --bars 10000
    python scripts/backfill_futures_5min.py --merge-csv vendor_export.csv
    python scripts/backfill_futures_5min.py --dry-run     # inspect, write nothing
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "nifty_fut_5min.csv"
SPOT = ROOT / "data" / "nifty_5min.csv"
COLS = ["open", "high", "low", "close", "volume"]

SYMBOL, EXCHANGE = "NIFTY1!", "NSE"

# Below this, writing the file would shrink the training set instead of
# enriching it — see the guard in main().
MIN_COVERAGE = 0.90


def fetch_tv(n_bars: int) -> pd.DataFrame:
    """Pull NSE:NIFTY1! 5-min straight from TvDatafeed — no fallback chain."""
    try:
        from tvDatafeed import Interval, TvDatafeed
    except ImportError:
        print("  tvDatafeed not installed — pip install tvdatafeed")
        return pd.DataFrame()

    user, pwd = os.getenv("TV_USERNAME", ""), os.getenv("TV_PASSWORD", "")
    if user and pwd:
        print(f"  logging in as {user}")
        tv = TvDatafeed(username=user, password=pwd)
    else:
        print("  no TV_USERNAME/TV_PASSWORD — anonymous session, less history")
        tv = TvDatafeed()

    raw = tv.get_hist(symbol=SYMBOL, exchange=EXCHANGE,
                      interval=Interval.in_5_minute, n_bars=n_bars)
    if raw is None or len(raw) == 0:
        print("  TradingView returned nothing")
        return pd.DataFrame()

    df = raw.rename(columns=str.lower)
    keep = [c for c in COLS if c in df.columns]
    df = df[keep].copy()
    df.index.name = "date"
    return df


def validate(df: pd.DataFrame, source: str) -> bool:
    """Refuse anything that isn't really futures volume."""
    if df.empty:
        print(f"  REJECT {source}: no rows")
        return False
    if "volume" not in df.columns:
        print(f"  REJECT {source}: no volume column — this is not futures data")
        return False
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    nz = float((v > 0).mean())
    if nz < 0.50:
        print(f"  REJECT {source}: only {nz:.1%} of bars have volume > 0.")
        print("         That is the spot index, not the futures contract")
        print("         (tv_fetcher.py:114 maps NIFTY1! -> ^NSEI in fallback).")
        return False
    if v[v > 0].nunique() < 10:
        print(f"  REJECT {source}: volume is near-constant — synthetic, not real")
        return False
    print(f"  OK {source}: {len(df):,} bars, {nz:.1%} with volume, "
          f"median {v[v>0].median():,.0f}")
    return True


def merge(new: pd.DataFrame) -> pd.DataFrame:
    """Union with whatever is already on disk. Never lose coverage."""
    if OUT.exists():
        old = pd.read_csv(OUT, parse_dates=["date"]).set_index("date")
        print(f"  existing file: {len(old):,} bars "
              f"({old.index.min():%Y-%m-%d} .. {old.index.max():%Y-%m-%d})")
        out = pd.concat([old, new])
    else:
        out = new
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def coverage(df: pd.DataFrame) -> float:
    """Return futures coverage as a fraction of the spot history."""
    if not SPOT.exists():
        return 1.0
    spot = pd.read_csv(SPOT, usecols=["date"], parse_dates=["date"])
    s0, s1 = spot["date"].min(), spot["date"].max()
    have = df.index.normalize().nunique()
    need = spot["date"].dt.normalize().nunique()
    print(f"\n  spot history : {s0:%Y-%m-%d} .. {s1:%Y-%m-%d}  ({need:,} days)")
    print(f"  futures have : {df.index.min():%Y-%m-%d} .. "
          f"{df.index.max():%Y-%m-%d}  ({have:,} days)")
    frac = have / need
    print(f"  coverage     : {frac:.1%} of the spot history")
    return frac


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Backfill NIFTY futures 5-min")
    ap.add_argument("--bars", type=int, default=20000,
                    help="bars to request from TradingView (it will cap this)")
    ap.add_argument("--merge-csv", default=None,
                    help="also merge a vendor CSV (date,open,high,low,close,volume)")
    ap.add_argument("--force", action="store_true",
                    help="write even with partial coverage (do NOT retrain after)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    frames = []

    if a.merge_csv:
        p = Path(a.merge_csv)
        print(f"reading vendor file {p.name} ...")
        v = pd.read_csv(p)
        dcol = next((c for c in v.columns
                     if c.lower() in ("date", "datetime", "timestamp")), None)
        if dcol is None:
            raise SystemExit("vendor CSV needs a date/datetime/timestamp column")
        v[dcol] = pd.to_datetime(v[dcol], errors="coerce")
        v = v.dropna(subset=[dcol]).set_index(dcol)
        v.index.name = "date"
        v = v.rename(columns=str.lower)
        v = v[[c for c in COLS if c in v.columns]]
        if validate(v, p.name):
            frames.append(v)

    print("fetching NSE:NIFTY1! 5-min from TradingView ...")
    tv = fetch_tv(a.bars)
    if validate(tv, "TradingView"):
        frames.append(tv)

    if not frames:
        print("\nNothing usable was fetched. data/nifty_fut_5min.csv unchanged.")
        print("The pipeline keeps using the spot proxy — which is correct")
        print("behaviour, not a silent failure.")
        raise SystemExit(1)

    new = pd.concat(frames)
    new = new[~new.index.duplicated(keep="last")].sort_index()
    merged = merge(new)
    frac = coverage(merged)

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # ── THE GUARD ────────────────────────────────────────────────────────
    # Writing this file is not inert: train_model_v9.py:1504 does
    #     use_real_vwap = not feat_fut.empty
    # which is ALL-OR-NOTHING for the whole 11-year build. With partial
    # coverage, merge_asof leaves NaN on every bar with no futures data and
    # train()'s dropna() then deletes them -- silently collapsing the
    # training set to whatever the futures file covers.
    #
    # V10 already died this way: option/IV features cost 86% of the training
    # rows for 7.7x worse recall. Refuse to repeat it by accident.
    if frac < MIN_COVERAGE:
        print(f"\n{'!'*70}")
        print(f"REFUSING TO WRITE — coverage {frac:.1%} < {MIN_COVERAGE:.0%}")
        print(f"{'!'*70}")
        print("This file is a switch, not just data. Creating it flips the")
        print("ENTIRE training build to real-futures mode, and every bar")
        print("without a futures quote is then dropped by train()'s dropna().")
        print(f"You would go from ~212,000 training rows to roughly the "
              f"{int(frac*2852):,} days this file covers.")
        print("\nThat is exactly how V10 failed (86% of data lost).")
        print("\nOptions:")
        print("  * keep running this periodically — it merges, coverage grows")
        print("  * buy vendor intraday history and --merge-csv it in")
        print("  * --force to write anyway (only if you are NOT retraining)")
        if not a.force:
            raise SystemExit(2)
        print("\n--force given: writing anyway. Do NOT retrain against this.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index_label="date")
    print(f"\nwrote {OUT} — {len(merged):,} bars")
    print("\nNEXT: retrain and the pipeline switches itself over.")
    print("  docker compose run --rm -T --build --entrypoint python \\")
    print("      nifty-trader scripts/train_model_v11.py")
    print("Confirm the log says 'VWAP/CMF source: REAL FUTURES' — if it still")
    print("says SPOT PROXY, the file was not picked up.")


if __name__ == "__main__":
    main()
