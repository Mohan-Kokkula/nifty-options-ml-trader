"""
test_trend_day_or_filter.py — should trend_day_brain also require an
opening-range break?

Came out of a user question: "whenever a trade is good we should take it,
instead of a timing window?" Testing that produced a table where every
pure 'has the session established itself' proxy UNDERperformed the clock
(session range >=5xATR scored 1.054, beyond-opening-range 0.993, against
the clock's 1.096) -- but the CLOCK AND the opening-range break together
scored 1.140, +3.26 pts/trade against +2.32.

That combined number was picked as the best of 8 variants measured over
the whole of 2015-2026, TEST window included. It is therefore a candidate,
not a finding, and the point of this script is to try to kill it.

HONESTY NOTE, up front: the TEST window has ALREADY been seen by that
sweep. A TEST number here is not virgin evidence and is reported with that
caveat rather than dressed up as confirmation. The per-year table below is
the stronger test -- an edge that survives in most individual years is
believable; one that appears only in aggregate is one or two good years
wearing a disguise. trend_day_brain itself was accepted on 3/3 folds, not
on an aggregate.

THE RULE (unchanged from core/trend_day_brain.py)
    window 11:30-14:00, one trade per day
    PUT  : close < vwap_proxy - 25 AND 3 lower closes AND close < day open
    CALL : mirror
    stop : day extreme +/- 10pt buffer, capped at 60pts
    target: 2R, else exit at 15:15
    costs: 5.9 pts round trip (futures)

THE ADDITION BEING TESTED
    require close beyond the 09:15-09:45 opening range in the trade's
    direction (below OR low for a PUT, above OR high for a CALL)

USAGE
    python scripts/test_trend_day_or_filter.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSV_5M = ROOT / "data" / "nifty_5min.csv"

FRICTION = 5.9
MAX_STOP, BUF, RR = 60.0, 10.0, 2.0
MIN_DEV, SEQ = 25.0, 3
WIN_START, WIN_END, EOD = "11:30", "14:00", "15:15"
OR_BARS = 6                      # 09:15-09:45
DEV_END = "2025-09-02"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_5M, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["tp"] = (df.high + df.low + df.close) / 3
    g = df.groupby(df.index.normalize())
    df["vwap"] = g["tp"].cumsum() / (g.cumcount() + 1)
    df["dopen"] = g["open"].transform("first")
    df["dhi"] = g["high"].cummax()
    df["dlo"] = g["low"].cummin()
    d = df["close"].diff()
    df["up3"] = (d > 0).rolling(SEQ).sum().eq(SEQ)
    df["dn3"] = (d < 0).rolling(SEQ).sum().eq(SEQ)
    df["hhmm"] = df.index.strftime("%H:%M")
    df["or_hi"] = g["high"].transform(lambda s: s.iloc[:OR_BARS].max())
    df["or_lo"] = g["low"].transform(lambda s: s.iloc[:OR_BARS].min())
    return df


def run(df: pd.DataFrame, require_or: bool) -> pd.DataFrame:
    """One trade per day, exactly the brain's rule. Returns per-trade rows."""
    rows = []
    for day, s in df.groupby(df.index.normalize()):
        H, L, C = s.high.values, s.low.values, s.close.values
        V, O = s.vwap.values, s.dopen.values
        HI, LO = s.dhi.values, s.dlo.values
        U3, D3, T = s.up3.values, s.dn3.values, s.hhmm.values
        ORH, ORL = s.or_hi.values, s.or_lo.values
        for i in range(len(s)):
            if not (WIN_START <= T[i] <= WIN_END):
                continue
            dirn = 0
            if C[i] < V[i] - MIN_DEV and D3[i] and C[i] < O[i]:
                dirn = -1
            elif C[i] > V[i] + MIN_DEV and U3[i] and C[i] > O[i]:
                dirn = 1
            if dirn == 0:
                continue
            if require_or:
                broke = (C[i] < ORL[i]) if dirn < 0 else (C[i] > ORH[i])
                if not broke:
                    continue
            e = C[i]
            stop = min(MAX_STOP,
                       abs((HI[i] + BUF) - e if dirn < 0 else e - (LO[i] - BUF)))
            if stop <= 0:
                continue
            sl = e + stop if dirn < 0 else e - stop
            tgt = e - stop * RR if dirn < 0 else e + stop * RR
            pnl = None
            for j in range(i + 1, len(s)):
                if dirn < 0:
                    if H[j] >= sl:
                        pnl = -stop
                        break
                    if L[j] <= tgt:
                        pnl = stop * RR
                        break
                else:
                    if L[j] <= sl:
                        pnl = -stop
                        break
                    if H[j] >= tgt:
                        pnl = stop * RR
                        break
                if T[j] >= EOD:
                    pnl = (e - C[j]) if dirn < 0 else (C[j] - e)
                    break
            if pnl is None:
                pnl = (e - C[-1]) if dirn < 0 else (C[-1] - e)
            rows.append({"day": pd.Timestamp(day), "pnl": pnl - FRICTION})
            break                                    # one trade per day
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return {"n": 0, "pf": 0.0, "avg": 0.0, "win": 0.0, "total": 0.0}
    a = t["pnl"].values
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    return {"n": len(a), "pf": float(w / l) if l > 0 else float("inf"),
            "avg": float(a.mean()), "win": float((a > 0).mean() * 100),
            "total": float(a.sum())}


def line(tag: str, s: dict) -> str:
    pf = "  inf " if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{tag:<26} {pf} {s['n']:>6} {s['avg']:>+8.2f} "
            f"{s['win']:>6.1f}% {s['total']:>+9.0f}")


HDR = f"{'variant':<26} {'PF':>6} {'n':>6} {'avg':>8} {'win':>7} {'total':>9}"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    df = load()
    print(f"bars {len(df):,}  {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}")
    print("NOTE: the sweep that produced this candidate already saw the TEST")
    print("window. The per-year table is the real evidence, not the TEST row.\n")

    base = run(df, require_or=False)
    withor = run(df, require_or=True)
    cut = pd.Timestamp(DEV_END)

    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    print(HDR)
    for tag, t in (("clock only", base), ("clock + OR break", withor)):
        print(line(f"{tag}  ALL", stats(t)))
    print("-" * 78)
    for tag, t in (("clock only", base), ("clock + OR break", withor)):
        print(line(f"{tag}  DEV", stats(t[t.day <= cut])))
    print("-" * 78)
    for tag, t in (("clock only", base), ("clock + OR break", withor)):
        print(line(f"{tag}  TEST*", stats(t[t.day > cut])))
    print("* TEST was already seen by the selection sweep — weak evidence.")

    print("\n" + "=" * 78)
    print("PER YEAR — the test that matters")
    print("=" * 78)
    print(f"{'year':<6} {'clock PF':>9} {'+OR PF':>8} {'dPF':>8} "
          f"{'clock n':>8} {'+OR n':>7} {'better?':>8}")
    print("-" * 78)
    years = sorted(set(base.day.dt.year) | set(withor.day.dt.year))
    wins = 0
    counted = 0
    for y in years:
        b = stats(base[base.day.dt.year == y])
        o = stats(withor[withor.day.dt.year == y])
        if b["n"] < 20 or o["n"] < 20:
            continue
        counted += 1
        d = o["pf"] - b["pf"]
        better = d > 0
        wins += better
        print(f"{y:<6} {b['pf']:>9.3f} {o['pf']:>8.3f} {d:>+8.3f} "
              f"{b['n']:>8} {o['n']:>7} {'YES' if better else 'no':>8}")
    print("-" * 78)
    print(f"the OR filter improved PF in {wins} of {counted} years")

    print("\nVERDICT")
    if counted == 0:
        print("  INCONCLUSIVE — not enough trades per year to judge.")
    elif wins / counted >= 0.7:
        print(f"  HOLDS UP: better in {wins}/{counted} years. Consistent enough")
        print("  to justify a walk-forward run against the brain's own folds.")
    elif wins / counted >= 0.5:
        print(f"  MARGINAL: better in only {wins}/{counted} years. That is close")
        print("  to a coin flip — the aggregate gain is carried by a few years,")
        print("  not by a stable effect. Do not adopt on this evidence.")
    else:
        print(f"  REJECT: better in only {wins}/{counted} years. The aggregate")
        print("  improvement is an artefact of selection across 8 variants.")


if __name__ == "__main__":
    main()
