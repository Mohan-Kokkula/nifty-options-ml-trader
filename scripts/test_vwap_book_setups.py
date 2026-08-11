"""
test_vwap_book_setups.py — do Trader Dale's VWAP setups work on NIFTY?

Source: "VWAP" (Dale Woods / trader-dale.com, 2024 edition), 122pp. Three
mechanical setups are extracted verbatim from the text; everything else in
the book is discretionary (order flow, volume profile) and is not testable
here.

    S1  REACTION TO VWAP        p.14
        Price above VWAP and pulls back to touch it   -> LONG
        Price below VWAP and retraces up to touch it  -> SHORT

    S2  VWAP ROTATION           p.50   (bands horizontal = rotation)
        Touch lower 1st deviation from above          -> LONG,  TP = VWAP
        Touch upper 1st deviation from below          -> SHORT, TP = VWAP

    S3  VWAP TREND              p.54   (bands vertical = trend)
        Upper dev rising  AND price above it, touch   -> LONG
        Lower dev falling AND price below it, touch   -> SHORT

    Entry confirmation (p.78): either take the first touch, or wait for one
    candle to close back on the correct side of the level.
    ATR exits (p.94, p.104): SL and TP each 10-20% of average daily ATR.

HONEST LIMITATION, STATED UP FRONT
    NIFTY spot carries no volume (data/nifty_5min.csv volume==0 for
    2015-2025). A true volume-weighted average price cannot be computed.
    What is computed here is the anchored TYPICAL-PRICE average --- VWAP
    with equal weights --- the same "SPOT PROXY" train_model_v9.py falls
    back to, and the same level the trend-day brain trades successfully.
    Every result below therefore tests the book's GEOMETRY, not its volume
    thesis. If the geometry fails, the volume version is not rescued by it;
    if it passes, the volume version might do better.

METHOD
    DEV  2015-01-21 .. 2025-09-02   choose one variant here
    TEST 2025-09-03 .. 2026-08-06   confirm it once, report whatever comes
    Costs 5.9 pts round trip (futures friction, EXECUTION_MODE=futures).
    Entries fill at the bar CLOSE, never at the level itself --- assuming a
    limit fill at the exact touch price would flatter every number here.

USAGE
    python scripts/test_vwap_book_setups.py
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
CSV_DAY = ROOT / "data" / "nifty_day.csv"

FRICTION = 5.9
DEV_END = "2025-09-02"
TEST_START = "2025-09-03"
EOD = "15:15"
ENTRY_START = "09:20"
ENTRY_END = "15:00"


# ══════════════════════════════════════════════════════════════════════════
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_5M, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["day"] = df.index.normalize()
    df["hhmm"] = df.index.strftime("%H:%M")
    return df


def daily_atr_points() -> float:
    """Average daily volatility, ATR(200) on the daily chart (book p.94)."""
    d = pd.read_csv(CSV_DAY, parse_dates=["date"]).set_index("date").sort_index()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"],
                    (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(200).mean().dropna().mean())


def anchored(df: pd.DataFrame, key: pd.Series) -> tuple:
    """Anchored VWAP-proxy and its 1st standard deviation band.

    With volume identically zero the volume-weighted mean degenerates to the
    plain running mean of typical price, so that is what is computed. sigma
    is the running population std of tp about that mean, giving the book's
    "1st deviation" bands at +/- 1 sigma.
    """
    g = df.groupby(key)["tp"]
    n = g.cumcount() + 1
    csum = g.cumsum()
    vwap = csum / n
    csq = df.groupby(key)["tp"].transform(lambda s: (s ** 2).cumsum())
    var = (csq / n) - vwap ** 2
    sigma = np.sqrt(var.clip(lower=0))
    return vwap, sigma


def touched(lo: float, hi: float, level: float) -> bool:
    return lo <= level <= hi


# ══════════════════════════════════════════════════════════════════════════
def run(df: pd.DataFrame, setup: str, anchor: str, entry: str,
        sl_pts: float, tp_pts: float, lo_i: int, hi_i: int) -> dict:
    """Walk bars [lo_i, hi_i), one position at a time, book rules only."""
    key = df["day"] if anchor == "day" else df.index.to_period("W").astype(str)
    vwap, sigma = anchored(df, key)
    up, dn = vwap + sigma, vwap - sigma

    H, L, C = df["high"].values, df["low"].values, df["close"].values
    V, S = vwap.values, sigma.values
    U, D = up.values, dn.values
    hhmm = df["hhmm"].values
    days = df["day"].values

    trades: list[float] = []
    pos = None                 # (dir, entry_px, sl, tp, use_vwap_tp)
    pend = None                # awaiting close-confirmation
    cur_day = None
    away = False               # price has moved away from the level first
    slope_n = 6                # 30 min of band slope for rotation vs trend

    for i in range(lo_i, hi_i):
        if days[i] != cur_day:
            cur_day, pos, pend, away = days[i], None, None, False

        # ── manage open position ─────────────────────────────────────────
        if pos is not None:
            d, epx, sl, tp, vtp = pos
            target = V[i] if vtp else tp
            hit = None
            if d == 1:
                if L[i] <= sl:
                    hit = sl - epx
                elif H[i] >= target:
                    hit = target - epx
            else:
                if H[i] >= sl:
                    hit = epx - sl
                elif L[i] <= target:
                    hit = epx - target
            if hit is None and hhmm[i] >= EOD:
                hit = (C[i] - epx) if d == 1 else (epx - C[i])
            if hit is not None:
                trades.append(hit - FRICTION)
                pos = None
            continue

        if not np.isfinite(V[i]) or not np.isfinite(S[i]) or S[i] <= 0:
            continue

        # ── a pending confirmed entry resolves on the next close ─────────
        if pend is not None:
            d, lvl = pend
            pend = None
            ok = (C[i] > lvl) if d == 1 else (C[i] < lvl)
            if ok and ENTRY_START <= hhmm[i] < ENTRY_END:
                epx = C[i]
                pos = (d, epx,
                       epx - sl_pts if d == 1 else epx + sl_pts,
                       epx + tp_pts if d == 1 else epx - tp_pts,
                       setup == "S2")
                away = False
            continue

        if not (ENTRY_START <= hhmm[i] < ENTRY_END):
            continue

        # ── signal ───────────────────────────────────────────────────────
        d = 0
        lvl = np.nan
        if setup == "S1":
            # needs a prior excursion of >= 1 sigma, else every bar "touches"
            if not away:
                away = abs(C[i] - V[i]) >= S[i]
            elif touched(L[i], H[i], V[i]):
                d, lvl = (1 if C[i - 1] > V[i - 1] else -1), V[i]
        elif setup == "S2":
            if i - slope_n >= 0:
                horiz = (abs(U[i] - U[i - slope_n]) < 0.25 * S[i]
                         and abs(D[i] - D[i - slope_n]) < 0.25 * S[i])
                if horiz and touched(L[i], H[i], D[i]) and C[i - 1] > D[i - 1]:
                    d, lvl = 1, D[i]
                elif horiz and touched(L[i], H[i], U[i]) and C[i - 1] < U[i - 1]:
                    d, lvl = -1, U[i]
        elif setup == "S3":
            if i - slope_n >= 0:
                rising = U[i] - U[i - slope_n] > 0.25 * S[i]
                falling = D[i] - D[i - slope_n] < -0.25 * S[i]
                if rising and C[i - 1] > U[i - 1] and touched(L[i], H[i], U[i]):
                    d, lvl = 1, U[i]
                elif falling and C[i - 1] < D[i - 1] and touched(L[i], H[i], D[i]):
                    d, lvl = -1, D[i]

        if d == 0:
            continue
        if entry == "confirm":
            pend = (d, lvl)
            continue
        epx = C[i]
        pos = (d, epx,
               epx - sl_pts if d == 1 else epx + sl_pts,
               epx + tp_pts if d == 1 else epx - tp_pts,
               setup == "S2")
        away = False

    return stats(trades)


def stats(t: list) -> dict:
    a = np.array(t, dtype=float)
    if len(a) == 0:
        return {"n": 0, "pf": 0.0, "avg": 0.0, "win": 0.0, "total": 0.0}
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    return {"n": len(a), "pf": float(w / l) if l > 0 else float("inf"),
            "avg": float(a.mean()), "win": float((a > 0).mean() * 100),
            "total": float(a.sum())}


def row(label: str, s: dict) -> str:
    pf = "  inf " if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{label:<40} {pf}  {s['n']:>6}  {s['avg']:+7.2f}  "
            f"{s['win']:5.1f}%  {s['total']:+10.0f}")


HDR = f"{'variant':<40} {'PF':>6}  {'n':>6}  {'avg':>7}  {'win':>6}  {'total':>10}"


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    df = load()
    adr = daily_atr_points()
    print(f"bars {len(df):,}  {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}")
    print(f"average daily ATR(200) = {adr:.0f} pts  ->  book says SL/TP = "
          f"10-20% = {adr*0.10:.0f}..{adr*0.20:.0f} pts")
    print(f"NOTE: volume is 0 for 2015-2025, so VWAP here is the equal-weight "
          f"typical-price anchor (spot proxy).")

    dev_hi = int(df.index.searchsorted(pd.Timestamp(DEV_END) + pd.Timedelta(days=1)))
    te_lo = int(df.index.searchsorted(pd.Timestamp(TEST_START)))
    print(f"\nDEV  {df.index[0]:%Y-%m-%d} .. {DEV_END}   ({dev_hi:,} bars)")
    print(f"TEST {TEST_START} .. {df.index[-1]:%Y-%m-%d}   "
          f"({len(df)-te_lo:,} bars)\n")

    pcts = [0.10, 0.15, 0.20]
    best = None
    print("=" * 92)
    print("DEV — variant selection (book's own parameter ranges only)")
    print("=" * 92)
    print(HDR)
    for setup in ("S1", "S2", "S3"):
        for anchor in ("day", "week"):
            for entry in ("touch", "confirm"):
                for p in pcts:
                    sl = tp = adr * p
                    s = run(df, setup, anchor, entry, sl, tp, 0, dev_hi)
                    if s["n"] < 100:
                        continue
                    tag = f"{setup} {anchor:<4} {entry:<7} sl=tp={p:.0%}"
                    print(row(tag, s))
                    if best is None or s["pf"] > best[0]["pf"]:
                        best = (s, setup, anchor, entry, p)
        print("-" * 92)

    if best is None:
        print("\nNo variant produced enough trades to judge. Nothing to confirm.")
        return

    s, setup, anchor, entry, p = best
    print(f"\nDEV winner: {setup} anchor={anchor} entry={entry} "
          f"sl=tp={p:.0%} of daily ATR ({adr*p:.0f} pts)")
    print(f"            PF {s['pf']:.3f}  n={s['n']}  avg {s['avg']:+.2f} pts")

    print("\n" + "=" * 92)
    print("TEST — confirmed once, reported as-is")
    print("=" * 92)
    print(HDR)
    t = run(df, setup, anchor, entry, adr * p, adr * p, te_lo, len(df))
    print(row(f"{setup} {anchor} {entry} sl=tp={p:.0%}", t))

    # ── Second experiment: S3 as the book actually specifies it ──────────
    # p.54: the Trend strategy "offers the advantage of trading with a
    # positive Risk Reward Ratio and allows for trailing your Take Profit".
    # Testing it at a fixed 1:1 above was a strawman of its own rules, so
    # the asymmetric and ride-to-close variants get their own pass.
    print("\n" + "=" * 92)
    print("DEV — S3 with the positive-RRR / ride-the-trend exit it asks for")
    print("=" * 92)
    print(HDR)
    best2 = None
    for anchor2 in ("day", "week"):
        for entry2 in ("touch", "confirm"):
            for p2 in pcts:
                for rr, rlbl in ((2.0, "tp=2R"), (3.0, "tp=3R"),
                                 (1e9, "ride to close")):
                    sl2 = adr * p2
                    s2 = run(df, "S3", anchor2, entry2, sl2,
                             sl2 * rr if rr < 1e8 else 1e9, 0, dev_hi)
                    if s2["n"] < 100:
                        continue
                    tag = f"S3 {anchor2:<4} {entry2:<7} sl={p2:.0%} {rlbl}"
                    print(row(tag, s2))
                    if best2 is None or s2["pf"] > best2[0]["pf"]:
                        best2 = (s2, anchor2, entry2, p2, rr, rlbl)

    if best2 is not None:
        s2, a2, e2, p2, rr, rlbl = best2
        print(f"\nDEV winner #2: S3 anchor={a2} entry={e2} sl={p2:.0%} {rlbl}")
        print(f"               PF {s2['pf']:.3f}  n={s2['n']}  "
              f"avg {s2['avg']:+.2f} pts")
        print("\n" + "=" * 92)
        print("TEST — second query. The holdout has now been used TWICE, so")
        print("this number is weaker evidence than the one above. Stated, not hidden.")
        print("=" * 92)
        print(HDR)
        t2 = run(df, "S3", a2, e2, adr * p2,
                 adr * p2 * rr if rr < 1e8 else 1e9, te_lo, len(df))
        print(row(f"S3 {a2} {e2} sl={p2:.0%} {rlbl}", t2))
        if t2["pf"] > t["pf"]:
            t = t2

    print("\nVERDICT")
    if t["n"] < 30:
        print(f"  INCONCLUSIVE — only {t['n']} trades on TEST, too few to judge.")
    elif t["pf"] > 1.0 and t["avg"] > 0:
        print(f"  PASSES: PF {t['pf']:.3f}, {t['avg']:+.2f} pts/trade after "
              f"{FRICTION} pts friction.")
    else:
        print(f"  FAILS: PF {t['pf']:.3f}, {t['avg']:+.2f} pts/trade. The DEV "
              f"result did not survive out-of-sample.")


if __name__ == "__main__":
    main()
