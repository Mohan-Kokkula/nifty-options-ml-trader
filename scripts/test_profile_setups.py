"""
test_profile_setups.py — do Market/Volume Profile setups work on NIFTY?

New feature family, tested standalone. Nothing here touches the live pilot
and the pilot imports none of it. If it fails, it stays out; if it passes,
it becomes its own brain the way trend_day_brain.py did.

SETUPS (classic Market Profile, all mechanical, all from prior-session or
strictly-causal developing profiles — no look-ahead)

    VP1  VALUE FADE       prior day balanced; price touches prior VAH from
                          below -> SHORT, or prior VAL from above -> LONG.
                          Target the prior POC. This is the "value holds"
                          trade: the market is rotating inside known value.

    VP2  VALUE BREAKOUT   price accepts outside prior value (closes beyond
                          VAH/VAL for 2 consecutive bars) -> trade with it.
                          The "value migrates" trade, opposite premise.

    VP3  POC MAGNET       price is > k*ATR from the DEVELOPING POC of the
                          session in progress -> trade back toward it.

EXITS      SL at 10/15/20% of average daily ATR (same scale as the VWAP
           test, so results are comparable), with three targets: the
           profile level itself (POC), a fixed 2R, or ride to the close.

SPLIT      DEV  .. 2025-09-02      TEST 2025-09-03 .. 2026-08-06
           Same holdout as the model work. Costs 5.9 pts round trip.
           Entries fill at the bar CLOSE, never at the level.

USAGE      python scripts/test_profile_setups.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.profile_features import (  # noqa: E402
    developing_profiles, profile_agreement, session_profiles,
)

CSV_5M = ROOT / "data" / "nifty_5min.csv"
CSV_DAY = ROOT / "data" / "nifty_day.csv"
CACHE = ROOT / "data" / ".profile_cache.pkl"

FRICTION = 5.9
DEV_END = "2025-09-02"
TEST_START = "2025-09-03"
EOD, ENTRY_START, ENTRY_END = "15:15", "09:20", "15:00"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_5M, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["hhmm"] = df.index.strftime("%H:%M")
    return df


def daily_atr_points() -> float:
    d = pd.read_csv(CSV_DAY, parse_dates=["date"]).set_index("date").sort_index()
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(200).mean().dropna().mean())


def bar_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill().values


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-session profile levels and causal developing levels."""
    sp = session_profiles(df, use_volume=False)
    prior = sp.shift(1).add_prefix("p_")          # yesterday's finished profile
    day = df.index.normalize()
    for c in prior.columns:
        df[c] = prior[c].reindex(day).values
    dev = developing_profiles(df)
    return df.join(dev)


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
    return (f"{label:<38} {pf}  {s['n']:>6}  {s['avg']:+7.2f}  "
            f"{s['win']:5.1f}%  {s['total']:+10.0f}")


HDR = (f"{'variant':<38} {'PF':>6}  {'n':>6}  {'avg':>7}  {'win':>6}  "
       f"{'total':>10}")


def run(d: dict, setup: str, sl_pts: float, target: str,
        lo_i: int, hi_i: int, k_atr: float = 1.5,
        return_trades: bool = False):
    H, L, C, O = d["H"], d["L"], d["C"], d["O"]
    pv, ph, pl = d["p_poc"], d["p_vah"], d["p_val"]
    dp = d["d_poc"]
    atr, hhmm, days = d["atr"], d["hhmm"], d["days"]

    trades: list[float] = []
    pos = None
    cur_day = None
    streak_up = streak_dn = 0

    for i in range(lo_i, hi_i):
        if days[i] != cur_day:
            cur_day, pos, streak_up, streak_dn = days[i], None, 0, 0

        if pos is not None:
            dirn, epx, sl, tgt = pos
            hit = None
            if dirn == 1:
                if L[i] <= sl:
                    hit = sl - epx
                elif np.isfinite(tgt) and H[i] >= tgt:
                    hit = tgt - epx
            else:
                if H[i] >= sl:
                    hit = epx - sl
                elif np.isfinite(tgt) and L[i] <= tgt:
                    hit = epx - tgt
            if hit is None and hhmm[i] >= EOD:
                hit = (C[i] - epx) if dirn == 1 else (epx - C[i])
            if hit is not None:
                trades.append(hit - FRICTION)
                pos = None
            continue

        if not (np.isfinite(pv[i]) and np.isfinite(ph[i]) and np.isfinite(pl[i])):
            continue

        # acceptance streaks for VP2 (must be updated every bar)
        streak_up = streak_up + 1 if C[i] > ph[i] else 0
        streak_dn = streak_dn + 1 if C[i] < pl[i] else 0

        if not (ENTRY_START <= hhmm[i] < ENTRY_END):
            continue

        dirn, lvl = 0, np.nan
        if setup == "VP1":
            balanced = (ph[i] - pl[i]) < 1.2 * atr[i] * 14
            if balanced and L[i] <= ph[i] <= H[i] and C[i - 1] < ph[i - 1]:
                dirn, lvl = -1, pv[i]
            elif balanced and L[i] <= pl[i] <= H[i] and C[i - 1] > pl[i - 1]:
                dirn, lvl = 1, pv[i]
        elif setup == "VP2":
            if streak_up == 2:
                dirn, lvl = 1, np.nan
            elif streak_dn == 2:
                dirn, lvl = -1, np.nan
        elif setup == "VP3":
            if np.isfinite(dp[i]) and atr[i] > 0:
                if C[i] - dp[i] > k_atr * atr[i]:
                    dirn, lvl = -1, dp[i]
                elif dp[i] - C[i] > k_atr * atr[i]:
                    dirn, lvl = 1, dp[i]

        if dirn == 0:
            continue
        epx = C[i]
        if target == "level":
            tgt = lvl if np.isfinite(lvl) else epx + dirn * 2 * sl_pts
        elif target == "2R":
            tgt = epx + dirn * 2 * sl_pts
        else:
            tgt = np.nan                      # ride to close
        # a target already behind price is not a trade
        if np.isfinite(tgt) and ((dirn == 1 and tgt <= epx)
                                 or (dirn == -1 and tgt >= epx)):
            continue
        pos = (dirn, epx, epx - dirn * sl_pts, tgt)

    return trades if return_trades else stats(trades)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    df = load()

    print("=" * 88)
    print("STEP 1 — is the TPO proxy faithful to a real Volume Profile?")
    print("=" * 88)
    volo = df[df["volume"] > 0]
    ag = profile_agreement(volo)
    if ag.get("n", 0) < 10:
        print(f"  only {ag.get('n',0)} sessions with volume — cannot check")
    else:
        print(f"  sessions with real volume: {ag['n']}")
        print(f"  POC   corr {ag['poc_corr']:.4f}   mean abs err "
              f"{ag['poc_mae_pts']:.1f} pts  ({ag['poc_mae_va']:.1%} of value width)")
        print(f"  VAH   mean abs err {ag['vah_mae_pts']:.1f} pts")
        print(f"  VAL   mean abs err {ag['val_mae_pts']:.1f} pts")
        print(f"  VA width corr {ag['vaw_corr']:.4f}")

    if CACHE.exists():
        print(f"\nloading cached profiles from {CACHE.name}")
        df = joblib.load(CACHE)
    else:
        print("\nbuilding profiles (causal, recomputed each bar)...")
        df = build_frame(df)
        joblib.dump(df, CACHE, compress=3)
        print(f"  cached ({time.time()-t0:.0f}s)")

    adr = daily_atr_points()
    d = {"H": df["high"].values, "L": df["low"].values,
         "C": df["close"].values, "O": df["open"].values,
         "p_poc": df["p_poc"].values, "p_vah": df["p_vah"].values,
         "p_val": df["p_val"].values, "d_poc": df["d_poc"].values,
         "atr": bar_atr(df), "hhmm": df["hhmm"].values,
         "days": df.index.normalize().values}

    dev_hi = int(df.index.searchsorted(pd.Timestamp(DEV_END) + pd.Timedelta(days=1)))
    te_lo = int(df.index.searchsorted(pd.Timestamp(TEST_START)))
    print(f"\nDEV  {df.index[0]:%Y-%m-%d} .. {DEV_END}  ({dev_hi:,} bars)")
    print(f"TEST {TEST_START} .. {df.index[-1]:%Y-%m-%d}  ({len(df)-te_lo:,} bars)")
    print(f"avg daily ATR(200) = {adr:.0f} pts\n")

    print("=" * 88)
    print("STEP 2 — DEV, variant selection")
    print("=" * 88)
    print(HDR)
    best = None
    for setup in ("VP1", "VP2", "VP3"):
        for p in (0.10, 0.15, 0.20):
            for tgt in ("level", "2R", "ride"):
                s = run(d, setup, adr * p, tgt, 0, dev_hi)
                if s["n"] < 100:
                    continue
                print(row(f"{setup} sl={p:.0%} tp={tgt}", s))
                if best is None or s["pf"] > best[0]["pf"]:
                    best = (s, setup, p, tgt)
        print("-" * 88)

    if best is None:
        print("\nNo variant produced enough trades. Nothing to confirm.")
        return
    s, setup, p, tgt = best
    print(f"\nDEV winner: {setup} sl={p:.0%} ({adr*p:.0f} pts) tp={tgt}")
    print(f"            PF {s['pf']:.3f}  n={s['n']}  avg {s['avg']:+.2f}")

    print("\n" + "=" * 88)
    print("STEP 3 — TEST, confirmed once, reported as-is")
    print("=" * 88)
    print(HDR)
    t = run(d, setup, adr * p, tgt, te_lo, len(df))
    print(row(f"{setup} sl={p:.0%} tp={tgt}", t))

    # ── STEP 4 ───────────────────────────────────────────────────────────
    # A rule that loses on DEV and wins on TEST has NOT been validated. The
    # honest question is: if the DEV distribution is the truth, how often
    # does a window of TEST's size look this good by chance? Contiguous
    # windows, not iid resampling, so regime persistence is preserved.
    print("\n" + "=" * 88)
    print("STEP 4 — is the TEST result distinguishable from luck?")
    print("=" * 88)
    dev_tr = np.array(run(d, setup, adr * p, tgt, 0, dev_hi, return_trades=True))
    w = t["n"]
    if len(dev_tr) < 3 * w or w < 10:
        print(f"  not enough DEV trades ({len(dev_tr)}) for {w}-trade windows")
        pctile = None
    else:
        pfs = []
        for i in range(0, len(dev_tr) - w):
            seg = dev_tr[i:i + w]
            wins, loss = seg[seg > 0].sum(), -seg[seg < 0].sum()
            pfs.append(wins / loss if loss > 0 else np.inf)
        pfs = np.array(pfs)
        pctile = float((pfs < t["pf"]).mean() * 100)
        print(f"  DEV PF over all contiguous {w}-trade windows "
              f"(n={len(pfs):,} windows):")
        print(f"     median {np.median(pfs):.3f} | p75 {np.percentile(pfs,75):.3f} "
              f"| p90 {np.percentile(pfs,90):.3f} | p95 {np.percentile(pfs,95):.3f}")
        print(f"  TEST PF {t['pf']:.3f} sits at the {pctile:.1f}th percentile "
              f"of what this rule already did on DEV by chance.")
        print(f"  {100-pctile:.1f}% of DEV windows were this good or better.")

    print("\nVERDICT")
    if t["n"] < 30:
        print(f"  INCONCLUSIVE — {t['n']} trades on TEST is too few to judge.")
    elif s["pf"] <= 1.0:
        print(f"  FAILS. The rule LOSES on DEV: PF {s['pf']:.3f} over "
              f"{s['n']:,} trades and 10.5 years.")
        print(f"  TEST PF {t['pf']:.3f} on {t['n']} trades is one year, and "
              f"{'' if pctile is None else f'{100-pctile:.0f}% of DEV windows '}"
              f"{'' if pctile is None else 'matched or beat it. '}"
              f"A decade of losses is the stronger evidence.")
        print(f"  Do not adopt. This is a favourable draw, not an edge.")
    elif t["pf"] > 1.0 and t["avg"] > 0:
        print(f"  PASSES: DEV {s['pf']:.3f} and TEST {t['pf']:.3f}, "
              f"{t['avg']:+.2f} pts/trade. Worth building as its own brain.")
    else:
        print(f"  FAILS: PF {t['pf']:.3f}, {t['avg']:+.2f} pts/trade. "
              f"Does not survive out-of-sample — leave it out.")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
