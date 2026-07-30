"""
fetch_futidx_bhavcopy.py — Priority 1: NIFTY index-futures basis/carry features
================================================================================
Built 2026-06-16. Downloads the NSE UDiFF F&O bhavcopy, extracts NIFTY index
futures (FinInstrmTp=IDF), builds basis/carry/roll features, merges them
leak-free (shift(1)) onto the leak-clean V9 frame, and runs the full
pre-registered validation gate (Pearson+Spearman IC, BH-FDR, friction,
walk-forward sign-stability, bootstrap CI).

REAL DATA ONLY — no synthetic fallback. If the download is unavailable the
script errors rather than fabricating data. NO feature is wired into model
training or live trading; this is the validation gate that must pass first.

USAGE
    python scripts/fetch_futidx_bhavcopy.py 2024-06-01 2026-06-04
    python scripts/fetch_futidx_bhavcopy.py            # default span
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import io
import sys
import time
import zipfile
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R_RATE, Q_YIELD = 0.065, 0.012     # from backtest_options (carry)
FWD_BARS = 3                        # 15-min forward target
N_FOLDS = 8
FRICTION_FLOOR_PCT = 0.12
CACHE = ROOT / "data/nse_bhavcopy/futidx_daily.csv"
_HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36", "Accept": "*/*",
         "Accept-Language": "en-US,en;q=0.9"}


# ------------------------------------------------------------------ download
def _udiff_url(d: date) -> str:
    return (f"https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")


def _fetch_day(d: date) -> pd.DataFrame | None:
    """Download one UDiFF FO bhavcopy; return NIFTY index-future rows or None."""
    try:
        req = urllib.request.Request(_udiff_url(d), headers=_HDRS)
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
    except Exception:
        return None                               # holiday / weekend / missing
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = zf.namelist()[0]
        df = pd.read_csv(zf.open(csv_name))
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "FinInstrmTp" not in df.columns or "TckrSymb" not in df.columns:
        return None
    fut = df[(df["FinInstrmTp"] == "IDF") & (df["TckrSymb"] == "NIFTY")].copy()
    if fut.empty:
        return None
    fut["TradDt"] = pd.to_datetime(fut["TradDt"]).dt.normalize()
    fut["XpryDt"] = pd.to_datetime(fut["XpryDt"]).dt.normalize()
    keep = ["TradDt", "XpryDt", "ClsPric", "SttlmPric", "OpnIntrst",
            "ChngInOpnIntrst", "UndrlygPric"]
    keep = [c for c in keep if c in fut.columns]
    return fut[keep].sort_values("XpryDt")


def download_range(start: date, end: date) -> pd.DataFrame:
    """Download all trading days in [start, end]; cache the parsed result."""
    if CACHE.exists():
        cached = pd.read_csv(CACHE, parse_dates=["TradDt", "XpryDt"])
        print(f"  using cached FUTIDX: {len(cached)} rows "
              f"{cached.TradDt.min().date()}..{cached.TradDt.max().date()}")
        return cached
    rows, d, ok, miss = [], start, 0, 0
    while d <= end:
        if d.weekday() < 5:                        # skip weekends
            day = _fetch_day(d)
            if day is not None and len(day):
                rows.append(day); ok += 1
            else:
                miss += 1
            time.sleep(0.3)                        # be polite to NSE
        d += timedelta(days=1)
        if (ok + miss) % 50 == 0 and (ok + miss):
            print(f"    progress: {ok} days fetched, {miss} skipped, at {d}")
    if not rows:
        raise RuntimeError("No FUTIDX data downloaded — NSE unreachable here. "
                           "Run where archives.nseindia.com is reachable.")
    out = pd.concat(rows, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    print(f"  downloaded FUTIDX: {ok} trading days -> {CACHE}")
    return out


# ------------------------------------------------------------------ features
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Daily near/far futures basis, carry, roll features (as of each date)."""
    raw = raw.sort_values(["TradDt", "XpryDt"])
    recs = []
    for d, g in raw.groupby("TradDt"):
        g = g[g["XpryDt"] >= d]
        if len(g) < 1:
            continue
        near = g.iloc[0]
        far = g.iloc[1] if len(g) > 1 else None
        S = float(near.get("UndrlygPric", 0) or 0)
        Fn = float(near.get("ClsPric", 0) or near.get("SttlmPric", 0) or 0)
        if S <= 0 or Fn <= 0:
            continue
        dte = max((near["XpryDt"] - d).days, 0)
        basis_pts = Fn - S
        rec = {
            "date": d,
            "near_expiry": near["XpryDt"],
            "basis_bps": basis_pts / S,
            "excess_basis": (basis_pts - S * (R_RATE - Q_YIELD) * dte / 365.0) / S,
            "near_oi": float(near.get("OpnIntrst", 0) or 0),
        }
        if far is not None:
            Sf = float(far.get("UndrlygPric", S) or S)
            Ff = float(far.get("ClsPric", 0) or far.get("SttlmPric", 0) or 0)
            rec["far_basis_bps"] = (Ff - Sf) / Sf if (Sf > 0 and Ff > 0) else np.nan
            rec["far_oi"] = float(far.get("OpnIntrst", 0) or 0)
        else:
            rec["far_basis_bps"] = np.nan
            rec["far_oi"] = np.nan
        recs.append(rec)
    f = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    roll = f["near_expiry"] != f["near_expiry"].shift(1)     # expiry rolled today
    f["basis_z20"] = (f["basis_bps"] - f["basis_bps"].rolling(20).mean()) \
        / f["basis_bps"].rolling(20).std()
    f["basis_chg"] = f["basis_bps"].diff().where(~roll)
    f["term_slope"] = f["far_basis_bps"] - f["basis_bps"]
    f["fut_oi_chg"] = f["near_oi"].pct_change().where(~roll)
    f["rollover_ratio"] = f["far_oi"] / (f["near_oi"] + f["far_oi"]).replace(0, np.nan)
    feat_cols = ["basis_bps", "excess_basis", "basis_z20", "basis_chg",
                 "term_slope", "fut_oi_chg", "rollover_ratio"]
    # shift(1): a bar on day D sees only day D-1 values (leak-free)
    f[feat_cols] = f[feat_cols].shift(1)
    return f[["date"] + feat_cols], feat_cols


# ------------------------------------------------------------------ validation
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank; prev = min(prev, p[idx] * n / i); q[idx] = prev
    return q


def validate(fday: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    from backtest_threshold_sweep import build_frame
    feat, _ = build_frame()
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat.index).normalize()
    fday = fday.set_index("date")
    for c in feat_cols:
        feat[c] = feat["date"].map(fday[c])
    px = feat["close"]
    y = (px.shift(-FWD_BARS) / px - 1).values
    fwd_std = np.nanstd(y)
    n = len(feat)
    print(f"  Frame {n:,} bars | fwd-15m std={fwd_std*100:.3f}%")
    rows, pvals = [], []
    bnds = np.linspace(0, n, N_FOLDS + 1).astype(int)
    rng = np.random.default_rng(3)
    for c in feat_cols:
        x = feat[c].values
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 500:
            rows.append(dict(feat=c, pearson=np.nan, spear=np.nan, p=1.0,
                             stab="0/0", ci_lo=np.nan, ci_hi=np.nan,
                             captured=0.0, clears=False)); pvals.append(1.0); continue
        pr, _ = stats.pearsonr(x[m], y[m])
        sr, sp = stats.spearmanr(x[m], y[m])
        signs = []
        for k in range(N_FOLDS):
            lo, hi = bnds[k], bnds[k+1]
            xx, yy = x[lo:hi], y[lo:hi]; mm = ~np.isnan(xx) & ~np.isnan(yy)
            if mm.sum() < 200: continue
            rr, _ = stats.spearmanr(xx[mm], yy[mm])
            if not np.isnan(rr): signs.append(np.sign(rr))
        nf_pos = int(np.sum(np.array(signs) == np.sign(sr))) if signs else 0
        xm, ym = x[m], y[m]
        bs = [stats.spearmanr(xm[i], ym[i])[0]
              for i in (rng.integers(0, len(xm), len(xm)) for _ in range(1000))]
        lo_ci, hi_ci = np.nanpercentile(bs, [2.5, 97.5])
        cap = abs(sr) * fwd_std * 100
        rows.append(dict(feat=c, pearson=pr, spear=sr, p=sp,
                         stab=f"{nf_pos}/{len(signs)}", ci_lo=lo_ci, ci_hi=hi_ci,
                         captured=cap, clears=cap > FRICTION_FLOOR_PCT))
        pvals.append(sp)
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p"].fillna(1.0).values)
    return res


def main():
    a = sys.argv[1:]
    start = datetime.strptime(a[0], "%Y-%m-%d").date() if len(a) > 0 else date(2024, 6, 1)
    end = datetime.strptime(a[1], "%Y-%m-%d").date() if len(a) > 1 else date(2026, 6, 4)
    print(f"FUTIDX basis validation | span {start}..{end}")
    raw = download_range(start, end)
    fday, feat_cols = build_features(raw)
    print(f"  futures features: {len(fday)} days, cols={feat_cols}")
    res = validate(fday, feat_cols)
    print("\n" + "=" * 100)
    print("  FUTURES BASIS — IC VALIDATION (leak-free shift(1), vs 15-min fwd)")
    print("=" * 100)
    print(f"  {'feature':<16}{'Pear':>8}{'Spear':>8}{'p':>8}{'q(BH)':>8}"
          f"{'CI_lo':>9}{'CI_hi':>9}{'fold+':>7}{'cap%':>7}  verdict")
    print("  " + "-" * 96)
    npass = 0
    for _, r in res.iterrows():
        ci_excl0 = (r.ci_lo > 0) or (r.ci_hi < 0)
        ok = (r.q_bh < 0.01) and bool(r.clears) and ci_excl0
        npass += ok
        print(f"  {r.feat:<16}{r.pearson:>+8.3f}{r.spear:>+8.3f}{r.p:>8.3f}"
              f"{r.q_bh:>8.3f}{r.ci_lo:>+9.3f}{r.ci_hi:>+9.3f}{r.stab:>7}"
              f"{r.captured:>7.3f}  {'PASS' if ok else 'fail'}")
    print("=" * 100)
    print(f"  Acceptance: BH q<0.01 AND captured>{FRICTION_FLOOR_PCT}% AND bootstrap CI excludes 0")
    print(f"  FEATURE GROUP VERDICT: {'PASS' if npass else 'FAIL'} ({npass}/{len(res)} features pass)")
    res.to_csv(ROOT / "logs/futidx_basis_ic.csv", index=False)


if __name__ == "__main__":
    main()
