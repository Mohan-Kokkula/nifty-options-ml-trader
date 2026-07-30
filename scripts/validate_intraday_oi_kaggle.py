"""
validate_intraday_oi_kaggle.py — PRELIMINARY intraday-OI IC validation
=======================================================================
Built 2026-06-16. Uses the Kaggle intraday 1-min NSE F&O dump
(data/kaggle/archive*/NSE_FNO_DATA_*.csv) which carries REAL per-contract
NIFTY option OI at 1-minute cadence — the one piece of intraday-OI history
obtainable without forward archiving.

Pipeline (leak-safe, causal):
  1. Load all NSE_FNO_DATA_*.csv, dedup by (Date,Time,Ticker).
  2. Filter NIFTY *index* options, parse strike / CE-PE / expiry; keep nearest
     expiry per day.
  3. Per minute: ATM = strike nearest spot (spot from data/nifty_5min.csv,
     ffilled to 1-min); build the oi_archiver ATM schema
     (timestamp,spot,atm_strike,call_oi,put_oi,pcr).
  4. compute_features() -> pcr_velocity, atm_ce/pe_oi_chg_pct,
     oi_buildup_signed, net_oi_flow.
  5. Forward 15-min return target from spot; Spearman/Pearson IC, BH-FDR,
     friction, bootstrap CI.

PRELIMINARY: ~12 trading days only — too thin for the full >=5/8-fold gate.
A pass here is a green light for the forward archive, not a deployment verdict.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "data" / "kaggle"
NIFTY5 = ROOT / "data" / "nifty_5min.csv"
FWD_MIN = 15
FRICTION_FLOOR_PCT = 0.12
_TICK = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


def load_oi() -> pd.DataFrame:
    files = sorted(set(KAGGLE.glob("**/NSE_FNO_DATA_*.csv"))
                   | set(KAGGLE.glob("**/NSE_FNO_DATA_*.CSV"))
                   | set(KAGGLE.glob("**/NSE_FNO_*.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c.strip() in
                             ("Ticker", "Date", "Time", "Close", "Open Interest"))
        except Exception:
            df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        if "Ticker" not in df.columns or "Open Interest" not in df.columns:
            continue
        df = df[df["Ticker"].astype(str).str.startswith("NIFTY")]
        frames.append(df)
    if not frames:
        raise RuntimeError("No NSE_FNO_DATA files with OI found under data/kaggle")
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Date", "Time", "Ticker"])
    # parse ticker
    base = raw["Ticker"].str.replace(".NFO", "", regex=False)
    m = base.str.extract(_TICK)
    raw = raw.assign(expiry=m[0], strike=pd.to_numeric(m[1], errors="coerce"),
                     opt=m[2])
    raw = raw.dropna(subset=["strike", "opt"])
    dt = pd.to_datetime(raw["Date"] + " " + raw["Time"],
                        format="%d/%m/%Y %H:%M:%S", errors="coerce")
    raw = raw.assign(ts=dt).dropna(subset=["ts"])
    raw["oi"] = pd.to_numeric(raw["Open Interest"], errors="coerce").fillna(0)
    # nearest expiry per day
    raw["day"] = raw["ts"].dt.normalize()
    raw["exp_dt"] = pd.to_datetime(raw["expiry"], format="%d%b%y", errors="coerce")
    near = raw.groupby("day")["exp_dt"].transform("min")
    raw = raw[raw["exp_dt"] == near]
    return raw[["ts", "day", "strike", "opt", "oi"]]


def spot_series() -> pd.Series:
    s = pd.read_csv(NIFTY5)
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index("date")["close"].sort_index()
    return s


def build_atm_series(oi: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    # per-minute spot (ffill 5-min onto 1-min grid of the OI timestamps)
    oi = oi.sort_values("ts")
    spot_1m = spot.reindex(spot.index.union(oi["ts"].unique())).sort_index().ffill()
    oi["spot"] = oi["ts"].map(spot_1m)
    oi = oi.dropna(subset=["spot"])
    rows = []
    for ts, g in oi.groupby("ts"):
        sp = float(g["spot"].iloc[0])
        atm = int(round(sp / 50) * 50)
        ce_tot = g.loc[g["opt"] == "CE", "oi"].sum()
        pe_tot = g.loc[g["opt"] == "PE", "oi"].sum()
        atm_ce = g[(g["opt"] == "CE") & (g["strike"] == atm)]["oi"].sum()
        atm_pe = g[(g["opt"] == "PE") & (g["strike"] == atm)]["oi"].sum()
        rows.append({"timestamp": ts, "spot": sp, "atm_strike": atm,
                     "call_oi": int(atm_ce), "put_oi": int(atm_pe),
                     "call_iv": 0.0, "put_iv": 0.0,
                     "pcr": round(pe_tot / max(ce_tot, 1), 4)})
    return pd.DataFrame(rows)


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank; prev = min(prev, p[idx] * n / i); q[idx] = prev
    return q


def main():
    print("Loading Kaggle intraday OI...")
    oi = load_oi()
    print(f"  NIFTY option rows: {len(oi):,} | days: {oi['day'].nunique()} "
          f"({oi['day'].min().date()}..{oi['day'].max().date()})")
    spot = spot_series()
    atm = build_atm_series(oi, spot)
    print(f"  ATM 1-min series: {len(atm):,} snapshots")

    sys.path.insert(0, str(ROOT))
    from scripts.oi_features import compute_features
    feats = compute_features(atm).reset_index()
    feats = feats.merge(atm[["timestamp", "spot"]], left_on="timestamp",
                        right_on="timestamp", how="left")
    feats = feats.sort_values("timestamp").set_index("timestamp")

    # forward 15-min return from spot (causal: feature at t -> ret t..t+15m)
    fwd = feats["spot"].shift(-FWD_MIN) / feats["spot"] - 1
    # only compare within the same day (no overnight gap leakage)
    same_day = pd.Series(feats.index.normalize(), index=feats.index)
    fwd = fwd.where(same_day.values == pd.Series(feats.index, index=feats.index)
                    .shift(-FWD_MIN).dt.normalize().values)
    y = fwd.values
    fwd_std = np.nanstd(y)
    print(f"  fwd-15m std={fwd_std*100:.3f}%\n")

    cols = ["pcr_velocity", "atm_ce_oi_chg_pct", "atm_pe_oi_chg_pct",
            "oi_buildup_signed", "net_oi_flow"]
    rows, pvals = [], []
    rng = np.random.default_rng(5)
    for c in cols:
        x = feats[c].values
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 200:
            rows.append(dict(feat=c, pearson=np.nan, spear=np.nan, p=1.0,
                             ci_lo=np.nan, ci_hi=np.nan, captured=0.0, n=int(m.sum())))
            pvals.append(1.0); continue
        pr, _ = stats.pearsonr(x[m], y[m])
        sr, sp = stats.spearmanr(x[m], y[m])
        xm, ym = x[m], y[m]
        bs = [stats.spearmanr(xm[i], ym[i])[0]
              for i in (rng.integers(0, len(xm), len(xm)) for _ in range(1000))]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(feat=c, pearson=pr, spear=sr, p=sp, ci_lo=lo, ci_hi=hi,
                         captured=abs(sr) * fwd_std * 100, n=int(m.sum())))
        pvals.append(sp)
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p"].fillna(1.0).values)

    print("=" * 96)
    print("  INTRADAY-OI PRELIMINARY IC  (Kaggle ~12 days, leak-safe, vs 15-min fwd)")
    print("=" * 96)
    print(f"  {'feature':<20}{'n':>8}{'Pear':>8}{'Spear':>8}{'p':>8}{'q(BH)':>8}"
          f"{'CI_lo':>9}{'CI_hi':>9}{'cap%':>7}  verdict")
    print("  " + "-" * 92)
    npass = 0
    for _, r in res.iterrows():
        ci0 = (r.ci_lo > 0) or (r.ci_hi < 0)
        ok = (r.q_bh < 0.01) and (r.captured > FRICTION_FLOOR_PCT) and ci0
        npass += ok
        print(f"  {r.feat:<20}{r.n:>8}{r.pearson:>+8.3f}{r.spear:>+8.3f}{r.p:>8.3f}"
              f"{r.q_bh:>8.3f}{r.ci_lo:>+9.3f}{r.ci_hi:>+9.3f}{r.captured:>7.3f}  "
              f"{'PASS' if ok else 'fail'}")
    print("=" * 96)
    print(f"  PRELIMINARY VERDICT: {'SIGNAL — justify forward archive' if npass else 'NO SIGNAL'} "
          f"({npass}/{len(res)}). NOTE: ~12 days only; not the full >=5/8-fold gate.")
    res.to_csv(ROOT / "logs/intraday_oi_kaggle_ic.csv", index=False)


if __name__ == "__main__":
    main()
