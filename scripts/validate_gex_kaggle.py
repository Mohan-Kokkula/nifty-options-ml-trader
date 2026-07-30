"""
validate_gex_kaggle.py — PRELIMINARY dealer-GEX IC validation (Kaggle 13 days)
==============================================================================
Built 2026-06-16. Derives IV from the Kaggle intraday option Close prices via
Black-Scholes (ATM IV per minute), computes gamma-weighted net dealer exposure
across near-ATM strikes, and tests GEX features both directionally (the gate)
and vs forward realized range (GEX's actual claim = vol suppression).

GEX_t = Σ_{|K-ATM|<=band} γ(S,K,T,atmIV) · S² · 0.01 · (OI_CE(K) - OI_PE(K))
zero-gamma proxy: gamma-weighted net-OI centroid strike; spot_minus_zg = (S-zg)/S

Leak-safe: feature at minute t, forward return t..t+15m (same-day only).
PRELIMINARY: ~13 days only — not the full >=5/8-fold gate.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import sys
from math import log, sqrt, exp, erf, pi
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "data" / "kaggle"
NIFTY5 = ROOT / "data" / "nifty_5min.csv"
R, Q = 0.065, 0.012
FWD_MIN = 15
BAND = 12 * 50          # +/- 12 strikes around ATM (gamma negligible beyond)
FRICTION_FLOOR_PCT = 0.12
_TICK = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


def _norm_pdf(x): return exp(-0.5 * x * x) / sqrt(2 * pi)
def _norm_cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))


def _d1(S, K, T, sig):
    if S <= 0 or K <= 0 or T <= 0 or sig <= 0:
        return 0.0
    return (log(S / K) + (R - Q + 0.5 * sig * sig) * T) / (sig * sqrt(T))


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, sig); d2 = d1 - sig * sqrt(T)
    return S * exp(-Q * T) * _norm_cdf(d1) - K * exp(-R * T) * _norm_cdf(d2)


def bs_gamma(S, K, T, sig):
    if S <= 0 or T <= 0 or sig <= 0:
        return 0.0
    d1 = _d1(S, K, T, sig)
    return exp(-Q * T) * _norm_pdf(d1) / (S * sig * sqrt(T))


def implied_vol_call(price, S, K, T):
    """Bisection IV from a call price; returns NaN if unsolvable."""
    if price <= 0 or T <= 0 or price >= S:
        return np.nan
    lo, hi = 0.01, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def load_oi_close() -> pd.DataFrame:
    files = sorted(set(KAGGLE.glob("**/NSE_FNO_DATA_*.csv"))
                   | set(KAGGLE.glob("**/NSE_FNO_DATA_*.CSV"))
                   | set(KAGGLE.glob("**/NSE_FNO_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        if not {"Ticker", "Open Interest", "Close"} <= set(df.columns):
            continue
        df = df[df["Ticker"].astype(str).str.startswith("NIFTY")]
        frames.append(df[["Ticker", "Date", "Time", "Close", "Open Interest"]])
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Time", "Ticker"])
    base = raw["Ticker"].str.replace(".NFO", "", regex=False)
    m = base.str.extract(_TICK)
    raw = raw.assign(expiry=m[0], strike=pd.to_numeric(m[1], errors="coerce"), opt=m[2])
    raw = raw.dropna(subset=["strike", "opt"])
    raw["ts"] = pd.to_datetime(raw["Date"] + " " + raw["Time"],
                               format="%d/%m/%Y %H:%M:%S", errors="coerce")
    raw = raw.dropna(subset=["ts"])
    raw["oi"] = pd.to_numeric(raw["Open Interest"], errors="coerce").fillna(0)
    raw["close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw["day"] = raw["ts"].dt.normalize()
    raw["exp_dt"] = pd.to_datetime(raw["expiry"], format="%d%b%y", errors="coerce")
    near = raw.groupby("day")["exp_dt"].transform("min")
    raw = raw[raw["exp_dt"] == near]
    return raw[["ts", "day", "strike", "opt", "oi", "close", "exp_dt"]]


def main():
    print("Loading Kaggle intraday OI+prices...")
    oi = load_oi_close()
    s = pd.read_csv(NIFTY5); s["date"] = pd.to_datetime(s["date"])
    spot = s.set_index("date")["close"].sort_index()
    spot_1m = spot.reindex(spot.index.union(oi["ts"].unique())).sort_index().ffill()
    oi["spot"] = oi["ts"].map(spot_1m)
    oi = oi.dropna(subset=["spot"])
    print(f"  rows={len(oi):,} days={oi['day'].nunique()} "
          f"({oi['day'].min().date()}..{oi['day'].max().date()})")

    recs = []
    for ts, g in oi.groupby("ts"):
        S = float(g["spot"].iloc[0])
        atm = round(S / 50) * 50
        exp_dt = g["exp_dt"].iloc[0]
        T = max((exp_dt + pd.Timedelta(hours=15, minutes=30) - ts).total_seconds()
                / (365.25 * 24 * 3600), 1e-6)
        # ATM IV from the ATM call price
        atm_ce = g[(g["opt"] == "CE") & (g["strike"] == atm)]["close"]
        if atm_ce.empty or atm_ce.iloc[0] <= 0:
            continue
        iv = implied_vol_call(float(atm_ce.iloc[0]), S, atm, T)
        if not (iv and 0.02 < iv < 2.5):
            continue
        band = g[(g["strike"] >= atm - BAND) & (g["strike"] <= atm + BAND)]
        ce = band[band["opt"] == "CE"].set_index("strike")["oi"]
        pe = band[band["opt"] == "PE"].set_index("strike")["oi"]
        strikes = sorted(set(ce.index) | set(pe.index))
        gex = 0.0; cum = 0.0; wsum = 0.0
        for K in strikes:
            net = float(ce.get(K, 0)) - float(pe.get(K, 0))
            gK = bs_gamma(S, K, T, iv)
            gex += gK * S * S * 0.01 * net
            wsum += abs(gK * net); cum += K * abs(gK * net)
        zg = cum / wsum if wsum > 0 else S
        recs.append({"ts": ts, "spot": S, "gex": gex,
                     "gex_norm": gex / wsum if wsum > 0 else 0.0,
                     "spot_minus_zg": (S - zg) / S, "atm_iv": iv})
    df = pd.DataFrame(recs).sort_values("ts").set_index("ts")
    print(f"  GEX snapshots: {len(df):,} | median ATM IV={df['atm_iv'].median()*100:.1f}%")

    same = pd.Series(df.index, index=df.index)
    fwd = df["spot"].shift(-FWD_MIN) / df["spot"] - 1
    fwd = fwd.where(same.dt.normalize().values ==
                    same.shift(-FWD_MIN).dt.normalize().values)
    y_dir = fwd.values
    y_rng = np.abs(fwd.values)                     # forward realized range (GEX's claim)
    dstd = np.nanstd(y_dir)

    def bh(p):
        p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); pr = 1.0
        for r, i in enumerate(reversed(o)):
            k = n - r; pr = min(pr, p[i] * n / k); q[i] = pr
        return q

    feats = ["gex", "gex_norm", "spot_minus_zg"]
    rng_state = np.random.default_rng(9)
    print("\n" + "=" * 96)
    print("  DEALER-GEX PRELIMINARY IC (Kaggle ~13 days, IV from option prices)")
    print("=" * 96)
    print(f"  {'feature':<15}{'n':>7}{'dir_Spear':>11}{'dir_p':>8}{'dir_q':>8}"
          f"{'dir_cap%':>9}{'RANGE_Spear':>13}{'range_p':>9}  dir_verdict")
    print("  " + "-" * 92)
    pvals, results = [], []
    for c in feats:
        x = df[c].values
        m = ~np.isnan(x) & ~np.isnan(y_dir)
        if m.sum() < 200:
            results.append((c, m.sum(), np.nan, 1.0, 0.0, np.nan, 1.0)); pvals.append(1.0); continue
        sr, sp = stats.spearmanr(x[m], y_dir[m])
        mr = ~np.isnan(x) & ~np.isnan(y_rng)
        rr, rp = stats.spearmanr(x[mr], y_rng[mr])
        results.append((c, int(m.sum()), sr, sp, abs(sr) * dstd * 100, rr, rp))
        pvals.append(sp)
    q = bh(pvals)
    npass = 0
    for (c, n, sr, sp, cap, rr, rp), qq in zip(results, q):
        ok = (qq < 0.01) and (cap > FRICTION_FLOOR_PCT)
        npass += ok
        print(f"  {c:<15}{n:>7}{sr:>+11.3f}{sp:>8.3f}{qq:>8.3f}{cap:>9.3f}"
              f"{rr:>+13.3f}{rp:>9.3f}  {'PASS' if ok else 'fail'}")
    print("=" * 96)
    print(f"  DIRECTIONAL VERDICT: {'SIGNAL' if npass else 'NO SIGNAL'} ({npass}/{len(feats)}).")
    print("  (RANGE columns = GEX vs forward |return|, its real vol-suppression claim — "
          "informational; the trading gate is directional.)")
    print("  NOTE: ~13 days only; preliminary, not the full >=5/8-fold gate.")
    df.to_csv(ROOT / "logs/gex_kaggle_ic.csv")


if __name__ == "__main__":
    main()
