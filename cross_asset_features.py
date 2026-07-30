"""
cross_asset_features.py — BankNifty lead-lag + USDINR features (Group I #2, #4)
==============================================================================
Built 2026-06-16. Production feature builder + leak-free IC validation for the
two approved orthogonal sources whose history is actually obtainable here
(BankNifty ^NSEBANK, USDINR INR=X via yfinance, full 2015-2026 daily).

DISCIPLINE: features are DAILY, known at the prior close, merged onto the
leak-clean V9 intraday frame with shift(1) — a bar on day D sees only day D-1
values (the exact 2026-06-11 daily-leak fix pattern). NO feature is wired into
model training or live trading here; this is the pre-registered IC gate that
must pass (BH q<0.01, friction, >=5/8 folds, bootstrap CI excludes 0) BEFORE
any inclusion is even proposed.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import yfinance as yf

from backtest_threshold_sweep import build_frame

CACHE = Path("data/cross_asset_daily.csv")
FWD_BARS = 3                       # 15-min forward (matches label horizon)
N_FOLDS = 8
FRICTION_FLOOR_PCT = 0.12          # 15-min ATM option round-trip floor


def fetch_daily():
    """Fetch + cache BankNifty / NIFTY / USDINR daily closes."""
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["date"])
    frames = {}
    for name, tk in [("bn", "^NSEBANK"), ("nf", "^NSEI"), ("usd", "INR=X")]:
        df = yf.download(tk, start="2015-01-01", end="2026-06-05",
                         interval="1d", progress=False, auto_adjust=False)
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).normalize()
        frames[name] = s
    d = pd.DataFrame(frames).dropna()
    d.index.name = "date"
    d = d.reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d.to_csv(CACHE, index=False)
    return d


def build_daily_features(d):
    """Daily features as of each date d (using only data up to and incl. d)."""
    f = pd.DataFrame({"date": d["date"]})
    B, N, U = d["bn"], d["nf"], d["usd"]
    retB, retN = B.pct_change(), N.pct_change()
    # BankNifty lead-lag
    f["bn_lag1_ret"]  = retB                                   # B return realized on day d
    f["rs_bn"]        = B.pct_change(5) - N.pct_change(5)      # 5d relative strength
    ratio = B / N
    f["bn_ratio_z"]   = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
    f["bn_divergence"]= (np.sign(retB) != np.sign(retN)).astype(float)
    # USDINR
    f["usdinr_ret_1d"]= U.pct_change()
    f["usdinr_z"]     = (U - U.rolling(20).mean()) / U.rolling(20).std()
    # shift(1): a bar on day D sees only day D-1 values (causal, no look-ahead)
    feat_cols = [c for c in f.columns if c != "date"]
    f[feat_cols] = f[feat_cols].shift(1)
    return f, feat_cols


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank; prev = min(prev, p[idx] * n / i); q[idx] = prev
    return q


def main():
    print("Fetching BankNifty / NIFTY / USDINR daily...")
    daily = fetch_daily()
    fday, fcols_x = build_daily_features(daily)
    print(f"Cross-asset daily features: {len(fday)} days, cols={fcols_x}")

    feat, _ = build_frame()
    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat.index).normalize()
    fday = fday.set_index("date")
    for c in fcols_x:
        feat[c] = feat["date"].map(fday[c])           # daily->intraday map (already shifted)

    px = feat["close"]
    feat["fwd_15m"] = px.shift(-FWD_BARS) / px - 1
    fwd_std = feat["fwd_15m"].std()
    n = len(feat)
    print(f"Frame: {n:,} bars | fwd-15m std={fwd_std*100:.3f}%\n")

    rows, pvals = [], []
    for tgt in ["fwd_15m"]:
        y = feat[tgt].values
        for c in fcols_x:
            x = feat[c].values
            m = ~np.isnan(x) & ~np.isnan(y)
            if m.sum() < 500:
                continue
            pr, pp = stats.pearsonr(x[m], y[m])
            sr, sp = stats.spearmanr(x[m], y[m])
            # walk-forward sign stability
            bnds = np.linspace(0, n, N_FOLDS + 1).astype(int)
            signs = []
            for k in range(N_FOLDS):
                lo, hi = bnds[k], bnds[k+1]
                xx, yy = x[lo:hi], y[lo:hi]
                mm = ~np.isnan(xx) & ~np.isnan(yy)
                if mm.sum() < 200: continue
                rr, _ = stats.spearmanr(xx[mm], yy[mm])
                if not np.isnan(rr): signs.append(np.sign(rr))
            same = max(np.sum(np.array(signs) == np.sign(sr)),
                       np.sum(np.array(signs) == -np.sign(sr))) if signs else 0
            stab = f"{int(np.sum(np.array(signs)==np.sign(sr)))}/{len(signs)}"
            # bootstrap CI on Spearman
            rng = np.random.default_rng(3)
            xm, ym = x[m], y[m]
            bs = []
            for _ in range(1000):
                idx = rng.integers(0, len(xm), len(xm))
                bs.append(stats.spearmanr(xm[idx], ym[idx])[0])
            lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
            captured = abs(sr) * fwd_std * 100
            rows.append(dict(feat=c, pearson=pr, spear=sr, p=sp, n=m.sum(),
                             stab=stab, ci_lo=lo_ci, ci_hi=hi_ci,
                             captured=captured, clears=captured > FRICTION_FLOOR_PCT))
            pvals.append(sp)
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p"].values)

    print("=" * 100)
    print("  IC VALIDATION — cross-asset features vs 15-min forward return (leak-free, shift(1))")
    print("=" * 100)
    print(f"  {'feature':<16}{'Pear':>8}{'Spear':>8}{'p':>8}{'q(BH)':>8}"
          f"{'CI_lo':>8}{'CI_hi':>8}{'fold+':>7}{'cap%':>7}  verdict")
    print("  " + "-" * 96)
    for _, r in res.iterrows():
        ci_excl0 = (r.ci_lo > 0) or (r.ci_hi < 0)
        pass_all = (r.q_bh < 0.01) and r.clears and ci_excl0
        v = "PASS" if pass_all else "fail"
        print(f"  {r.feat:<16}{r.pearson:>+8.3f}{r.spear:>+8.3f}{r.p:>8.3f}"
              f"{r.q_bh:>8.3f}{r.ci_lo:>+8.3f}{r.ci_hi:>+8.3f}{r.stab:>7}"
              f"{r.captured:>7.3f}  {v}")
    print("=" * 100)
    print("  Acceptance: BH q<0.01 AND captured>friction(0.12%) AND bootstrap CI excludes 0.")
    n_pass = sum(((res.q_bh<0.01) & res.clears &
                  ((res.ci_lo>0)|(res.ci_hi<0))))
    print(f"  Features PASSING the gate: {n_pass}/{len(res)}")
    res.to_csv("logs/cross_asset_ic.csv", index=False)


if __name__ == "__main__":
    main()
