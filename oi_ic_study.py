"""
oi_ic_study.py — Daily OI/PCR Information-Coefficient study (real bhavcopy data)
================================================================================
Built 2026-06-16 for the OI/PCR predictive-information review.

DATA REALITY (decided before writing this):
  * Intraday 1-min OI archive (data/openalgo_oi/) is EMPTY — the genuinely-new
    intraday OI signal cannot be tested yet (archiver started ~Jun 12, well
    short of the pre-registered 8-12 weeks).  The only prior intraday
    "diagnostics" were SYNTHETIC (reports/oi_feature_diagnostics.json).
  * The ONLY OI/PCR data with real sample is daily NSE bhavcopy:
    data/nse_bhavcopy/nifty_pcr_daily.csv  (~495 trading days, 2024-05..2026-06).
    This study therefore measures DAILY OI/PCR predictive content only.

METHOD
  * Single index (NIFTY) => no cross-section => classic cross-sectional IC
    degenerates to a TIME-SERIES correlation of feature_t vs forward return.
    We report it as such and add monthly-IC stability + rolling IC.
  * Target: next-day return (EOD feature at day t predicts t -> t+1), causal.
  * Pearson + Spearman IC, full-sample p-values, monthly-IC IR (mean/std of
    per-month IC), 60-day rolling IC, walk-forward OOS IC (5 folds, 1-day
    embargo), Benjamini-Hochberg FDR correction across all features x targets.
  * Friction translation: implied directional tilt captured = IC * std(fwd_ret),
    compared to a NIFTY-options round-trip friction floor.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

PCR = "data/nse_bhavcopy/nifty_pcr_daily.csv"
DAY = "data/nifty_day.csv"

# ---------------------------------------------------------------- load + build
def load():
    p = pd.read_csv(PCR, parse_dates=["date", "expiry"])
    p = p[p["expiry"] >= p["date"]]                      # upcoming expiries only
    # near-expiry row per day = smallest expiry >= date
    near = p.sort_values("expiry").groupby("date", as_index=False).first()
    near = near.rename(columns={"ce_oi": "ce_near", "pe_oi": "pe_near",
                                "pcr_oi": "pcr_near"})
    # all-expiry aggregate per day (smoother totals)
    agg = p.groupby("date").agg(ce_all=("ce_oi", "sum"),
                                pe_all=("pe_oi", "sum")).reset_index()
    agg["pcr_all"] = agg["pe_all"] / agg["ce_all"].clip(lower=1)
    d = near.merge(agg, on="date").sort_values("date").reset_index(drop=True)
    d["roll"] = (d["expiry"] != d["expiry"].shift(1))     # expiry rolled today

    # NIFTY daily close
    day = pd.read_csv(DAY)
    day["date"] = pd.to_datetime(day["date"]).dt.normalize()
    day = day[["date", "open", "close"]].dropna().drop_duplicates("date")
    d["date"] = d["date"].dt.normalize()
    d = d.merge(day, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return d


def build_features(d):
    f = pd.DataFrame({"date": d["date"]})
    px = d["close"]
    ret = px.pct_change()
    # masked changes: NaN on expiry-roll days (OI delta invalid across re-reference)
    dpcr_near = d["pcr_near"].diff().where(~d["roll"])
    # ---- features A..H ----
    f["A_pcr_level"]    = d["pcr_near"]                                 # static / level
    f["B_pcr_change"]   = d["pcr_all"].diff()                           # rate-of-change
    f["C_pcr_accel"]    = d["pcr_all"].diff().diff()                    # acceleration
    f["D_ce_buildup"]   = d["ce_all"].pct_change()                      # CE OI build-up
    f["E_pe_buildup"]   = d["pe_all"].pct_change()                      # PE OI build-up
    f["F_oi_imbalance"] = (d["pe_all"] - d["ce_all"]) / (d["pe_all"] + d["ce_all"])  # imbalance
    # G divergence: price & OI-imbalance moving oppositely (z-scored product, negated)
    dimb = f["F_oi_imbalance"].diff()
    zr = (ret - ret.mean()) / ret.std()
    zi = (dimb - dimb.mean()) / dimb.std()
    f["G_oi_divergence"] = -(zr * zi)                                   # divergence
    # H trend persistence: 3-day momentum of near PCR (sign-consistent drift)
    f["H_oi_persistence"] = d["pcr_near"] - d["pcr_near"].shift(3)      # regime/trend
    # ---- targets (forward, causal: known at t, realized t+1) ----
    f["fwd_cc"] = px.shift(-1) / px - 1                                 # next-day close-close
    f["fwd_oc"] = d["close"].shift(-1) / d["open"].shift(-1) - 1        # next-day open->close
    return f


FEATS = ["A_pcr_level", "B_pcr_change", "C_pcr_accel", "D_ce_buildup",
         "E_pe_buildup", "F_oi_imbalance", "G_oi_divergence", "H_oi_persistence"]
KIND = {"A_pcr_level": "static/level", "B_pcr_change": "rate-of-change",
        "C_pcr_accel": "acceleration", "D_ce_buildup": "rate-of-change",
        "E_pe_buildup": "rate-of-change", "F_oi_imbalance": "static/level",
        "G_oi_divergence": "divergence", "H_oi_persistence": "regime/trend"}


def ic(x, y):
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 30:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp, len(x)


def bh_fdr(pvals):
    """Benjamini-Hochberg; return q-values aligned to input order."""
    p = np.array(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        val = min(prev, p[idx] * n / i)
        q[idx] = val
        prev = val
    return q


def walk_forward_ic(f, feat, tgt, folds=5, embargo=1):
    df = f[[feat, tgt]].dropna().reset_index(drop=True)
    n = len(df)
    bnds = np.linspace(0, n, folds + 1).astype(int)
    ics = []
    for k in range(folds):
        lo, hi = bnds[k], bnds[k + 1]
        seg = df.iloc[lo + embargo: hi]   # embargo at fold start
        if len(seg) >= 25:
            r, _ = stats.spearmanr(seg[feat], seg[tgt])
            ics.append(r)
    return np.array(ics)


def main():
    d = load()
    f = build_features(d)
    print(f"Daily OI/PCR IC study | {len(f)} trading days | "
          f"{f.date.min().date()} -> {f.date.max().date()}")
    fwd_std = f["fwd_cc"].std()
    print(f"Next-day close-close return std = {fwd_std*100:.2f}%/day\n")

    rows = []
    pvals = []
    for tgt in ["fwd_cc", "fwd_oc"]:
        for feat in FEATS:
            pr, pp, sr, sp, nn = ic(f[feat], f[tgt])
            # monthly IC stability
            tmp = f[[feat, tgt]].copy()
            tmp["m"] = f["date"].dt.to_period("M")
            mics = []
            for _, g in tmp.dropna().groupby("m"):
                if len(g) >= 8:
                    mics.append(stats.spearmanr(g[feat], g[tgt])[0])
            mics = np.array([x for x in mics if not np.isnan(x)])
            ic_ir = mics.mean() / mics.std() if len(mics) > 2 and mics.std() > 0 else np.nan
            pos_mo = (mics > 0).mean() * 100 if len(mics) else np.nan
            # walk-forward OOS
            wf = walk_forward_ic(f, feat, tgt)
            wf_mean = np.nanmean(wf) if len(wf) else np.nan
            wf_pos = (wf > 0).mean() * 100 if len(wf) else np.nan
            rows.append(dict(target=tgt, feat=feat, kind=KIND[feat], n=nn,
                             pearson=pr, p_pearson=pp, spearman=sr, p_spear=sp,
                             ic_ir=ic_ir, pos_mo=pos_mo,
                             wf_ic=wf_mean, wf_pos=wf_pos))
            pvals.append(sp)
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p_spear"].values)

    for tgt in ["fwd_cc", "fwd_oc"]:
        sub = res[res.target == tgt].copy()
        tname = "next-day CLOSE->CLOSE" if tgt == "fwd_cc" else "next-day OPEN->CLOSE"
        print("=" * 104)
        print(f"  TARGET: {tname}")
        print("=" * 104)
        print(f"  {'Feature':<17}{'kind':<15}{'Pear':>7}{'Spear':>7}"
              f"{'p':>8}{'q(BH)':>8}{'IC_IR':>7}{'+mo%':>6}{'wfIC':>7}{'wf+%':>6}  sig")
        print("  " + "-" * 100)
        for _, r in sub.iterrows():
            sig = "***" if r.q_bh < 0.01 else ("**" if r.q_bh < 0.05 else
                  ("*" if r.q_bh < 0.10 else ""))
            print(f"  {r.feat:<17}{r.kind:<15}{r.pearson:>+7.3f}{r.spearman:>+7.3f}"
                  f"{r.p_spear:>8.3f}{r.q_bh:>8.3f}"
                  f"{(r.ic_ir if pd.notna(r.ic_ir) else 0):>+7.2f}"
                  f"{(r.pos_mo if pd.notna(r.pos_mo) else 0):>6.0f}"
                  f"{(r.wf_ic if pd.notna(r.wf_ic) else 0):>+7.3f}"
                  f"{(r.wf_pos if pd.notna(r.wf_pos) else 0):>6.0f}  {sig}")
        print()

    # friction translation (primary target)
    print("=" * 104)
    print("  FRICTION CHECK (next-day close->close)")
    print("=" * 104)
    print("  Captured directional tilt = |Spearman IC| x daily-return std.")
    print("  NIFTY options round-trip friction floor ~ 0.30-0.50% of UNDERLYING-equivalent")
    print("  (ATM premium spread + theta over a 1-day hold + STT/brokerage/GST).")
    sub = res[res.target == "fwd_cc"]
    for _, r in sub.iterrows():
        captured = abs(r.spearman) * fwd_std * 100
        verdict = "clears" if captured > 0.40 else "BELOW friction"
        print(f"    {r.feat:<17} captured ~{captured:5.3f}% / day   -> {verdict}")
    print("\n  *** q<0.01  ** q<0.05  * q<0.10  (Benjamini-Hochberg FDR across 16 tests)")
    res.to_csv("logs/oi_ic_results.csv", index=False)
    print("  Full results -> logs/oi_ic_results.csv")


if __name__ == "__main__":
    main()
