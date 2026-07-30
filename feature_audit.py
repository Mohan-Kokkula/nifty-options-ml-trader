"""
feature_audit.py — full leak-clean feature audit for OpenClaw
=============================================================
Built 2026-06-16. For every model feature on the leak-fixed V9 frame:
  1. Information Coefficient  (Spearman of feature vs FORWARD return; the
     continuous target the label is built from), + BH-FDR q-value.
  2. Mutual information       (vs the 3-class label, subsampled).
  3. Importance               (XGBoost gain — SHAP proxy; fast + stable rank).
  4. Walk-forward stability   (sign-consistency of IC across 6 time folds).
  5. Friction-adjusted value  (|IC| x fwd-return std => captured tilt vs a
     15-min option round-trip friction floor).
Classify each: A strong / B weak / C redundant / D noise.

No model/threshold/architecture changes — diagnostics only.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

from backtest_threshold_sweep import build_frame   # reuses leak-clean V9 build

FWD_BARS = 3                      # 15-min forward (matches label horizon)
N_FOLDS  = 6
CORR_REDUNDANT = 0.95
# 15-min ATM option round-trip friction as fraction of underlying-equivalent
# (spread dominates at this horizon; theta small). Conservative floor.
FRICTION_FLOOR_PCT = 0.12


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        prev = min(prev, p[idx] * n / i); q[idx] = prev
    return q


def main():
    feat, fcols = build_frame()
    px = feat["close"]
    fwd = (px.shift(-FWD_BARS) / px - 1)
    valid = fwd.notna()
    F = feat.loc[valid, fcols]
    y_fwd = fwd[valid].values
    y_lbl = feat.loc[valid, "label"].values
    fwd_std = np.nanstd(y_fwd)
    n = len(F)
    print(f"Audit frame: {n:,} rows x {len(fcols)} features | fwd-15m std={fwd_std*100:.3f}%\n")

    # ---- 1. IC (Spearman vs forward return) ----
    ic, pval = {}, {}
    fvals = F.values
    for j, c in enumerate(fcols):
        col = fvals[:, j]
        m = ~np.isnan(col)
        if m.sum() < 200 or np.nanstd(col) == 0:
            ic[c], pval[c] = 0.0, 1.0; continue
        r, p = stats.spearmanr(col[m], y_fwd[m])
        ic[c] = 0.0 if np.isnan(r) else r
        pval[c] = 1.0 if np.isnan(p) else p
    q = bh_fdr([pval[c] for c in fcols])
    qmap = dict(zip(fcols, q))

    # ---- 2. Mutual information (vs label, subsampled) ----
    sub = np.random.default_rng(42).choice(n, min(30000, n), replace=False)
    Xs = np.nan_to_num(fvals[sub])
    mi = mutual_info_classif(Xs, y_lbl[sub], discrete_features=False, random_state=42)
    mimap = dict(zip(fcols, mi))

    # ---- 3. Importance (XGB gain) ----
    Xtr = np.nan_to_num(fvals)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                            subsample=0.7, colsample_bytree=0.5, num_class=3,
                            objective="multi:softprob", n_jobs=-1, verbosity=0)
    clf.fit(Xtr, y_lbl)
    imp = dict(zip(fcols, clf.feature_importances_))

    # ---- 4. Walk-forward IC sign stability ----
    bnds = np.linspace(0, n, N_FOLDS + 1).astype(int)
    stab = {}
    for c in fcols:
        col = fvals[:, list(fcols).index(c)]
        full_sign = np.sign(ic[c])
        agree = 0; tot = 0
        for k in range(N_FOLDS):
            lo, hi = bnds[k], bnds[k + 1]
            cc, yy = col[lo:hi], y_fwd[lo:hi]
            mm = ~np.isnan(cc)
            if mm.sum() < 50: continue
            r, _ = stats.spearmanr(cc[mm], yy[mm])
            if not np.isnan(r):
                tot += 1; agree += (np.sign(r) == full_sign)
        stab[c] = agree / tot if tot else 0.0

    # ---- redundancy: corr > 0.95 with a higher-|IC| feature ----
    rank = sorted(fcols, key=lambda c: -abs(ic[c]))
    Fc = pd.DataFrame(np.nan_to_num(fvals), columns=fcols)
    corr = Fc.corr().abs()
    redundant = {}
    kept = []
    for c in rank:
        red = any(corr.loc[c, k] > CORR_REDUNDANT for k in kept)
        redundant[c] = red
        if not red: kept.append(c)

    # ---- classify ----
    rows = []
    for c in fcols:
        captured = abs(ic[c]) * fwd_std * 100
        clears = captured > FRICTION_FLOOR_PCT
        if redundant[c]:
            cat = "C_redundant"
        elif qmap[c] < 0.05 and clears and stab[c] >= 0.7:
            cat = "A_strong"
        elif abs(ic[c]) >= 0.015 and stab[c] >= 0.67:
            cat = "B_weak"
        else:
            cat = "D_noise"
        rows.append(dict(feature=c, ic=ic[c], q=qmap[c], mi=mimap[c],
                         imp=imp[c], stab=stab[c], captured_pct=captured,
                         clears_friction=clears, category=cat))
    df = pd.DataFrame(rows).sort_values("imp", ascending=False).reset_index(drop=True)
    df.to_csv("logs/feature_audit.csv", index=False)

    # ---- summary ----
    print("=" * 92)
    print("  CATEGORY SUMMARY")
    print("=" * 92)
    vc = df.category.value_counts()
    for k in ["A_strong", "B_weak", "C_redundant", "D_noise"]:
        print(f"    {k:<14} {vc.get(k,0):>4}  features")
    print(f"    {'TOTAL':<14} {len(df):>4}")
    print(f"\n  Min BH q-value across ALL features: {df.q.min():.3f}")
    print(f"  Features with q<0.05 (signif IC):   {(df.q<0.05).sum()}")
    print(f"  Features clearing 15-min friction:  {df.clears_friction.sum()}")

    print("\n" + "=" * 92)
    print("  TOP 20 FEATURES BY XGB IMPORTANCE")
    print("=" * 92)
    print(f"  {'feature':<26}{'IC':>8}{'q(BH)':>8}{'MI':>8}{'imp':>8}{'stab':>7}{'cap%':>7}  cat")
    print("  " + "-" * 88)
    for _, r in df.head(20).iterrows():
        print(f"  {r.feature:<26}{r.ic:>+8.3f}{r.q:>8.3f}{r.mi:>8.4f}"
              f"{r.imp:>8.4f}{r.stab:>7.2f}{r.captured_pct:>7.3f}  {r.category.split('_')[1]}")

    print("\n  TOP 10 BY |IC| (predictive, regardless of importance):")
    for _, r in df.reindex(df.ic.abs().sort_values(ascending=False).index).head(10).iterrows():
        print(f"    {r.feature:<26} IC={r.ic:+.3f} q={r.q:.3f} stab={r.stab:.2f} "
              f"cap={r.captured_pct:.3f}% [{r.category.split('_')[1]}]")
    print("\n  Full table -> logs/feature_audit.csv")


if __name__ == "__main__":
    main()
