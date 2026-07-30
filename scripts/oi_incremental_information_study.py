"""
oi_incremental_information_study.py -- does the OI archive add INCREMENTAL
information on top of the existing production feature set, not just raw
correlation? Explainability-only; no production strategy is trained, no
production files touched, no thresholds/labels/architecture changed.

Answers, per the task: conditional MI, reduced pairwise PID, SHAP +
SHAP interaction values, permutation importance conditioned on existing
features, incremental gain (blocked-CV R^2), forward selection, backward
elimination, hierarchical clustering, redundancy analysis, info gain after
conditioning on the top-20 production features (feature_audit.csv's own
ranking, reused -- not recomputed).

HONESTY CONSTRAINT carried over from every prior study in this arc: the
OI archive only joins to ~28 trading days / ~2,054 bars. Every method
below is run on that window and ONLY that window (never allowed to see
the OI-free 211k-bar history, since that would make the "baseline"
model's comparison unfair -- it must be trained on the identical rows).
n=2054, p=32 (20 prod + 12 OI) is a thin regime for SHAP interactions,
permutation importance, and full high-dimensional PID -- flagged
explicitly at each step, not glossed over. Regression target (forward
return) is used throughout instead of the 3-class label, for sample
efficiency and because production's own train_fold() refuses below
n=5000 -- a small, explicitly-diagnostic model is fit here instead,
never saved, never wired into any production path.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import build_frame

RNG_SEED = 42
N_FOLDS = 4          # blocked CV (NOT the production's 3-month walk-forward -- too short for that)
PURGE_BARS = 6        # ~30 min purge at each fold boundary


# ---------------------------------------------------------------------------
def build_oi_features():
    files = sorted(glob.glob(str(ROOT / "data/oi_archive/oi_2026-*.csv")))
    raw = pd.concat([pd.read_csv(f, parse_dates=["snapshot_ts"]) for f in files], ignore_index=True)
    atm = raw[raw["strike"] == raw["atm_strike"]].copy()
    atm = atm.sort_values("snapshot_ts").drop_duplicates("snapshot_ts", keep="last")
    atm["bar5"] = atm["snapshot_ts"].dt.ceil("5min")
    bar = atm.groupby("bar5").last()[["spot", "atm_strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "pcr_total"]]
    bar = bar.sort_index()

    f = pd.DataFrame(index=bar.index)
    f["atm_ce_oi_chg_pct"] = bar["ce_oi"].pct_change()
    f["atm_pe_oi_chg_pct"] = bar["pe_oi"].pct_change()
    f["net_oi_change"] = (bar["ce_oi"] - bar["pe_oi"]).diff()
    f["oi_momentum_3"] = (bar["ce_oi"] - bar["pe_oi"]).diff(3)
    f["oi_acceleration"] = f["net_oi_change"].diff()
    f["ce_pe_oi_ratio"] = bar["ce_oi"] / bar["pe_oi"].replace(0, np.nan)
    f["pcr"] = bar["pcr_total"]
    ce_roll_mean = bar["ce_oi"].rolling(20, min_periods=10).mean()
    ce_roll_std = bar["ce_oi"].rolling(20, min_periods=10).std()
    f["oi_zscore"] = (bar["ce_oi"] - ce_roll_mean) / ce_roll_std.replace(0, np.nan)
    f["oi_rolling_vol"] = bar["ce_oi"].pct_change().rolling(10, min_periods=5).std()
    price_ret = bar["spot"].pct_change()
    oi_ret = (bar["ce_oi"] - bar["pe_oi"]).pct_change()
    f["oi_price_divergence"] = (np.sign(price_ret) != np.sign(oi_ret)).astype(float)
    oi_net = bar["ce_oi"] - bar["pe_oi"]
    oi_band = oi_net.rolling(20, min_periods=10)
    f["oi_breakout"] = (oi_net - oi_band.mean()) / oi_band.std().replace(0, np.nan)
    f["oi_trend_persistence"] = np.sign(f["net_oi_change"]).rolling(5, min_periods=3).apply(
        lambda x: (x == x.iloc[-1]).sum(), raw=False)
    return f


def blocked_folds(index, n_folds, purge_bars):
    """Contiguous blocked CV over the 28-day window (not an expanding
    walk-forward -- too little data for that). Purge PURGE_BARS on each
    side of the test block from the training rows."""
    n = len(index)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        test_pos = np.arange(lo, hi)
        train_pos = np.array([j for j in range(n)
                               if j < lo - purge_bars or j > hi + purge_bars - 1])
        folds.append((train_pos, test_pos))
    return folds


def fit_diag_model(X, y, seed=RNG_SEED):
    m = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_alpha=0.0,
                          reg_lambda=0.3, min_child_weight=3, verbosity=0,
                          n_jobs=-1, random_state=seed)
    m.fit(X, y)
    # sanity guard: the original reg_alpha=1.0/reg_lambda=2.0/min_child_weight=10
    # config produced trees with ZERO splits against this tiny-variance target
    # (y std ~0.0008), silently degenerating every downstream metric
    # (permutation importance, SHAP, incremental gain) to exactly 0. Fail
    # loudly instead of repeating that mistake with a different config.
    tdf = m.get_booster().trees_to_dataframe()
    n_splits = int((tdf["Feature"] != "Leaf").sum())
    if n_splits == 0:
        raise RuntimeError(
            "fit_diag_model produced a model with ZERO split nodes -- "
            "every prediction would be a constant, silently zeroing out "
            "every explainability metric downstream. Loosen regularization.")
    return m


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
def main():
    print("Building OI features + production frame ...")
    oi = build_oi_features()
    oi_cols = list(oi.columns)
    feat, fcols_all = build_frame()

    prod_audit = pd.read_csv(ROOT / "logs/feature_audit.csv")
    top20 = prod_audit.sort_values("imp", ascending=False).head(20)["feature"].tolist()
    print(f"Top-20 production features (reused from feature_audit.csv): {top20}")

    joined = feat.join(oi, how="inner").dropna(subset=oi_cols + top20)
    fwd_ret = feat["close"].pct_change().shift(-3)
    y = fwd_ret.loc[joined.index]
    valid = y.notna() & np.isfinite(y)
    joined, y = joined[valid], y[valid]
    joined = joined.sort_index()
    y = y.loc[joined.index]
    n = len(joined)
    print(f"Final analysis window: n={n} bars, {joined.index.date[0]} -> {joined.index.date[-1]}")

    def _clean(df_):
        # atm_ce/pe_oi_chg_pct's pct_change() divides by a zero prior OI
        # reading on a handful of bars, producing inf -- fillna() alone
        # does not catch inf, so replace first.
        return df_.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_prod = _clean(joined[top20])
    X_oi = _clean(joined[oi_cols])
    X_all = pd.concat([X_prod, X_oi], axis=1)

    results = {"n_bars": n, "n_days": int(pd.Series(joined.index.date).nunique()),
               "top20_production_features": top20, "oi_features": oi_cols}

    # =======================================================================
    # 1. INCREMENTAL GAIN: blocked-CV R^2, baseline (top20 prod) vs augmented (+OI)
    # =======================================================================
    print("\n[1/9] Incremental gain analysis (blocked CV, baseline vs augmented) ...")
    folds = blocked_folds(joined.index, N_FOLDS, PURGE_BARS)
    r2_base, r2_aug = [], []
    for k, (tr, te) in enumerate(folds, 1):
        if len(tr) < 200 or len(te) < 20:
            continue
        m_base = fit_diag_model(X_prod.iloc[tr], y.iloc[tr])
        m_aug = fit_diag_model(X_all.iloc[tr], y.iloc[tr])
        pb = m_base.predict(X_prod.iloc[te])
        pa = m_aug.predict(X_all.iloc[te])
        rb, ra = r2(y.iloc[te].values, pb), r2(y.iloc[te].values, pa)
        r2_base.append(rb); r2_aug.append(ra)
        print(f"  fold {k}: n_test={len(te)}  R2_baseline={rb:+.4f}  R2_augmented={ra:+.4f}  delta={ra-rb:+.4f}")
    delta = np.array(r2_aug) - np.array(r2_base)
    # paired test across folds (small n_folds -> report both t-test and sign)
    t_stat, t_p = stats.ttest_rel(r2_aug, r2_base) if len(delta) > 1 else (np.nan, np.nan)
    results["incremental_gain"] = dict(
        r2_baseline_per_fold=r2_base, r2_augmented_per_fold=r2_aug,
        delta_per_fold=delta.tolist(), mean_delta=float(delta.mean()),
        folds_improved=int((delta > 0).sum()), n_folds=len(delta),
        paired_ttest_stat=float(t_stat) if np.isfinite(t_stat) else None,
        paired_ttest_p=float(t_p) if np.isfinite(t_p) else None)
    print(f"  Mean delta R2: {delta.mean():+.4f}  folds improved: {(delta>0).sum()}/{len(delta)}  "
          f"paired t-test p={t_p:.4f}" if np.isfinite(t_p) else "  (t-test n/a, too few folds)")

    # =======================================================================
    # 2. Fit ONE full-window diagnostic model (80/20 time split) for SHAP/perm-imp
    # =======================================================================
    print("\n[2/9] Fitting single diagnostic model (80/20 time split) for SHAP/permutation importance ...")
    cut = int(n * 0.8)
    Xtr, Xte = X_all.iloc[:cut], X_all.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    model = fit_diag_model(Xtr, ytr)
    test_r2 = r2(yte.values, model.predict(Xte))
    print(f"  full-augmented-model holdout R2={test_r2:+.4f} (n_test={len(Xte)})")
    results["diagnostic_model_holdout_r2"] = float(test_r2)

    # =======================================================================
    # 3. Permutation importance CONDITIONED on existing features (in augmented model)
    # =======================================================================
    print("\n[3/9] Permutation importance (conditioned on top-20 production features already in model) ...")
    rng = np.random.default_rng(RNG_SEED)
    base_pred = model.predict(Xte)
    base_r2 = r2(yte.values, base_pred)
    perm_imp = {}
    for c in oi_cols:
        drops = []
        for _ in range(30):
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base_r2 - r2(yte.values, model.predict(Xp)))
        perm_imp[c] = dict(mean_r2_drop=float(np.mean(drops)), std_r2_drop=float(np.std(drops)))
    results["permutation_importance_conditioned"] = perm_imp
    for c, v in sorted(perm_imp.items(), key=lambda kv: -kv[1]["mean_r2_drop"]):
        print(f"  {c:<22} mean R2 drop={v['mean_r2_drop']:+.5f} (+/-{v['std_r2_drop']:.5f})")

    # =======================================================================
    # 4. SHAP + SHAP interaction values
    # =======================================================================
    print("\n[4/9] SHAP values + SHAP interaction values (holdout set) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)
    mean_abs_shap = dict(zip(X_all.columns, np.abs(shap_values).mean(axis=0)))
    try:
        shap_inter = explainer.shap_interaction_values(Xte)
        inter_mag = np.abs(shap_inter).mean(axis=0)  # (p, p)
        cols = list(X_all.columns)
        oi_interactions = {}
        for oc in oi_cols:
            i = cols.index(oc)
            with_prod = {cols[j]: float(inter_mag[i, j]) for j in range(len(cols)) if cols[j] in top20}
            top_partner = max(with_prod, key=with_prod.get)
            oi_interactions[oc] = dict(mean_abs_shap=float(mean_abs_shap[oc]),
                                        strongest_interaction_partner=top_partner,
                                        strongest_interaction_strength=with_prod[top_partner],
                                        mean_interaction_with_top20=float(np.mean(list(with_prod.values()))))
        shap_computed = True
    except Exception as e:
        print(f"  SHAP interaction values failed ({e}); falling back to main-effect SHAP only")
        oi_interactions = {oc: dict(mean_abs_shap=float(mean_abs_shap[oc])) for oc in oi_cols}
        shap_computed = False
    results["shap"] = dict(computed_interactions=shap_computed, per_oi_feature=oi_interactions,
                            mean_abs_shap_top20_avg=float(np.mean([mean_abs_shap[c] for c in top20])))
    print(f"  (for context) mean |SHAP| averaged over the top-20 production features: "
          f"{results['shap']['mean_abs_shap_top20_avg']:.6f}")
    for c, v in sorted(oi_interactions.items(), key=lambda kv: -kv[1]["mean_abs_shap"]):
        print(f"  {c:<22} mean|SHAP|={v['mean_abs_shap']:.6f}")

    # =======================================================================
    # 5. Conditional Mutual Information (residualization proxy)
    # =======================================================================
    print("\n[5/9] Conditional Mutual Information I(OI_feature ; forward_return | top-20 production) ...")
    resid_model_y = fit_diag_model(X_prod, y)
    y_resid = y.values - resid_model_y.predict(X_prod)
    cmi = {}
    for c in oi_cols:
        resid_model_x = GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=RNG_SEED)
        resid_model_x.fit(X_prod, X_oi[c])
        x_resid = X_oi[c].values - resid_model_x.predict(X_prod)
        mi_raw = float(mutual_info_regression(X_oi[[c]].values, y.values, random_state=RNG_SEED)[0])
        mi_cond = float(mutual_info_regression(x_resid.reshape(-1, 1), y_resid, random_state=RNG_SEED)[0])
        cmi[c] = dict(mi_unconditional=mi_raw, mi_conditional_on_top20=mi_cond,
                       info_retained_pct=100 * mi_cond / mi_raw if mi_raw > 1e-12 else None)
    results["conditional_mutual_information"] = cmi
    for c, v in sorted(cmi.items(), key=lambda kv: -kv[1]["mi_conditional_on_top20"]):
        ret = f"{v['info_retained_pct']:.0f}%" if v['info_retained_pct'] is not None else "n/a"
        print(f"  {c:<22} MI_raw={v['mi_unconditional']:.5f}  MI|top20={v['mi_conditional_on_top20']:.5f}  retained={ret}")

    # =======================================================================
    # 6. Reduced pairwise PID (OI feature, best-matched production feature, target)
    # =======================================================================
    print("\n[6/9] Reduced pairwise PID (NOT full high-dim PID -- infeasible at n=2054, p=196; "
          "3-bin discretized 2-source Imin decomposition against each OI feature's single most-"
          "correlated top-20 production feature) ...")

    def imin_pid(x1, x2, y_, bins=3):
        def disc(v):
            try:
                return pd.qcut(v, bins, labels=False, duplicates="drop")
            except Exception:
                return pd.cut(v, bins, labels=False)
        x1d, x2d, yd = disc(pd.Series(x1)), disc(pd.Series(x2)), disc(pd.Series(y_))
        df_ = pd.DataFrame({"x1": x1d, "x2": x2d, "y": yd}).dropna()
        if len(df_) < 50:
            return None
        def mi_disc(a, b):
            ct = pd.crosstab(a, b)
            from sklearn.metrics import mutual_info_score
            return mutual_info_score(a, b)
        i_x1y = mi_disc(df_["x1"], df_["y"])
        i_x2y = mi_disc(df_["x2"], df_["y"])
        i_x1x2y = mutual_info_score(df_["x1"].astype(str) + "_" + df_["x2"].astype(str), df_["y"]) \
            if len(df_) else np.nan
        redundancy = min(i_x1y, i_x2y)  # Williams & Beer Imin (simplified, 2-source)
        unique_x1 = i_x1y - redundancy
        unique_x2 = i_x2y - redundancy
        synergy = i_x1x2y - unique_x1 - unique_x2 - redundancy
        return dict(I_oi_y=float(i_x1y), I_partner_y=float(i_x2y), I_joint_y=float(i_x1x2y),
                    redundancy=float(redundancy), unique_oi=float(unique_x1),
                    unique_partner=float(unique_x2), synergy=float(synergy))

    from sklearn.metrics import mutual_info_score
    pid_results = {}
    corr_matrix_all = X_all.corr()
    for c in oi_cols:
        corrs = corr_matrix_all.loc[c, top20].abs()
        partner = corrs.idxmax()
        r = imin_pid(X_oi[c].values, X_prod[partner].values, y.values)
        pid_results[c] = dict(partner=partner, partner_corr=float(corr_matrix_all.loc[c, partner]), **(r or {}))
    results["reduced_pairwise_pid"] = pid_results
    for c, v in pid_results.items():
        if "unique_oi" in v:
            print(f"  {c:<22} vs {v['partner']:<15} unique_OI={v['unique_oi']:+.5f} "
                  f"redundancy={v['redundancy']:.5f} synergy={v['synergy']:+.5f}")

    # =======================================================================
    # 7. Forward feature selection (top20 fixed, greedily add OI features)
    # =======================================================================
    print("\n[7/9] Forward selection (top-20 production fixed as base, greedily add OI features by CV R2) ...")
    selected, remaining = [], list(oi_cols)
    fwd_history = []
    cur_best = np.mean(r2_base)  # baseline CV R2 with no OI features (from step 1's folds)
    while remaining:
        gains = {}
        for c in remaining:
            cols_try = top20 + selected + [c]
            rs = []
            for tr, te in folds:
                if len(tr) < 200 or len(te) < 20:
                    continue
                m = fit_diag_model(X_all[cols_try].iloc[tr], y.iloc[tr])
                rs.append(r2(y.iloc[te].values, m.predict(X_all[cols_try].iloc[te])))
            gains[c] = np.mean(rs) - cur_best
        best_c = max(gains, key=gains.get)
        if gains[best_c] <= 0.0005:  # stop if the best remaining candidate adds negligible R2
            break
        selected.append(best_c); remaining.remove(best_c)
        cur_best += gains[best_c]
        fwd_history.append(dict(added=best_c, cv_r2_after=float(cur_best), gain=float(gains[best_c])))
        print(f"  + {best_c:<22} CV R2 -> {cur_best:+.4f} (gain {gains[best_c]:+.4f})")
    if not fwd_history:
        print("  no OI feature improved CV R2 by more than 0.0005 -- none selected")
    results["forward_selection"] = dict(selected_order=fwd_history, final_selected=selected)

    # =======================================================================
    # 8. Backward elimination (start with all 32, drop weakest by conditioned perm importance)
    # =======================================================================
    print("\n[8/9] Backward elimination (start with all 12 OI features + top20, drop weakest OI by "
          "conditioned permutation importance, one at a time) ...")
    active_oi = list(oi_cols)
    back_history = []
    while len(active_oi) > 0:
        cols_try = top20 + active_oi
        m = fit_diag_model(X_all[cols_try].iloc[:cut], y.iloc[:cut])
        Xte_try = X_all[cols_try].iloc[cut:]
        base_r2_try = r2(yte.values, m.predict(Xte_try))
        drops = {}
        for c in active_oi:
            Xp = Xte_try.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops[c] = base_r2_try - r2(yte.values, m.predict(Xp))
        weakest = min(drops, key=drops.get)
        back_history.append(dict(removed=weakest, importance_when_removed=float(drops[weakest]),
                                  r2_before_removal=float(base_r2_try)))
        if drops[weakest] > 0.001:  # stop once even the weakest remaining feature matters
            print(f"  stopping: weakest remaining ({weakest}) still has importance {drops[weakest]:+.4f} > 0.001")
            break
        print(f"  - removed {weakest:<22} (importance {drops[weakest]:+.5f}, R2 before={base_r2_try:+.4f})")
        active_oi.remove(weakest)
    results["backward_elimination"] = dict(history=back_history, features_surviving=active_oi)

    # =======================================================================
    # 9. Hierarchical clustering + redundancy analysis
    # =======================================================================
    print("\n[9/9] Hierarchical clustering + redundancy analysis (top-20 production + 12 OI) ...")
    dist = 1 - corr_matrix_all.abs()
    condensed = squareform(dist.values, checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    cluster_ids = hierarchy.fcluster(Z, t=0.6, criterion="distance")
    clusters = {}
    for col, cid in zip(corr_matrix_all.columns, cluster_ids):
        clusters.setdefault(int(cid), []).append(col)
    oi_clusters = {c: cid for col, cid in zip(corr_matrix_all.columns, cluster_ids) for c in oi_cols if c == col}
    redundancy = {}
    for c in oi_cols:
        max_corr = corr_matrix_all.loc[c, top20].abs().max()
        max_partner = corr_matrix_all.loc[c, top20].abs().idxmax()
        redundancy[c] = dict(cluster_id=oi_clusters.get(c), cluster_mates=[x for x in clusters[oi_clusters[c]] if x != c],
                              max_abs_corr_with_prod=float(max_corr), most_correlated_prod_feature=max_partner)
    results["clustering_redundancy"] = dict(clusters={str(k): v for k, v in clusters.items()},
                                             per_oi_feature=redundancy)
    for c, v in redundancy.items():
        print(f"  {c:<22} cluster={v['cluster_id']} mates={v['cluster_mates']} "
              f"max|corr| w/ prod={v['max_abs_corr_with_prod']:.3f} ({v['most_correlated_prod_feature']})")

    with open(ROOT / "logs/oi_incremental_information_study.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\nWrote logs/oi_incremental_information_study.json")


if __name__ == "__main__":
    main()
