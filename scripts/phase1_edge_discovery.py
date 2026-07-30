"""
phase1_edge_discovery.py — PHASE 1: which feature families retain genuine
out-of-sample predictive power after leakage removal?

Method (no gate changes, no threshold optimization):
  1. Family mapping of all clean-pipeline features (+ option context).
  2. Grouped permutation importance on the CLEAN full model's frozen test set.
  3. Family-only / combo retrains (identical hyperparams + purged split).
  4. Each run evaluated on: production dir_acc (CONF=0.22) val/test, and the
     FIXED gate-sim harness (NEW config, unmodified) on the test segment:
     PF / trades / maxDD.
Option-chain families (IV/OI) only exist 2024-06+ → evaluated on that window
with an in-window purged split, alongside a price-only control on the SAME
window for a fair comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    create_labels, purged_train_val_test_split, _fit_ensemble,
)
from scripts.train_model_v10 import (                         # noqa: E402
    load_option_features, compute_daily_option_context,
    merge_option_context_onto_5min,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.validate_fixes_sim import run_variant            # noqa: E402

RES = ROOT / "data/validation_results"
CONF = 0.22

NEW_VARIANT = dict(thr_call=0.32, thr_put=0.25, skip_ceil=0.65,
                   g2_call=72, g2_put=58, new_regime_floors=True,
                   pe_chop_cap=True, f_rsi=True, f_slt=True, f_rev=True,
                   f_mtf=True, cr_pen=5, cr_floor=65)

IV_COLS = ["atm_iv", "atm_iv_rank30", "atm_iv_zscore", "atm_iv_rising",
           "atm_iv_extreme", "iv_skew", "term_struct", "atm_theta_pct",
           "atm_delta", "iv_contango"]
OI_COLS = ["pcr_near", "pcr_all", "pcr_bullish", "pcr_bearish",
           "oi_ce_atm_log", "oi_pe_atm_log", "oi_ce_chg_dir", "oi_pe_chg_dir"]


def map_families(cols):
    fam = {}
    for c in cols:
        if c in IV_COLS:
            fam[c] = "iv"
        elif c in OI_COLS:
            fam[c] = "oi"
        elif c.startswith(("tf15_", "tf30_", "tf60_")) or c.startswith(
                ("day_ema", "day_rsi", "day_atr", "day_week_pos")):
            fam[c] = "htf"
        elif c.startswith(("vix_",)):
            fam[c] = "regime_vix"
        elif c.startswith(("is_", "dow", "dte", "expiry")):
            fam[c] = "regime_session"
        elif c.startswith(("dist_vwap", "above_vwap", "below_vwap",
                           "vwap_dist", "bars_above", "bars_below", "cmf")):
            fam[c] = "vwap"
        elif c in ("a10", "a14", "ar", "ava", "vov", "vol_regime", "park",
                   "gkv") or c.startswith("rv"):
            fam[c] = "volatility_proxy"
        elif c.startswith(("above_prev", "below_prev", "dist_prev",
                           "dist_week", "intraday", "day_move", "day_gap",
                           "day_big_gap")) or "sweep" in c or "fvg" in c \
                or "_ob" in c or "choch" in c:
            fam[c] = "market_structure"
        else:
            fam[c] = "price_action"
    return fam


def build_aux(df):
    aux = pd.DataFrame(index=df.index)
    aux["date"] = df.index.date
    aux["minute_of_day"] = df.index.hour * 60 + df.index.minute
    g = pd.Series(df.index.date, index=df.index)
    aux["day_start_idx"] = np.arange(len(df)) - g.groupby(g.values).cumcount().values
    df15 = df[["open", "high", "low", "close"]].resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr15 = pd.concat([df15.high - df15.low,
                      (df15.high - df15.close.shift()).abs(),
                      (df15.low - df15.close.shift()).abs()], axis=1).max(axis=1)
    aux["atr15"] = tr15.rolling(14).mean().reindex(df.index, method="ffill").fillna(30.0)
    e9, e21 = (df15.close.ewm(span=s, adjust=False).mean() for s in (9, 21))
    aux["ema15_bull"] = (e9 > e21).reindex(df.index, method="ffill").fillna(False)
    df60 = df[["close"]].resample("60min").last().dropna()
    e9h, e21h = (df60.close.ewm(span=s, adjust=False).mean() for s in (9, 21))
    aux["ema60_bull"] = (e9h > e21h).reindex(df.index, method="ffill").fillna(False)
    return aux


def dir_acc_eval(models, Xe, ye):
    p = np.mean([m.predict_proba(Xe) for m in models.values()], axis=0)
    s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
    dm = (s != 2) & (ye != 2)
    return (float((ye[dm] == s[dm]).mean()) if dm.any() else None,
            int((s != 2).sum()), p)


def sample_weights(ytr, tue):
    call_n, put_n = (ytr == 0).sum(), (ytr == 1).sum()
    dir_n = call_n + put_n
    skip_pct = (ytr == 2).mean()
    trade_w = skip_pct / (1 - skip_pct) if skip_pct < 1 else 2.0
    cb = dir_n / (2.0 * call_n) if call_n else 1.0
    pb = dir_n / (2.0 * put_n) if put_n else 1.0
    w = np.where(ytr == 0, trade_w * cb, np.where(ytr == 1, trade_w * pb, 1.0))
    return w * np.where(tue, 6.0, 1.0)


def trade_metrics(tr):
    if tr.empty:
        return {"trades": 0, "pf": None, "dd": 0, "net": 0}
    pnl = tr["pnl"]
    daily = tr.groupby(pd.to_datetime(tr["ts"]).dt.date)["pnl"].sum().cumsum()
    return {"trades": len(tr),
            "pf": round(pnl[pnl > 0].sum() / max(abs(pnl[pnl <= 0].sum()), 1e-9), 2),
            "dd": round(float((daily - daily.cummax()).min()), 0),
            "net": round(float(pnl.sum()), 0)}


def run_ablation(tag, dfm, aux, fcols_sub, results):
    sub = dfm.dropna(subset=fcols_sub + ["label"])
    if len(sub) < 5000:
        results[tag] = {"error": f"too few rows ({len(sub)})"}
        return
    X = sub[fcols_sub].values
    y = sub["label"].values
    tr_i, va_i, te_i = purged_train_val_test_split(len(X))
    sc = StandardScaler()
    Xtr, Xvl, Xte = sc.fit_transform(X[tr_i]), sc.transform(X[va_i]), sc.transform(X[te_i])
    tue = (sub["expiry_is_tue"].values[tr_i].astype(bool)
           if "expiry_is_tue" in sub.columns else np.zeros(len(tr_i), bool))
    models = _fit_ensemble(Xtr, y[tr_i], Xvl, y[va_i], sample_weights(y[tr_i], tue))
    va_acc, va_n, _ = dir_acc_eval(models, Xvl, y[va_i])
    te_acc, te_n, pte = dir_acc_eval(models, Xte, y[te_i])

    # fixed-harness trade sim on the TEST segment
    te_rows = sub.iloc[te_i]
    dft = te_rows[["open", "high", "low", "close"]].copy()
    dft["call_p"], dft["put_p"], dft["skip_p"] = pte[:, 0], pte[:, 1], pte[:, 2]
    dft["rsi14"] = te_rows["rsi14"] if "rsi14" in te_rows.columns else 50.0
    auxt = build_aux(dft)
    sim = trade_metrics(run_variant(dft, auxt, NEW_VARIANT))
    results[tag] = {
        "n_features": len(fcols_sub), "rows": len(sub),
        "test_range": [str(te_rows.index[0].date()), str(te_rows.index[-1].date())],
        "val_dir_acc": round(va_acc, 4) if va_acc else None,
        "test_dir_acc": round(te_acc, 4) if te_acc else None,
        "test_signals": te_n, "sim": sim,
    }
    print(f"  {tag:24s} feats={len(fcols_sub):3d} "
          f"val={va_acc if va_acc else 0:.3f} test={te_acc if te_acc else 0:.3f} "
          f"sig={te_n:5d} | sim: n={sim['trades']:4d} pf={sim['pf']} "
          f"dd={sim['dd']} net={sim['net']}")


def main():
    print("[1/4] Building clean matrix + option context...")
    dfm = build_features()
    bhav = load_option_features(str(ROOT / "data/v10_training_features.csv"))
    opt_ctx = compute_daily_option_context(bhav) if not bhav.empty else pd.DataFrame()
    dfm = merge_option_context_onto_5min(dfm, opt_ctx)
    dfm = create_labels(dfm)

    fcols9 = joblib.load(ROOT / "models/feature_cols_v9.pkl")
    fam = map_families(fcols9 + IV_COLS + OI_COLS)
    pd.Series(fam).rename("family").to_csv(RES / "phase1_family_map.csv")
    groups = {}
    for c, f in fam.items():
        groups.setdefault(f, []).append(c)
    print("  families:", {k: len(v) for k, v in groups.items()})

    results = {"family_sizes": {k: len(v) for k, v in groups.items()}}

    # ── [2/4] grouped permutation importance on clean full model ──────
    print("[2/4] Permutation importance (clean model, frozen test)...")
    models = joblib.load(ROOT / "models/sandbox_clean/models.pkl")
    sc = joblib.load(ROOT / "models/sandbox_clean/scaler.pkl")
    sub = dfm.dropna(subset=fcols9 + ["label"])
    _, _, te_i = purged_train_val_test_split(len(sub))
    Xte = sc.transform(sub[fcols9].values[te_i])
    yte = sub["label"].values[te_i]
    base_acc, base_n, _ = dir_acc_eval(models, Xte, yte)
    print(f"  baseline test dir_acc={base_acc:.4f} (n_signals={base_n})")
    rng = np.random.default_rng(11)
    perm = {}
    col_ix = {c: i for i, c in enumerate(fcols9)}
    for famname, cols in groups.items():
        ixs = [col_ix[c] for c in cols if c in col_ix]
        if not ixs:
            continue
        drops = []
        for _ in range(5):
            Xp = Xte.copy()
            sh = rng.permutation(len(Xp))
            Xp[:, ixs] = Xp[sh][:, ixs]
            a, _, _ = dir_acc_eval(models, Xp, yte)
            drops.append(base_acc - (a or 0))
        perm[famname] = round(float(np.mean(drops)), 4)
        print(f"  {famname:18s} dir_acc drop when permuted: {perm[famname]:+.4f}")
    results["permutation_importance"] = perm
    results["perm_baseline"] = {"dir_acc": base_acc, "n_signals": base_n}

    # ── [3/4] ablation retrains — full window ─────────────────────────
    print("[3/4] Ablation retrains (full window 2015-2026)...")
    g = groups
    run_ablation("price_only", dfm, None, g["price_action"], results)
    run_ablation("structure_only", dfm, None, g["market_structure"], results)
    run_ablation("htf_only", dfm, None, g["htf"], results)
    run_ablation("volatility_only", dfm, None, g["volatility_proxy"], results)
    run_ablation("vwap_only", dfm, None, g["vwap"], results)
    run_ablation("regime_only", dfm, None,
                 g["regime_vix"] + g["regime_session"], results)
    run_ablation("price_plus_htf", dfm, None,
                 g["price_action"] + g["htf"], results)

    # ── [4/4] option-window runs (2024-06+) ───────────────────────────
    print("[4/4] Option-window retrains (2024-06 → 2026-06)...")
    dwin = dfm.dropna(subset=["atm_iv"])
    print(f"  option window rows: {len(dwin):,} "
          f"({dwin.index[0].date()} → {dwin.index[-1].date()})")
    run_ablation("optwin_price_control", dwin, None, g["price_action"], results)
    run_ablation("optwin_optchain_only", dwin, None,
                 IV_COLS + OI_COLS, results)
    run_ablation("optwin_oi_only", dwin, None, OI_COLS, results)
    run_ablation("optwin_iv_only", dwin, None, IV_COLS, results)
    run_ablation("optwin_price_plus_opt", dwin, None,
                 g["price_action"] + IV_COLS + OI_COLS, results)

    json.dump(results, open(RES / "phase1_report.json", "w"), indent=1, default=str)
    print(f"\nReport → {RES / 'phase1_report.json'}")


if __name__ == "__main__":
    main()
