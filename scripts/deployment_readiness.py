"""
deployment_readiness.py — Steps 2+3 of the Phase-1 implementation order.

2. Retrains V10-style WITHOUT the negative daily-OI features (keeps IV) on
   the clean pipeline → models/sandbox_v10_no_oi/  (production untouched).
3. Deployment readiness report on a COMMON out-of-sample window (beyond every
   candidate's train+val data) for four configurations:
     A  live-today        : leaked V10 weights + LEAKED features (running state)
     B  post-restart      : leaked V10 weights + FIXED features (the hazard)
     C  clean V9          : sandbox_clean + fixed features
     D  clean V10 no-OI   : new sandbox + fixed features
   Metrics: dir_acc (vs labels from the FIXED matrix = single ground truth),
   gate-sim (NEW config) trades/PF/DD/net, friction-adjusted PF.

NO model is promoted by this script.
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
    create_labels, purged_train_val_test_split, _fit_ensemble, merge_htf,
)
from scripts.train_model_v10 import (                         # noqa: E402
    load_option_features, compute_daily_option_context,
    merge_option_context_onto_5min,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.phase0_leakage_proof import (                    # noqa: E402
    build as build_either, leaked_features_daily, leaked_merge_htf,
)
from scripts.phase1_edge_discovery import (                   # noqa: E402
    IV_COLS, OI_COLS, sample_weights, build_aux,
)
from scripts.validate_fixes_sim import run_variant            # noqa: E402

RES = ROOT / "data/validation_results"
OUT = ROOT / "models/sandbox_v10_no_oi"
OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.22
DELTA, LOT = 0.50, 65

NEW_VARIANT = dict(thr_call=0.32, thr_put=0.25, skip_ceil=0.65,
                   g2_call=72, g2_put=58, new_regime_floors=True,
                   pe_chop_cap=True, f_rsi=True, f_slt=True, f_rev=True,
                   f_mtf=True, cr_pen=5, cr_floor=65)


def predict(models, scaler, fcols, mat, rows):
    sub = mat.loc[rows]
    X = sub.reindex(columns=fcols, fill_value=0.0).values
    X = np.nan_to_num(X, nan=0.0)
    p = np.mean([m.predict_proba(scaler.transform(X))
                 for m in models.values()], axis=0)
    return p


def dir_acc(p, y):
    s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
    dm = (s != 2) & (y != 2)
    return (float((y[dm] == s[dm]).mean()) if dm.any() else None,
            int((s != 2).sum()))


def friction(tr, rng):
    if tr.empty:
        return tr
    out = tr.copy()
    out["pnl"] = out["pnl"] - 4.0 * DELTA * LOT - 80.0 - 200.0
    w = out.index[out["pnl"] > 0].to_numpy()
    drop = rng.choice(w, size=int(len(w) * 0.07), replace=False) if len(w) > 10 else []
    return out.drop(index=drop)


def sim_metrics(tr):
    if tr.empty:
        return {"trades": 0, "pf": None, "dd": 0, "net": 0}
    pnl = tr["pnl"]
    daily = tr.groupby(pd.to_datetime(tr["ts"]).dt.date)["pnl"].sum().cumsum()
    return {"trades": len(tr), "wr": round(float((pnl > 0).mean()), 3),
            "pf": round(float(pnl[pnl > 0].sum() / max(abs(pnl[pnl <= 0].sum()), 1e-9)), 2),
            "dd": round(float((daily - daily.cummax()).min()), 0),
            "net": round(float(pnl.sum()), 0)}


def main():
    print("[1/5] Building FIXED matrix + option context...")
    mat_fixed = build_features()
    bhav = load_option_features(str(ROOT / "data/v10_training_features.csv"))
    opt_ctx = compute_daily_option_context(bhav)
    mat_fixed = merge_option_context_onto_5min(mat_fixed, opt_ctx)

    print("[2/5] Building LEAKED matrix (for live-today row)...")
    mat_leaked = build_either(leaked_features_daily, leaked_merge_htf)
    mat_leaked = merge_option_context_onto_5min(mat_leaked, opt_ctx)

    print("[3/5] Labels (single ground truth, fixed matrix)...")
    lab = create_labels(mat_fixed.copy())["label"]

    print("[4/5] Training clean V10-no-OI sandbox...")
    fcols9 = joblib.load(ROOT / "models/feature_cols_v9.pkl")
    cols_d = fcols9 + [c for c in IV_COLS if c in mat_fixed.columns]
    sub_d = mat_fixed.dropna(subset=cols_d).join(lab.rename("label"), how="inner")
    sub_d = sub_d.dropna(subset=["label"])
    X, y = sub_d[cols_d].values, sub_d["label"].values
    tr_i, va_i, te_i = purged_train_val_test_split(len(X))
    sc_d = StandardScaler()
    Xtr, Xvl, Xte = (sc_d.fit_transform(X[tr_i]), sc_d.transform(X[va_i]),
                     sc_d.transform(X[te_i]))
    tue = (sub_d["expiry_is_tue"].values[tr_i].astype(bool)
           if "expiry_is_tue" in sub_d.columns else np.zeros(len(tr_i), bool))
    models_d = _fit_ensemble(Xtr, y[tr_i], Xvl, y[va_i],
                             sample_weights(y[tr_i], tue))
    joblib.dump(models_d, OUT / "models.pkl")
    joblib.dump(sc_d, OUT / "scaler.pkl")
    joblib.dump(cols_d, OUT / "feature_cols.pkl")
    d_val_end = sub_d.index[va_i].max()
    p_dte = np.mean([m.predict_proba(Xte) for m in models_d.values()], axis=0)
    d_native_acc, d_native_n = dir_acc(p_dte, y[te_i])
    print(f"   D native frozen-test dir_acc={d_native_acc} "
          f"(n_signals={d_native_n}, window {sub_d.index[te_i].min().date()} "
          f"→ {sub_d.index[te_i].max().date()})")

    # leaked V10's val-end (same split proportions on its dropna rows)
    fcols10 = joblib.load(ROOT / "models/feature_cols_v10.pkl")
    sub10 = mat_leaked.dropna(subset=[c for c in fcols10 if c in mat_leaked.columns])
    _, va10, te10 = purged_train_val_test_split(len(sub10))
    v10_val_end = sub10.index[va10].max()

    common_start = max(d_val_end, v10_val_end,
                       pd.Timestamp("2025-07-07")) + pd.Timedelta(days=1)
    print(f"[5/5] Common OOS window: {common_start.date()} → "
          f"{mat_fixed.index.max().date()}")

    rows_f = mat_fixed.index[(mat_fixed.index >= common_start)]
    rows_l = mat_leaked.index[(mat_leaked.index >= common_start)]
    rows = rows_f.intersection(rows_l)
    ycommon = lab.reindex(rows).fillna(2).astype(int).values

    v10m = joblib.load(ROOT / "models/nifty_v10_models.pkl")
    v10s = joblib.load(ROOT / "models/nifty_v10_scaler.pkl")
    v9c = joblib.load(ROOT / "models/sandbox_clean/models.pkl")
    v9cs = joblib.load(ROOT / "models/sandbox_clean/scaler.pkl")
    v9cf = joblib.load(ROOT / "models/sandbox_clean/feature_cols.pkl")

    configs = {
        "A_live_today_leakedV10_leakedfeat": (v10m, v10s, fcols10, mat_leaked),
        "B_post_restart_leakedV10_fixedfeat": (v10m, v10s, fcols10, mat_fixed),
        "C_clean_V9": (v9c, v9cs, v9cf, mat_fixed),
        "D_clean_V10_no_OI": (models_d, sc_d, cols_d, mat_fixed),
    }
    rng = np.random.default_rng(7)
    report = {"common_window": [str(rows.min()), str(rows.max())],
              "common_bars": len(rows),
              "D_native_test": {"dir_acc": d_native_acc, "n": d_native_n},
              "note": "C native frozen-test dir_acc=0.5868 (2025-07→2026-06)"}
    print(f"\n{'config':38s} {'dir_acc':>8s} {'sig':>5s} | sim n/wr/PF/DD/net | fricPF")
    for tag, (m, s, fc, mat) in configs.items():
        p = predict(m, s, fc, mat, rows)
        da, nsig = dir_acc(p, ycommon)
        dfs = mat.loc[rows, ["open", "high", "low", "close"]].copy()
        dfs["call_p"], dfs["put_p"], dfs["skip_p"] = p[:, 0], p[:, 1], p[:, 2]
        dfs["rsi14"] = mat.loc[rows, "rsi14"] if "rsi14" in mat.columns else 50.0
        trades = run_variant(dfs, build_aux(dfs), NEW_VARIANT)
        base = sim_metrics(trades)
        fric = sim_metrics(friction(trades, rng))
        report[tag] = {"dir_acc": round(da, 4) if da else None,
                       "n_signals": nsig, "sim": base, "friction_sim": fric}
        print(f"{tag:38s} {da if da else 0:8.4f} {nsig:5d} | "
              f"{base['trades']}/{base.get('wr')}/{base['pf']}/{base['dd']}/"
              f"{base['net']} | {fric['pf']}")

    json.dump(report, open(RES / "deployment_readiness.json", "w"),
              indent=1, default=str)
    print(f"\nReport → {RES / 'deployment_readiness.json'}")
    print("NO model promoted (per instruction #4).")


if __name__ == "__main__":
    main()
