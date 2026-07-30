"""
shadow_compare.py — PART E: offline shadow validation, clean V9 champion vs
HTF-36 challenger, on the SAME frozen test rows through the SAME unmodified
gate harness, base and friction-adjusted.

Outputs: reports/shadow_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    create_labels, purged_train_val_test_split, FWD_BARS,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.phase1_edge_discovery import build_aux           # noqa: E402
from scripts.validate_fixes_sim import run_variant            # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
CONF = 0.22
DELTA, LOT = 0.50, 65
NEW_VARIANT = dict(thr_call=0.32, thr_put=0.25, skip_ceil=0.65,
                   g2_call=72, g2_put=58, new_regime_floors=True,
                   pe_chop_cap=True, f_rsi=True, f_slt=True, f_rev=True,
                   f_mtf=True, cr_pen=5, cr_floor=65)


def sim_metrics(tr):
    if tr.empty:
        return {"trades": 0}
    pnl = tr["pnl"]
    daily = tr.groupby(pd.to_datetime(tr["ts"]).dt.date)["pnl"].sum().cumsum()
    return {"trades": len(tr), "win_rate": round(float((pnl > 0).mean()), 3),
            "pf": round(float(pnl[pnl > 0].sum() / max(abs(pnl[pnl <= 0].sum()), 1e-9)), 2),
            "expectancy": round(float(pnl.mean()), 0),
            "max_dd": round(float((daily - daily.cummax()).min()), 0),
            "net": round(float(pnl.sum()), 0)}


def friction(tr, rng):
    if tr.empty:
        return tr
    out = tr.copy()
    out["pnl"] = out["pnl"] - 4.0 * DELTA * LOT - 80.0 - 200.0
    w = out.index[out["pnl"] > 0].to_numpy()
    drop = (rng.choice(w, size=int(len(w) * 0.07), replace=False)
            if len(w) > 10 else [])
    return out.drop(index=drop)


def main():
    print("[1/3] Matrix + labels + test split...")
    dfm = build_features()
    dfm = create_labels(dfm)
    dfm["fwd_ret"] = dfm["close"].shift(-FWD_BARS) / dfm["close"] - 1
    fcols_v9 = joblib.load(ROOT / "models/sandbox_clean/feature_cols.pkl")
    sub = dfm.dropna(subset=fcols_v9 + ["label"])
    _, _, te_i = purged_train_val_test_split(len(sub))
    te = sub.iloc[te_i]
    yte = te["label"].values
    fwd = te["fwd_ret"].values
    rng = np.random.default_rng(21)

    candidates = {
        "clean_V9_champion": ROOT / "models/sandbox_clean",
        "HTF36_challenger": ROOT / "models/htf36",
    }
    report = {"test_window": [str(te.index.min()), str(te.index.max())],
              "test_bars": len(te), "harness": "NEW gate config, unmodified",
              "models": {}}

    print("[2/3] Evaluate both candidates on identical rows...")
    for tag, d in candidates.items():
        models = joblib.load(d / "models.pkl")
        scaler = joblib.load(d / "scaler.pkl")
        fc = joblib.load(d / "feature_cols.pkl")
        X = scaler.transform(te[fc].values)
        p = np.mean([m.predict_proba(X) for m in models.values()], axis=0)
        s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
        dm = (s != 2) & (yte != 2)
        da = float((yte[dm] == s[dm]).mean()) if dm.any() else None
        msk = ~np.isnan(fwd)
        ic, icp = spearmanr((p[:, 0] - p[:, 1])[msk], fwd[msk])
        dft = te[["open", "high", "low", "close"]].copy()
        dft["call_p"], dft["put_p"], dft["skip_p"] = p[:, 0], p[:, 1], p[:, 2]
        dft["rsi14"] = te["rsi14"] if "rsi14" in te.columns else 50.0
        trades = run_variant(dft, build_aux(dft), NEW_VARIANT)
        conf_dist = (p.max(axis=1)[s != 2])
        report["models"][tag] = {
            "n_features": len(fc),
            "signals": int((s != 2).sum()),
            "mean_signal_confidence": round(float(conf_dist.mean()), 4)
            if len(conf_dist) else None,
            "dir_acc": round(da, 4) if da else None,
            "IC": round(float(ic), 4), "IC_p": round(float(icp), 6),
            "sim": sim_metrics(trades),
            "sim_friction": sim_metrics(friction(trades, rng)),
        }
        m = report["models"][tag]
        print(f"   {tag:20s} f={m['n_features']:3d} acc={m['dir_acc']} "
              f"IC={m['IC']:+.4f}(p={m['IC_p']:.4f}) sim={m['sim']} ")

    print("[3/3] Save...")
    json.dump(report, open(REPORTS / "shadow_validation.json", "w"), indent=1)
    print(f"Saved -> {REPORTS / 'shadow_validation.json'}")


if __name__ == "__main__":
    main()
