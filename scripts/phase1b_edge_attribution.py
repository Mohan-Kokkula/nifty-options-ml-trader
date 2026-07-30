"""
phase1b_edge_attribution.py — completes the edge-attribution matrix:
  (a) drop-one-family ablations (full minus each family)
  (b) incremental feature-add (strongest family first, add one at a time)

Identical hyperparams (_fit_ensemble), identical purged 85/7.5/7.5 split,
clean pipeline only, no tuning. Metrics per run on the frozen test segment:
dir_acc (CONF=0.22), Spearman IC of (call_p-put_p) vs fwd 3-bar return,
gate-sim (NEW config): trades, PF, maxDD, expectancy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    create_labels, purged_train_val_test_split, _fit_ensemble, FWD_BARS,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.phase1_edge_discovery import (                   # noqa: E402
    map_families, sample_weights, build_aux, IV_COLS, OI_COLS,
)
from scripts.validate_fixes_sim import run_variant            # noqa: E402

RES = ROOT / "data/validation_results"
CONF = 0.22
NEW_VARIANT = dict(thr_call=0.32, thr_put=0.25, skip_ceil=0.65,
                   g2_call=72, g2_put=58, new_regime_floors=True,
                   pe_chop_cap=True, f_rsi=True, f_slt=True, f_rev=True,
                   f_mtf=True, cr_pen=5, cr_floor=65)


def evaluate(tag, dfm, cols, results):
    sub = dfm.dropna(subset=cols + ["label", "fwd_ret"])
    X, y = sub[cols].values, sub["label"].values
    tr_i, va_i, te_i = purged_train_val_test_split(len(X))
    sc = StandardScaler()
    Xtr, Xvl, Xte = (sc.fit_transform(X[tr_i]), sc.transform(X[va_i]),
                     sc.transform(X[te_i]))
    tue = (sub["expiry_is_tue"].values[tr_i].astype(bool)
           if "expiry_is_tue" in sub.columns else np.zeros(len(tr_i), bool))
    models = _fit_ensemble(Xtr, y[tr_i], Xvl, y[va_i],
                           sample_weights(y[tr_i], tue))
    p = np.mean([m.predict_proba(Xte) for m in models.values()], axis=0)
    yte = y[te_i]
    s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
    dm = (s != 2) & (yte != 2)
    da = float((yte[dm] == s[dm]).mean()) if dm.any() else None
    ic, icp = spearmanr(p[:, 0] - p[:, 1], sub["fwd_ret"].values[te_i])

    te_rows = sub.iloc[te_i]
    dft = te_rows[["open", "high", "low", "close"]].copy()
    dft["call_p"], dft["put_p"], dft["skip_p"] = p[:, 0], p[:, 1], p[:, 2]
    dft["rsi14"] = te_rows["rsi14"] if "rsi14" in te_rows.columns else 50.0
    tr = run_variant(dft, build_aux(dft), NEW_VARIANT)
    if tr.empty:
        sim = {"trades": 0, "pf": None, "dd": 0, "net": 0, "exp": None}
    else:
        pnl = tr["pnl"]
        daily = tr.groupby(pd.to_datetime(tr["ts"]).dt.date)["pnl"].sum().cumsum()
        sim = {"trades": len(tr),
               "pf": round(float(pnl[pnl > 0].sum() / max(abs(pnl[pnl <= 0].sum()), 1e-9)), 2),
               "dd": round(float((daily - daily.cummax()).min()), 0),
               "net": round(float(pnl.sum()), 0),
               "exp": round(float(pnl.mean()), 0)}
    results[tag] = {"n_features": len(cols),
                    "test_dir_acc": round(da, 4) if da else None,
                    "IC": round(float(ic), 4), "IC_p": round(float(icp), 6),
                    "sim": sim}
    print(f"  {tag:28s} f={len(cols):3d} acc={da if da else 0:.4f} "
          f"IC={ic:+.4f}(p={icp:.4f}) | n={sim['trades']:3d} pf={sim['pf']} "
          f"dd={sim['dd']} exp={sim['exp']}")


def main():
    print("Building clean matrix...")
    dfm = build_features()
    dfm = create_labels(dfm)
    dfm["fwd_ret"] = dfm["close"].shift(-FWD_BARS) / dfm["close"] - 1

    fcols9 = joblib.load(ROOT / "models/feature_cols_v9.pkl")
    fam = map_families(fcols9)
    groups = {}
    for c, f in fam.items():
        groups.setdefault(f, []).append(c)
    groups["regime"] = groups.pop("regime_vix", []) + groups.pop("regime_session", [])

    results = {}
    print("\n[A] Drop-one-family ablations (full minus family):")
    for famname in ("price_action", "htf", "market_structure",
                    "volatility_proxy", "vwap", "regime"):
        cols = [c for c in fcols9 if c not in set(groups.get(famname, []))]
        evaluate(f"FULL_minus_{famname}", dfm, cols, results)
    evaluate("FULL_baseline", dfm, fcols9, results)

    print("\n[B] Incremental feature-add (strongest-first order from Phase 1):")
    order = ["htf", "price_action", "market_structure",
             "volatility_proxy", "vwap", "regime"]
    acc = []
    for famname in order:
        acc = acc + groups.get(famname, [])
        evaluate(f"INC_{'+'.join(o[:5] for o in order[:order.index(famname)+1])}",
                 dfm, list(acc), results)

    json.dump(results, open(RES / "phase1b_report.json", "w"), indent=1)
    print(f"\nReport → {RES / 'phase1b_report.json'}")


if __name__ == "__main__":
    main()
