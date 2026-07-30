"""
phase1_optwin_fix.py — re-evaluate the option-window families with metrics
robust to majority-class collapse (24k training rows make the regularized
ensemble argmax-degenerate, so CONF-based dir_acc reads zero).

Metrics per run, on the frozen in-window test segment:
  - IC: Spearman corr of (call_p - put_p) vs forward 3-bar return
  - top-decile dir_acc: among the 10% most directionally-confident test bars
    that have a directional label, fraction where sign(call_p-put_p) matches
  - same for price-control on the SAME window (fair comparison)
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
from scripts.train_model_v10 import (                         # noqa: E402
    load_option_features, compute_daily_option_context,
    merge_option_context_onto_5min,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.phase1_edge_discovery import (                   # noqa: E402
    IV_COLS, OI_COLS, map_families, sample_weights,
)

RES = ROOT / "data/validation_results"


def main():
    print("Building matrix...")
    dfm = build_features()
    bhav = load_option_features(str(ROOT / "data/v10_training_features.csv"))
    dfm = merge_option_context_onto_5min(
        dfm, compute_daily_option_context(bhav))
    dfm = create_labels(dfm)
    dfm["fwd_ret"] = dfm["close"].shift(-FWD_BARS) / dfm["close"] - 1

    fcols9 = joblib.load(ROOT / "models/feature_cols_v9.pkl")
    fam = map_families(fcols9 + IV_COLS + OI_COLS)
    price = [c for c, f in fam.items() if f == "price_action"]

    dwin = dfm.dropna(subset=["atm_iv"])
    print(f"option window: {len(dwin):,} rows "
          f"({dwin.index[0].date()} → {dwin.index[-1].date()})")

    configs = {
        "price_control": price,
        "optchain_only": IV_COLS + OI_COLS,
        "oi_only": OI_COLS,
        "iv_only": IV_COLS,
        "price_plus_opt": price + IV_COLS + OI_COLS,
    }
    out = {}
    for tag, cols in configs.items():
        sub = dwin.dropna(subset=cols + ["label", "fwd_ret"])
        X, y = sub[cols].values, sub["label"].values
        tr_i, va_i, te_i = purged_train_val_test_split(len(X))
        sc = StandardScaler()
        Xtr, Xvl, Xte = (sc.fit_transform(X[tr_i]), sc.transform(X[va_i]),
                         sc.transform(X[te_i]))
        tue = (sub["expiry_is_tue"].values[tr_i].astype(bool)
               if "expiry_is_tue" in sub.columns
               else np.zeros(len(tr_i), bool))
        models = _fit_ensemble(Xtr, y[tr_i], Xvl, y[va_i],
                               sample_weights(y[tr_i], tue))
        p = np.mean([m.predict_proba(Xte) for m in models.values()], axis=0)
        edge = p[:, 0] - p[:, 1]
        fwd = sub["fwd_ret"].values[te_i]
        yte = y[te_i]
        ic, ic_p = spearmanr(edge, fwd)
        # top-decile directional accuracy
        k = max(50, int(0.10 * len(edge)))
        top = np.argsort(-np.abs(edge))[:k]
        dirlab = top[yte[top] != 2]
        if len(dirlab):
            pred_dir = np.where(edge[dirlab] > 0, 0, 1)
            tda = float((pred_dir == yte[dirlab]).mean())
        else:
            # fall back to sign-of-forward-return agreement
            pred_dir = edge[top] > 0
            tda = float((pred_dir == (fwd[top] > 0)).mean())
        out[tag] = {"n_features": len(cols), "test_bars": int(len(edge)),
                    "IC_spearman": round(float(ic), 4),
                    "IC_pvalue": round(float(ic_p), 6),
                    "top_decile_dir_acc": round(tda, 4),
                    "n_top_decile_dir_labeled": int(len(dirlab))}
        print(f"  {tag:16s} feats={len(cols):3d} IC={ic:+.4f} (p={ic_p:.4f}) "
              f"top-decile dir_acc={tda:.3f} (n={len(dirlab)})")

    json.dump(out, open(RES / "phase1_optwin_report.json", "w"), indent=1)
    print(f"Saved → {RES / 'phase1_optwin_report.json'}")


if __name__ == "__main__":
    main()
