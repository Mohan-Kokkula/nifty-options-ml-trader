"""
train_htf36.py — PART A: HTF-36 challenger model.

Trains on ONLY tf15_*/tf30_*/tf60_* features (the family with the strongest
standalone evidence: test dir_acc 0.608, the study's only significant IC
+0.0177 p=0.027). Same clean pipeline, same leakage fixes, same purged
85/7.5/7.5 split, same hyperparameters, same eval methodology.

Outputs (sandbox only — never touches production paths):
  models/htf36/models.pkl        (dict of fitted estimators — codebase convention)
  models/htf36/scaler.pkl
  models/htf36/feature_cols.pkl
  models/htf36/metadata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    create_labels, purged_train_val_test_split, _fit_ensemble, FWD_BARS,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.phase1_edge_discovery import sample_weights      # noqa: E402

OUT = ROOT / "models" / "htf36"
OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.22


def main():
    print("[1/3] Clean matrix + labels...")
    dfm = build_features()
    base_fcols = joblib.load(ROOT / "models/sandbox_clean/feature_cols.pkl")
    fcols = [c for c in base_fcols if c.startswith(("tf15_", "tf30_", "tf60_"))]
    print(f"   HTF feature set: {len(fcols)} columns")
    dfm = create_labels(dfm)
    dfm["fwd_ret"] = dfm["close"].shift(-FWD_BARS) / dfm["close"] - 1
    sub = dfm.dropna(subset=fcols + ["label"])

    print("[2/3] Train (identical hyperparams + purged split)...")
    X, y = sub[fcols].values, sub["label"].values
    tr_i, va_i, te_i = purged_train_val_test_split(len(X))
    sc = StandardScaler()
    Xtr, Xvl, Xte = (sc.fit_transform(X[tr_i]), sc.transform(X[va_i]),
                     sc.transform(X[te_i]))
    tue = (sub["expiry_is_tue"].values[tr_i].astype(bool)
           if "expiry_is_tue" in sub.columns
           else np.zeros(len(tr_i), bool))
    models = _fit_ensemble(Xtr, y[tr_i], Xvl, y[va_i],
                           sample_weights(y[tr_i], tue))

    print("[3/3] Evaluate (production metric CONF=0.22 + IC)...")

    def _eval(Xe, ye, fwd):
        p = np.mean([m.predict_proba(Xe) for m in models.values()], axis=0)
        s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
        dm = (s != 2) & (ye != 2)
        da = float((ye[dm] == s[dm]).mean()) if dm.any() else None
        msk = ~np.isnan(fwd)
        ic, icp = spearmanr((p[:, 0] - p[:, 1])[msk], fwd[msk])
        return {"dir_acc": round(da, 4) if da else None,
                "n_signals": int((s != 2).sum()),
                "IC": round(float(ic), 4), "IC_p": round(float(icp), 6)}

    vm = _eval(Xvl, y[va_i], sub["fwd_ret"].values[va_i])
    tm = _eval(Xte, y[te_i], sub["fwd_ret"].values[te_i])
    meta = {
        "model": "HTF36 challenger", "n_features": len(fcols),
        "features": fcols,
        "train_end": str(sub.index[tr_i].max()),
        "val_end": str(sub.index[va_i].max()),
        "test_range": [str(sub.index[te_i].min()), str(sub.index[te_i].max())],
        "val": vm, "test": tm,
        "clean_baseline_ref": {"test_dir_acc": 0.5868, "test_IC": 0.0028},
        "phase1b_ref": {"htf_only_acc": 0.6082, "htf_only_IC": 0.0177},
    }
    joblib.dump(models, OUT / "models.pkl")
    joblib.dump(sc, OUT / "scaler.pkl")
    joblib.dump(fcols, OUT / "feature_cols.pkl")
    json.dump(meta, open(OUT / "metadata.json", "w"), indent=1)
    print(f"   VAL : {vm}")
    print(f"   TEST: {tm}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
