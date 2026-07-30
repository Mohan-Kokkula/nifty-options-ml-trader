"""
phase0_predcache_clean.py — predictions from the CLEAN retrained model over
the leak-fixed feature matrix. Output mirrors validation_predcache.pkl
(columns call_v9/put_v9/skip_v9 so validate_fixes_sim.py reads it as-is).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("predclean")

from scripts.phase0_retrain_clean import build_features  # noqa: E402

OUT = ROOT / "data/validation_predcache_clean.pkl"
MODELS = ROOT / "models/sandbox_clean"


def main():
    log.info("Building CLEAN feature matrix...")
    df_feat = build_features()
    models = joblib.load(MODELS / "models.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    fcols = joblib.load(MODELS / "feature_cols.pkl")
    for c in [c for c in fcols if c not in df_feat.columns]:
        df_feat[c] = 0.0
    sub = df_feat.dropna(subset=fcols)
    log.info(f"Predicting {len(sub):,} bars with CLEAN model...")
    X = sub[fcols].values
    probs = np.zeros((len(sub), 3))
    for i in range(0, len(sub), 50_000):
        Xs = scaler.transform(X[i:i + 50_000])
        probs[i:i + 50_000] = np.mean(
            [m.predict_proba(Xs) for m in models.values()], axis=0)
        log.info(f"  {min(i + 50_000, len(sub)):,}/{len(sub):,}")
    out = sub[["open", "high", "low", "close"]].copy()
    out["rsi14"] = sub["rsi14"] if "rsi14" in sub.columns else np.nan
    out["call_v9"], out["put_v9"], out["skip_v9"] = (
        probs[:, 0], probs[:, 1], probs[:, 2])
    out.index.name = "ts"
    out.to_pickle(OUT)
    log.info(f"Saved {len(out):,} rows → {OUT}")


if __name__ == "__main__":
    main()
