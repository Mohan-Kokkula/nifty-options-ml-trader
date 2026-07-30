"""
calibration.py — PART B: probability calibration for the clean V9 champion.

Fits BOTH Isotonic Regression and Platt scaling (per class, one-vs-rest) on
the VALIDATION segment's predictions; method selection by validation ECE;
test-segment numbers reported but never used for selection.

No trading logic changes — produces a calibrator artifact only.

Outputs:
  models/calibration/calibrator.pkl        {method, per-class calibrators}
  models/calibration/calibration_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    create_labels, purged_train_val_test_split,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.train_model_v10 import (                         # noqa: E402
    load_option_features, compute_daily_option_context,
    merge_option_context_onto_5min, V10_FEATURES_CSV,
)

OUT = ROOT / "models" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["CALL", "PUT", "SKIP"]


def ece(prob, hit, bins=10):
    """Expected + max calibration error, equal-width bins, with reliability
    curve data (the numeric form of a reliability diagram)."""
    edges = np.linspace(0, 1, bins + 1)
    e, mce, n = 0.0, 0.0, len(prob)
    curve = []
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] + (i == bins - 1))
        if m.sum() == 0:
            continue
        conf, acc = float(prob[m].mean()), float(hit[m].mean())
        gap = abs(conf - acc)
        curve.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                      "n": int(m.sum()), "mean_prob": round(conf, 4),
                      "observed_rate": round(acc, 4),
                      "gap": round(gap, 4)})
        e += m.sum() / n * gap
        mce = max(mce, gap)
    return float(e), float(mce), curve


def brier(prob, hit):
    return float(np.mean((prob - hit) ** 2))


def main():
    # --live: calibrate whichever model core/ml_engine.py's load_model() will
    # load as the production champion (V10 preferred, V9 fallback — same
    # candidate order), so the calibrator matches the deployed model's
    # probability distribution.
    live = "--live" in sys.argv
    version = None
    if live:
        for version, mp, sp, fp in (
            ("V10", "models/nifty_v10_models.pkl", "models/nifty_v10_scaler.pkl", "models/feature_cols_v10.pkl"),
            ("V9", "models/nifty_v9_models.pkl", "models/nifty_v9_scaler.pkl", "models/feature_cols_v9.pkl"),
        ):
            if (ROOT / mp).exists() and (ROOT / sp).exists() and (ROOT / fp).exists():
                mdir = f"LIVE champion ({version})"
                break
        else:
            print("No live champion model found (models/nifty_v10_*.pkl or nifty_v9_*.pkl)")
            sys.exit(1)
    else:
        mdir, mp, sp, fp = ("sandbox_clean", "models/sandbox_clean/models.pkl",
                            "models/sandbox_clean/scaler.pkl",
                            "models/sandbox_clean/feature_cols.pkl")
    print(f"[1/4] Matrix + labels + predictions from {mdir}...")
    dfm = build_features()
    if version == "V10":
        print("[1.5/4] Merging V10 option-chain context (Bhavcopy) onto feature matrix...")
        bhav_df = load_option_features(str(ROOT / V10_FEATURES_CSV))
        opt_ctx = compute_daily_option_context(bhav_df) if not bhav_df.empty else pd.DataFrame()
        dfm = merge_option_context_onto_5min(dfm, opt_ctx)
    dfm = create_labels(dfm)
    fcols = joblib.load(ROOT / fp)
    models = joblib.load(ROOT / mp)
    scaler = joblib.load(ROOT / sp)
    sub = dfm.dropna(subset=fcols + ["label"])
    X, y = sub[fcols].values, sub["label"].values
    _, va_i, te_i = purged_train_val_test_split(len(X))

    def predict(ix):
        p = np.zeros((len(ix), 3))
        for s in range(0, len(ix), 50_000):
            Xs = scaler.transform(X[ix[s:s + 50_000]])
            p[s:s + 50_000] = np.mean(
                [m.predict_proba(Xs) for m in models.values()], axis=0)
        return p

    pv, pt = predict(va_i), predict(te_i)
    yv, yt = y[va_i], y[te_i]

    print("[2/4] Fit Platt + Isotonic per class on VAL...")
    fits = {"platt": [], "isotonic": []}
    for k in range(3):
        hit_v = (yv == k).astype(float)
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(pv[:, [k]], hit_v)
        fits["platt"].append(lr)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pv[:, k], hit_v)
        fits["isotonic"].append(iso)

    def apply(method, p):
        out = np.zeros_like(p)
        for k in range(3):
            if method == "platt":
                out[:, k] = fits["platt"][k].predict_proba(p[:, [k]])[:, 1]
            else:
                out[:, k] = fits["isotonic"][k].predict(p[:, k])
        rs = out.sum(axis=1, keepdims=True)
        return out / np.where(rs > 0, rs, 1.0)

    print("[3/4] Evaluate (selection on VAL, report TEST)...")
    report = {"model_source": mdir, "n_val": len(yv), "n_test": len(yt),
              "methods": {}}
    for method in ("raw", "platt", "isotonic"):
        rec = {}
        for split, p, yy in (("val", pv, yv), ("test", pt, yt)):
            pc = p if method == "raw" else apply(method, p)
            eces, mces, briers, curves = [], [], [], {}
            for k in range(3):
                hit = (yy == k).astype(float)
                e, mc, curve = ece(pc[:, k], hit)
                eces.append(e)
                mces.append(mc)
                briers.append(brier(pc[:, k], hit))
                curves[CLASSES[k]] = curve
            rec[split] = {"ece_per_class": [round(x, 4) for x in eces],
                          "ece_mean": round(float(np.mean(eces)), 4),
                          "mce_per_class": [round(x, 4) for x in mces],
                          "mce_max": round(float(np.max(mces)), 4),
                          "brier_per_class": [round(x, 4) for x in briers],
                          "brier_mean": round(float(np.mean(briers)), 4),
                          "reliability_diagram": curves if split == "test" else None}
        report["methods"][method] = rec
        print(f"   {method:9s} val ECE={rec['val']['ece_mean']:.4f} "
              f"MCE={rec['val']['mce_max']:.4f} "
              f"Brier={rec['val']['brier_mean']:.4f} | test "
              f"ECE={rec['test']['ece_mean']:.4f} "
              f"MCE={rec['test']['mce_max']:.4f} "
              f"Brier={rec['test']['brier_mean']:.4f}")

    chosen = min(("platt", "isotonic"),
                 key=lambda m: report["methods"][m]["val"]["ece_mean"])
    report["chosen"] = chosen
    report["selection_basis"] = "lowest validation ECE (test never used)"

    # "If the calibrator says 70%, does ~70% happen?" — TEST-set spot check
    # on the calibrated probabilities, all classes pooled, 0.60-0.80 buckets.
    pc_t = apply(chosen, pt)
    checks = []
    for lo, hi in ((0.60, 0.70), (0.65, 0.75), (0.70, 0.80)):
        probs, hits = [], []
        for k in range(3):
            m = (pc_t[:, k] >= lo) & (pc_t[:, k] < hi)
            probs.append(pc_t[m, k])
            hits.append((yt[m] == k).astype(float))
        probs = np.concatenate(probs)
        hits = np.concatenate(hits)
        if len(probs):
            checks.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": int(len(probs)),
                           "stated_prob": round(float(probs.mean()), 4),
                           "observed_rate": round(float(hits.mean()), 4)})
    report["seventy_pct_verification"] = checks
    print(f"[4/4] Chosen: {chosen} | 70% check: {checks}")

    joblib.dump({"method": chosen, "calibrators": fits[chosen],
                 "classes": CLASSES, "model_source": mdir},
                OUT / "calibrator.pkl")
    json.dump(report, open(OUT / "calibration_report.json", "w"), indent=1)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
