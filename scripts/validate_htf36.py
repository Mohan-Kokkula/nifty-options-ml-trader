"""
validate_htf36.py — HTF-36 challenger validation vs current V9 champion.

Scope (per validation brief):
  - Challenger: HTF trend-state features ONLY (tf15_*, tf30_*, tf60_* — 28
    columns; "no other families" excludes the day_ema/day_rsi/day_atr/
    day_week_pos columns that the phase1 "htf" family (36) also contained).
  - Champion: current production V9 (models/sandbox_clean, 170 features,
    the clean-pipeline model that was promoted as the live champion).
  - Same clean feature pipeline (phase0_retrain_clean.build_features),
    same purged 85/7.5/7.5 split (train_model_v9.purged_train_val_test_split),
    same hyperparameters (train_model_v9._fit_ensemble — both models were
    already trained that way; this script does NOT retrain).
  - Same frozen test, same gate stack (NEW_VARIANT, CONF=0.22). No tuning,
    no gate changes.

Metrics (val + frozen test): dir_acc (production metric, CONF=0.22), IC
(Spearman of (call_p-put_p) vs fwd 3-bar return), plus NEW-gate trade-sim
on the frozen test segment: trades, win rate, PF, expectancy, Sharpe,
Sortino, maxDD, CAGR, recovery. Bootstrap 95% CIs for dir_acc/IC (row
resample, test set) and for the trade-sim metrics (trade resample).

Output: data/validation_results/htf36_report.json
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
from scripts.phase1_edge_discovery import build_aux, NEW_VARIANT  # noqa: E402
from scripts.validate_fixes_sim import run_variant, metrics, CAPITAL  # noqa: E402

RES = ROOT / "data/validation_results"
CONF = 0.22
N_BOOT_ROWS = 2000      # bootstrap for dir_acc / IC (test rows)
N_BOOT_TRADES = 5000    # bootstrap for trade-sim metrics
SEED = 42


def predict(models, X):
    return np.mean([m.predict_proba(X) for m in models.values()], axis=0)


def signal(p):
    return np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)


def dir_acc_ic(p, y, fwd):
    s = signal(p)
    dm = (s != 2) & (y != 2)
    da = float((y[dm] == s[dm]).mean()) if dm.any() else None
    msk = ~np.isnan(fwd)
    if msk.sum() > 2:
        ic, icp = spearmanr((p[:, 0] - p[:, 1])[msk], fwd[msk])
    else:
        ic, icp = np.nan, np.nan
    return da, int((s != 2).sum()), float(ic), float(icp)


def ci(arr, lo=2.5, hi=97.5):
    arr = [v for v in arr if v is not None and np.isfinite(v)]
    if not arr:
        return [None, None]
    return [float(np.percentile(arr, lo)), float(np.percentile(arr, hi))]


def bootstrap_dir_acc_ic(p, y, fwd, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, ics = [], []
    for _ in range(n_boot):
        ridx = rng.integers(0, n, n)
        da, _, ic, _ = dir_acc_ic(p[ridx], y[ridx], fwd[ridx])
        if da is not None:
            accs.append(da)
        if np.isfinite(ic):
            ics.append(ic)
    return {"dir_acc_ci": ci(accs), "IC_ci": ci(ics)}


def bootstrap_trade_metrics(tr, yrs, n_boot, seed):
    if tr.empty or len(tr) < 3:
        return None
    n = len(tr)
    rng = np.random.default_rng(seed)
    keys = ["wr", "pf", "sharpe", "sortino", "maxdd", "expectancy", "net", "recovery"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        ridx = rng.integers(0, n, n)
        trb = tr.iloc[ridx].reset_index(drop=True)
        m = metrics(trb, yrs)
        for k in keys:
            v = m.get(k)
            if v is not None:
                samples[k].append(v)
    return {f"{k}_ci": ci(v) for k, v in samples.items()}


def clean(obj):
    """Recursively round floats and replace NaN/Inf with None for JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def load_artifacts(name):
    d = ROOT / "models" / name
    return (joblib.load(d / "models.pkl"), joblib.load(d / "scaler.pkl"),
            joblib.load(d / "feature_cols.pkl"))


def evaluate(tag, dfm, fcols, models, sc):
    sub = dfm.dropna(subset=fcols + ["label", "fwd_ret"])
    X = sub[fcols].values
    y = sub["label"].values
    fwd = sub["fwd_ret"].values
    tr_i, va_i, te_i = purged_train_val_test_split(len(X))

    Xvl, Xte = sc.transform(X[va_i]), sc.transform(X[te_i])
    p_val, p_te = predict(models, Xvl), predict(models, Xte)

    va_acc, va_n, va_ic, va_icp = dir_acc_ic(p_val, y[va_i], fwd[va_i])
    te_acc, te_n, te_ic, te_icp = dir_acc_ic(p_te, y[te_i], fwd[te_i])

    val_boot = bootstrap_dir_acc_ic(p_val, y[va_i], fwd[va_i], N_BOOT_ROWS, SEED)
    test_boot = bootstrap_dir_acc_ic(p_te, y[te_i], fwd[te_i], N_BOOT_ROWS, SEED + 1)

    # NEW-gate trade sim on the frozen test segment (unmodified gate stack)
    te_rows = sub.iloc[te_i]
    dft = te_rows[["open", "high", "low", "close"]].copy()
    dft["call_p"], dft["put_p"], dft["skip_p"] = p_te[:, 0], p_te[:, 1], p_te[:, 2]
    dft["rsi14"] = (te_rows["rsi14"] if "rsi14" in te_rows.columns
                    else pd.Series(50.0, index=te_rows.index)).fillna(50.0)
    auxt = build_aux(dft)
    trades = run_variant(dft, auxt, NEW_VARIANT)

    test_start, test_end = te_rows.index[0], te_rows.index[-1]
    yrs = max((test_end - test_start).days / 365.25, 0.01)
    sim = metrics(trades, yrs)
    sim_boot = bootstrap_trade_metrics(trades, yrs, N_BOOT_TRADES, SEED + 2)

    print(f"\n[{tag}] n_features={len(fcols)} rows={len(sub):,}")
    print(f"  VAL : dir_acc={va_acc} (n={va_n})  IC={va_ic:+.4f} (p={va_icp:.4f})")
    print(f"  TEST: dir_acc={te_acc} (n={te_n})  IC={te_ic:+.4f} (p={te_icp:.4f})")
    print(f"  TEST range: {test_start} -> {test_end}  ({yrs:.3f} yrs)")
    print(f"  SIM (NEW gate): trades={sim.get('trades')} pf={sim.get('pf')} "
          f"exp={sim.get('expectancy')} maxdd={sim.get('maxdd')} "
          f"sharpe={sim.get('sharpe')} net={sim.get('net')}")

    return {
        "n_features": len(fcols),
        "rows": int(len(sub)),
        "train_end": str(sub.index[tr_i].max()),
        "val_end": str(sub.index[va_i].max()),
        "test_range": [str(test_start), str(test_end)],
        "test_span_years": yrs,
        "val": {"dir_acc": va_acc, "n_signals": va_n, "IC": va_ic, "IC_p": va_icp,
                **val_boot},
        "test": {"dir_acc": te_acc, "n_signals": te_n, "IC": te_ic, "IC_p": te_icp,
                 **test_boot},
        "sim": sim,
        "sim_ci": sim_boot,
    }


def main():
    print("[1/3] Building clean feature matrix + labels...")
    dfm = build_features()
    dfm = create_labels(dfm)
    dfm["fwd_ret"] = dfm["close"].shift(-FWD_BARS) / dfm["close"] - 1

    print("\n[2/3] Loading pre-trained artifacts (no retraining)...")
    champ_models, champ_sc, champ_fcols = load_artifacts("sandbox_clean")
    htf_models, htf_sc, htf_fcols = load_artifacts("htf36")
    print(f"  champion (sandbox_clean): {len(champ_fcols)} features")
    print(f"  HTF36 (tf15/tf30/tf60)  : {len(htf_fcols)} features")
    assert set(htf_fcols).issubset(set(champ_fcols)), \
        "HTF36 features must be a subset of the champion's clean feature set"
    assert all(c.startswith(("tf15_", "tf30_", "tf60_")) for c in htf_fcols), \
        "HTF36 must use ONLY tf15/tf30/tf60 families"

    print("\n[3/3] Evaluating champion vs HTF36 (frozen test, NEW gate, CONF=0.22)...")
    champion = evaluate("CHAMPION (V9 clean, 170f)", dfm, champ_fcols, champ_models, champ_sc)
    htf36 = evaluate("HTF36 (28f tf15/tf30/tf60)", dfm, htf_fcols, htf_models, htf_sc)

    report = {
        "spec": {
            "challenger": "HTF36 — tf15_*/tf30_*/tf60_* only (28 cols), "
                           "no other families",
            "champion": "current V9 production champion (models/sandbox_clean, "
                        "170-feature clean pipeline)",
            "pipeline": "scripts/phase0_retrain_clean.build_features + "
                        "train_model_v9.create_labels (clean, leak-fixed)",
            "split": "train_model_v9.purged_train_val_test_split "
                     "(85/7.5/7.5, purge=FWD_BARS, embargo=EMBARGO_BARS)",
            "hyperparams": "train_model_v9._fit_ensemble (unchanged, "
                           "pre-trained artifacts loaded as-is)",
            "gate": "NEW_VARIANT (phase1_edge_discovery), CONF=0.22, unmodified",
            "bootstrap": {
                "dir_acc_IC": f"{N_BOOT_ROWS}x row resample with replacement "
                               "on frozen-test predictions",
                "trade_sim": f"{N_BOOT_TRADES}x trade resample with replacement, "
                              "metrics() recomputed each draw",
            },
            "no_tuning": True, "no_gate_changes": True,
        },
        "champion": champion,
        "htf36": htf36,
    }

    out = clean(report)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "htf36_report.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nReport -> {RES / 'htf36_report.json'}")


if __name__ == "__main__":
    main()
