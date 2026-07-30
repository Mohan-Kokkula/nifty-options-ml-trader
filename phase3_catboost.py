"""
phase3_catboost.py — CatBoost as a 5th brain, comparing all 5 architectures.

Same leak-clean V9 frame, same 8-fold walk-forward, same friction model.
Reports XGB, LGB, RF, NN, CatBoost as standalone brains + averaged ensembles.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, gc, json
from pathlib import Path
from datetime import date, timedelta
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import (
    build_frame, train_fold, _proba3, signals_from_probas,
    EMBARGO_DAYS, add_months,
)
from backtest_multibrain import train_rf_fold
from backtest_multibrain_v2 import train_nn_fold
from backtest_options import simulate_trades, build_iv_map
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("Installing catboost..."); import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostClassifier

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65


def _sample_weights(sub):
    y = sub["label"].values
    skip_pct = (y == 2).mean()
    trade_w = skip_pct / max(1 - skip_pct, 1e-9)
    w = np.where(y == 2, 1.0, trade_w)
    return w


def train_cat_fold(feat, fcols, train_mask):
    sub = feat[train_mask]
    if len(sub) < 5000: return None, None
    cut = int(len(sub) * 0.95)
    tr, ev = sub.iloc[:cut], sub.iloc[cut:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    Xev = sc.transform(ev[fcols].values)
    ytr, yev = tr["label"].values, ev["label"].values
    w = _sample_weights(tr)
    m = CatBoostClassifier(
        iterations=700, depth=5, learning_rate=0.03,
        loss_function="MultiClass", classes_count=3,
        l2_leaf_reg=3.0, random_state=42, verbose=False,
        early_stopping_rounds=50,
    )
    m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xev, yev), verbose=False)
    return m, sc


def sim_and_report(test, probas, iv, exp, name):
    sig = signals_from_probas(probas, CALL_THR, PUT_THR, SKIP_CEIL)
    tdf = simulate_trades(test, sig, probas, iv, exp)
    return tdf


def metrics(tdf):
    if tdf is None or tdf.empty: return dict(n=0, pf=float("nan"), wr=float("nan"), net=0.0, dd=0.0)
    net = tdf.net_option.values
    g = net[net > 0].sum(); l = -net[net <= 0].sum()
    pf = g / l if l > 0 else float("inf")
    eq = np.cumsum(net); dd = float((eq - np.maximum.accumulate(eq)).min())
    return dict(n=len(net), pf=pf, wr=(net > 0).mean() * 100,
                net=float(net.sum()), dd=dd)


def main():
    print("Phase 3 — 5-brain comparison (XGB, LGB, RF, NN, CatBoost)")
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    folds = []; f0 = date(2024, 7, 1); f_end = date(2026, 5, 1)
    while f0 < f_end:
        f1 = min(add_months(f0, 3), f_end); folds.append((f0, f1)); f0 = f1

    per_brain = {b: [] for b in ("xgb", "lgb", "rf", "nn", "cat")}
    per_ensemble = {b: [] for b in ("avg2_xl", "avg3_xlr", "avg4_xlrn", "avg5_all")}

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50: continue
        print(f"\n[Fold {k}/{len(folds)}] train<{cutoff} -> test {a}..{b}")

        models, sc_xl = train_fold(feat, fcols, tr_mask)
        if models is None: continue
        test = feat[te_mask]
        Xte_xl = sc_xl.transform(test[fcols].values)
        p_xgb = _proba3(list(models.values())[0], Xte_xl)
        p_lgb = _proba3(list(models.values())[1], Xte_xl)
        del models; gc.collect()
        print("  XGB+LGB done.")

        rf, sc_rf = train_rf_fold(feat, fcols, tr_mask)
        Xte_rf = sc_rf.transform(test[fcols].values)
        p_rf = _proba3(rf, Xte_rf)
        del rf; gc.collect()
        print("  RF done.")

        nn, sc_nn = train_nn_fold(feat, fcols, tr_mask)
        Xte_nn = sc_nn.transform(test[fcols].values)
        p_nn = _proba3(nn, Xte_nn)
        del nn; gc.collect()
        print("  NN done.")

        cat, sc_cat = train_cat_fold(feat, fcols, tr_mask)
        Xte_cat = sc_cat.transform(test[fcols].values)
        p_cat = _proba3(cat, Xte_cat)
        del cat; gc.collect()
        print("  CatBoost done.")

        # standalone per-brain sims
        for name, p in [("xgb", p_xgb), ("lgb", p_lgb), ("rf", p_rf),
                        ("nn", p_nn), ("cat", p_cat)]:
            tdf = sim_and_report(test, p, iv, exp, name)
            if len(tdf): tdf["fold"] = k; per_brain[name].append(tdf)

        # ensembles
        p_xl  = (p_xgb + p_lgb) / 2
        p_xlr = (p_xgb + p_lgb + p_rf) / 3
        p_xlrn = (p_xgb + p_lgb + p_rf + p_nn) / 4
        p_all = (p_xgb + p_lgb + p_rf + p_nn + p_cat) / 5
        for name, p in [("avg2_xl", p_xl), ("avg3_xlr", p_xlr),
                        ("avg4_xlrn", p_xlrn), ("avg5_all", p_all)]:
            tdf = sim_and_report(test, p, iv, exp, name)
            if len(tdf): tdf["fold"] = k; per_ensemble[name].append(tdf)

    # aggregate
    rows = []
    print("\n" + "=" * 90)
    print("  Phase 3 — 5-brain comparison table")
    print("=" * 90)
    print(f"  {'model':<14}{'trades':>7}{'PF':>7}{'WR%':>7}{'Net':>15}{'MaxDD':>15}")
    print("  " + "-" * 86)

    def emit(label, trades):
        full = pd.concat(trades, ignore_index=True) if trades else None
        m = metrics(full)
        print(f"  {label:<14}{m['n']:>7}{m['pf']:>7.3f}{m['wr']:>6.0f}%"
              f"{('Rs.' + format(m['net'], '+,.0f')):>15}"
              f"{('Rs.' + format(m['dd'], '+,.0f')):>15}")
        rows.append({"config": label, **m})
        if full is not None: full.to_csv(ROOT / f"logs/phase3_{label}.csv", index=False)

    for name in ("xgb", "lgb", "rf", "nn", "cat"): emit(name, per_brain[name])
    print("  " + "-" * 86)
    for name in ("avg2_xl", "avg3_xlr", "avg4_xlrn", "avg5_all"): emit(name, per_ensemble[name])
    print("=" * 90)

    with open(ROOT / "logs/phase3_summary.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
