"""
backtest_multibrain.py — multi-brain voting test on EXISTING data
==================================================================
Built 2026-06-21. Tests whether adding RandomForest as a 3rd brain to the
existing XGB+LGB ensemble improves walk-forward profitability, using exactly
the same leak-clean V9 frame, the same 8-fold structure, same friction model.

Configurations tested (all on identical OOS predictions per fold):
  A. baseline   XGB + LGB (current production ensemble, averaged probs)
  B. avg3       XGB + LGB + RF averaged probs
  C. vote_2of3  Each brain emits CALL/PUT/SKIP; trade only if >=2 agree on
                directional class; otherwise force SKIP
  D. vote_3of3  Unanimous agreement only

Same gate as A_current: CALL_THR=0.32, PUT_THR=0.25, SKIP_CEIL=0.65, MIN_EDGE=0.05.

Step 1 of the "one by one" multi-brain experiment. NN deferred to Step 2 only
if this step produces PF > 1.0 in any voting config.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import (
    build_frame, train_fold, _proba3, signals_from_probas,
    month_floor, add_months, EMBARGO_DAYS, MIN_EDGE,
)
from backtest_options import simulate_trades, build_iv_map
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65  # config A_current gate


def train_rf_fold(feat, fcols, train_mask):
    """Train Random Forest brain on same data as XGB+LGB. Returns (model, scaler).
    Hyperparams chosen to be comparable depth/capacity to XGB n_estimators=700/depth=5."""
    sub = feat[train_mask]
    if len(sub) < 5000:
        return None, None
    cut = int(len(sub) * 0.95)
    tr = sub.iloc[:cut]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    ytr = tr["label"].values
    # sample weights — same construction as XGB+LGB
    skip_pct = (ytr == 2).mean()
    trade_w = skip_pct / max(1 - skip_pct, 1e-9)
    w = np.where(ytr == 2, 1.0, trade_w)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=12,
        min_samples_leaf=30, max_features="sqrt",
        n_jobs=-1, random_state=42,
        class_weight=None,                # sample_weight handles imbalance
    )
    rf.fit(Xtr, ytr, sample_weight=w)
    return rf, sc


def vote_signals(probas_list, thresholds):
    """Per-bar consensus voting across N brains.
    Each brain emits CALL/PUT/SKIP using the same gate; trade only if all
    voting-rule-required brains agree on the directional class."""
    sigs_per_brain = [signals_from_probas(p, *thresholds) for p in probas_list]
    sigs = np.stack(sigs_per_brain, axis=1)  # (n_bars, n_brains)
    n_bars, n_brains = sigs.shape
    out = np.full(n_bars, 2)  # default SKIP

    # required agreement: 2 of n or n of n (unanimous)
    def _consensus(required):
        result = np.full(n_bars, 2)
        for direction in (0, 1):           # CALL=0, PUT=1
            agree_count = (sigs == direction).sum(axis=1)
            result[agree_count >= required] = direction
        return result

    return {
        "vote_2of3": _consensus(2),
        "vote_3of3": _consensus(3),
    }


def metrics(tdf):
    if tdf is None or len(tdf) == 0:
        return dict(n=0, pf=float("nan"), sharpe=float("nan"),
                    net=0.0, wr=float("nan"), avg=float("nan"), dd=0.0)
    net = tdf.net_option.values
    n = len(net)
    g = net[net > 0].sum(); l = -net[net <= 0].sum()
    pf = g / l if l > 0 else float("inf")
    std = net.std(ddof=1) if n > 1 else 0.0
    t = tdf.sort_values("time")
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days, 1)
    tpy = n / (span / 365.25)
    sharpe = (net.mean() / std * np.sqrt(tpy)) if std > 0 else 0.0
    eq = np.cumsum(net); dd = float((eq - np.maximum.accumulate(eq)).min())
    return dict(n=n, pf=pf, sharpe=sharpe, net=net.sum(),
                wr=(net > 0).mean() * 100, avg=net.mean(), dd=dd)


def main():
    feat, fcols = build_frame()
    iv, exp = build_iv_map()
    test_start, test_end, step = date(2024, 7, 1), date(2026, 5, 1), 3
    folds = []
    f0 = test_start
    while f0 < test_end:
        f1 = min(add_months(f0, step), test_end)
        folds.append((f0, f1)); f0 = f1

    cfg_trades = {k: [] for k in ("baseline", "avg3", "vote_2of3", "vote_3of3")}

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50:
            continue
        print(f"\n  Fold {k}/{len(folds)}: train<{cutoff} -> test {a}..{b}")

        # XGB+LGB (existing brains)
        models, sc = train_fold(feat, fcols, tr_mask)
        if models is None:
            continue
        test = feat[te_mask]
        Xte_xgb = sc.transform(test[fcols].values)
        probas_xgb_lgb = [_proba3(m, Xte_xgb) for m in models.values()]
        probas_baseline = np.mean(probas_xgb_lgb, axis=0)
        print(f"    XGB+LGB trained.")

        # RF (3rd brain)
        rf, sc_rf = train_rf_fold(feat, fcols, tr_mask)
        if rf is None:
            continue
        Xte_rf = sc_rf.transform(test[fcols].values)
        probas_rf = _proba3(rf, Xte_rf)
        print(f"    RF trained.")

        # Averaged 3-brain
        probas_avg3 = np.mean([*probas_xgb_lgb, probas_rf], axis=0)

        # Signals
        sig_baseline = signals_from_probas(probas_baseline, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_avg3     = signals_from_probas(probas_avg3,     CALL_THR, PUT_THR, SKIP_CEIL)
        vote_sigs = vote_signals(
            [probas_xgb_lgb[0], probas_xgb_lgb[1], probas_rf],
            (CALL_THR, PUT_THR, SKIP_CEIL),
        )

        # Simulate each config (same simulate_trades for apples-apples friction)
        for name, sig in [("baseline", sig_baseline),
                          ("avg3", sig_avg3),
                          ("vote_2of3", vote_sigs["vote_2of3"]),
                          ("vote_3of3", vote_sigs["vote_3of3"])]:
            tdf = simulate_trades(test, sig, probas_baseline, iv, exp)
            if len(tdf):
                tdf["fold"] = k
                cfg_trades[name].append(tdf)
            print(f"    {name:<12} {len(tdf):>4} trades net Rs.{tdf.net_option.sum():+,.0f}"
                  if len(tdf) else f"    {name:<12}    0 trades")

    # Aggregate report
    print("\n" + "=" * 100)
    print("  MULTI-BRAIN AGGREGATE (real option P&L, post-friction, identical gate)")
    print("=" * 100)
    print(f"  {'Config':<12}{'Trd':>6}{'Win%':>7}{'PF':>7}{'Sharpe':>8}"
          f"{'MaxDD':>13}{'AvgTrd':>10}{'Net':>13}")
    print("  " + "-" * 96)
    rows = {}
    for name in ("baseline", "avg3", "vote_2of3", "vote_3of3"):
        full = pd.concat(cfg_trades[name], ignore_index=True) if cfg_trades[name] else None
        m = metrics(full)
        rows[name] = m
        if m["n"] == 0:
            print(f"  {name:<12} 0 trades")
            continue
        print(f"  {name:<12}{m['n']:>6}{m['wr']:>6.0f}%{m['pf']:>7.2f}{m['sharpe']:>8.2f}"
              f"{('Rs.'+format(abs(m['dd']),',.0f')):>13}"
              f"{('Rs.'+format(m['avg'],'+,.0f')):>10}"
              f"{('Rs.'+format(m['net'],'+,.0f')):>13}")
    print("=" * 100)

    # Verdict
    baseline_pf = rows["baseline"].get("pf", 0)
    best_voting = max(rows["vote_2of3"].get("pf", 0),
                      rows["vote_3of3"].get("pf", 0))
    print(f"\n  Baseline (XGB+LGB) PF: {baseline_pf:.2f}")
    print(f"  Best voting config PF: {best_voting:.2f}")
    if best_voting > 1.0:
        print(f"  [STEP-1 PASS] Voting clears PF > 1.0 — consider NN as 4th brain (Step 2)")
    elif best_voting > baseline_pf * 1.10:
        print(f"  [STEP-1 PARTIAL] Voting improves PF >10% over baseline but stays sub-1.0")
        print(f"                  NN as 4th brain may help; modest expected gain.")
    else:
        print(f"  [STEP-1 FAIL] Voting does NOT materially improve PF.")
        print(f"                NN unlikely to fix. Multi-brain hypothesis closed.")

    # Persist for inspection
    for name in ("baseline", "avg3", "vote_2of3", "vote_3of3"):
        if cfg_trades[name]:
            pd.concat(cfg_trades[name], ignore_index=True).to_csv(
                ROOT / f"logs/multibrain_{name}.csv", index=False)
    print("\n  Per-config trade CSVs -> logs/multibrain_<config>.csv\n")


if __name__ == "__main__":
    main()
