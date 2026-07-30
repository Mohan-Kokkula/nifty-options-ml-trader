"""
backtest_multibrain_v2.py — 4-brain test: XGB + LGB + RF + NN
==============================================================
Built 2026-06-22. Step 2 of "one by one" multi-brain experiment.

Adds Neural Network (sklearn MLPClassifier — 3 hidden layers) as 4th brain
alongside existing XGB, LGB, RF. Same leak-clean V9 frame, same 8-fold
walk-forward, same friction model, same gate (CALL_THR=0.32, PUT_THR=0.25,
SKIP_CEIL=0.65, MIN_EDGE=0.05).

Voting configurations tested (all on identical OOS predictions per fold):
  baseline      XGB+LGB averaged (current production)
  avg4          all 4 brains averaged probabilities
  any_brain     "if ANY brain says direction X AND no brain says opposite,
                 take X. else SKIP." (per user request — most permissive)
  vote_2of4     at least 2 brains agree, no opposing vote
  vote_3of4     majority (3 of 4)
  vote_4of4     unanimous

Each brain emits CALL/PUT/SKIP via the same signals_from_probas() gate.
Voting operates over the per-brain emitted signals, NOT over raw probas.
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
    build_frame, _proba3, signals_from_probas,
    add_months, EMBARGO_DAYS, MIN_EDGE,
)
from backtest_multibrain import train_rf_fold, metrics
from backtest_options import simulate_trades, build_iv_map
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65


def train_nn_fold(feat, fcols, train_mask):
    """Train MLP brain. 3 hidden layers, ReLU, dropout via L2 regularization,
    early stopping on internal validation split. Conservative regularization
    to discourage overfitting on this small-by-DL-standards dataset (~190k)."""
    sub = feat[train_mask]
    if len(sub) < 5000:
        return None, None
    cut = int(len(sub) * 0.95)
    tr = sub.iloc[:cut]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    ytr = tr["label"].values
    # sample weights — same construction as XGB/LGB/RF
    skip_pct = (ytr == 2).mean()
    trade_w = skip_pct / max(1 - skip_pct, 1e-9)
    w = np.where(ytr == 2, 1.0, trade_w)
    # MLPClassifier doesn't accept sample_weight directly → oversample trade rows
    rng = np.random.default_rng(42)
    trade_idx = np.where(ytr != 2)[0]
    skip_idx = np.where(ytr == 2)[0]
    oversample_n = int(len(trade_idx) * (trade_w - 1))
    oversampled = rng.choice(trade_idx, oversample_n, replace=True)
    idx = np.concatenate([np.arange(len(ytr)), oversampled])
    rng.shuffle(idx)
    Xtr_w, ytr_w = Xtr[idx], ytr[idx]

    nn = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,                  # L2
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=False,
    )
    nn.fit(Xtr_w, ytr_w)
    return nn, sc


def vote_signals_4brain(brain_sigs):
    """brain_sigs: list of 4 arrays of CALL/PUT/SKIP per bar.
    Returns dict of {rule_name: aggregate signal array}."""
    sigs = np.stack(brain_sigs, axis=1)  # (n_bars, 4)
    n_bars = sigs.shape[0]
    n_call = (sigs == 0).sum(axis=1)
    n_put = (sigs == 1).sum(axis=1)

    def _rule(required):
        out = np.full(n_bars, 2)
        out[(n_call >= required) & (n_put == 0)] = 0
        out[(n_put >= required) & (n_call == 0)] = 1
        return out

    return {
        "any_brain": _rule(1),     # ≥1 agrees, no opposing — user's request
        "vote_2of4": _rule(2),
        "vote_3of4": _rule(3),
        "vote_4of4": _rule(4),
    }


def main():
    feat, fcols = build_frame()
    iv, exp = build_iv_map()
    test_start, test_end, step = date(2024, 7, 1), date(2026, 5, 1), 3
    folds = []
    f0 = test_start
    while f0 < test_end:
        f1 = min(add_months(f0, step), test_end)
        folds.append((f0, f1)); f0 = f1

    cfg_trades = {k: [] for k in
                  ("baseline", "avg4", "any_brain",
                   "vote_2of4", "vote_3of4", "vote_4of4")}

    # need train_fold from threshold_sweep for XGB+LGB
    from backtest_threshold_sweep import train_fold

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50:
            continue
        print(f"\n  Fold {k}/{len(folds)}: train<{cutoff} -> test {a}..{b}")

        # 1. XGB + LGB
        models, sc = train_fold(feat, fcols, tr_mask)
        if models is None:
            print("    XGB+LGB train failed, skipping"); continue
        test = feat[te_mask]
        Xte_xl = sc.transform(test[fcols].values)
        probas_xgb = _proba3(list(models.values())[0], Xte_xl)
        probas_lgb = _proba3(list(models.values())[1], Xte_xl)
        probas_xl_avg = (probas_xgb + probas_lgb) / 2
        print(f"    XGB+LGB trained.")

        # 2. RF
        rf, sc_rf = train_rf_fold(feat, fcols, tr_mask)
        if rf is None:
            print("    RF train failed, skipping"); continue
        Xte_rf = sc_rf.transform(test[fcols].values)
        probas_rf = _proba3(rf, Xte_rf)
        print(f"    RF trained.")

        # 3. NN
        nn, sc_nn = train_nn_fold(feat, fcols, tr_mask)
        if nn is None:
            print("    NN train failed, skipping"); continue
        Xte_nn = sc_nn.transform(test[fcols].values)
        probas_nn = _proba3(nn, Xte_nn)
        print(f"    NN trained.")

        # Per-brain signals (each at the same gate)
        sig_xgb = signals_from_probas(probas_xgb, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_lgb = signals_from_probas(probas_lgb, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_rf  = signals_from_probas(probas_rf,  CALL_THR, PUT_THR, SKIP_CEIL)
        sig_nn  = signals_from_probas(probas_nn,  CALL_THR, PUT_THR, SKIP_CEIL)

        # Configurations
        sig_baseline = signals_from_probas(probas_xl_avg, CALL_THR, PUT_THR, SKIP_CEIL)
        probas_avg4  = (probas_xgb + probas_lgb + probas_rf + probas_nn) / 4
        sig_avg4     = signals_from_probas(probas_avg4, CALL_THR, PUT_THR, SKIP_CEIL)
        vote_sigs    = vote_signals_4brain([sig_xgb, sig_lgb, sig_rf, sig_nn])

        # Simulate each (same friction, same simulate_trades)
        for name, sig in [
            ("baseline",  sig_baseline),
            ("avg4",      sig_avg4),
            ("any_brain", vote_sigs["any_brain"]),
            ("vote_2of4", vote_sigs["vote_2of4"]),
            ("vote_3of4", vote_sigs["vote_3of4"]),
            ("vote_4of4", vote_sigs["vote_4of4"]),
        ]:
            tdf = simulate_trades(test, sig, probas_xl_avg, iv, exp)
            if len(tdf):
                tdf["fold"] = k
                cfg_trades[name].append(tdf)
            tag = f"net Rs.{tdf.net_option.sum():+,.0f}" if len(tdf) else "(no trades)"
            print(f"    {name:<11} {len(tdf):>4} trades  {tag}")

    # Aggregate report
    print("\n" + "=" * 100)
    print("  4-BRAIN AGGREGATE (real option P&L, post-friction, identical gate)")
    print("=" * 100)
    print(f"  {'Config':<12}{'Trd':>6}{'Win%':>7}{'PF':>7}{'Sharpe':>8}"
          f"{'MaxDD':>13}{'AvgTrd':>10}{'Net':>13}")
    print("  " + "-" * 96)
    rows = {}
    for name in ("baseline", "avg4", "any_brain", "vote_2of4", "vote_3of4", "vote_4of4"):
        full = pd.concat(cfg_trades[name], ignore_index=True) if cfg_trades[name] else None
        m = metrics(full)
        rows[name] = m
        if m["n"] == 0:
            print(f"  {name:<12} 0 trades"); continue
        print(f"  {name:<12}{m['n']:>6}{m['wr']:>6.0f}%{m['pf']:>7.2f}{m['sharpe']:>8.2f}"
              f"{('Rs.'+format(abs(m['dd']),',.0f')):>13}"
              f"{('Rs.'+format(m['avg'],'+,.0f')):>10}"
              f"{('Rs.'+format(m['net'],'+,.0f')):>13}")
    print("=" * 100)

    # Verdict
    baseline_pf = rows["baseline"].get("pf", 0)
    best_pf_name = max(rows, key=lambda k: rows[k].get("pf", 0))
    best_pf = rows[best_pf_name].get("pf", 0)
    print(f"\n  Baseline (XGB+LGB) PF: {baseline_pf:.2f}")
    print(f"  Best config: {best_pf_name} PF: {best_pf:.2f}")
    print(f"  any_brain PF (user's requested rule): {rows['any_brain'].get('pf', 0):.2f}  "
          f"trades: {rows['any_brain'].get('n', 0)}")
    if best_pf > 1.0:
        print(f"  [PASS] At least one config clears PF > 1.0")
    elif best_pf > baseline_pf * 1.10:
        print(f"  [PARTIAL] >10% improvement over baseline but sub-1.0")
    else:
        print(f"  [FAIL] No material improvement. Architecture exhausted.")

    for name, trs in cfg_trades.items():
        if trs:
            pd.concat(trs, ignore_index=True).to_csv(
                ROOT / f"logs/multibrain_v2_{name}.csv", index=False)
    print("\n  Per-config trade CSVs -> logs/multibrain_v2_<config>.csv\n")


if __name__ == "__main__":
    main()
