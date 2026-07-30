"""
backtest_rf_alone.py — RF as STANDALONE brain (not ensemble)
=============================================================
Built 2026-06-22. Honest gap-fill: prior multibrain test ran RF only as a
member of voting/averaged ensembles. This script runs RF ALONE through the
exact same 8-fold harness, same gate, same friction model, for direct
comparison vs the XGB+LGB baseline.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from backtest_threshold_sweep import (
    build_frame, _proba3, signals_from_probas,
    add_months, EMBARGO_DAYS,
)
from backtest_multibrain import train_rf_fold, metrics
from backtest_options import simulate_trades, build_iv_map

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65


def main():
    feat, fcols = build_frame()
    iv, exp = build_iv_map()
    test_start, test_end, step = date(2024, 7, 1), date(2026, 5, 1), 3
    folds = []
    f0 = test_start
    while f0 < test_end:
        f1 = min(add_months(f0, step), test_end); folds.append((f0, f1)); f0 = f1

    rf_trades, fold_summary = [], []

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50:
            continue
        print(f"\n  Fold {k}/{len(folds)}: train<{cutoff} -> test {a}..{b}")
        rf, sc = train_rf_fold(feat, fcols, tr_mask)
        if rf is None:
            print("    skipped"); continue
        test = feat[te_mask]
        Xte = sc.transform(test[fcols].values)
        probas = _proba3(rf, Xte)
        sig = signals_from_probas(probas, CALL_THR, PUT_THR, SKIP_CEIL)
        tdf = simulate_trades(test, sig, probas, iv, exp)
        if len(tdf):
            tdf["fold"] = k
            rf_trades.append(tdf)
            net = tdf.net_option.sum()
            wr = (tdf.net_option > 0).mean() * 100
            fold_summary.append((f"{a}..{b}", len(tdf), wr, net))
            print(f"    rf_alone {len(tdf):>4} trades net Rs.{net:+,.0f} winrate {wr:.0f}%")

    if not rf_trades:
        print("\n  No RF trades generated."); return
    full = pd.concat(rf_trades, ignore_index=True)
    m = metrics(full)

    print("\n" + "=" * 92)
    print("  RF-ALONE AGGREGATE (real option P&L, post-friction, same gate as baseline)")
    print("=" * 92)
    print(f"  {'Config':<14}{'Trd':>6}{'Win%':>7}{'PF':>7}{'Sharpe':>8}"
          f"{'MaxDD':>13}{'AvgTrd':>10}{'Net':>13}")
    print("  " + "-" * 88)
    print(f"  {'rf_alone':<14}{m['n']:>6}{m['wr']:>6.0f}%{m['pf']:>7.2f}{m['sharpe']:>8.2f}"
          f"{('Rs.'+format(abs(m['dd']),',.0f')):>13}"
          f"{('Rs.'+format(m['avg'],'+,.0f')):>10}"
          f"{('Rs.'+format(m['net'],'+,.0f')):>13}")
    print("  (For comparison: baseline XGB+LGB = PF 0.75, vote_2of3 = PF 0.80)")
    print("=" * 92)

    print("\n  Per-fold:")
    print(f"  {'Fold window':<22}{'Trades':>8}{'Net Win%':>10}{'Net P&L':>16}")
    print("  " + "-" * 55)
    for w, n, wr, net in fold_summary:
        print(f"  {w:<22}{n:>8}{wr:>9.0f}%{('Rs.' + format(net, '+,.0f')):>16}")

    full.to_csv(ROOT / "logs/multibrain_rf_alone.csv", index=False)
    print(f"\n  Trade detail -> logs/multibrain_rf_alone.csv ({len(full)} trades)")


if __name__ == "__main__":
    main()
