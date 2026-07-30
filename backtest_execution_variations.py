"""
backtest_execution_variations.py — execution / geometry / structure sweep
==========================================================================
Built 2026-06-16. Tests whether OpenClaw's losses are driven by frequency,
option structure, SL/TP geometry, or signal quality — WITHOUT touching
features, labels, model, or indicators.

Method: leak-clean V9 walk-forward. Train ONCE per fold (the expensive part),
cache (test_df, probas, signals=config-A gate), then re-run the cheap
simulate_trades under every execution variant on the same OOS predictions:

  FREQUENCY  : baseline trades tagged with entry directional confidence,
               then post-hoc kept by confidence percentile (top 75/50/25/10%).
  GEOMETRY   : monkeypatch SL/TP ATR multipliers (wider TP, tighter SL, both).
  STRUCTURE  : SPREAD_HEDGE_WIDTH > 0 -> debit spread instead of naked long.

Reports PF / annualized Sharpe / Net / trade-count for every variation.
"""
import warnings; warnings.filterwarnings("ignore")
from datetime import date, timedelta
import numpy as np, pandas as pd

import backtest_options as bo
from backtest_options import simulate_trades, build_iv_map
from backtest_threshold_sweep import (
    build_frame, train_fold, _proba3, signals_from_probas,
    month_floor, add_months, EMBARGO_DAYS,
)

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65   # config A (current live gate)
BASE_SL_MULT = bo.SL_ATR_MULTIPLIER
BASE_TP_MULT = bo.TP_ATR_MULTIPLIER


def metrics(tdf):
    if tdf is None or len(tdf) == 0:
        return dict(n=0, pf=float("nan"), sharpe=float("nan"), net=0.0,
                    wr=float("nan"), avg=float("nan"))
    net = tdf.net_option.values
    n = len(net)
    g = net[net > 0].sum(); l = -net[net <= 0].sum()
    pf = g / l if l > 0 else float("inf")
    std = net.std(ddof=1) if n > 1 else 0.0
    t = tdf.sort_values("time")
    span = max((t.time.iloc[-1] - t.time.iloc[0]).days, 1)
    tpy = n / (span / 365.25)
    sharpe = (net.mean() / std * np.sqrt(tpy)) if std > 0 else 0.0
    return dict(n=n, pf=pf, sharpe=sharpe, net=net.sum(),
                wr=(net > 0).mean() * 100, avg=net.mean())


def run_sim(test, signals, probas, iv, exp):
    return simulate_trades(test, signals, probas, iv, exp)


def main():
    feat, fcols = build_frame()
    iv, exp = build_iv_map()
    test_start, test_end, step = date(2024, 7, 1), date(2026, 5, 1), 3
    folds = []
    f0 = test_start
    while f0 < test_end:
        f1 = min(add_months(f0, step), test_end); folds.append((f0, f1)); f0 = f1

    # variants that need a re-simulation (monkeypatched globals)
    GEOM = {
        "baseline":      dict(sl=BASE_SL_MULT, tp=BASE_TP_MULT, sw=0),
        "widerTP_1.5x":  dict(sl=BASE_SL_MULT, tp=BASE_TP_MULT * 1.5, sw=0),
        "tighterSL_.6x": dict(sl=BASE_SL_MULT * 0.6, tp=BASE_TP_MULT, sw=0),
        "tightSL+wideTP":dict(sl=BASE_SL_MULT * 0.6, tp=BASE_TP_MULT * 1.5, sw=0),
        "debit_spread":  dict(sl=BASE_SL_MULT, tp=BASE_TP_MULT, sw=100),
    }
    geom_trades = {k: [] for k in GEOM}
    base_with_conf = []      # baseline trades tagged with confidence

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50:
            continue
        print(f"  Fold {k}/{len(folds)}: train<{cutoff} -> test {a}..{b}")
        models, sc = train_fold(feat, fcols, tr_mask)
        if models is None:
            continue
        test = feat[te_mask]
        Xte = sc.transform(test[fcols].values)
        probas = np.mean([_proba3(m, Xte) for m in models.values()], axis=0)
        signals = signals_from_probas(probas, CALL_THR, PUT_THR, SKIP_CEIL)

        for name, p in GEOM.items():
            bo.SL_ATR_MULTIPLIER = p["sl"]
            bo.TP_ATR_MULTIPLIER = p["tp"]
            bo.SPREAD_HEDGE_WIDTH = p["sw"]
            tdf = run_sim(test, signals, probas, iv, exp)
            bo.SL_ATR_MULTIPLIER = BASE_SL_MULT
            bo.TP_ATR_MULTIPLIER = BASE_TP_MULT
            bo.SPREAD_HEDGE_WIDTH = 0
            if len(tdf):
                geom_trades[name].append(tdf)
                if name == "baseline":
                    # tag each trade with entry directional confidence
                    idxmap = {t: j for j, t in enumerate(test.index)}
                    confs, dirs = [], []
                    for _, row in tdf.iterrows():
                        j = idxmap.get(pd.Timestamp(row["entry_time"]))
                        if j is None:
                            confs.append(np.nan); continue
                        dcode = 0 if row["dir"] == "CALL" else 1
                        confs.append(float(probas[j, dcode]))
                    tt = tdf.copy(); tt["conf"] = confs
                    base_with_conf.append(tt)

    # ---------------- GEOMETRY / STRUCTURE report ----------------
    print("\n" + "=" * 78)
    print("  EXECUTION / GEOMETRY / STRUCTURE  (aggregate OOS, post-friction)")
    print("=" * 78)
    print(f"  {'variant':<17}{'trades':>7}{'PF':>7}{'Sharpe':>8}{'WinRt':>7}{'Net':>14}")
    print("  " + "-" * 74)
    base_m = None
    for name in GEOM:
        full = pd.concat(geom_trades[name], ignore_index=True) if geom_trades[name] else None
        m = metrics(full)
        if name == "baseline": base_m = m
        print(f"  {name:<17}{m['n']:>7}{m['pf']:>7.2f}{m['sharpe']:>8.2f}"
              f"{m['wr']:>6.0f}%{('Rs.'+format(m['net'],'+,.0f')):>14}")

    # ---------------- FREQUENCY report ----------------
    print("\n" + "=" * 78)
    print("  FREQUENCY — keep only top-confidence trades (post-hoc on baseline)")
    print("=" * 78)
    allb = pd.concat(base_with_conf, ignore_index=True)
    allb = allb.dropna(subset=["conf"])
    print(f"  {'keep':<14}{'trades':>7}{'PF':>7}{'Sharpe':>8}{'WinRt':>7}{'Net':>14}{'AvgTrd':>9}")
    print("  " + "-" * 74)
    for label, pct in [("ALL (100%)", 1.00), ("top 75%", 0.75),
                       ("top 50%", 0.50), ("top 25%", 0.25), ("top 10%", 0.10)]:
        if pct < 1.0:
            thr = allb["conf"].quantile(1 - pct)
            sub = allb[allb["conf"] >= thr]
        else:
            sub = allb
        m = metrics(sub)
        print(f"  {label:<14}{m['n']:>7}{m['pf']:>7.2f}{m['sharpe']:>8.2f}"
              f"{m['wr']:>6.0f}%{('Rs.'+format(m['net'],'+,.0f')):>14}"
              f"{('Rs.'+format(m['avg'],'+,.0f')):>9}")
    allb.to_csv("logs/exec_baseline_trades.csv", index=False)
    print("\n  baseline+conf trades -> logs/exec_baseline_trades.csv")


if __name__ == "__main__":
    main()
