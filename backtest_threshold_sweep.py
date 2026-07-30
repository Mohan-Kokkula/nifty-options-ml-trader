"""
backtest_threshold_sweep.py — LEAK-CLEAN walk-forward gate-threshold sweep
==========================================================================
Built 2026-06-16 for the Risk-Committee gate-threshold review.

WHY A NEW SCRIPT
----------------
backtest_walkforward_options.py imports its feature pipeline from
train_model_v8, whose features_daily() still has the pre-2026-06-11
daily look-ahead leak (intraday bars see functions of the SAME day's
15:30 close). Its absolute numbers are the known "92.9% fiction".

This script imports the LEAK-FIXED V9 pipeline (train_model_v9, which
shifts all close-derived daily features by 1 day) and is otherwise a
faithful clone of the true walk-forward:
  * rolls train/test windows forward, retrains XGB+LGB per fold on
    pre-test data only, with a 3-day embargo;
  * prices every OOS trade as a real option (theta, spread, STT,
    brokerage, GST) via backtest_options.simulate_trades;
  * TRAINS ONCE PER FOLD, then scores the SAME out-of-sample
    probabilities under all 5 gate-threshold configs, so the configs
    are compared on identical predictions (the only thing that varies
    is the gate).

Gate rule (matches ml_engine.predict signal selection), per config:
    CALL if call_p >= call_thr and call_p-put_p >= MIN_EDGE and skip_p < skip_ceil
    PUT  if put_p  >= put_thr  and put_p-call_p >= MIN_EDGE and skip_p < skip_ceil
    else SKIP

USAGE
    python backtest_threshold_sweep.py                 # 2024-07..2026-05 step 3
    python backtest_threshold_sweep.py 2024-07 2026-05 3
    python backtest_threshold_sweep.py quick           # last fold only
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---- LEAK-FIXED V9 pipeline (NOT v8) ---------------------------------
from scripts.train_model_v9 import (
    load_5min, load_higher, resample_from_5min,
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix, load_vix,
    merge_htf, merge_daily_onto_5min, merge_vix_onto_5min,
    add_intraday_context, create_labels,
    CSV_VIX,
)
from backtest_options import build_iv_map, simulate_trades

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

CSV5  = "data/nifty_5min.csv"
CSV15 = "data/nifty_15min.csv"
CSV30 = "data/nifty_30min.csv"
CSV60 = "data/nifty_60min.csv"
CSVD  = "data/nifty_day.csv"

EMBARGO_DAYS = 3
MIN_EDGE     = 0.05

# (name, call_thr, put_thr, skip_ceil)
CONFIGS = [
    ("A_current", 0.32, 0.25, 0.65),
    ("B_put020",  0.32, 0.20, 0.65),
    ("C_skip070", 0.32, 0.25, 0.70),
    ("D_both",    0.32, 0.20, 0.70),
    ("E_loose",   0.32, 0.20, 0.75),
]

SKIP_COLS = {"label", "open", "high", "low", "close", "day_open",
             "intraday_high", "intraday_low"}


def build_frame():
    print("Building V9 (leak-fixed) feature + label frame...")
    df5 = load_5min(CSV5)
    if Path(CSV15).exists():
        up = load_higher(CSV15, "15-min")
        rec = resample_from_5min(df5[df5.index > up.index[-1]], "15min")
        df15 = pd.concat([up, rec]).sort_index()
    else:
        df15 = resample_from_5min(df5, "15min")
    df30 = load_higher(CSV30, "30-min") if Path(CSV30).exists() else resample_from_5min(df5, "30min")
    df60 = load_higher(CSV60, "60min")  if Path(CSV60).exists() else resample_from_5min(df5, "60min")
    if Path(CSVD).exists():
        dfd = load_higher(CSVD, "Daily")
        dfd = dfd[~dfd.index.duplicated(keep="last")]
    else:
        dfd = resample_from_5min(df5, "D")

    feat = features_5min(df5)
    feat = merge_htf(feat, features_15min(df15), "15m")
    feat = merge_htf(feat, features_30min(df30), "30m")
    feat = merge_htf(feat, features_60min(df60), "60m")
    feat = merge_daily_onto_5min(feat, features_daily(dfd))
    try:
        dfvix = load_vix(CSV_VIX)
        if not dfvix.empty:
            feat = merge_vix_onto_5min(feat, features_vix(dfvix))
    except Exception as e:
        print(f"  VIX merge skipped: {e}")
    feat = add_intraday_context(feat)
    feat = create_labels(feat)
    feat.dropna(inplace=True)

    fcols = [c for c in feat.columns if c not in SKIP_COLS
             and feat[c].dtype in (np.float64, np.int64, np.int32, float, int)]
    print(f"  Frame: {len(feat):,} bars x {len(fcols)} features "
          f"| {feat.index[0].date()} -> {feat.index[-1].date()}")
    # report training label distribution (Phase 5)
    lbl = feat["label"].value_counts(normalize=True).sort_index()
    names = {0: "CALL", 1: "PUT", 2: "SKIP"}
    print("  TRAIN LABEL DISTRIBUTION:")
    for k in (0, 1, 2):
        print(f"    {names[k]:<5} {lbl.get(k, 0)*100:5.1f}%")
    return feat, fcols


def _sample_weights(sub):
    y = sub["label"].values
    skip_pct = (y == 2).mean()
    trade_pct = 1 - skip_pct
    trade_w = skip_pct / trade_pct if trade_pct > 0 else 2.0
    w = np.where(y == 2, 1.0, trade_w)
    try:
        if "expiry_is_tue" in sub.columns:
            tue = sub["expiry_is_tue"].values.astype(bool)
            w = w * np.where(tue, 6.0, 1.0)
    except Exception:
        pass
    try:
        bd = sub.index.date
        d_open = sub.groupby(bd)["open"].first()
        d_close = sub.groupby(bd)["close"].last()
        d_move = (d_close - d_open).abs()
        trend_days = set(d_move[d_move > 150.0].index)
        tmask = np.array([d in trend_days for d in bd], dtype=bool)
        w = w * np.where(tmask, 3.0, 1.0)
    except Exception:
        pass
    return w


def train_fold(feat, fcols, train_mask):
    sub = feat[train_mask]
    if len(sub) < 5000:
        return None, None
    cut = int(len(sub) * 0.95)
    tr, ev = sub.iloc[:cut], sub.iloc[cut:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    Xev = sc.transform(ev[fcols].values)
    ytr, yev = tr["label"].values, ev["label"].values
    w = _sample_weights(tr)
    models = {}
    mx = xgb.XGBClassifier(
        n_estimators=700, max_depth=5, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.5, min_child_weight=15,
        gamma=0.3, reg_alpha=1.5, reg_lambda=3.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=50,
        verbosity=0, n_jobs=-1)
    mx.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xev, yev)], verbose=False)
    models["xgb"] = mx
    ml = lgb.LGBMClassifier(
        n_estimators=700, max_depth=5, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.5, min_child_samples=30,
        reg_alpha=1.5, reg_lambda=3.0, num_leaves=28,
        objective="multiclass", num_class=3, verbose=-1, n_jobs=-1)
    ml.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xev, yev)],
           callbacks=[lgb.early_stopping(50, verbose=False),
                      lgb.log_evaluation(-1)])
    models["lgb"] = ml
    return models, sc


def _proba3(model, X):
    p = model.predict_proba(X)
    if p.shape[1] == 3:
        return p
    out = np.zeros((len(X), 3))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def signals_from_probas(probas, call_thr, put_thr, skip_ceil):
    sig = np.full(len(probas), 2)
    sig[(probas[:, 0] >= call_thr) &
        (probas[:, 0] - probas[:, 1] >= MIN_EDGE) &
        (probas[:, 2] < skip_ceil)] = 0
    sig[(probas[:, 1] >= put_thr) &
        (probas[:, 1] - probas[:, 0] >= MIN_EDGE) &
        (probas[:, 2] < skip_ceil)] = 1
    return sig


def month_floor(s):
    y, m = map(int, s.split("-"));  return date(y, m, 1)


def add_months(d, n):
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def metrics(tdf):
    """Full metric set for one config's aggregated OOS trades."""
    if tdf is None or len(tdf) == 0:
        return None
    net = tdf.net_option.values
    n = len(net)
    wins = net > 0
    gross_profit = net[net > 0].sum()
    gross_loss = -net[net <= 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    # equity curve / max drawdown (chronological)
    t = tdf.sort_values("time")
    eq = t.net_option.cumsum().values
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = -dd.min() if len(dd) else 0.0
    net_total = net.sum()
    avg = net.mean()
    std = net.std(ddof=1) if n > 1 else 0.0
    sharpe_trade = avg / std if std > 0 else 0.0
    # annualized Sharpe: span in years -> trades/yr
    span_days = max((t.time.iloc[-1] - t.time.iloc[0]).days, 1)
    trades_per_yr = n / (span_days / 365.25)
    sharpe_ann = sharpe_trade * np.sqrt(trades_per_yr) if std > 0 else 0.0
    calmar = (net_total / max_dd) if max_dd > 0 else float("inf")
    put = tdf[tdf["dir"] == "PUT"]
    call = tdf[tdf["dir"] == "CALL"]
    put_wr = (put.net_option > 0).mean() * 100 if len(put) else float("nan")
    call_wr = (call.net_option > 0).mean() * 100 if len(call) else float("nan")
    return dict(
        trades=n, win_rate=wins.mean() * 100, pf=pf,
        sharpe_trade=sharpe_trade, sharpe_ann=sharpe_ann, calmar=calmar,
        max_dd=max_dd, avg_trade=avg, ev=avg, net=net_total,
        n_put=len(put), n_call=len(call), put_wr=put_wr, call_wr=call_wr,
    )


def main():
    args = [a for a in sys.argv[1:]]
    quick = "quick" in args
    args = [a for a in args if a != "quick"]
    test_start = month_floor(args[0]) if len(args) > 0 else date(2024, 7, 1)
    test_end   = month_floor(args[1]) if len(args) > 1 else date(2026, 5, 1)
    step       = int(args[2]) if len(args) > 2 else 3

    feat, fcols = build_frame()
    iv_by_date, expiries = build_iv_map()
    print(f"IV calibrated for {len(iv_by_date)} days\n")

    folds = []
    f0 = test_start
    while f0 < test_end:
        f1 = min(add_months(f0, step), test_end)
        folds.append((f0, f1));  f0 = f1
    if quick:
        folds = folds[-1:]

    print("=" * 70)
    print(f"  LEAK-CLEAN (V9) WALK-FORWARD THRESHOLD SWEEP | {len(folds)} fold(s)")
    print(f"  test {test_start} -> {test_end} | embargo {EMBARGO_DAYS}d | XGB+LGB")
    print("=" * 70)

    # per-config accumulated trades
    cfg_trades = {name: [] for name, *_ in CONFIGS}

    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        train_mask = feat.index.date < cutoff
        test_mask = (feat.index.date >= a) & (feat.index.date < b)
        n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
        print(f"\n  Fold {k}/{len(folds)}: train < {cutoff} ({n_tr:,}) "
              f"-> test {a}..{b} ({n_te:,})")
        if n_tr < 5000 or n_te < 50:
            print("    skipped — insufficient data");  continue
        models, sc = train_fold(feat, fcols, train_mask)
        if models is None:
            print("    skipped — train failed");  continue
        test = feat[test_mask]
        Xte = sc.transform(test[fcols].values)
        probas = np.mean([_proba3(m, Xte) for m in models.values()], axis=0)
        # score every config on the SAME probas
        for name, call_thr, put_thr, skip_ceil in CONFIGS:
            sig = signals_from_probas(probas, call_thr, put_thr, skip_ceil)
            tdf = simulate_trades(test, sig, probas, iv_by_date, expiries)
            if len(tdf):
                tdf["fold"] = k
                cfg_trades[name].append(tdf)
            print(f"    {name:<11} {len(tdf):>4} trades  "
                  f"net Rs.{tdf.net_option.sum():+,.0f}" if len(tdf)
                  else f"    {name:<11}    0 trades")

    # aggregate + report
    print("\n" + "=" * 100)
    print("  AGGREGATE OUT-OF-SAMPLE PERFORMANCE BY CONFIG (real option P&L, post-friction)")
    print("=" * 100)
    hdr = (f"  {'Config':<11}{'Trd':>5}{'Win%':>7}{'PF':>7}{'Shrp':>7}"
           f"{'Calmar':>8}{'MaxDD':>12}{'AvgTrd':>10}{'Net':>14}"
           f"{'PUTwr':>7}{'CALLwr':>8}")
    print(hdr);  print("  " + "-" * 98)
    rows = {}
    for name, *_ in CONFIGS:
        full = pd.concat(cfg_trades[name], ignore_index=True) if cfg_trades[name] else None
        m = metrics(full)
        rows[name] = m
        if m is None:
            print(f"  {name:<11}  no trades");  continue
        print(f"  {name:<11}{m['trades']:>5}{m['win_rate']:>6.0f}%"
              f"{m['pf']:>7.2f}{m['sharpe_ann']:>7.2f}{m['calmar']:>8.2f}"
              f"{('Rs.'+format(m['max_dd'],',.0f')):>12}"
              f"{('Rs.'+format(m['avg_trade'],'+,.0f')):>10}"
              f"{('Rs.'+format(m['net'],'+,.0f')):>14}"
              f"{m['put_wr']:>6.0f}%{m['call_wr']:>7.0f}%")
    print("=" * 100)
    print("  Sharp = annualized Sharpe (trade-level).  Calmar = NetProfit / MaxDD (proxy).")
    print("  PUTwr/CALLwr = per-side net win rate.  All P&L after spread+STT+brokerage+GST.")

    # save per-config trade CSVs for deeper inspection
    for name, *_ in CONFIGS:
        if cfg_trades[name]:
            pd.concat(cfg_trades[name], ignore_index=True).to_csv(
                f"logs/sweep_{name}.csv", index=False)
    print("\n  Per-config trade CSVs -> logs/sweep_<config>.csv\n")


if __name__ == "__main__":
    main()
