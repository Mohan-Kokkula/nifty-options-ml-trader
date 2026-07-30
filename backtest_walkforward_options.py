"""
backtest_walkforward_options.py — TRUE walk-forward, option-priced
==================================================================
WHY THIS EXISTS
----------------
scripts/backtest_walkforward.py is NOT real walk-forward: it loads
ONE pre-trained model (trained on data that INCLUDES the test days)
and just loops over dates. That cannot detect overfitting — the
model has already seen the answers.

This script does it properly:
  * Rolls a TRAIN window / TEST window forward through time.
  * For every fold it RE-TRAINS the model from scratch on data
    BEFORE the test window only (with an embargo gap so forward-
    return labels cannot leak), then predicts on the unseen test
    window.
  * Trades in the test window are priced as REAL OPTIONS using the
    Black-Scholes engine in backtest_options.py (theta, spread,
    STT, brokerage, GST — all included).
  * Aggregates every out-of-sample trade into one honest report.

If the strategy's edge is real it will survive here. If the
86.6%/Rs.38.6L came from overfitting, this is where it collapses.

USAGE
    python backtest_walkforward_options.py
    python backtest_walkforward_options.py 2024-07 2026-05 3
        args: test_start(YYYY-MM)  test_end(YYYY-MM)  step_months
    python backtest_walkforward_options.py quick
        single most-recent fold only — fast smoke test
    python backtest_walkforward_options.py live
        apply the restrictive live-bot gates (85% confidence floor +
        11:00 morning block) — compare against the default run to see
        whether the live over-tightening helped or hurt.

REQUIREMENTS
    xgboost + lightgbm installed (same as model training).
    Each fold retrains 2 gradient-boosted models — budget a few
    minutes per fold. A full 2024-2026 run is ~20-40 min.
"""

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---- training pipeline (feature builders + labels) -------------------
from scripts.train_model_v8 import (
    load_5min, load_higher, resample_from_5min,
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix, load_vix,
    merge_htf, merge_daily_onto_5min, merge_vix_onto_5min,
    add_intraday_context, create_labels,
    CSV_VIX,
)
# ---- option-pricing engine (the bit we are "wiring in") --------------
from backtest_options import build_iv_map, simulate_trades, report

# Prefer XGBoost+LightGBM (production model). Fall back to sklearn
# HistGradientBoosting if they are not installed — same fallback that
# train_model_v8.py uses, so the script still runs (lower fidelity).
try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_XLG = True
except ImportError:
    HAS_XLG = False
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

# data files
CSV5  = "data/nifty_5min.csv"
CSV15 = "data/nifty_15min.csv"
CSV30 = "data/nifty_30min.csv"
CSV60 = "data/nifty_60min.csv"
CSVD  = "data/nifty_day.csv"

# label leakage guard: forward-return labels look ahead a few bars, and
# intraday context spans the day — embargo a few TRADING days between
# the end of train and the start of test so nothing leaks.
EMBARGO_DAYS = 3

# signal thresholds — identical to backtest.py / _build_full_features
SIG_CONFIDENCE = 0.30
SIG_MIN_EDGE   = 0.05
SIG_SKIP_CEIL  = 0.60

# "live mode" gates — approximate the restrictive live claude_pilot config.
# NOTE: an approximation. The exact live confidence uses a probability
# calibrator that cannot be faithfully replicated here; this uses the raw
# directional conviction p[dir]/(p_call+p_put). The morning block is exact.
# The dozens of micro-filters (V-RECOVERY, RSI-divergence, VWAP structure,
# IV gate, ...) are NOT replicated — so "live mode" is a floor on how
# restrictive the live bot is, not an exact replica.
LIVE_CONF_GATE       = 0.85       # live pilot min_confidence (raised to 85%)
LIVE_MORNING_BLOCK   = 11 * 60    # no entries before 11:00 (105-min hard block)

SKIP_COLS = {"label", "open", "high", "low", "close", "day_open",
             "intraday_high", "intraday_low"}


# ----------------------------------------------------------------------
def build_frame():
    """Build the full multi-TF feature+label frame ONCE (mirrors train())."""
    print("Building feature + label frame (one-time, ~1-2 min)...")
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
    return feat, fcols


def _sample_weights(sub):
    """Class weight + expiry-Tue boost + trending-day boost (mirrors train())."""
    y = sub["label"].values
    skip_pct = (y == 2).mean()
    trade_pct = 1 - skip_pct
    trade_w = skip_pct / trade_pct if trade_pct > 0 else 2.0
    w = np.where(y == 2, 1.0, trade_w)
    # expiry-day boost
    try:
        if "expiry_is_tue" in sub.columns:
            tue = sub["expiry_is_tue"].values.astype(bool)
            w = w * np.where(tue, 6.0, 1.0)
    except Exception:
        pass
    # trending-day boost
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
    return w, trade_w


def train_fold(feat, fcols, train_mask):
    """Retrain XGB+LGB on train rows only. Returns (models, scaler)."""
    sub = feat[train_mask]
    if len(sub) < 5000:
        return None, None
    # time-ordered 95/5 split inside the train window for early stopping
    cut = int(len(sub) * 0.95)
    tr, ev = sub.iloc[:cut], sub.iloc[cut:]

    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[fcols].values)
    Xev = sc.transform(ev[fcols].values)
    ytr, yev = tr["label"].values, ev["label"].values
    w, trade_w = _sample_weights(tr)

    models = {}
    if HAS_XLG:
        # production model: XGBoost + LightGBM (same params as train_model_v8)
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
    else:
        # fallback: sklearn HistGradientBoosting (same as train_model_v8)
        for i, name in enumerate(["hgb_a", "hgb_b", "hgb_c"]):
            m = HistGradientBoostingClassifier(
                max_iter=600, max_depth=5,
                learning_rate=0.02 + i * 0.007,
                min_samples_leaf=30, l2_regularization=3.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=40, random_state=42 + i * 17)
            m.fit(Xtr, ytr, sample_weight=w)
            models[name] = m
    return models, sc


def _proba3(model, X):
    """predict_proba remapped to a fixed [CALL,PUT,SKIP] 3-column array."""
    p = model.predict_proba(X)
    if p.shape[1] == 3:
        return p
    out = np.zeros((len(X), 3))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def signals_from_probas(probas):
    """Same signal rule as backtest.py _build_full_features."""
    sig = np.full(len(probas), 2)
    sig[(probas[:, 0] >= SIG_CONFIDENCE) &
        (probas[:, 0] - probas[:, 1] >= SIG_MIN_EDGE) &
        (probas[:, 2] < SIG_SKIP_CEIL)] = 0
    sig[(probas[:, 1] >= SIG_CONFIDENCE) &
        (probas[:, 1] - probas[:, 0] >= SIG_MIN_EDGE) &
        (probas[:, 2] < SIG_SKIP_CEIL)] = 1
    return sig


def month_floor(s):
    y, m = map(int, s.split("-"))
    return date(y, m, 1)


def add_months(d, n):
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


# ----------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:]]
    quick = "quick" in args
    live  = "live" in args          # apply the live bot's restrictive gates
    args = [a for a in args if a not in ("quick", "live")]
    test_start = month_floor(args[0]) if len(args) > 0 else date(2024, 7, 1)
    test_end   = month_floor(args[1]) if len(args) > 1 else date(2026, 5, 1)
    step       = int(args[2]) if len(args) > 2 else 3

    feat, fcols = build_frame()
    iv_by_date, expiries = build_iv_map()
    print(f"IV calibrated for {len(iv_by_date)} days "
          f"({'bhavcopy' if iv_by_date else 'VIX fallback only'})\n")

    # build the fold list
    folds = []
    f0 = test_start
    while f0 < test_end:
        f1 = min(add_months(f0, step), test_end)
        folds.append((f0, f1))
        f0 = f1
    if quick:
        folds = folds[-1:]

    algo = "XGBoost+LightGBM" if HAS_XLG else "HistGradientBoosting (fallback)"
    mode = "LIVE-CONFIG (85% conf gate + 11:00 morning block)" if live \
           else "validated config (backtest.py thresholds)"
    print("=" * 70)
    print(f"  TRUE WALK-FORWARD  |  {len(folds)} fold(s)  |  "
          f"test {test_start} -> {test_end}  |  embargo {EMBARGO_DAYS}d")
    print(f"  Config: {mode}")
    print(f"  Each fold retrains {algo} on PRE-test data only.")
    if not HAS_XLG:
        print("  NOTE: xgboost/lightgbm not installed — using fallback model.")
        print("        Install them to validate the ACTUAL production model.")
    print("=" * 70)

    all_trades = []
    fold_rows = []
    for k, (a, b) in enumerate(folds, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        train_mask = feat.index.date < cutoff
        test_mask = (feat.index.date >= a) & (feat.index.date < b)
        n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
        print(f"\n  Fold {k}/{len(folds)}: train < {cutoff} ({n_tr:,} bars) "
              f"-> test {a}..{b} ({n_te:,} bars)")
        if n_tr < 5000 or n_te < 50:
            print("    skipped — not enough data")
            continue

        models, sc = train_fold(feat, fcols, train_mask)
        if models is None:
            print("    skipped — training failed")
            continue

        test = feat[test_mask]
        Xte = sc.transform(test[fcols].values)
        probas = np.mean([_proba3(m, Xte) for m in models.values()], axis=0)
        signals = signals_from_probas(probas)

        if live:
            # raise confidence bar to ~85% directional conviction
            p0, p1 = probas[:, 0], probas[:, 1]
            denom = np.where((p0 + p1) > 1e-9, p0 + p1, 1.0)
            dir_conf = np.where(signals == 0, p0 / denom,
                        np.where(signals == 1, p1 / denom, 0.0))
            signals = np.where(dir_conf < LIVE_CONF_GATE, 2, signals)
            # morning hard block — no entries before 11:00
            mins = np.array([t.hour * 60 + t.minute for t in test.index])
            signals = np.where(mins < LIVE_MORNING_BLOCK, 2, signals)

        tdf = simulate_trades(test, signals, probas, iv_by_date, expiries)
        if len(tdf) == 0:
            print("    no trades in this fold")
            continue
        tdf["fold"] = k
        all_trades.append(tdf)
        net = tdf.net_option.sum()
        wr = (tdf.net_option > 0).mean() * 100
        fold_rows.append((f"{a}..{b}", len(tdf), wr, net))
        print(f"    OUT-OF-SAMPLE: {len(tdf)} trades | "
              f"net win {wr:.0f}% | Rs.{net:+,.0f}")

    if not all_trades:
        print("\nNo out-of-sample trades generated.")
        return

    print("\n" + "=" * 70)
    print("  PER-FOLD (OUT-OF-SAMPLE) SUMMARY")
    print("=" * 70)
    print(f"  {'Fold window':<22}{'Trades':>8}{'Net Win%':>10}{'Net P&L':>16}")
    print(f"  {'-'*55}")
    for w, n, wr, net in fold_rows:
        print(f"  {w:<22}{n:>8}{wr:>9.0f}%{('Rs.' + format(net, '+,.0f')):>16}")


    full = pd.concat(all_trades, ignore_index=True)
    report(full, label="AGGREGATE OUT-OF-SAMPLE RESULT", yearly=True)

    out = "logs/walkforward_options_trades.csv"
    try:
        full.to_csv(out, index=False)
        print(f"  Per-trade detail -> {out}")
    except Exception as e:
        print(f"  (could not write CSV: {e})")

    net = full.net_option.sum()
    wr = (full.net_option > 0).mean() * 100
    pos_folds = sum(1 for *_, x in fold_rows if x > 0)
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VERDICT  (out-of-sample, real option P&L)")
    print("=" * 70)
    print(f"  [{'PASS' if net > 0 else 'FAIL'}] Net P&L positive        "
          f"(Rs.{net:+,.0f})")
    print(f"  [{'PASS' if wr >= 55 else 'FAIL'}] Net win rate >= 55%     "
          f"({wr:.0f}%)")
    print(f"  [{'PASS' if pos_folds >= len(fold_rows)*0.6 else 'FAIL'}] "
          f">=60% of folds profitable ({pos_folds}/{len(fold_rows)})")
    print("=" * 70)
    print("  If these FAIL while in-sample backtest.py shows 86%+, the")
    print("  edge was overfitting - no SL/TP tuning will fix that.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
