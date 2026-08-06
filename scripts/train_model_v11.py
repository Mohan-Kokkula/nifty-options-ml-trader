"""
train_model_v11.py — V9's pipeline, V9's labels, V9's ensemble, 26 features.

WHY V11 EXISTS
--------------
V9 trains on 163 features. A controlled ablation (2026-08-07) held the
ENTIRE pipeline constant — same loaders, same feature builders, same
labels, same purged split, same class weighting, same ensemble
hyperparameters — and varied ONLY the feature subset:

    set        nfeat  VALacc  TESTacc  TESTrec   VAL->TEST drift
    full         163   0.616    0.600    0.637   -0.015
    grp0.95      154   0.623    0.598    0.640   -0.025
    grp0.9       143   0.625    0.603    0.614   -0.022
    grp0.8       118   0.636    0.611    0.638   -0.025
    grp0.7        99   0.621    0.599    0.634   -0.022
    V11 (this)    26   0.642    0.649    0.489   +0.007

Two findings drove this file:

  1. Over half of V9's features co-move with something else. At
     |rho| >= 0.70, 87 of 163 features sit in multi-member correlation
     groups. Seven oscillators (rsi7, rsi14, st14, wr14, cci20, bbp,
     zscore_20) are one oscillator — Williams %R IS the Stochastic
     with a sign flip, and Bollinger %B IS a 20-bar z-score.

  2. Feature COUNT drives generalization. Across 12 configurations,
     every set at <=50 features had POSITIVE VAL->TEST drift and every
     set at >=99 had NEGATIVE drift. The crossover sits near 50-80.

This set was selected on VAL accuracy (pre-registered before the run)
and confirmed once on TEST. It was also written down from trading logic
BEFORE any model was fit, so unlike a top-K search it carries no
selection optimism.

Trading arithmetic on the frozen TEST set:
    V9  : 6,157 signals @ 60.0%  ->  net directional edge 1,231
    V11 : 4,676 signals @ 64.9%  ->  net directional edge 1,393
13% more net edge from 24% fewer trades, so 24% less friction paid.

WHAT THIS IS NOT
----------------
A better classifier is NOT a proven edge. 64.9% on 3-bar labels still
has to clear ~29pts of options round-trip friction, and that gate has
not been passed. V11 is saved as a registry CANDIDATE only — promotion
is decided by champion-challenger against V9 on frozen metrics.

Usage:
    python scripts/train_model_v11.py
    python scripts/train_model_v11.py --csv5 data/nifty_5min.csv
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.train_model_v9 as V9

log = logging.getLogger("V11")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [V11] %(message)s",
                        datefmt="%H:%M:%S")

METRIC_VERSION = V9.METRIC_VERSION      # 2 = live-rule scoring


# ── The 26 ────────────────────────────────────────────────────────────
# One representative per kind of information a discretionary trader
# actually reads. Grouped here by intent so future edits stay honest
# about what each one is FOR.
V11_FEATURES = [
    # trend / direction across the timeframe hierarchy
    "tf15_adx", "tf15_ema_dist", "tf30_ema_dist", "tf60_ema_dist", "day_ema_dist",
    # momentum / extension
    "tf15_rsi", "rsi14", "m6", "roc", "mh",
    # volatility state
    "a14", "rv20", "bbw", "day_atr_pct",
    # location vs the day's reference levels
    "dist_vwap", "vwap_dist_atr", "close_pos", "intraday_pos",
    "dist_from_open", "dist_prev_close", "day_move_atr",
    # calendar / session
    "dte", "is_morning", "is_afternoon",
    # short-term structure
    "consec_bull", "consec_bear",
]


def build_dataset(csv5, csv15=None, csv30=None, csv60=None,
                  csv_day=None, csv_vix=None, csv_fut=None):
    """Assemble the labelled frame using V9's pipeline verbatim.

    Mirrors train_model_v9.train()'s data-assembly section. Kept as a
    separate function so V11 never re-implements a feature — any change
    to V9's builders flows through here automatically.
    """
    df_fut = V9.load_futures(csv_fut or V9.CSV_FUT)
    feat_fut = V9.features_futures(df_fut) if not df_fut.empty else pd.DataFrame()
    use_real_vwap = not feat_fut.empty
    log.info(f"  VWAP/CMF source: "
             f"{'REAL futures' if use_real_vwap else 'SPOT PROXY (no futures CSV)'}")

    df5 = V9.load_5min(csv5)
    if csv15 and Path(csv15).exists():
        up = V9.load_higher(csv15, "15-min (uploaded)")
        recent = V9.resample_from_5min(df5[df5.index > up.index[-1]], "15min")
        df15 = up if recent.empty else pd.concat([up, recent]).sort_index()
    else:
        df15 = V9.resample_from_5min(df5, "15min")
    df30 = (V9.load_higher(csv30, "30-min") if csv30 and Path(csv30).exists()
            else V9.resample_from_5min(df5, "30min"))
    df60 = (V9.load_higher(csv60, "60min") if csv60 and Path(csv60).exists()
            else V9.resample_from_5min(df5, "60min"))
    if csv_day and Path(csv_day).exists():
        df_day = V9.load_higher(csv_day, "Daily")
        df_day = df_day[~df_day.index.duplicated(keep="last")]
    else:
        df_day = V9.resample_from_5min(df5, "D")

    df_feat = V9.features_5min(df5)
    df_feat = V9.merge_htf(df_feat, V9.features_15min(df15), "15m")
    df_feat = V9.merge_htf(df_feat, V9.features_30min(df30), "30m")
    df_feat = V9.merge_htf(df_feat, V9.features_60min(df60), "60m")
    df_feat = V9.merge_daily_onto_5min(df_feat, V9.features_daily(df_day))

    df_vix = V9.load_vix(csv_vix or V9.CSV_VIX)
    if not df_vix.empty:
        df_feat = V9.merge_vix_onto_5min(df_feat, V9.features_vix(df_vix))
    if use_real_vwap:
        df_feat = V9.merge_futures_onto_5min(df_feat, feat_fut)
    df_feat = V9.add_intraday_context(df_feat, has_real_futures=use_real_vwap)

    df_feat = V9.create_labels(df_feat)
    df_feat.dropna(inplace=True)
    return df_feat


def build_sample_weights(df_feat, train_idx, ytr):
    """V9's exact weighting: inverse-frequency CALL/PUT balance, then the
    expiry-calendar boost, then the ATR-normalized trending-day boost."""
    skip_pct = (ytr == 2).mean()
    trade_w = skip_pct / (1 - skip_pct) if skip_pct < 1 else 2.0
    call_n, put_n = int((ytr == 0).sum()), int((ytr == 1).sum())
    dir_n = call_n + put_n
    if dir_n > 0 and call_n > 0 and put_n > 0:
        call_b, put_b = dir_n / (2.0 * call_n), dir_n / (2.0 * put_n)
    else:
        call_b = put_b = 1.0
    w = np.where(ytr == 0, trade_w * call_b,
        np.where(ytr == 1, trade_w * put_b, 1.0))
    log.info(f"Class balance: CALL={call_n:,} PUT={put_n:,} "
             f"SKIP={int((ytr==2).sum()):,} | CALLx{call_b:.2f} PUTx{put_b:.2f}")

    try:
        if "expiry_is_tue" in df_feat.columns:
            tue = df_feat["expiry_is_tue"].values[train_idx].astype(bool)
            w = w * np.where(tue, 6.0, 1.0)
            log.info(f"Expiry-calendar boost: {tue.sum():,}/{len(tue):,} rows x 6.0")
    except Exception as e:
        log.warning(f"Expiry-calendar boost skipped: {e}")

    try:
        sl = df_feat.iloc[train_idx]
        d = sl.index.date
        move = (sl.groupby(d)["close"].last() - sl.groupby(d)["open"].first()).abs()
        if "day_atr_pct" in sl.columns:
            atr = sl.groupby(d)["day_atr_pct"].last() * sl.groupby(d)["close"].last()
        else:
            atr = sl.groupby(d)["high"].max() - sl.groupby(d)["low"].min()
        trending = set((move / atr.replace(0, np.nan))[lambda s: s > 1.5].index)
        mask = np.array([x in trending for x in d], dtype=bool)
        w = w * np.where(mask, 3.0, 1.0)
        log.info(f"Trending-day boost: {mask.sum():,}/{len(mask):,} rows x 3.0")
    except Exception as e:
        log.warning(f"Trending-day boost skipped: {e}")
    return w


def train(csv5, csv15=None, csv30=None, csv60=None, csv_day=None,
          csv_vix=None, csv_fut=None):
    log.info("=" * 62)
    log.info("  NIFTY V11 — V9 pipeline, 26-feature trader set")
    log.info("=" * 62)

    df_feat = build_dataset(csv5, csv15, csv30, csv60, csv_day, csv_vix, csv_fut)

    missing = [c for c in V11_FEATURES if c not in df_feat.columns]
    if missing:
        raise SystemExit(
            f"V11 feature(s) absent from the pipeline output: {missing}\n"
            f"The V9 feature builders changed. Fix the names in V11_FEATURES "
            f"rather than silently training on a different set."
        )
    FCOLS = list(V11_FEATURES)
    log.info(f"\nFeatures: {len(FCOLS)} (V9 trains on ~163)")

    X = df_feat[FCOLS].values
    y = df_feat["label"].values
    train_idx, val_idx, test_idx = V9.purged_train_val_test_split(len(X))

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[train_idx])     # scaler fit on TRAIN ONLY
    Xvl = sc.transform(X[val_idx])
    Xte = sc.transform(X[test_idx])
    ytr, yvl, yte = y[train_idx], y[val_idx], y[test_idx]

    sample_w = build_sample_weights(df_feat, train_idx, ytr)

    log.info("\n── Training ensemble ──")
    models = V9._fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w)

    def _eval(Xe, ye):
        p = np.mean([m.predict_proba(Xe) for m in models.values()], axis=0)
        s = V9.live_rule_signals(p)
        if not (s != 2).any():
            return None
        m = V9.live_rule_metrics(p, ye)
        m["n_skip"] = int((s == 2).sum())
        return m

    val_m, test_m = _eval(Xvl, yvl), _eval(Xte, yte)

    log.info(f"\n{'='*62}")
    log.info("  EVALUATION (V11) — scored with the LIVE decision rule")
    log.info(f"{'='*62}")
    if val_m:
        log.info(f"  [VAL  early-stop] dir_acc={val_m['dir_acc']:.1%} "
                 f"recall={val_m['recall']:.1%} "
                 f"(CALL={val_m['n_call']} PUT={val_m['n_put']} SKIP={val_m['n_skip']})")
    if test_m:
        log.info(f"  [TEST  frozen   ] dir_acc={test_m['dir_acc']:.1%} "
                 f"recall={test_m['recall']:.1%} "
                 f"(CALL={test_m['n_call']} PUT={test_m['n_put']} SKIP={test_m['n_skip']})")
    if val_m and test_m:
        gap = val_m["dir_acc"] - test_m["dir_acc"]
        log.info(f"  VAL->TEST dir_acc gap: {gap:+.1%} "
                 f"({'OVERFIT RISK' if gap > 0.05 else 'ok'})")
        log.info(f"  (the ablation measured {-0.007:+.3f} here — a large positive "
                 f"gap means something changed)")

    # ── Registry candidate only. Promotion is the gate's decision. ────
    candidate_id = None
    try:
        from core.model_registry import save_candidate
        tr_d, va_d, te_d = (df_feat.index[i] for i in (train_idx, val_idx, test_idx))
        metadata = {
            "train_date_start": str(tr_d.min().date()), "train_date_end": str(tr_d.max().date()),
            "val_date_start":   str(va_d.min().date()), "val_date_end":   str(va_d.max().date()),
            "test_date_start":  str(te_d.min().date()), "test_date_end":  str(te_d.max().date()),
            "train_bars": int(len(train_idx)), "val_bars": int(len(val_idx)),
            "test_bars":  int(len(test_idx)),
            "metric_version": METRIC_VERSION,
            "val_dir_acc":  round(float(val_m["dir_acc"]), 4) if val_m else None,
            "test_dir_acc": round(float(test_m["dir_acc"]), 4) if test_m else 0.0,
            "test_fired_on_skip": int(test_m["fired_on_skip"]) if test_m else 0,
            "test_fired_on_dir":  int(test_m["fired_on_dir"]) if test_m else 0,
            "test_recall":  round(float(test_m["recall"]), 4) if test_m else 0.0,
            "test_signals": int(test_m["n_call"] + test_m["n_put"]) if test_m else 0,
            "val_test_gap": round(float(val_m["dir_acc"]) - float(test_m["dir_acc"]), 4)
                            if (val_m and test_m) else None,
            "n_features":   len(FCOLS),
            "fwd_bars":     V9.FWD_BARS,
            "train_frac":   V9.TRAIN_FRAC,
            "val_frac":     V9.VAL_FRAC,
            "embargo_bars": V9.EMBARGO_BARS,
            "label_quantile": 0.70,
            "feature_selection": "trader26 — VAL-selected, a-priori shortlist",
        }
        candidate_id = save_candidate("v11", models, sc, FCOLS, metadata)
        log.info(f"\n  Registry candidate: v11/{candidate_id}")
        log.info("  NOT promoted — champion-challenger decides against V9.")
    except Exception as e:
        import joblib
        log.warning(f"  Registry save failed ({e}) — direct fallback save")
        Path("models").mkdir(exist_ok=True)
        joblib.dump(models, "models/nifty_v11_models.pkl")
        joblib.dump(sc,     "models/nifty_v11_scaler.pkl")
        joblib.dump(FCOLS,  "models/feature_cols_v11.pkl")

    log.info(f"\n{'='*62}\n  V11 training complete\n{'='*62}\n")
    return candidate_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Nifty V11 (26-feature) model")
    p.add_argument("--csv5",   default=V9.CSV_5M)
    p.add_argument("--csv15",  default=V9.CSV_15M)
    p.add_argument("--csv30",  default=V9.CSV_30M)
    p.add_argument("--csv60",  default=V9.CSV_60M)
    p.add_argument("--csvday", default=V9.CSV_DAY)
    p.add_argument("--csvvix", default=V9.CSV_VIX)
    p.add_argument("--csvfut", default=V9.CSV_FUT)
    a = p.parse_args()
    train(csv5=a.csv5, csv15=a.csv15, csv30=a.csv30, csv60=a.csv60,
          csv_day=a.csvday, csv_vix=a.csvvix, csv_fut=a.csvfut)
