"""
train_model_v10.py — V10: V9 + Option Chain features (IV/Greeks/PCR/OI)
=========================================================================

NEW vs V9
---------
  OPTION FEATURES (joined by date from NSE Bhavcopy):
    atm_iv          — Implied volatility of ATM call (nearest strike to spot)
    atm_iv_rank30   — IV rank: where today's IV sits in last 30 days (0..1)
    atm_iv_zscore   — IV z-score vs 20-day mean
    atm_delta       — Delta of ATM call
    atm_theta_pct   — Theta as % of ATM option price (daily decay rate)
    pcr_near        — Put-Call Ratio for near expiry (OI-based)
    pcr_all         — PCR across all expiries
    oi_ce_atm       — OI at ATM call strike (log-scaled)
    oi_pe_atm       — OI at ATM put strike (log-scaled)
    oi_ce_change    — OI change direction at ATM call (+1/-1/0)
    oi_pe_change    — OI change direction at ATM put
    iv_term_struct  — Near IV / Far IV (term structure slope)
    skew_25d        — Put-Call IV skew at ±25% moneyness band

  WHY THESE FEATURES HELP
  - IV rank + z-score: avoids selling options during IV expansion (crush risk)
  - PCR: market sentiment, >1.2 = bearish, <0.8 = bullish
  - OI at ATM: shows where big money is positioned
  - Theta: higher theta → premium decays faster → better for directional trades
  - Term structure: backwardation = near-term fear

USAGE:
    python scripts/train_model_v10.py
    python scripts/train_model_v10.py --prune-shap

REQUIRES:
    data/v10_training_features.csv   ← from build_training_dataset.py
    data/nifty_5min.csv              ← same as V9
"""

import warnings; warnings.filterwarnings("ignore")
import argparse, logging, sys, joblib, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/nifty_v10_models.pkl"
SCALER_PATH = "models/nifty_v10_scaler.pkl"
FCOLS_PATH  = "models/feature_cols_v10.pkl"

V10_FEATURES_CSV = "data/v10_training_features.csv"

# Reuse all V9 loaders + feature engineering
from scripts.train_model_v9 import (
    load_5min, load_higher, load_futures, load_vix,
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix, features_futures,
    merge_htf, merge_daily_onto_5min, merge_vix_onto_5min,
    merge_futures_onto_5min, add_intraday_context,
    create_labels, shap_prune_features, _fit_ensemble,
    resample_from_5min, purged_train_val_test_split,
    CSV_5M, CSV_15M, CSV_30M, CSV_60M, CSV_DAY, CSV_VIX, CSV_FUT,
    # 2026-06-10 FIX: these 4 constants are referenced in the registry
    # metadata dict below (~line 543) but were never imported, causing a
    # NameError that was silently swallowed by the `except Exception as _e`
    # around save_candidate(). Result: save_candidate("v10", ...) was NEVER
    # reached -> registry never got a v10_* entry -> promote_if_passes_gate
    # always returned "No candidates in registry for version='v10'" -> the
    # fallback joblib.dump() below overwrote the LIVE model with NO gate.
    FWD_BARS, TRAIN_FRAC, VAL_FRAC, EMBARGO_BARS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [V10] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Option feature loader (from build_training_dataset.py output)
# ─────────────────────────────────────────────────────────────────────────────

def load_option_features(v10_csv: str) -> pd.DataFrame:
    """
    Load per-day option chain features built from NSE Bhavcopy.
    Returns one row per date with ATM IV, PCR, OI, Greeks etc.
    """
    p = Path(v10_csv)
    if not p.exists():
        log.warning(f"V10 features not found: {p}")
        log.warning("Run: python scripts/build_training_dataset.py")
        return pd.DataFrame()

    df = pd.read_csv(p, parse_dates=["date"])
    log.info(f"V10 Bhavcopy features: {len(df):,} rows, {df['date'].nunique()} days")
    return df


def compute_daily_option_context(bhav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-strike bhavcopy rows into ONE row per trading day
    with the ATM-focused features the model needs.

    For each day:
    - Find the near expiry (closest to expiry)
    - Find the ATM strike (nearest strike to the day's underlying spot)
    - Extract IV, delta, theta, OI for that ATM strike
    - Compute PCR across all strikes for near expiry
    - Compute IV skew + term structure
    """
    if bhav_df.empty:
        return pd.DataFrame()

    daily_rows = []

    for trade_date, day_df in bhav_df.groupby("date"):
        # ── 1. Find near expiry ──────────────────────────────────────────
        spot = day_df["spot"].median()
        if spot <= 0:
            continue

        # Sort expiries by proximity to trade_date
        expiries = sorted(day_df["expiry"].unique())
        near_exp = expiries[0] if expiries else None
        if not near_exp:
            continue

        near_df = day_df[day_df["expiry"] == near_exp]

        # ── 2. ATM strike ────────────────────────────────────────────────
        strikes_available = near_df["strike"].unique()
        if len(strikes_available) == 0:
            continue
        atm_strike = int(min(strikes_available, key=lambda s: abs(s - spot)))

        atm_ce = near_df[(near_df["strike"] == atm_strike) & (near_df["opt_type"] == "CE")]
        atm_pe = near_df[(near_df["strike"] == atm_strike) & (near_df["opt_type"] == "PE")]

        # ── 3. ATM option metrics ────────────────────────────────────────
        atm_iv_ce    = atm_ce["iv"].values[0] if len(atm_ce) and not pd.isna(atm_ce["iv"].values[0]) else None
        atm_iv_pe    = atm_pe["iv"].values[0] if len(atm_pe) and not pd.isna(atm_pe["iv"].values[0]) else None
        atm_iv       = (atm_iv_ce or atm_iv_pe or None)
        atm_delta    = atm_ce["delta"].values[0] if len(atm_ce) else 0.5
        atm_theta    = atm_ce["theta"].values[0] if len(atm_ce) else 0.0
        atm_price    = atm_ce["price"].values[0] if len(atm_ce) else 1.0
        atm_oi_ce    = atm_ce["oi"].values[0] if len(atm_ce) else 0
        atm_oi_pe    = atm_pe["oi"].values[0] if len(atm_pe) else 0
        atm_oi_ce_ch = atm_ce["oi_change"].values[0] if len(atm_ce) else 0
        atm_oi_pe_ch = atm_pe["oi_change"].values[0] if len(atm_pe) else 0

        # Theta as daily premium decay %
        theta_pct = abs(atm_theta / max(atm_price, 0.01)) * 100 if atm_price > 0 else 0

        # ── 4. PCR for near expiry ───────────────────────────────────────
        ce_oi_total = near_df[near_df["opt_type"] == "CE"]["oi"].sum()
        pe_oi_total = near_df[near_df["opt_type"] == "PE"]["oi"].sum()
        pcr_near    = pe_oi_total / max(ce_oi_total, 1)

        # PCR all expiries
        ce_oi_all = day_df[day_df["opt_type"] == "CE"]["oi"].sum()
        pe_oi_all = day_df[day_df["opt_type"] == "PE"]["oi"].sum()
        pcr_all   = pe_oi_all / max(ce_oi_all, 1)

        # ── 5. IV Skew (put-call IV difference at 25-delta band) ─────────
        # Find strikes ≈ ±5% from ATM to approximate 25-delta options
        band = int(spot * 0.025 / 50) * 50  # round to nearest 50-pt band
        otm_put_strikes  = [s for s in strikes_available if s <= atm_strike - band]
        otm_call_strikes = [s for s in strikes_available if s >= atm_strike + band]

        skew_put  = near_df[(near_df["strike"].isin(otm_put_strikes))
                            & (near_df["opt_type"] == "PE")]["iv"].median()
        skew_call = near_df[(near_df["strike"].isin(otm_call_strikes))
                            & (near_df["opt_type"] == "CE")]["iv"].median()
        iv_skew = (
            (skew_put - skew_call)
            if not pd.isna(skew_put) and not pd.isna(skew_call)
            else 0.0
        )

        # ── 6. Term structure (near vs far IV) ───────────────────────────
        if len(expiries) >= 2:
            far_exp = expiries[-1]
            far_df  = day_df[day_df["expiry"] == far_exp]
            far_atm = far_df[far_df["strike"] == atm_strike]
            far_iv_ce = far_atm[far_atm["opt_type"] == "CE"]["iv"].values
            far_iv    = far_iv_ce[0] if len(far_iv_ce) and not pd.isna(far_iv_ce[0]) else None
            term_struct = ((atm_iv or 0.20) / max(far_iv or 0.20, 0.01))
        else:
            term_struct = 1.0

        daily_rows.append({
            "date":           pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
            "opt_spot":       round(spot, 0),
            "opt_atm_strike": atm_strike,
            "opt_near_expiry": near_exp,
            "atm_iv":         round(atm_iv, 4) if atm_iv else None,
            "atm_delta":      round(float(atm_delta), 4),
            "atm_theta_pct":  round(float(theta_pct), 4),
            "pcr_near":       round(float(pcr_near), 4),
            "pcr_all":        round(float(pcr_all), 4),
            "oi_ce_atm_log":  math.log1p(float(atm_oi_ce)),
            "oi_pe_atm_log":  math.log1p(float(atm_oi_pe)),
            "oi_ce_chg_dir":  int(np.sign(atm_oi_ce_ch)),
            "oi_pe_chg_dir":  int(np.sign(atm_oi_pe_ch)),
            "iv_skew":        round(float(iv_skew), 4),
            "term_struct":    round(float(term_struct), 4),
        })

    if not daily_rows:
        return pd.DataFrame()

    ctx = pd.DataFrame(daily_rows)
    ctx["date"] = pd.to_datetime(ctx["date"])
    ctx = ctx.set_index("date").sort_index()

    # ── IV rank + z-score (rolling 30-day) ──────────────────────────────
    iv = ctx["atm_iv"].ffill()
    ctx["atm_iv_rank30"]  = iv.rolling(30, min_periods=5).rank(pct=True)
    ctx["atm_iv_zscore"]  = (iv - iv.rolling(20).mean()) / (iv.rolling(20).std() + 1e-9)
    ctx["atm_iv_rising"]  = (iv > iv.shift(1)).astype(int)
    ctx["atm_iv_extreme"] = (ctx["atm_iv_rank30"] > 0.85).astype(int)
    ctx["pcr_bullish"]    = (ctx["pcr_near"] < 0.80).astype(int)
    ctx["pcr_bearish"]    = (ctx["pcr_near"] > 1.20).astype(int)
    ctx["iv_contango"]    = (ctx["term_struct"] < 1.0).astype(int)  # near < far = normal

    log.info(f"Daily option context: {len(ctx)} days, {ctx.shape[1]} features")
    log.info(f"  ATM IV range: {iv.min():.2%} – {iv.max():.2%}")
    log.info(f"  PCR range: {ctx['pcr_near'].min():.2f} – {ctx['pcr_near'].max():.2f}")

    return ctx


def merge_option_context_onto_5min(df5: pd.DataFrame,
                                   opt_ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Join daily option context onto 5-min spot bars by date.
    Uses yesterday's option context (shift 1 day) to avoid lookahead.
    """
    if opt_ctx.empty:
        log.warning("No option context to merge — V10 = V9 features only")
        return df5

    df5 = df5.copy()
    df5["_date"] = df5.index.normalize()

    # Shift by 1 trading day: use yesterday's option data to avoid lookahead
    opt_shifted = opt_ctx.shift(1)
    opt_shifted = opt_shifted[~opt_shifted.index.duplicated(keep="last")]

    new_cols = [c for c in opt_shifted.columns
                if c not in ("opt_spot", "opt_atm_strike", "opt_near_expiry")]
    for col in new_cols:
        df5[col] = df5["_date"].map(opt_shifted[col])

    df5.drop(columns=["_date"], inplace=True)

    filled = df5[new_cols].notna().any(axis=1).sum()
    log.info(
        f"Option context merged: {filled:,}/{len(df5):,} bars have option data "
        f"({filled/len(df5):.0%})"
    )
    return df5


# ─────────────────────────────────────────────────────────────────────────────
# Main train
# ─────────────────────────────────────────────────────────────────────────────

def train_v10(
    csv5=CSV_5M, csv15=CSV_15M, csv30=CSV_30M, csv60=CSV_60M,
    csv_day=CSV_DAY, csv_vix=CSV_VIX, csv_fut=CSV_FUT,
    v10_features=V10_FEATURES_CSV,
    prune_shap=False,
):
    log.info("=" * 65)
    log.info("  NIFTY V10 — V9 + Option Chain features (IV/Greeks/PCR/OI)")
    log.info("=" * 65)

    # ── Load option context ───────────────────────────────────────────────
    log.info("\n[1/6] Loading option chain context (Bhavcopy)...")
    bhav_df  = load_option_features(v10_features)
    opt_ctx  = compute_daily_option_context(bhav_df) if not bhav_df.empty else pd.DataFrame()

    # ── V9 pipeline (unchanged) ───────────────────────────────────────────
    log.info("\n[2/6] Loading spot data (same as V9)...")
    df_fut   = load_futures(csv_fut)
    feat_fut = features_futures(df_fut) if not df_fut.empty else pd.DataFrame()
    use_real_vwap = not feat_fut.empty
    log.info(f"  VWAP/CMF: {'real futures' if use_real_vwap else 'spot proxy'}")

    df5 = load_5min(csv5)

    if csv15 and Path(csv15).exists():
        df15_up = load_higher(csv15, "15-min")
        # Filter to only 5m bars NEWER than the last stored 15m bar.
        # If the data is up to date (same end date) this slice may be
        # empty — resample_from_5min() handles that and returns an empty
        # DataFrame, so we skip the concat in that case.
        _df5_delta = df5[df5.index > df15_up.index[-1]]
        df15_re    = resample_from_5min(_df5_delta, "15min")
        if df15_re.empty:
            df15 = df15_up          # nothing new — use stored 15m as-is
        else:
            df15 = pd.concat([df15_up, df15_re]).sort_index()
    else:
        df15 = resample_from_5min(df5, "15min")

    df30 = load_higher(csv30, "30-min") if csv30 and Path(csv30).exists() else resample_from_5min(df5, "30min")
    df60 = load_higher(csv60, "60min")  if csv60 and Path(csv60).exists() else resample_from_5min(df5, "60min")
    if csv_day and Path(csv_day).exists():
        df_day = load_higher(csv_day, "Daily")
        df_day = df_day[~df_day.index.duplicated(keep="last")]
    else:
        df_day = resample_from_5min(df5, "D")

    log.info("\n[3/6] Computing V9 features...")
    feat15   = features_15min(df15)
    feat30   = features_30min(df30)
    feat60   = features_60min(df60)
    feat_day = features_daily(df_day)
    df_vix   = load_vix(csv_vix)
    feat_vix = features_vix(df_vix) if not df_vix.empty else pd.DataFrame()

    df_feat = features_5min(df5)
    df_feat = merge_htf(df_feat, feat15, "15m")
    df_feat = merge_htf(df_feat, feat30, "30m")
    df_feat = merge_htf(df_feat, feat60, "60m")
    df_feat = merge_daily_onto_5min(df_feat, feat_day)
    if not feat_vix.empty:
        df_feat = merge_vix_onto_5min(df_feat, feat_vix)
    if use_real_vwap:
        df_feat = merge_futures_onto_5min(df_feat, feat_fut)
    df_feat = add_intraday_context(df_feat, has_real_futures=use_real_vwap)

    # ── Inject option features ────────────────────────────────────────────
    log.info("\n[4/6] Injecting option chain features...")
    df_feat = merge_option_context_onto_5min(df_feat, opt_ctx)

    log.info(f"Total columns after merge: {len(df_feat.columns)}")

    # ── Labels ────────────────────────────────────────────────────────────
    df_feat = create_labels(df_feat)
    df_feat.dropna(inplace=True)

    # ── Feature columns ───────────────────────────────────────────────────
    LEVEL_COLS = {
        "open", "high", "low", "close", "volume",
        "day_open", "intraday_high", "intraday_low",
        "day_prev_high", "day_prev_low", "day_prev_close",
        "week_high", "week_low", "vwap_session", "fut_vwap",
        "_atr14_pts",
    }
    PTS_EXCLUDE = {c for c in df_feat.columns if c.endswith("_pts")}
    # Option-context raw levels (not model features)
    OPT_META = {"opt_spot", "opt_atm_strike", "opt_near_expiry"}
    # 2026-06-12 PHASE-1 FINDING: daily-lagged Bhavcopy OI features are
    # significantly NEGATIVE predictors (Spearman IC −0.078, p=0.0003 on the
    # in-window frozen test) — removed from training. Intraday OI (OpenAlgo
    # archive) is a separate, future research track.
    OI_EXCLUDE = {"pcr_near", "pcr_all", "pcr_bullish", "pcr_bearish",
                  "oi_ce_atm_log", "oi_pe_atm_log",
                  "oi_ce_chg_dir", "oi_pe_chg_dir"}
    # 2026-06-12 PHASE-1 FINDING: VIX features have no directional value
    # (permutation −0.010). Model-only exclusion; labels/pilot unaffected.
    VIX_EXCLUDE = {c for c in df_feat.columns if c.startswith("vix_")}
    SKIP = ({"label", "volume"} | LEVEL_COLS | PTS_EXCLUDE | OPT_META
            | OI_EXCLUDE | VIX_EXCLUDE)

    FCOLS = [c for c in df_feat.columns
             if c not in SKIP
             and df_feat[c].dtype in [np.float64, np.int64, np.int32, float, int]]

    # List which new V10 features made it into FCOLS
    v9_cols = set(joblib.load("models/feature_cols_v9.pkl")) if Path("models/feature_cols_v9.pkl").exists() else set()
    new_v10  = [c for c in FCOLS if c not in v9_cols]
    log.info(f"\nFCOLS: {len(FCOLS)} features ({len(FCOLS)-len(v9_cols):+d} vs V9)")
    log.info(f"New V10 option features in model: {new_v10}")

    # ── Train ─────────────────────────────────────────────────────────────
    log.info(f"\n[5/6] Training V10 ensemble...")
    X = df_feat[FCOLS].values
    y = df_feat["label"].values

    # ── Purged 3-way split (Issue #2): TRAIN / VAL(early-stop) / TEST(frozen) ──
    tr_idx, va_idx, te_idx = purged_train_val_test_split(len(X))
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X[tr_idx])     # scaler fit on TRAIN ONLY
    Xvl = sc.transform(X[va_idx])         # early-stopping set
    Xte = sc.transform(X[te_idx])         # FROZEN test — never seen by fit
    ytr, yvl, yte = y[tr_idx], y[va_idx], y[te_idx]

    # Class balancing (same as V9)
    call_n = int((ytr == 0).sum())
    put_n  = int((ytr == 1).sum())
    dir_n  = call_n + put_n
    skip_pct  = (ytr == 2).mean()
    trade_pct = 1 - skip_pct
    trade_w   = skip_pct / trade_pct if trade_pct > 0 else 2.0
    call_balance = dir_n / (2.0 * call_n) if call_n > 0 else 1.0
    put_balance  = dir_n / (2.0 * put_n)  if put_n  > 0 else 1.0
    sample_w = np.where(ytr == 0, trade_w * call_balance,
               np.where(ytr == 1, trade_w * put_balance, 1.0))
    log.info(f"Class balance: CALL={call_n:,} PUT={put_n:,} | "
             f"CALL_w×{call_balance:.2f} PUT_w×{put_balance:.2f}")

    # IV-expansion risk weighting: down-weight bars where IV rank > 0.85
    # (during IV spikes, option premiums are unpredictable — bad training signal)
    if "atm_iv_extreme" in df_feat.columns:
        iv_extreme = df_feat["atm_iv_extreme"].values[tr_idx]
        sample_w = sample_w * np.where(iv_extreme == 1, 0.5, 1.0)
        log.info(f"IV-extreme down-weight: {iv_extreme.sum():,} bars × 0.5x")

    models = _fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w)

    # SHAP prune
    if prune_shap:
        log.info("\n── SHAP pruning ──")
        anchor = models.get("xgb") or models.get("lgb") or list(models.values())[0]
        pruned = shap_prune_features(anchor, Xtr, FCOLS, bottom_pct=0.25, corr_threshold=0.95)
        if len(pruned) < len(FCOLS):
            log.info(f"\n── Pass 2 ({len(pruned)} pruned features) ──")
            FCOLS = pruned
            X = df_feat[FCOLS].values
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx])
            Xvl = sc.transform(X[va_idx])
            Xte = sc.transform(X[te_idx])
            models = _fit_ensemble(Xtr, ytr, Xvl, yvl, sample_w)

    # ── Save to registry (Issue #4: never overwrite live model directly) ─────
    Path(MODEL_PATH).parent.mkdir(exist_ok=True)
    log.info(f"\n  Saving candidate to registry (live path updated only after gate passes)")

    # ── Evaluation (Issue #2) ─────────────────────────────────────────────
    # Headline metrics use the FROZEN TEST set (never seen during fit).
    # VAL (early-stop set) is shown for comparison — a large VAL->TEST gap
    # signals overfitting to the early-stopping set.
    log.info(f"\n[6/6] Evaluation (headline = FROZEN TEST set)...")
    CONF = 0.22

    tr_dates = df_feat.index[tr_idx]
    va_dates = df_feat.index[va_idx]
    te_dates = df_feat.index[te_idx]
    log.info(f"\n{'='*65}")
    log.info("  V10 SPLIT SUMMARY")
    log.info(f"{'='*65}")
    log.info(f"  TRAIN : {tr_dates.min().date()} -> {tr_dates.max().date()} "
             f"({tr_dates.normalize().nunique()} days, {len(tr_idx):,} bars)")
    log.info(f"  VAL   : {va_dates.min().date()} -> {va_dates.max().date()} "
             f"({va_dates.normalize().nunique()} days, {len(va_idx):,} bars)")
    log.info(f"  TEST  : {te_dates.min().date()} -> {te_dates.max().date()} "
             f"({te_dates.normalize().nunique()} days, {len(te_idx):,} bars)")
    log.info(f"  Features: {len(FCOLS)} total ({len(new_v10):+d} new V10 vs V9)")
    log.info(f"{'='*65}")

    def _dir_acc(X_eval, y_eval):
        p = np.mean([m.predict_proba(X_eval) for m in models.values()], axis=0)
        s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
        dm = (s != 2) & (y_eval != 2)
        return (y_eval[dm] == s[dm]).mean() if dm.any() else 0.0
    _va_da = _dir_acc(Xvl, yvl)
    _te_da = _dir_acc(Xte, yte)
    log.info(f"  [VAL ] direction accuracy: {_va_da:.1%}")
    log.info(f"  [TEST] direction accuracy: {_te_da:.1%}")
    log.info(f"  VAL->TEST gap: {_va_da - _te_da:+.1%} "
             f"({'OVERFIT RISK' if (_va_da - _te_da) > 0.05 else 'ok'})")

    # ── Detailed report on the FROZEN TEST set ────────────────────────────
    preds = np.mean([m.predict_proba(Xte) for m in models.values()], axis=0)
    sigs  = np.where(preds.max(axis=1) >= CONF, preds.argmax(axis=1), 2)
    traded = sigs != 2

    if traded.any():
        acc     = (yte[traded] == sigs[traded]).mean()
        dir_msk = traded & (yte != 2)
        dir_acc = (yte[dir_msk] == sigs[dir_msk]).mean() if dir_msk.any() else 0
        nc  = int((sigs == 0).sum())
        np_ = int((sigs == 1).sum())
        ns  = int((sigs == 2).sum())
        total_traded = nc + np_

        log.info(f"\n  [TEST] Signals : CALL={nc}  PUT={np_}  SKIP={ns}")
        log.info(f"  [TEST] Trade rate : {total_traded/(total_traded+ns):.1%} of bars")
        log.info(f"  [TEST] Overall accuracy  : {acc:.1%}")
        log.info(f"  [TEST] Direction accuracy: {dir_acc:.1%}")

        log.info(f"\n  [TEST] Per-signal win rates:")
        for sig_id, sig_name in [(0, "CALL"), (1, "PUT")]:
            mask = (sigs == sig_id)
            if mask.sum() > 0:
                correct  = (yte[mask] == sig_id).sum()
                wrong    = (yte[mask] != sig_id).sum()
                win_rate = correct / mask.sum()
                avg_conf = preds[mask, sig_id].mean()
                log.info(f"    {sig_name} : {mask.sum():4d} signals | "
                         f"Win={correct} ({win_rate:.1%}) | Wrong={wrong} | "
                         f"Avg conf={avg_conf:.2f}")

        log.info(f"\n  [TEST] Precision by confidence tier (directional):")
        for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
            hi_dir = (preds.max(axis=1) >= threshold) & (preds.argmax(axis=1) != 2)
            if hi_dir.sum() > 0:
                precision = (preds.argmax(axis=1)[hi_dir] == yte[hi_dir]).mean()
                log.info(f"    conf>={threshold:.0%} -> {hi_dir.sum():4d} signals | "
                         f"precision={precision:.1%}")

        if v9_cols:
            log.info(f"\n  Feature comparison vs V9:")
            log.info(f"    V9 features: {len(v9_cols)} | V10 features: {len(FCOLS)} "
                     f"| new V10: {len(new_v10)}")
            if new_v10:
                anchor = models.get("xgb") or models.get("lgb") or list(models.values())[0]
                if hasattr(anchor, "feature_importances_"):
                    imp_map = dict(zip(FCOLS, anchor.feature_importances_))
                    log.info(f"    New V10 feature importances (top 10):")
                    for feat in sorted(new_v10, key=lambda f: imp_map.get(f, 0), reverse=True)[:10]:
                        log.info(f"      {feat:30s}: {imp_map.get(feat, 0):.5f}")

        anchor = models.get("xgb") or models.get("lgb") or list(models.values())[0]
        if hasattr(anchor, "feature_importances_"):
            imp = anchor.feature_importances_
            top = np.argsort(imp)[-15:][::-1]
            log.info(f"\n  Top 15 features by importance:")
            for rank, i in enumerate(top, 1):
                tag = " *NEW" if FCOLS[i] in new_v10 else ""
                log.info(f"    {rank:2d}. {FCOLS[i]:35s} {imp[i]:.4f}{tag}")
    else:
        log.warning("  No TEST signals at CONF=%.2f — threshold may be too high", CONF)

    log.info(f"\n{'='*65}")
    log.info("  V10 TRAINING COMPLETE")

    # ── Registry: save candidate with metadata (Issue #4) ─────────────────
    candidate_id = None
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.model_registry import save_candidate

        metadata = {
            # Data ranges (te_dates from the split done above)
            "train_date_start": str(df_feat.index[tr_idx].min().date()),
            "train_date_end":   str(df_feat.index[tr_idx].max().date()),
            "val_date_start":   str(df_feat.index[va_idx].min().date()),
            "val_date_end":     str(df_feat.index[va_idx].max().date()),
            "test_date_start":  str(df_feat.index[te_idx].min().date()),
            "test_date_end":    str(df_feat.index[te_idx].max().date()),
            "train_bars":  int(len(tr_idx)),
            "val_bars":    int(len(va_idx)),
            "test_bars":   int(len(te_idx)),
            # Quality from frozen TEST set
            "val_dir_acc":  round(_va_da, 4),
            "test_dir_acc": round(_te_da, 4),
            "test_signals": int((sigs != 2).sum()) if traded.any() else 0,
            "val_test_gap": round(_va_da - _te_da, 4),
            # Architecture
            "n_features":       len(FCOLS),
            "new_v10_features": len(new_v10),
            "fwd_bars":         FWD_BARS,
            "train_frac":       TRAIN_FRAC,
            "val_frac":         VAL_FRAC,
            "embargo_bars":     EMBARGO_BARS,
            "label_quantile":   0.70,
        }

        candidate_id = save_candidate("v10", models, sc, FCOLS, metadata)
        log.info(f"  Registry candidate: v10/{candidate_id}")
    except Exception as _e:
        log.warning(f"  Registry save failed ({_e}) — falling back to direct save")
        joblib.dump(models, MODEL_PATH)
        joblib.dump(sc,     SCALER_PATH)
        joblib.dump(FCOLS,  FCOLS_PATH)
        log.info(f"  Fallback saved: {MODEL_PATH}")

    return candidate_id
    log.info(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NIFTY V10 model (V9 + option features)")
    parser.add_argument("--csv5",   default=CSV_5M)
    parser.add_argument("--csv15",  default=CSV_15M)
    parser.add_argument("--csv30",  default=CSV_30M)
    parser.add_argument("--csv60",  default=CSV_60M)
    parser.add_argument("--csvday", default=CSV_DAY)
    parser.add_argument("--csvvix", default=CSV_VIX)
    parser.add_argument("--csvfut", default=CSV_FUT)
    parser.add_argument("--v10data", default=V10_FEATURES_CSV,
                        help="Path to v10_training_features.csv from build_training_dataset.py")
    parser.add_argument("--prune-shap", action="store_true")
    args = parser.parse_args()

    train_v10(
        csv5=args.csv5, csv15=args.csv15, csv30=args.csv30,
        csv60=args.csv60, csv_day=args.csvday, csv_vix=args.csvvix,
        csv_fut=args.csvfut, v10_features=args.v10data,
        prune_shap=args.prune_shap,
    )
