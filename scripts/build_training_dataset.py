"""
build_training_dataset.py — Merge NSE Bhavcopy + spot/VIX into V10 training features.

Input files required:
  data/nse_bhavcopy/nifty_options_merged.csv   ← from download_nse_bhavcopy.py
  data/nifty_day.csv                           ← existing daily NIFTY spot
  data/india_vix.csv                           ← existing VIX data

  Optionally: 5-min spot data from Truedata manual download:
  data/truedata_archive/index/NIFTY_I.csv      ← if available, adds intraday features

Output:
  data/v10_training_features.csv               ← ready for train_model_v10.py

USAGE:
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --min-strikes 5  # only liquid options
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RISK_FREE = 0.065   # India 10yr ≈ 6.5%


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes (standalone, no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _bsm(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if opt == "CE" else max(0.0, K - S)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    if opt == "CE":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _iv(price, S, K, T, r, opt) -> float:
    """Newton-Raphson IV, returns NaN on failure."""
    if price <= 0 or T <= 0 or S <= 0:
        return float("nan")
    sig = 0.25
    for _ in range(60):
        p = _bsm(S, K, T, r, sig, opt)
        sq = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sq)
        vega = S * _ncdf(d1) * sq * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        if vega < 1e-8:
            break
        sig -= (p - price) / vega
        sig = max(0.01, min(sig, 5.0))
        if abs(p - price) < 0.01:
            return sig
    # bisection fallback
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _bsm(S, K, T, r, mid, opt) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            return (lo + hi) / 2
    return float("nan")


def _delta(S, K, T, r, sigma, opt) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if (opt == "CE" and S > K) else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if opt == "CE" else _ncdf(d1) - 1


def _theta_per_day(S, K, T, r, sigma, opt) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    phi = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    if opt == "CE":
        return (-S * phi * sigma / (2 * sq) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    return (-S * phi * sigma / (2 * sq) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365


def _gamma(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    phi = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return phi / (S * sigma * math.sqrt(T))


def _time_to_expiry(trade_date: str, expiry_str: str) -> float:
    """
    Returns T in years.
    NSE new Bhavcopy expiry format: '2026-05-26' (ISO)
    Also handles legacy: '26-MAY-2026'
    """
    for fmt in ["%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d"]:
        try:
            exp   = datetime.strptime(expiry_str.strip(), fmt).date()
            trade = datetime.strptime(str(trade_date)[:10], "%Y-%m-%d").date()
            days  = (exp - trade).days
            return max(0.0, days / 365.0)
        except ValueError:
            continue
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nifty_daily() -> pd.DataFrame:
    """Load NIFTY daily OHLCV from data/nifty_day.csv."""
    p = DATA_DIR / "nifty_day.csv"
    if not p.exists():
        log.warning(f"Missing: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    log.info(f"NIFTY daily: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def load_vix_daily() -> pd.Series:
    """Load INDIAVIX daily close from data/india_vix.csv."""
    p = DATA_DIR / "india_vix.csv"
    if not p.exists():
        log.warning(f"Missing VIX: {p} — using constant 18")
        return pd.Series(dtype=float)
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    vix = df["close"] if "close" in df.columns else df.iloc[:, 0]
    log.info(f"VIX daily: {len(vix)} rows")
    return vix


def load_bhavcopy() -> pd.DataFrame:
    """Load merged Bhavcopy."""
    p = DATA_DIR / "nse_bhavcopy" / "nifty_options_merged.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing: {p}\nRun first: python scripts/download_nse_bhavcopy.py"
        )
    df = pd.read_csv(p, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()  # strip time if any
    log.info(f"Bhavcopy: {len(df):,} rows, {df['date'].nunique()} days")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_spot_features(nifty: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Build daily spot-level features:
      - NIFTY returns at multiple lookbacks
      - Volatility (rolling std of returns)
      - VIX features
    """
    df = nifty.copy()
    df["ret_1d"]  = df["close"].pct_change(1)
    df["ret_3d"]  = df["close"].pct_change(3)
    df["ret_5d"]  = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)

    df["vol_5d"]  = df["ret_1d"].rolling(5).std() * math.sqrt(252)
    df["vol_20d"] = df["ret_1d"].rolling(20).std() * math.sqrt(252)

    df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]

    # EMA crossover (8/21)
    df["ema8"]  = df["close"].ewm(span=8).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema_cross"] = (df["ema8"] - df["ema21"]) / df["close"]

    # RSI (14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-9)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # VIX features
    if not vix.empty:
        vix_aligned = vix.reindex(df.index, method="ffill")
        df["vix"]          = vix_aligned
        df["vix_ret_1d"]   = vix_aligned.pct_change(1)
        df["vix_ma20"]     = vix_aligned.rolling(20).mean()
        df["vix_zscore"]   = (vix_aligned - vix_aligned.rolling(20).mean()) / (vix_aligned.rolling(20).std() + 1e-9)
        df["vix_rank_30d"] = vix_aligned.rolling(30).rank(pct=True)
        df["vix_extreme"]  = (vix_aligned > 25).astype(int) - (vix_aligned < 12).astype(int)
    else:
        df["vix"] = 18.0

    return df


def build_option_features(bhav: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, expiry, strike, opt_type) row, compute:
      - IV (Black-Scholes from settle price)
      - Delta, Theta, Gamma
      - Moneyness features
      - OI momentum
      - PCR (put-call ratio at strike level)
      - Next-day return label

    Uses NSE's own underlying price (UndrlygPric column) when available —
    more accurate than daily close from nifty_day.csv.
    """
    # Spot lookup: date -> NIFTY close (fallback if underlying not in bhavcopy)
    if "close" in spot_df.columns:
        spot_map = {d.strftime("%Y-%m-%d"): v for d, v in spot_df["close"].items()}
    else:
        spot_map = {}

    # Check if bhavcopy has embedded underlying price
    has_underlying = "underlying" in bhav.columns

    rows = []
    bhav_sorted = bhav.sort_values(["date", "expiry", "strike", "opt_type"])

    log.info(f"Computing option features for {len(bhav_sorted):,} rows...")

    # Group by date for PCR computation
    date_groups = bhav_sorted.groupby("date")

    prev_oi: dict[tuple, int] = {}   # (expiry, strike, opt_type) -> previous OI

    for trade_date, day_grp in date_groups:
        date_str = str(trade_date)[:10]

        # Prefer NSE embedded underlying price (most accurate)
        if has_underlying:
            underlying_vals = day_grp["underlying"].dropna()
            underlying_vals = underlying_vals[underlying_vals > 0]
            S = float(underlying_vals.median()) if len(underlying_vals) > 0 else None
        else:
            S = None

        # Fallback to daily NIFTY close
        if not S:
            S = spot_map.get(date_str, None)
        if not S:
            # Try nearest prior date
            from datetime import timedelta
            for offset in range(1, 6):
                nearby = (trade_date - timedelta(days=offset)).strftime("%Y-%m-%d")
                if nearby in spot_map:
                    S = spot_map[nearby]
                    break
        if not S or S <= 0:
            prev_oi = {}
            continue

        # Build PCR by expiry for this day
        pcr_by_exp: dict[str, float] = {}
        for exp, exp_grp in day_grp.groupby("expiry"):
            ce_oi = exp_grp[exp_grp["opt_type"] == "CE"]["oi"].sum()
            pe_oi = exp_grp[exp_grp["opt_type"] == "PE"]["oi"].sum()
            pcr_by_exp[exp] = pe_oi / max(ce_oi, 1)

        for _, row in day_grp.iterrows():
            strike   = float(row["strike"])
            opt_type = str(row["opt_type"]).strip().upper()
            expiry   = str(row["expiry"]).strip()
            price    = float(row.get("settle") or row.get("close") or 0)

            if strike <= 0 or price <= 0:
                continue

            T = _time_to_expiry(str(trade_date)[:10], expiry)
            if math.isnan(T) or T < 0:
                continue

            # IV + Greeks
            iv  = _iv(price, S, strike, T, RISK_FREE, opt_type)
            iv_ = iv if not math.isnan(iv) else 0.20

            dlt = _delta(S, strike, T, RISK_FREE, iv_, opt_type)
            th  = _theta_per_day(S, strike, T, RISK_FREE, iv_, opt_type)
            gam = _gamma(S, strike, T, RISK_FREE, iv_)

            # Moneyness
            mono     = (S - strike) / strike if opt_type == "CE" else (strike - S) / strike
            log_mono = math.log(S / strike) if opt_type == "CE" else math.log(strike / S)

            # OI change
            key    = (expiry, int(strike), opt_type)
            oi_now = int(row.get("oi") or 0)
            oi_chg = oi_now - prev_oi.get(key, oi_now)
            prev_oi[key] = oi_now

            # Premium quality: ratio of market price to BSM fair
            bsm_fair = _bsm(S, strike, T, RISK_FREE, iv_, opt_type)
            prem_ratio = price / max(bsm_fair, 0.01)

            rows.append({
                "date":          str(trade_date)[:10],
                "expiry":        expiry,
                "strike":        int(strike),
                "opt_type":      opt_type,
                "spot":          S,
                "price":         price,
                "volume":        int(row.get("volume") or 0),
                "oi":            oi_now,
                "oi_change":     oi_chg,
                "time_to_exp_d": round(T * 365, 2),
                "iv":            round(iv, 5) if not math.isnan(iv) else None,
                "delta":         round(dlt, 5),
                "theta":         round(th, 5),
                "gamma":         round(gam, 7),
                "moneyness":     round(mono, 5),
                "log_moneyness": round(log_mono, 5),
                "is_itm":        1 if mono > 0 else 0,
                "prem_ratio":    round(prem_ratio, 4),
                "pcr_expiry":    round(pcr_by_exp.get(expiry, 1.0), 4),
                "near_expiry":   1 if T * 365 <= 1 else 0,
            })

    return pd.DataFrame(rows)


def add_spot_features_to_options(opt_df: pd.DataFrame, spot_feats: pd.DataFrame) -> pd.DataFrame:
    """Join daily spot features onto option rows by date."""
    spot_reset = spot_feats.reset_index()
    spot_reset["date"] = spot_reset["date"].dt.strftime("%Y-%m-%d")

    cols_to_join = [
        "date", "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "vol_5d", "vol_20d", "high_low_pct", "close_open_pct",
        "ema_cross", "rsi14", "vix", "vix_ret_1d", "vix_zscore",
        "vix_rank_30d", "vix_extreme",
    ]
    available = [c for c in cols_to_join if c in spot_reset.columns]
    merged = opt_df.merge(spot_reset[available], on="date", how="left")
    return merged


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward-looking labels for training.

    Label 1: next-day option price return (regression target)
    Label 2: direction (1=up≥2%, -1=down≥2%, 0=neutral) — classification target
    Label 3: spot direction next day
    """
    df = df.sort_values(["expiry", "strike", "opt_type", "date"]).copy()

    # Next-day option price
    df["next_price"] = df.groupby(["expiry", "strike", "opt_type"])["price"].shift(-1)
    df["opt_ret_1d"] = (df["next_price"] - df["price"]) / df["price"]

    # Direction label (option)
    df["label_opt"] = 0
    df.loc[df["opt_ret_1d"] >= 0.02,  "label_opt"] = 1
    df.loc[df["opt_ret_1d"] <= -0.02, "label_opt"] = -1

    # Spot direction label
    df["next_spot"] = df.groupby(["date"])["spot"].transform(
        lambda x: x.shift(-1) if len(x) > 0 else x
    )
    # Simpler: just use the daily spot return
    df["spot_ret_1d"] = df.groupby("strike")["spot"].pct_change()  # approximate
    df["label_spot"] = 0
    df.loc[df["spot_ret_1d"] >= 0.005, "label_spot"] = 1
    df.loc[df["spot_ret_1d"] <= -0.005, "label_spot"] = -1

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-strikes", type=int, default=0,
                        help="Min days a strike must appear (filter illiquid)")
    parser.add_argument("--output", default="data/v10_training_features.csv")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Building V10 training dataset")
    log.info("=" * 60)

    # ── Load raw data ─────────────────────────────────────────────────────────
    bhav    = load_bhavcopy()
    nifty   = load_nifty_daily()
    vix     = load_vix_daily()

    if nifty.empty:
        log.error("No NIFTY daily data. Make sure data/nifty_day.csv exists.")
        return

    # ── Spot features ─────────────────────────────────────────────────────────
    log.info("Computing spot features...")
    spot_feats = build_spot_features(nifty, vix)

    # ── Option features ───────────────────────────────────────────────────────
    log.info("Computing option IV/Greeks features...")
    opt_df = build_option_features(bhav, spot_feats)
    log.info(f"Option features: {len(opt_df):,} rows")

    if opt_df.empty:
        log.error("No option features built. Check bhavcopy data.")
        return

    # ── Filter illiquid strikes ───────────────────────────────────────────────
    if args.min_strikes > 0:
        counts = opt_df.groupby(["expiry", "strike", "opt_type"]).size()
        valid  = counts[counts >= args.min_strikes].index
        before = len(opt_df)
        opt_df = opt_df.set_index(["expiry", "strike", "opt_type"]).loc[
            opt_df.set_index(["expiry", "strike", "opt_type"]).index.isin(valid)
        ].reset_index()
        log.info(f"Filtered illiquid: {before:,} -> {len(opt_df):,} rows")

    # ── Join spot features ────────────────────────────────────────────────────
    log.info("Joining spot features...")
    combined = add_spot_features_to_options(opt_df, spot_feats)

    # ── Add labels ────────────────────────────────────────────────────────────
    log.info("Adding forward labels...")
    combined = add_labels(combined)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = BASE_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V10 TRAINING DATASET READY")
    print("=" * 60)
    print(f"  Rows    : {len(combined):,}")
    print(f"  Columns : {combined.shape[1]}")
    print(f"  Dates   : {combined['date'].nunique()} trading days")
    print(f"  CE rows : {(combined['opt_type']=='CE').sum():,}")
    print(f"  PE rows : {(combined['opt_type']=='PE').sum():,}")
    if "iv" in combined.columns:
        iv_valid = combined["iv"].notna().sum()
        print(f"  IV computed : {iv_valid:,} / {len(combined):,} ({100*iv_valid//len(combined)}%)")
    print(f"\n  Output  : {out_path}")
    print()
    print("New V10 features added vs V9:")
    print("  iv, delta, theta, gamma, moneyness, log_moneyness,")
    print("  time_to_exp_d, is_itm, prem_ratio, pcr_expiry,")
    print("  oi, oi_change, near_expiry")
    print()
    print("Next step:")
    print("  python scripts/train_model_v10.py --data data/v10_training_features.csv")


if __name__ == "__main__":
    main()
