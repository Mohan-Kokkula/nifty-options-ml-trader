"""
truedata_parser.py — Parse Truedata historical files and build V10 feature matrix.

Truedata file format (no header):
  DD-MM-YYYY,HH:MM:SS,open,high,low,close,volume,oi,0.00,0.00

USAGE:
    from core.truedata_parser import TruedataParser

    parser = TruedataParser(data_dir="data/truedata_archive")
    df = parser.build_option_features(expiry="260519", strike=24000, opt_type="CE")
    # df has: datetime, open, high, low, close, vol, oi, moneyness, iv, delta, ...
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File reader
# ─────────────────────────────────────────────────────────────────────────────

_COLUMNS = ["date_str", "time_str", "open", "high", "low", "close", "volume", "oi", "_x", "_y"]
_DTYPES  = {"open": float, "high": float, "low": float, "close": float,
            "volume": int, "oi": int}


def read_truedata_file(path: Path | str) -> pd.DataFrame:
    """
    Read one Truedata .txt/.csv file into a DataFrame with a proper datetime index.
    Returns empty DataFrame on failure.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < 50:
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            path,
            header=None,
            names=_COLUMNS,
            dtype=str,
        )
        # Parse datetime
        df["datetime"] = pd.to_datetime(
            df["date_str"] + " " + df["time_str"],
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce",
        )
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime").sort_index()

        # Cast numerics
        for col, dtype in _DTYPES.items():
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)

        df = df[["open", "high", "low", "close", "volume", "oi"]]
        return df

    except Exception as e:
        log.warning(f"Failed to read {path}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers (lightweight, no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _bsm_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """Black-Scholes price. opt='CE' or 'PE'."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if opt == "CE" else max(0.0, K - S)
        return intrinsic
    sqT   = math.sqrt(T)
    d1    = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2    = d1 - sigma * sqT
    if opt == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _implied_vol(market_price: float, S: float, K: float, T: float, r: float, opt: str) -> float:
    """Newton-Raphson IV solver. Returns NaN on failure."""
    if T <= 0 or market_price <= 0:
        return float("nan")
    intrinsic = max(0.0, S - K) if opt == "CE" else max(0.0, K - S)
    if market_price < intrinsic - 0.01:
        return float("nan")

    sigma = 0.30  # initial guess
    for _ in range(50):
        price = _bsm_price(S, K, T, r, sigma, opt)
        sqT   = math.sqrt(T)
        d1    = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
        vega  = S * _norm_cdf(d1) * sqT * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        if vega < 1e-10:
            break
        sigma -= (price - market_price) / vega
        sigma  = max(0.001, min(sigma, 5.0))
        if abs(price - market_price) < 0.01:
            return sigma

    # Bisection fallback
    lo, hi = 0.001, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p   = _bsm_price(S, K, T, r, mid, opt)
        if p < market_price:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-5:
            return (lo + hi) / 2
    return float("nan")


def _delta(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if (opt == "CE" and S > K) else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if opt == "CE" else _norm_cdf(d1) - 1


def _theta(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """Theta (per day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqT  = math.sqrt(T)
    d1   = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2   = d1 - sigma * sqT
    phi  = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    if opt == "CE":
        th = (-S * phi * sigma / (2 * sqT) - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
    else:
        th = (-S * phi * sigma / (2 * sqT) + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365
    return th


# ─────────────────────────────────────────────────────────────────────────────
# Main parser class
# ─────────────────────────────────────────────────────────────────────────────

class TruedataParser:
    """
    Reads Truedata downloaded files and produces V10 training DataFrames.

    Directory layout expected:
        data_dir/
          index/NIFTY_I.csv        ← continuous futures (spot proxy)
          index/NIFTY.csv          ← spot
          index/INDIAVIX.csv       ← VIX
          options/260519/NIFTY26051924000CE.csv
          options/260519/NIFTY26051924000PE.csv
          ...
    """

    RISK_FREE = 0.065  # India 10yr govt bond ≈ 6.5%

    def __init__(self, data_dir: str | Path = "data/truedata_archive"):
        self.data_dir = Path(data_dir)
        self._spot_cache: dict[str, pd.DataFrame] = {}

    # ── low-level loaders ────────────────────────────────────────────────────

    def load_spot(self) -> pd.DataFrame:
        """Load NIFTY-I (continuous futures) as spot proxy."""
        if "spot" not in self._spot_cache:
            # Try NIFTY-I first, then NIFTY
            for name in ("NIFTY_I", "NIFTY"):
                p = self.data_dir / "index" / f"{name}.csv"
                df = read_truedata_file(p)
                if not df.empty:
                    self._spot_cache["spot"] = df
                    return df
            log.warning("No spot data found (NIFTY_I or NIFTY)")
            self._spot_cache["spot"] = pd.DataFrame()
        return self._spot_cache["spot"]

    def load_vix(self) -> pd.DataFrame:
        """Load INDIAVIX 5-min."""
        p = self.data_dir / "index" / "INDIAVIX.csv"
        return read_truedata_file(p)

    def load_option(self, expiry: str, strike: int, opt_type: str) -> pd.DataFrame:
        """Load one option contract e.g. expiry='260519', strike=24000, opt_type='CE'."""
        symbol = f"NIFTY{expiry}{strike}{opt_type}"
        p = self.data_dir / "options" / expiry / f"{symbol}.csv"
        return read_truedata_file(p)

    def available_expiries(self) -> list[str]:
        """List expiry codes that have downloaded data."""
        opt_root = self.data_dir / "options"
        if not opt_root.exists():
            return []
        return sorted(d.name for d in opt_root.iterdir() if d.is_dir())

    def available_strikes(self, expiry: str) -> list[tuple[int, str]]:
        """List (strike, opt_type) pairs downloaded for an expiry."""
        exp_dir = self.data_dir / "options" / expiry
        if not exp_dir.exists():
            return []
        result = []
        for f in sorted(exp_dir.glob("*.csv")):
            name = f.stem  # e.g. NIFTY26051924000CE
            prefix = f"NIFTY{expiry}"
            if name.startswith(prefix):
                rest = name[len(prefix):]  # e.g. 24000CE
                opt  = rest[-2:]
                try:
                    strike = int(rest[:-2])
                    result.append((strike, opt))
                except ValueError:
                    pass
        return result

    # ── feature builder ──────────────────────────────────────────────────────

    def build_option_features(
        self,
        expiry: str,
        strike: int,
        opt_type: str,
        trading_hours_only: bool = True,
    ) -> pd.DataFrame:
        """
        Build a row-per-5min feature DataFrame for one option contract.

        Columns:
          datetime, expiry, strike, opt_type,
          opt_open, opt_high, opt_low, opt_close, opt_volume, opt_oi,
          spot_close, spot_pct_chg, spot_range_pct,
          vix_close, vix_rank_30d,
          moneyness, log_moneyness, time_to_expiry_days,
          iv, delta, theta,
          iv_premium_pct (opt vs BSM),
          oi_change, vol_spike,
          is_itm, is_near_expiry,
          label_direction (1=rose, -1=fell, 0=unchanged over next 15min)
        """
        opt_df  = self.load_option(expiry, strike, opt_type)
        spot_df = self.load_spot()
        vix_df  = self.load_vix()

        if opt_df.empty or spot_df.empty:
            return pd.DataFrame()

        # Parse expiry date
        exp_date = _parse_expiry_date(expiry)

        # Restrict to trading hours (9:15–15:30)
        if trading_hours_only:
            opt_df  = _filter_trading_hours(opt_df)
            spot_df = _filter_trading_hours(spot_df)
            if not vix_df.empty:
                vix_df = _filter_trading_hours(vix_df)

        # Align on 5-min index
        opt_df = opt_df[~opt_df.index.duplicated(keep="first")]

        # Merge spot
        spot_aligned = spot_df["close"].reindex(opt_df.index, method="ffill")

        # Merge VIX
        if not vix_df.empty:
            vix_aligned = vix_df["close"].reindex(opt_df.index, method="ffill")
        else:
            vix_aligned = pd.Series(18.0, index=opt_df.index)

        rows = []
        prev_oi = None
        prev_close = None

        for idx, row in opt_df.iterrows():
            S    = float(spot_aligned.get(idx, float("nan")))
            vix  = float(vix_aligned.get(idx, 18.0))
            K    = float(strike)
            opt_c = float(row["close"])

            if math.isnan(S) or S <= 0 or opt_c <= 0:
                prev_oi    = int(row["oi"])
                prev_close = opt_c
                continue

            # Time to expiry
            T = _time_to_expiry(idx, exp_date)

            # IV
            iv = _implied_vol(opt_c, S, K, T, self.RISK_FREE, opt_type)

            # Greeks (use iv if valid, else fallback)
            iv_use = iv if not math.isnan(iv) else 0.20
            dlt    = _delta(S, K, T, self.RISK_FREE, iv_use, opt_type)
            th     = _theta(S, K, T, self.RISK_FREE, iv_use, opt_type)

            # BSM fair price
            bsm    = _bsm_price(S, K, T, self.RISK_FREE, iv_use, opt_type)
            iv_prem = (opt_c - bsm) / bsm * 100 if bsm > 0.01 else 0.0

            # Moneyness
            mono      = (S - K) / K if opt_type == "CE" else (K - S) / K
            log_mono  = math.log(S / K) if S > 0 and K > 0 else 0.0

            # OI change
            oi_chg = (int(row["oi"]) - prev_oi) if prev_oi is not None else 0

            # Volume spike (compare to rolling avg — simplified: just flag >1000)
            vol_spike = 1 if int(row["volume"]) > 500 else 0

            # Spot % change (vs previous bar)
            spot_pct  = 0.0
            spot_range = 0.0
            if prev_close is not None and prev_close > 0:
                spot_pct = (opt_c - prev_close) / prev_close * 100

            rows.append({
                "datetime":          idx,
                "expiry":            expiry,
                "strike":            strike,
                "opt_type":          opt_type,
                "opt_open":          float(row["open"]),
                "opt_high":          float(row["high"]),
                "opt_low":           float(row["low"]),
                "opt_close":         opt_c,
                "opt_volume":        int(row["volume"]),
                "opt_oi":            int(row["oi"]),
                "spot_close":        S,
                "vix_close":         vix,
                "moneyness":         round(mono, 6),
                "log_moneyness":     round(log_mono, 6),
                "time_to_expiry_d":  round(T * 365, 4),
                "iv":                round(iv, 6) if not math.isnan(iv) else None,
                "delta":             round(dlt, 6),
                "theta":             round(th, 6),
                "iv_premium_pct":    round(iv_prem, 4),
                "oi_change":         oi_chg,
                "vol_spike":         vol_spike,
                "is_itm":            1 if mono > 0 else 0,
                "is_near_expiry":    1 if T * 365 <= 1.0 else 0,
            })

            prev_oi    = int(row["oi"])
            prev_close = opt_c

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("datetime")

        # Add label: did option price rise in next 15 min (3 bars)?
        df["label_direction"] = _compute_labels(df["opt_close"], forward_bars=3)

        return df

    def build_full_training_set(
        self,
        expiries: list[str] | None = None,
        max_strikes_per_expiry: int | None = None,
        trading_hours_only: bool = True,
    ) -> pd.DataFrame:
        """
        Build V10 training dataset from all downloaded option files.

        Returns a single merged DataFrame with all contracts, all dates.
        Call scripts/train_model_v9.py or train_model_v10.py after this.
        """
        if expiries is None:
            expiries = self.available_expiries()

        all_frames = []
        for exp in expiries:
            pairs = self.available_strikes(exp)
            if max_strikes_per_expiry:
                pairs = pairs[:max_strikes_per_expiry]
            log.info(f"Expiry {exp}: processing {len(pairs)} contracts...")
            for strike, opt_type in pairs:
                df = self.build_option_features(exp, strike, opt_type, trading_hours_only)
                if not df.empty:
                    all_frames.append(df)

        if not all_frames:
            log.warning("No data frames built — check download first")
            return pd.DataFrame()

        combined = pd.concat(all_frames, axis=0).sort_index()
        log.info(f"Full training set: {len(combined):,} rows, {combined.shape[1]} features")
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _parse_expiry_date(code: str) -> date:
    """'260519' → date(2026, 5, 19)."""
    yy = int(code[0:2])
    mm = int(code[2:4])
    dd = int(code[4:6])
    return date(2000 + yy, mm, dd)


def _time_to_expiry(dt: datetime | pd.Timestamp, exp_date: date) -> float:
    """Time to expiry in years (simplified: calendar days / 365)."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    target = datetime(exp_date.year, exp_date.month, exp_date.day, 15, 30)
    secs   = (target - dt).total_seconds()
    return max(0.0, secs / (365.0 * 24 * 3600))


def _filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:15–15:30 bars."""
    if df.empty:
        return df
    mask = (
        (df.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
        (df.index.time <= datetime.strptime("15:30", "%H:%M").time())
    )
    return df[mask]


def _compute_labels(close: pd.Series, forward_bars: int = 3) -> pd.Series:
    """
    Label = 1 if close[t+forward_bars] > close[t] by ≥0.5%
            -1 if close[t+forward_bars] < close[t] by ≥0.5%
             0 otherwise
    """
    future = close.shift(-forward_bars)
    pct    = (future - close) / close * 100
    label  = pd.Series(0, index=close.index)
    label[pct >= 0.5]  = 1
    label[pct <= -0.5] = -1
    return label


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Try to load the one file the user already downloaded
    data_root = Path("data/truedata_archive")
    desktop = Path.home() / "OneDrive" / "Desktop"

    # Check if user's manually downloaded files exist
    manual_file = desktop / "20260519 115013 547" / "Trade 5 Minute" / "NIFTY26051925000CE.txt"
    if manual_file.exists():
        log.info(f"Reading manually downloaded file: {manual_file}")
        df = read_truedata_file(manual_file)
        print(f"Rows: {len(df)}")
        print(df.head(10))
    else:
        # Try archive
        parser = TruedataParser(data_root)
        expiries = parser.available_expiries()
        print(f"Available expiries: {expiries}")
        if expiries:
            pairs = parser.available_strikes(expiries[-1])
            print(f"Strikes in {expiries[-1]}: {pairs[:5]}...")
