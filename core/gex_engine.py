"""
gex_engine.py -- Gamma Exposure (GEX) Engine for Nifty
=======================================================
Computes GEX from options chain data and provides key levels:
  - Net GEX per strike
  - Call Wall (strike with max call GEX)
  - Put Wall (strike with max put GEX in absolute terms)
  - Gamma Flip (where net GEX crosses zero)
  - Total GEX (positive = dampening, negative = amplifying)

For backtesting without historical OI data, provides a proxy-based
GEX regime classifier using VIX + ATR ratio.

GEX Formula (per QuantPy video):
  GEX_call = gamma * OI * lot_size * spot^2 / 100
  GEX_put  = -gamma * OI * lot_size * spot^2 / 100
  (negative because MM is short puts -> short gamma)
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm


NIFTY_LOT_SIZE = 75


def bsm_gamma(spot, strike, r, T, sigma):
    if T <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (spot * sigma * math.sqrt(T))


def compute_gex_from_chain(chain_df, spot, days_to_expiry, r=0.065):
    T = max(days_to_expiry / 365.0, 1 / 365.0)
    results = []
    for _, row in chain_df.iterrows():
        strike = float(row["strike"])
        call_oi = float(row.get("call_oi", 0))
        put_oi = float(row.get("put_oi", 0))
        call_iv = float(row.get("call_iv", 0)) / 100.0
        put_iv = float(row.get("put_iv", 0)) / 100.0
        if call_iv <= 0:
            call_iv = 0.15
        if put_iv <= 0:
            put_iv = 0.15

        gamma_c = bsm_gamma(spot, strike, r, T, call_iv)
        gamma_p = bsm_gamma(spot, strike, r, T, put_iv)

        call_gex = gamma_c * call_oi * NIFTY_LOT_SIZE * spot**2 / 100.0
        put_gex = -gamma_p * put_oi * NIFTY_LOT_SIZE * spot**2 / 100.0
        net_gex = call_gex + put_gex

        results.append({
            "strike": strike, "call_oi": call_oi, "put_oi": put_oi,
            "call_gex": call_gex, "put_gex": put_gex, "net_gex": net_gex,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return {"gex_by_strike": df, "call_wall": 0, "put_wall": 0,
                "gamma_flip": spot, "total_gex": 0, "regime": "neutral"}

    call_wall = float(df.loc[df["call_gex"].idxmax(), "strike"])
    put_wall = float(df.loc[df["put_gex"].idxmin(), "strike"])
    total_gex = float(df["net_gex"].sum())
    gamma_flip = _find_gamma_flip(df, spot)

    regime = "positive" if total_gex > 0 else ("negative" if total_gex < 0 else "neutral")

    return {
        "gex_by_strike": df, "call_wall": call_wall, "put_wall": put_wall,
        "gamma_flip": gamma_flip, "total_gex": total_gex, "regime": regime,
    }


def _find_gamma_flip(gex_df, spot):
    df = gex_df.sort_values("strike").reset_index(drop=True)
    best_flip = spot
    min_dist = float("inf")
    for i in range(1, len(df)):
        g0, g1 = df.loc[i - 1, "net_gex"], df.loc[i, "net_gex"]
        if g0 * g1 < 0:
            s0, s1 = df.loc[i - 1, "strike"], df.loc[i, "strike"]
            flip = s0 + (s1 - s0) * abs(g0) / (abs(g0) + abs(g1))
            dist = abs(flip - spot)
            if dist < min_dist:
                min_dist = dist
                best_flip = flip
    return best_flip


def gex_proxy_regime(vix, atr_fast, atr_slow):
    """
    Proxy GEX regime when no options OI data available.
    VIX thresholds calibrated for India VIX.
    """
    if vix is None or vix <= 0:
        vix = 15.0
    atr_ratio = atr_fast / atr_slow if atr_slow > 0 else 1.0

    if atr_ratio < 0.80:
        return "positive", atr_ratio, vix
    elif vix < 15 and atr_ratio < 0.92:
        return "positive", atr_ratio, vix
    elif atr_ratio > 1.20:
        return "negative", atr_ratio, vix
    elif vix > 17 and atr_ratio > 1.05:
        return "negative", atr_ratio, vix
    else:
        return "neutral", atr_ratio, vix


def proxy_key_levels(prev_high, prev_low, prev_close, spot):
    round_above = math.ceil(spot / 100) * 100
    round_below = math.floor(spot / 100) * 100
    return {
        "call_wall": max(prev_high, round_above),
        "put_wall": min(prev_low, round_below),
        "gamma_flip": prev_close,
        "round_above": round_above,
        "round_below": round_below,
    }


def get_days_to_expiry(current_date, expiry_dates):
    future = [e for e in expiry_dates if e >= current_date]
    if not future:
        return 5, None
    nearest = min(future)
    return max((nearest - current_date).days, 1), nearest


if __name__ == "__main__":
    chain = pd.read_csv("data/nifty_options_chain_20261028.csv")
    spot = 22698.85
    result = compute_gex_from_chain(chain, spot, days_to_expiry=5)

    print(f"Spot:        {spot}")
    print(f"Call Wall:   {result['call_wall']}")
    print(f"Put Wall:    {result['put_wall']}")
    print(f"Gamma Flip:  {result['gamma_flip']:.0f}")
    print(f"Total GEX:   {result['total_gex']:,.0f}")
    print(f"Regime:      {result['regime']}")
    print()

    df = result["gex_by_strike"]
    df_near = df[(df["strike"] >= spot - 500) & (df["strike"] <= spot + 500)]
    print("Strike-wise GEX near ATM:")
    print(f"{'Strike':>8s}  {'Call GEX':>14s}  {'Put GEX':>14s}  {'Net GEX':>14s}")
    for _, r in df_near.iterrows():
        print(f"{r['strike']:8.0f}  {r['call_gex']:14,.0f}  {r['put_gex']:14,.0f}  {r['net_gex']:14,.0f}")
