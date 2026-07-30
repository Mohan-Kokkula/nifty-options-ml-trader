"""
validate_vol_straddle_kaggle.py — PRELIMINARY vol-strategy P&L test
====================================================================
Built 2026-06-16. Tests whether the GEX/range signal translates into a
friction-surviving NON-DIRECTIONAL (straddle) P&L — the disciplined check
before considering any pivot to vol structures.

REAL prices: ATM straddle priced from the Kaggle intraday option Close
(both legs), held a fixed horizon, exited at the SAME strikes' Close.
Full double-leg round-trip friction (spread+STT+brokerage+GST) via
backtest_options.round_trip_cost. GEX signal from logs/gex_kaggle_ic.csv.

Compares: unconditional LONG straddle, unconditional SHORT straddle, and
GEX-conditioned (LONG when GEX predicts a big move, SHORT when quiet).

PRELIMINARY: ~13 days, single regime. A pass justifies a real vol research
track; a fail closes the vol avenue too.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KAGGLE = ROOT / "data" / "kaggle"
GEX_CSV = ROOT / "logs" / "gex_kaggle_ic.csv"
from backtest_options import round_trip_cost, LOT_SIZE

HOLD_MIN = 30
ENTRY_EVERY = 15
ENTRY_FROM, ENTRY_TO = (9, 45), (14, 30)
_TICK = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


def load_prices() -> dict:
    files = sorted(set(KAGGLE.glob("**/NSE_FNO_DATA_*.csv"))
                   | set(KAGGLE.glob("**/NSE_FNO_DATA_*.CSV"))
                   | set(KAGGLE.glob("**/NSE_FNO_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        if not {"Ticker", "Close"} <= set(df.columns):
            continue
        df = df[df["Ticker"].astype(str).str.startswith("NIFTY")]
        frames.append(df[["Ticker", "Date", "Time", "Close"]])
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Time", "Ticker"])
    base = raw["Ticker"].str.replace(".NFO", "", regex=False)
    m = base.str.extract(_TICK)
    raw = raw.assign(strike=pd.to_numeric(m[1], errors="coerce"), opt=m[2],
                     expiry=m[0]).dropna(subset=["strike", "opt"])
    raw["ts"] = pd.to_datetime(raw["Date"] + " " + raw["Time"],
                               format="%d/%m/%Y %H:%M:%S", errors="coerce")
    raw = raw.dropna(subset=["ts"])
    raw["exp_dt"] = pd.to_datetime(raw["expiry"], format="%d%b%y", errors="coerce")
    near = raw.groupby(raw["ts"].dt.normalize())["exp_dt"].transform("min")
    raw = raw[raw["exp_dt"] == near]
    raw["close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw["key"] = (raw["ts"].astype("int64") // 10**9).astype(str) + "_" + \
                 raw["strike"].astype(int).astype(str) + "_" + raw["opt"]
    return dict(zip(raw["key"], raw["close"]))


def px(prices, ts, strike, opt):
    return prices.get(f"{int(ts.timestamp())}_{int(strike)}_{opt}", np.nan)


def pf(x):
    x = np.asarray(x); g = x[x > 0].sum(); l = -x[x <= 0].sum()
    return g / l if l > 0 else float("inf")


def main():
    print("Loading GEX signal + Kaggle option prices...")
    gx = pd.read_csv(GEX_CSV)
    gx["ts"] = pd.to_datetime(gx["ts"]); gx = gx.set_index("ts").sort_index()
    prices = load_prices()
    qty = LOT_SIZE

    trades = []
    for ts, row in gx.iterrows():
        hm = (ts.hour, ts.minute)
        if hm < ENTRY_FROM or hm > ENTRY_TO or ts.minute % ENTRY_EVERY != 0:
            continue
        S = float(row["spot"]); atm = round(S / 50) * 50
        exit_ts = ts + pd.Timedelta(minutes=HOLD_MIN)
        if exit_ts.normalize() != ts.normalize():
            continue
        ce0, pe0 = px(prices, ts, atm, "CE"), px(prices, ts, atm, "PE")
        ce1, pe1 = px(prices, exit_ts, atm, "CE"), px(prices, exit_ts, atm, "PE")
        if any(np.isnan(v) or v <= 0 for v in (ce0, pe0, ce1, pe1)):
            continue
        # double-leg round-trip friction
        cost = round_trip_cost(ce0, ce1, qty) + round_trip_cost(pe0, pe1, qty)
        long_gross = ((ce1 + pe1) - (ce0 + pe0)) * qty
        long_net = long_gross - cost
        short_net = -long_gross - cost
        trades.append({"ts": ts, "gex": row["gex"],
                       "spot_minus_zg": abs(row.get("spot_minus_zg", 0)),
                       "long_net": long_net, "short_net": short_net,
                       "straddle_cost": (ce0 + pe0) * qty})
    t = pd.DataFrame(trades)
    if t.empty:
        print("No straddle trades constructed."); return
    print(f"  straddle trades: {len(t)} | avg entry straddle premium "
          f"Rs.{(t['straddle_cost']).mean():,.0f}\n")

    gex_med = t["gex"].median()
    # GEX-conditioned: low/neg GEX -> expect big move -> LONG; high GEX -> SHORT
    cond = np.where(t["gex"] < gex_med, t["long_net"], t["short_net"])

    def rep(name, x):
        x = np.asarray(x)
        print(f"  {name:<26}{len(x):>5}  WR {((x>0).mean()*100):>3.0f}%  "
              f"PF {pf(x):>5.2f}  avg Rs.{x.mean():>+8.0f}  net Rs.{x.sum():>+10,.0f}")

    print("=" * 84)
    print("  VOL-STRATEGY P&L (ATM straddle, real prices, post double-leg friction)")
    print("=" * 84)
    print(f"  {'strategy':<26}{'n':>5}  {'win':>4}  {'PF':>7}  {'avg':>12}  {'net':>13}")
    print("  " + "-" * 80)
    rep("unconditional LONG", t["long_net"])
    rep("unconditional SHORT", t["short_net"])
    rep("GEX-conditioned", cond)
    print("=" * 84)
    best_pf = pf(cond)
    print(f"  PRELIMINARY VERDICT: {'VOL SIGNAL — justify a vol research track' if best_pf > 1.1 else 'NO VOL EDGE'} "
          f"(GEX-conditioned PF {best_pf:.2f}).")
    print(f"  NOTE: ~13 days single regime ({t['ts'].min().date()}..{t['ts'].max().date()}); "
          f"HOLD={HOLD_MIN}m. Preliminary, not the full multi-regime gate.")
    t.to_csv(ROOT / "logs/vol_straddle_kaggle.csv", index=False)


if __name__ == "__main__":
    main()
