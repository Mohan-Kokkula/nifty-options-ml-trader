"""
backtest_today.py — Backtest V9.3 logic on today's bars (FULL FEATURE PIPELINE)
================================================================================
Builds the same multi-TF feature frame the trainer uses, then calls
predict_precomputed() so live inference matches training exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import date

from scripts.train_model_v8 import (
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix, load_vix,
    merge_htf, merge_daily_onto_5min, merge_vix_onto_5min,
    add_intraday_context,
)
from core.ml_engine import predict_precomputed, load_model, is_ready

def _load(p):
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()

print("Loading data...")
df5  = _load("data/nifty_5min.csv")
df15 = _load("data/nifty_15min.csv")
df30 = _load("data/nifty_30min.csv")
df60 = _load("data/nifty_60min.csv")
dfd  = _load("data/nifty_day.csv")

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--date", default=str(date.today()), help="YYYY-MM-DD")
_args = _parser.parse_args()
today = date.fromisoformat(_args.date)
today_str = _args.date

# Cap to today (no future leak)
df5  = df5[df5.index.date  <= today]
df15 = df15[df15.index.date <= today]
df30 = df30[df30.index.date <= today]
df60 = df60[df60.index.date <= today]
dfd  = dfd[dfd.index.date  <= today]

print("Engineering features (full multi-TF pipeline)...")
feat5  = features_5min(df5.copy())
feat15 = features_15min(df15.copy())
feat30 = features_30min(df30.copy())
feat60 = features_60min(df60.copy())
featD  = features_daily(dfd.copy())

df_feat = feat5
df_feat = merge_htf(df_feat, feat15, "15m")
df_feat = merge_htf(df_feat, feat30, "30m")
df_feat = merge_htf(df_feat, feat60, "60m")
df_feat = merge_daily_onto_5min(df_feat, featD)
try:
    vix_raw = load_vix("data/india_vix.csv")
    feat_vix = features_vix(vix_raw)
    df_feat = merge_vix_onto_5min(df_feat, feat_vix)
except Exception as e:
    print(f"  VIX merge skipped: {e}")
df_feat = add_intraday_context(df_feat)

print(f"Feature frame: {len(df_feat):,} rows × {len(df_feat.columns)} cols")

load_model()
print(f"ML loaded: {is_ready()}")
print()

today_bars_full = df_feat[df_feat.index.date == today]
print(f"Today bars in feature frame: {len(today_bars_full)}")
print()

# Estimate VIX
vix = 16.0
try:
    vdf = pd.read_csv("data/india_vix.csv", parse_dates=["Date"])
    vix = float(vdf[vdf["Date"].dt.date <= today].iloc[-1].get("Close", 16.0))
except Exception:
    pass

# VWAP
today_raw = df5[df5.index.date == today]
cum_pv = 0; cum_vol = 0; vwap_series = {}
for ts, row in today_raw.iterrows():
    typical = (row["high"] + row["low"] + row["close"]) / 3
    cum_pv += typical * row["volume"]; cum_vol += row["volume"]
    vwap_series[ts] = cum_pv / cum_vol if cum_vol else row["close"]

results = []
prev_signal = 2

for ts in today_bars_full.index:
    h, m = ts.hour, ts.minute
    hm = h * 100 + m
    if hm < 930 or hm >= 1500:
        continue

    # Slice feature frame up to this ts (no leak)
    sub = df_feat[df_feat.index <= ts]
    try:
        signal, proba, conf, _ = predict_precomputed(sub)
    except Exception as e:
        continue

    raw = today_raw.loc[ts] if ts in today_raw.index else None
    if raw is None:
        continue
    spot = float(raw["close"])

    direction_str = "CALL" if signal == 0 else ("PUT" if signal == 1 else "SKIP")
    if signal == 2:
        prev_signal = signal
        continue

    # VIX confidence adjustment (matches live bot vix_regime.py)
    if vix >= 30:
        conf = max(0, conf - 0.20)
    elif vix >= 25:
        conf = max(0, conf - 0.10)
    elif vix >= 20:
        conf = max(0, conf - 0.05)
    elif vix >= 16:
        conf = max(0, conf - 0.03)
    elif vix < 12:
        conf = min(1, conf + 0.05)

    # Momentum override inputs — compute sequential closes + VWAP distance
    # at this bar from raw 5-min data (matches live claude_pilot logic).
    vwap_now = vwap_series.get(ts, spot)
    pa_vwap_distance = spot - vwap_now
    # Sequential 5-min closes in same direction (last up to 10 bars)
    prior = today_raw[today_raw.index <= ts].tail(10)
    closes_arr = prior["close"].astype(float).values
    pa_sequential_closes = 0
    if len(closes_arr) >= 2:
        last_dir = 1 if closes_arr[-1] > closes_arr[-2] else -1
        pa_sequential_closes = 1
        for i in range(len(closes_arr) - 2, 0, -1):
            bar_dir = 1 if closes_arr[i] > closes_arr[i - 1] else -1
            if bar_dir == last_dir:
                pa_sequential_closes += 1
            else:
                break
        pa_sequential_closes *= last_dir

    # Apply momentum override: drop threshold 60→55 when structure confirms direction.
    min_conf_pct = 60
    if signal == 1 and pa_vwap_distance <= -40 and pa_sequential_closes <= -3:
        min_conf_pct = 55
    elif signal == 0 and pa_vwap_distance >= +40 and pa_sequential_closes >= +3:
        min_conf_pct = 55

    # GATE: confidence ≥ threshold (55 or 60 depending on momentum)
    if conf * 100 < min_conf_pct:
        prev_signal = signal
        results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                        "conf": int(conf*100), "outcome": "SKIP_LOW_CONF", "pnl": 0})
        continue

    # GATE: morning trap 09:30-09:45 needs 75
    if hm < 945 and conf * 100 < 75:
        prev_signal = signal
        results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                        "conf": int(conf*100), "outcome": "SKIP_MORNING_TRAP", "pnl": 0})
        continue

    # GATE: 2-bar confirmation
    if signal != prev_signal:
        prev_signal = signal
        results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                        "conf": int(conf*100), "outcome": "SKIP_2BAR", "pnl": 0})
        continue

    # GATE: VWAP structure
    vwap = vwap_series.get(ts, spot)
    if direction_str == "CALL" and spot < vwap - 5:
        prev_signal = signal
        results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                        "conf": int(conf*100), "outcome": "SKIP_VWAP_BLOCK", "pnl": 0,
                        "note": f"spot {spot:.0f} < VWAP {vwap:.0f}"})
        continue
    if direction_str == "PUT" and spot > vwap + 5:
        prev_signal = signal
        results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                        "conf": int(conf*100), "outcome": "SKIP_VWAP_BLOCK", "pnl": 0,
                        "note": f"spot {spot:.0f} > VWAP {vwap:.0f}"})
        continue

    # GATE: expiry-day cutoff 14:30
    if hm >= 1430:
        prev_signal = signal
        results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                        "conf": int(conf*100), "outcome": "SKIP_EXPIRY_CUTOFF", "pnl": 0})
        continue

    # EXECUTE: simulate SL/TP
    sl_pts = 60; tp_pts = 64
    if direction_str == "CALL":
        sl_price = spot - sl_pts; tp_price = spot + tp_pts
    else:
        sl_price = spot + sl_pts; tp_price = spot - tp_pts

    future = today_raw[today_raw.index > ts]
    outcome = "TIME_EXIT"; pnl = 0; exit_price = spot; exit_time = ts
    for fts, fbar in future.iterrows():
        fh, fl = float(fbar["high"]), float(fbar["low"])
        if direction_str == "CALL":
            if fl <= sl_price:
                outcome="SL"; pnl=-sl_pts; exit_price=sl_price; exit_time=fts; break
            if fh >= tp_price:
                outcome="TP"; pnl=tp_pts; exit_price=tp_price; exit_time=fts; break
        else:
            if fh >= sl_price:
                outcome="SL"; pnl=-sl_pts; exit_price=sl_price; exit_time=fts; break
            if fl <= tp_price:
                outcome="TP"; pnl=tp_pts; exit_price=tp_price; exit_time=fts; break
        if fts.hour == 15 and fts.minute >= 20:
            exit_price = float(fbar["close"])
            pnl = (exit_price - spot) if direction_str == "CALL" else (spot - exit_price)
            exit_time = fts; break

    results.append({"time": ts.strftime("%H:%M"), "spot": spot, "ml": direction_str,
                    "conf": int(conf*100), "outcome": outcome, "pnl": round(pnl, 1),
                    "exit_time": exit_time.strftime("%H:%M"),
                    "exit_price": round(exit_price, 1)})
    prev_signal = signal

# REPORT
print("=" * 88)
print(f"BACKTEST RESULTS  {today_str} (NIFTY EXPIRY DAY)")
print("=" * 88)
print(f"{'Time':6}{'Spot':10}{'ML':6}{'Conf':6}{'Outcome':22}{'P&L':>8}  Note")
print("-" * 88)

executed = []; skipped = []
for r in results:
    note = r.get("note", "")
    if "exit_time" in r:
        note = f"-> {r['exit_time']} @ {r['exit_price']}"
        executed.append(r)
    else:
        skipped.append(r)
    print(f"{r['time']:6}{r['spot']:<10.1f}{r['ml']:6}{r['conf']:<6}{r['outcome']:22}{r['pnl']:>+8.1f}  {note}")

print()
print("=" * 88)
print("SUMMARY")
print("=" * 88)
print(f"Total ML signals (CALL/PUT) : {len(results)}")
print(f"Skipped by gates            : {len(skipped)}")
print(f"Executed                    : {len(executed)}")
if executed:
    wins = [r for r in executed if r["pnl"] > 0]
    losses = [r for r in executed if r["pnl"] < 0]
    total = sum(r["pnl"] for r in executed)
    print(f"  Wins / Losses             : {len(wins)} / {len(losses)}")
    print(f"  Win rate                  : {len(wins)/len(executed)*100:.0f}%")
    print(f"  Total P&L                 : {total:+.0f} pts  (Rs {total*65:+,.0f} @ 65 lot)")
else:
    print("No trades executed.")

print()
print("Skip reasons:")
sc = {}
for r in skipped: sc[r["outcome"]] = sc.get(r["outcome"], 0) + 1
for k, v in sorted(sc.items(), key=lambda x: -x[1]):
    print(f"  {k:25}: {v}")

print()
print(f"Day range: low={today_raw['low'].min():.0f} high={today_raw['high'].max():.0f} "
      f"open={today_raw['open'].iloc[0]:.0f} close={today_raw['close'].iloc[-1]:.0f}")
print(f"Day move : {today_raw['close'].iloc[-1] - today_raw['open'].iloc[0]:+.0f} pts")
