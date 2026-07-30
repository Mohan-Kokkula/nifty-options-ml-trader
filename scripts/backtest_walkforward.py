"""
backtest_walkforward.py — Walk-forward validation across multiple trading days.

Reuses the same V9.3 gate logic as backtest_today.py but loops over a list of
dates (default: last N Tuesday expiries) and aggregates results.

Usage:
    python scripts/backtest_walkforward.py
    python scripts/backtest_walkforward.py 2026-03-03 2026-03-10 2026-03-17
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import date, datetime

from scripts.train_model_v8 import (
    features_5min, features_15min, features_30min, features_60min,
    features_daily, features_vix, load_vix,
    merge_htf, merge_daily_onto_5min, merge_vix_onto_5min,
    add_intraday_context,
)
from core.ml_engine import predict_precomputed, load_model

# ── Build full feature frame ONCE (huge speedup) ──────────────────
def _load(p):
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()

print("Loading data + building feature frame (one-time)...")
df5  = _load("data/nifty_5min.csv")
df15 = _load("data/nifty_15min.csv")
df30 = _load("data/nifty_30min.csv")
df60 = _load("data/nifty_60min.csv")
dfd  = _load("data/nifty_day.csv")

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
    feat_vix = features_vix(load_vix("data/india_vix.csv"))
    df_feat = merge_vix_onto_5min(df_feat, feat_vix)
except Exception as e:
    print(f"  VIX merge skipped: {e}")
df_feat = add_intraday_context(df_feat)

load_model()
print(f"Feature frame: {len(df_feat):,} rows × {len(df_feat.columns)} cols\n")


def backtest_one_day(target: date) -> dict:
    today_bars = df_feat[df_feat.index.date == target]
    today_raw  = df5[df5.index.date == target]
    if len(today_bars) == 0 or len(today_raw) == 0:
        return None

    # VWAP
    cum_pv = 0; cum_vol = 0; vwap_series = {}
    for ts, row in today_raw.iterrows():
        typical = (row["high"] + row["low"] + row["close"]) / 3
        cum_pv += typical * row["volume"]; cum_vol += row["volume"]
        vwap_series[ts] = cum_pv / cum_vol if cum_vol else row["close"]

    results = []
    prev_signal = 2
    for ts in today_bars.index:
        hm = ts.hour * 100 + ts.minute
        if hm < 930 or hm >= 1500:
            continue
        sub = df_feat[df_feat.index <= ts]
        try:
            signal, proba, conf, _ = predict_precomputed(sub)
        except Exception:
            continue
        if ts not in today_raw.index:
            continue
        spot = float(today_raw.loc[ts]["close"])
        direction = "CALL" if signal == 0 else ("PUT" if signal == 1 else "SKIP")
        if signal == 2:
            prev_signal = signal; continue
        if conf * 100 < 60:
            prev_signal = signal
            results.append({"outcome": "SKIP_LOW_CONF", "pnl": 0}); continue
        if hm < 945 and conf * 100 < 75:
            prev_signal = signal
            results.append({"outcome": "SKIP_MORNING_TRAP", "pnl": 0}); continue
        if signal != prev_signal:
            prev_signal = signal
            results.append({"outcome": "SKIP_2BAR", "pnl": 0}); continue
        vwap = vwap_series.get(ts, spot)
        if direction == "CALL" and spot < vwap - 5:
            prev_signal = signal
            results.append({"outcome": "SKIP_VWAP", "pnl": 0}); continue
        if direction == "PUT" and spot > vwap + 5:
            prev_signal = signal
            results.append({"outcome": "SKIP_VWAP", "pnl": 0}); continue
        if hm >= 1430:
            prev_signal = signal
            results.append({"outcome": "SKIP_EXPIRY_CUTOFF", "pnl": 0}); continue

        # CHOP-DAY GATE: after 11:00, if day range is < 60 pts AND
        # 15m ADX < 20 → no trend → block all signals
        if hm >= 1100:
            day_so_far = today_raw[today_raw.index <= ts]
            day_range  = float(day_so_far["high"].max() - day_so_far["low"].min())
            row = today_bars.loc[ts] if ts in today_bars.index else None
            adx15 = float(row.get("tf15_adx", 25)) if row is not None else 25
            if day_range < 60 and adx15 < 20:
                prev_signal = signal
                results.append({"outcome": "SKIP_CHOP_DAY", "pnl": 0}); continue

        # Execute
        sl_pts, tp_pts = 60, 64
        if direction == "CALL":
            sl_price, tp_price = spot - sl_pts, spot + tp_pts
        else:
            sl_price, tp_price = spot + sl_pts, spot - tp_pts
        future = today_raw[today_raw.index > ts]
        outcome, pnl = "TIME_EXIT", 0
        for fts, fbar in future.iterrows():
            fh, fl = float(fbar["high"]), float(fbar["low"])
            if direction == "CALL":
                # Pessimistic: SL checked first
                if fl <= sl_price: outcome="SL"; pnl=-sl_pts; break
                if fh >= tp_price: outcome="TP"; pnl=tp_pts; break
            else:
                if fh >= sl_price: outcome="SL"; pnl=-sl_pts; break
                if fl <= tp_price: outcome="TP"; pnl=tp_pts; break
            if fts.hour == 15 and fts.minute >= 20:
                exit_p = float(fbar["close"])
                pnl = (exit_p - spot) if direction == "CALL" else (spot - exit_p)
                break
        # Apply realistic haircuts: 0.55 delta + 4pts slippage
        pnl_real = pnl * 0.55 - (4 if outcome != "TIME_EXIT" else 2)
        results.append({"outcome": outcome, "pnl": pnl, "pnl_real": pnl_real,
                        "direction": direction})
        prev_signal = signal

    executed = [r for r in results if "pnl_real" in r]
    skipped  = [r for r in results if "pnl_real" not in r]
    wins = [r for r in executed if r["pnl_real"] > 0]
    losses = [r for r in executed if r["pnl_real"] < 0]
    total_raw  = sum(r["pnl"] for r in executed)
    total_real = sum(r["pnl_real"] for r in executed)
    day_raw = today_raw
    day_move = float(day_raw["close"].iloc[-1] - day_raw["open"].iloc[0])

    return {
        "date": target.isoformat(),
        "signals": len(results),
        "executed": len(executed),
        "skipped": len(skipped),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(executed) * 100) if executed else 0.0,
        "pnl_raw_pts": total_raw,
        "pnl_real_pts": total_real,
        "pnl_real_rs": total_real * 65,
        "day_move": day_move,
    }


# ── Pick dates ────────────────────────────────────────────────────
if len(sys.argv) > 1:
    test_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in sys.argv[1:]]
else:
    # Default: every Tuesday in Mar 2026 + first 2 of Apr 2026
    test_dates = [
        date(2026, 3, 3),  date(2026, 3, 10), date(2026, 3, 17),
        date(2026, 3, 24), date(2026, 3, 31), date(2026, 4, 7),
        date(2026, 4, 9),
    ]

print(f"Walk-forward over {len(test_dates)} days:\n")
print(f"{'Date':12}{'Sig':6}{'Exec':6}{'W/L':8}{'WinRt':8}{'P&L raw':>10}{'P&L real':>11}{'Rs @65lot':>13}{'Day':>8}")
print("-" * 88)

agg = {"signals": 0, "executed": 0, "wins": 0, "losses": 0,
       "pnl_raw": 0.0, "pnl_real": 0.0}
profitable_days = 0
valid_days = 0
for d in test_dates:
    r = backtest_one_day(d)
    if r is None:
        print(f"{d.isoformat():12}(no data — holiday/weekend)")
        continue
    valid_days += 1
    print(f"{r['date']:12}{r['signals']:<6}{r['executed']:<6}"
          f"{r['wins']}/{r['losses']:<6}{r['win_rate']:<6.0f}% "
          f"{r['pnl_raw_pts']:>+9.0f} {r['pnl_real_pts']:>+10.0f} "
          f"{r['pnl_real_rs']:>+12,.0f} {r['day_move']:>+7.0f}")
    agg["signals"]  += r["signals"]
    agg["executed"] += r["executed"]
    agg["wins"]     += r["wins"]
    agg["losses"]   += r["losses"]
    agg["pnl_raw"]  += r["pnl_raw_pts"]
    agg["pnl_real"] += r["pnl_real_pts"]
    if r["pnl_real_pts"] > 0:
        profitable_days += 1

print("-" * 88)
total_wr = (agg["wins"] / agg["executed"] * 100) if agg["executed"] else 0
print(f"{'TOTAL':12}{agg['signals']:<6}{agg['executed']:<6}"
      f"{agg['wins']}/{agg['losses']:<6}{total_wr:<6.0f}% "
      f"{agg['pnl_raw']:>+9.0f} {agg['pnl_real']:>+10.0f} "
      f"{agg['pnl_real']*65:>+12,.0f}")

print()
print("=" * 88)
print("WALK-FORWARD VERDICT")
print("=" * 88)
print(f"Days tested              : {valid_days} (of {len(test_dates)} requested)")
print(f"Profitable days          : {profitable_days}/{valid_days}  "
      f"({profitable_days/max(valid_days,1)*100:.0f}%)")
print(f"Overall win rate         : {total_wr:.0f}%")
print(f"Avg trades/day           : {agg['executed']/len(test_dates):.1f}")
print(f"Total P&L (raw spot)     : {agg['pnl_raw']:+.0f} pts")
print(f"Total P&L (after haircut): {agg['pnl_real']:+.0f} pts  =  Rs {agg['pnl_real']*65:+,.0f} @ 65 lot")
print()
print("Pass criteria for live trading:")
print(f"  [{'PASS' if total_wr >= 60 else 'FAIL'}] Win rate >= 60%       (got {total_wr:.0f}%)")
print(f"  [{'PASS' if profitable_days >= valid_days*0.6 else 'FAIL'}] Profitable days >= 60% (got {profitable_days/max(valid_days,1)*100:.0f}%)")
print(f"  [{'PASS' if agg['pnl_real'] > 0 else 'FAIL'}] Net P&L positive       (got {agg['pnl_real']:+.0f} pts)")
