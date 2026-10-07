"""
backtest_s12.py — Backtest S12 Energy Flow (sequential AND-gate).
All 6 gates must pass: Volume → Force Index → ATR → RSI → ADX → Structure.
Fixed SL=35, TP=70 (2:1 R:R). Trail-after-TP: 20pts behind peak.
Max 1/dir/day, max 2 total/day. Cooldown 5 bars after SL.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import pandas as pd
import numpy as np


def _compute_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(period).mean()
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


SL = 35
TP = 70
SWING_LOOKBACK = 20
COOLDOWN_BARS = 5
MAX_PER_DIR_DAY = 1
MAX_PER_DAY = 2
TRAIL_AFTER_TP_STEP = 20
LOT_SIZE = 65


def fetch_data():
    print("Fetching 5m Nifty data from TradingView...")
    from core.tv_fetcher import TVFetcher
    tv = TVFetcher()
    df = tv.get_ohlcv("NIFTY", "NSE", interval_minutes=5, n_bars=5000)
    if df is None or df.empty:
        print("ERROR: No data")
        sys.exit(1)
    if "volume" not in df.columns:
        df["volume"] = 1000000
    print(f"Got {len(df)} bars: {df.index[0]} to {df.index[-1]}")
    return df


def run_backtest(df):
    adx = _compute_adx(df)
    atr = _compute_atr(df)
    atr_sma = atr.rolling(20).mean()
    rsi = _compute_rsi(df["close"])

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    opn = df["open"].values
    vol = df["volume"].values if "volume" in df.columns else np.full(len(df), 1000000.0)
    vol_avg = pd.Series(vol).rolling(20).mean().values

    swing_high = df["high"].rolling(SWING_LOOKBACK).max().values
    swing_low = df["low"].rolling(SWING_LOOKBACK).min().values
    prev_sh = np.roll(swing_high, 1); prev_sh[0] = np.nan
    prev_sl = np.roll(swing_low, 1); prev_sl[0] = np.nan

    close_change = np.diff(close, prepend=close[0])
    force_index = close_change * vol

    adx_v = adx.values
    atr_v = atr.values
    atr_sma_v = atr_sma.values
    rsi_v = rsi.values
    rsi_prev = np.roll(rsi_v, 1); rsi_prev[0] = np.nan

    trades = []
    position = None
    min_idx = 60
    last_sl_bar = {"CALL": -999, "PUT": -999}
    daily_dir_count = {}
    daily_total = 0
    last_date = None
    gate_stats = {"g1": 0, "g2": 0, "g3": 0, "g4": 0, "g5": 0, "g6": 0, "signal": 0, "bars": 0}

    for i in range(min_idx, len(df)):
        ts = df.index[i]
        spot = float(close[i])
        bar_high = float(high[i])
        bar_low = float(low[i])
        bar_open = float(opn[i])
        today = ts.date()

        if today != last_date:
            daily_dir_count = {"CALL": 0, "PUT": 0}
            daily_total = 0
            last_date = today

        # ── Exit ──
        if position is not None:
            entry = position["entry_nifty"]
            d = position["dir"]
            trailing = position.get("trailing", False)
            peak = position.get("peak", 0.0)

            unrealized = (entry - spot) if d == "PUT" else (spot - entry)
            hit_sl = hit_tp = eod = False

            if trailing:
                if d == "CALL":
                    peak = max(peak, bar_high)
                else:
                    peak = min(peak, bar_low)
                position["peak"] = peak
                trail_sl = peak - TRAIL_AFTER_TP_STEP if d == "CALL" else peak + TRAIL_AFTER_TP_STEP
                if d == "CALL" and bar_low <= trail_sl:
                    hit_sl = True
                    pnl_pts = trail_sl - entry
                elif d == "PUT" and bar_high >= trail_sl:
                    hit_sl = True
                    pnl_pts = entry - trail_sl
                else:
                    pnl_pts = unrealized
            else:
                if d == "PUT":
                    if bar_high >= entry + SL:
                        hit_sl, pnl_pts = True, -SL
                    elif bar_low <= entry - TP:
                        hit_tp, pnl_pts = True, TP
                    else:
                        pnl_pts = unrealized
                else:
                    if bar_low <= entry - SL:
                        hit_sl, pnl_pts = True, -SL
                    elif bar_high >= entry + TP:
                        hit_tp, pnl_pts = True, TP
                    else:
                        pnl_pts = unrealized

            if hit_tp:
                momentum_ok = (spot >= bar_open) if d == "CALL" else (spot <= bar_open)
                if momentum_ok:
                    hit_tp = False
                    position["trailing"] = True
                    position["peak"] = bar_high if d == "CALL" else bar_low
                    pnl_pts = unrealized

            hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 0
            if hm >= 1515:
                eod = True
            if hit_sl or hit_tp or eod:
                if hit_sl and not trailing:
                    last_sl_bar[position["dir"]] = i
                reason = "TRAIL_TP" if trailing else ("SL" if hit_sl else ("TP" if hit_tp else "EOD"))
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "dir": d, "entry_nifty": entry, "exit_nifty": spot,
                    "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
                    "exit_reason": reason,
                    "hour": position["entry_time"].hour if hasattr(position["entry_time"], 'hour') else 0,
                    "failed_at": "none",
                })
                position = None
            if position is not None:
                continue

        # ── Time ──
        hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 1000
        if hm >= 1500:
            continue

        gate_stats["bars"] += 1

        # ── Gate 1: VOLUME ──
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 0 or vol[i] <= 1.5 * vol_avg[i]:
            continue
        gate_stats["g1"] += 1

        # ── Gate 2: FORCE INDEX ──
        fi = force_index[i]
        if fi == 0:
            continue
        bull_force = fi > 0
        bear_force = fi < 0
        gate_stats["g2"] += 1

        # ── Gate 3: ATR expansion ──
        if np.isnan(atr_v[i]) or np.isnan(atr_sma_v[i]) or atr_sma_v[i] <= 0:
            continue
        if atr_v[i] <= atr_sma_v[i]:
            continue
        gate_stats["g3"] += 1

        # ── Gate 4: RSI momentum ──
        if np.isnan(rsi_v[i]) or np.isnan(rsi_prev[i]):
            continue
        bull_rsi = rsi_v[i] > 50 and rsi_v[i] > rsi_prev[i]
        bear_rsi = rsi_v[i] < 50 and rsi_v[i] < rsi_prev[i]
        if not (bull_rsi or bear_rsi):
            continue
        gate_stats["g4"] += 1

        # ── Gate 5: ADX trending ──
        if np.isnan(adx_v[i]) or adx_v[i] <= 20:
            continue
        gate_stats["g5"] += 1

        # ── Gate 6: Structure breakout ──
        if np.isnan(prev_sh[i]) or np.isnan(prev_sl[i]):
            continue
        bull_breakout = close[i] > prev_sh[i]
        bear_breakout = close[i] < prev_sl[i]

        direction = None
        if bull_force and bull_rsi and bull_breakout:
            direction = "CALL"
        elif bear_force and bear_rsi and bear_breakout:
            direction = "PUT"

        if direction is None:
            continue
        gate_stats["g6"] += 1

        # ── Cooldown ──
        if (i - last_sl_bar[direction]) < COOLDOWN_BARS:
            continue

        # ── Max per direction per day ──
        if daily_dir_count[direction] >= MAX_PER_DIR_DAY:
            continue
        if daily_total >= MAX_PER_DAY:
            continue

        gate_stats["signal"] += 1
        daily_dir_count[direction] += 1
        daily_total += 1
        position = {"dir": direction, "entry_nifty": spot, "entry_time": ts}

    if position is not None:
        spot = float(close[-1])
        d = position["dir"]
        pnl_pts = (position["entry_nifty"] - spot) if d == "PUT" else (spot - position["entry_nifty"])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": df.index[-1],
            "dir": d, "entry_nifty": position["entry_nifty"], "exit_nifty": spot,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
            "exit_reason": "DATA_END", "hour": 0, "failed_at": "none",
        })

    return trades, gate_stats


def print_results(trades, gate_stats, label):
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")

    print(f"\n  Gate funnel (bars evaluated: {gate_stats['bars']}):")
    print(f"    G1 Volume    : {gate_stats['g1']:5d} passed")
    print(f"    G2 Force Idx : {gate_stats['g2']:5d} passed")
    print(f"    G3 ATR expand: {gate_stats['g3']:5d} passed")
    print(f"    G4 RSI mom   : {gate_stats['g4']:5d} passed")
    print(f"    G5 ADX trend : {gate_stats['g5']:5d} passed")
    print(f"    G6 Structure : {gate_stats['g6']:5d} passed")
    print(f"    Signals      : {gate_stats['signal']:5d} (after cooldown/caps)")

    if not trades:
        print("  No trades generated.")
        return

    tdf = pd.DataFrame(trades)
    total = len(tdf)
    winners = tdf[tdf["pnl_pts"] > 0]
    losers = tdf[tdf["pnl_pts"] < 0]
    wr = len(winners) / total * 100
    total_pnl = tdf["pnl_pts"].sum()
    total_rs = tdf["pnl_rs"].sum()
    gp = winners["pnl_pts"].sum() if len(winners) else 0
    gl = abs(losers["pnl_pts"].sum()) if len(losers) else 1
    pf = gp / gl if gl > 0 else float('inf')
    avg_w = winners["pnl_pts"].mean() if len(winners) else 0
    avg_l = losers["pnl_pts"].mean() if len(losers) else 0
    cum = tdf["pnl_pts"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    print(f"\n  Trades     : {total}")
    print(f"  Winners    : {len(winners)} ({wr:.1f}%)")
    print(f"  Losers     : {len(losers)}")
    print(f"  PF         : {pf:.2f}")
    print(f"  Total PnL  : {total_pnl:+.1f} pts | Rs.{total_rs:+,.0f}")
    print(f"  Avg Win    : {avg_w:+.1f} pts")
    print(f"  Avg Loss   : {avg_l:+.1f} pts")
    print(f"  Max DD     : {max_dd:.1f} pts")

    print(f"\n  Exit breakdown:")
    for reason, grp in tdf.groupby("exit_reason"):
        print(f"    {reason:8s}: {len(grp):3d} trades, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Direction:")
    for d, grp in tdf.groupby("dir"):
        dwr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    {d:5s}: {len(grp):3d} trades, WR={dwr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Hourly:")
    for h, grp in tdf.groupby("hour"):
        if h == 0:
            continue
        hwr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    {h:02d}:xx : {len(grp):3d} trades, WR={hwr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Trade log:")
    print(f"  {'Entry':20s} {'Dir':5s} {'Entry':>8s} {'Exit':>8s} {'PnL':>8s} {'Rs':>10s} {'Exit':8s}")
    for _, t in tdf.iterrows():
        print(f"  {str(t['entry_time'])[:16]:20s} {t['dir']:5s} {t['entry_nifty']:8.1f} {t['exit_nifty']:8.1f} "
              f"{t['pnl_pts']:+8.1f} {t['pnl_rs']:+10,.0f} {t['exit_reason']:8s}")


def main():
    df = fetch_data()
    dates = sorted(set(df.index.date))
    print(f"Trading days: {len(dates)} | {dates[0]} to {dates[-1]}")

    trades, gate_stats = run_backtest(df)
    print_results(trades, gate_stats, "S12 Energy Flow — AND-gate SL=35 TP=70 Trail=20")


if __name__ == "__main__":
    main()
