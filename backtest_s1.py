"""
backtest_s1.py — Backtest S1 PSAR (dual-timeframe 5m+15m alignment).
OPEN_SETTLE=10:00 (was 09:20). SL=60, TP=120 (1:2 R:R).
Trail-after-TP: 20pts behind peak once TP reached.
EMA40 side filter, flat-day PUT filter, flip-only entry.
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
    atr = tr.rolling(period).mean()
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def compute_psar(high, low, close, af_start=0.03, af_step=0.02, af_max=0.2):
    n = len(high)
    sar = np.full(n, np.nan)
    direction = np.zeros(n)
    flips = np.zeros(n)

    if n < 2:
        return sar, direction, flips

    uptrend = close[1] > close[0]
    if uptrend:
        ep = high[1]
        prev_sar = low[0]
        prev_ep = high[1]
    else:
        ep = low[1]
        prev_sar = high[0]
        prev_ep = low[1]

    af = af_start
    sar[1] = prev_sar + af_start * (prev_ep - prev_sar)
    direction[0] = 1 if uptrend else -1
    direction[1] = 1 if uptrend else -1
    flips[1] = 1 if uptrend else -1
    next_bar_sar = sar[1] + af * (ep - sar[1])

    for i in range(2, n):
        first_trend_bar = False
        current_sar = next_bar_sar

        bear_flip = uptrend and current_sar > low[i]
        bull_flip = (not uptrend) and current_sar < high[i]

        if bear_flip:
            first_trend_bar = True
            uptrend = False
            current_sar = max(ep, high[i])
            ep = low[i]
            af = af_start
            flips[i] = -1
        elif bull_flip:
            first_trend_bar = True
            uptrend = True
            current_sar = min(ep, low[i])
            ep = high[i]
            af = af_start
            flips[i] = 1

        if not first_trend_bar:
            if uptrend and high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            elif not uptrend and low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)

        if uptrend:
            current_sar = min(current_sar, low[i - 1])
            if i >= 2:
                current_sar = min(current_sar, low[i - 2])
        else:
            current_sar = max(current_sar, high[i - 1])
            if i >= 2:
                current_sar = max(current_sar, high[i - 2])

        sar[i] = current_sar
        direction[i] = 1 if uptrend else -1
        next_bar_sar = current_sar + af * (ep - current_sar)

    return sar, direction, flips


def resample_15m(df5):
    df15 = df5.resample("15min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    return df15


SL = 60
TP = 120
OPEN_SETTLE = 1000
LUNCH_START = 1200
LUNCH_END = 1330
FLIP_MAX_BARS = 1
FLAT_THRESHOLD = 40
EMA_PERIOD = 40
TRAIL_AFTER_TP_STEP = 20
LOT_SIZE = 65


def adx_adaptive_af(adx_val):
    if adx_val is not None and not np.isnan(adx_val):
        if adx_val > 30:
            return 0.03, 0.025, 0.3
        elif adx_val > 20:
            return 0.045, 0.03, 0.3
        else:
            return 0.02, 0.015, 0.15
    return 0.03, 0.02, 0.2


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


def run_backtest(df5):
    df15 = resample_15m(df5)

    adx = _compute_adx(df5)
    ema40 = df5["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    close = df5["close"].values
    high = df5["high"].values
    low = df5["low"].values
    opn = df5["open"].values

    trades = []
    position = None
    min_idx = 60
    last_date = None
    day_open = None

    for i in range(min_idx, len(df5)):
        ts = df5.index[i]
        spot = float(close[i])
        bar_high = float(high[i])
        bar_low = float(low[i])
        bar_open = float(opn[i])
        today = ts.date()
        hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 1000

        if today != last_date:
            last_date = today
            today_bars = df5[df5.index.date == today]
            day_open = float(today_bars.iloc[0]["open"]) if not today_bars.empty else spot

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

            if hm >= 1515:
                eod = True
            if hit_sl or hit_tp or eod:
                reason = "TRAIL_TP" if trailing else ("SL" if hit_sl else ("TP" if hit_tp else "EOD"))
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "dir": d, "entry_nifty": entry, "exit_nifty": spot,
                    "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
                    "exit_reason": reason,
                    "hour": position["entry_time"].hour if hasattr(position["entry_time"], 'hour') else 0,
                })
                position = None
            if position is not None:
                continue

        # ── Time filters ──
        if hm >= 1500:
            continue
        if hm < OPEN_SETTLE:
            continue
        if LUNCH_START <= hm <= LUNCH_END:
            continue

        # ── Compute PSAR signals ──
        lookback_5m = df5.iloc[max(0, i - 100):i + 1]
        if len(lookback_5m) < 20:
            continue

        # ADX-adaptive AF
        adx_val = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else None
        af_start, af_step, af_max = adx_adaptive_af(adx_val)

        # 5m PSAR
        sar5, dir5, flip5 = compute_psar(
            lookback_5m["high"].values, lookback_5m["low"].values,
            lookback_5m["close"].values, af_start, af_step, af_max
        )
        d5 = int(dir5[-2])  # confirmed bar
        is_flip_5m = int(flip5[-2]) != 0

        # Count bars since flip
        bars_since_flip = 0
        for j in range(len(dir5) - 3, -1, -1):
            if int(dir5[j]) != d5:
                break
            bars_since_flip += 1

        if bars_since_flip > FLIP_MAX_BARS:
            continue

        # 15m PSAR
        ts_15m = df15.index[df15.index <= ts]
        if len(ts_15m) < 10:
            continue
        lookback_15m = df15.loc[ts_15m[-50:]]
        sar15, dir15, flip15 = compute_psar(
            lookback_15m["high"].values, lookback_15m["low"].values,
            lookback_15m["close"].values, af_start, af_step, af_max
        )
        d15 = int(dir15[-2]) if len(dir15) > 1 else 0

        both_bull = d5 == 1 and d15 == 1
        both_bear = d5 == -1 and d15 == -1

        if not (both_bull or both_bear):
            continue

        direction = None
        if both_bull:
            direction = "CALL"
        elif both_bear:
            if abs(spot - day_open) <= FLAT_THRESHOLD:
                continue
            direction = "PUT"

        if direction is None:
            continue

        position = {"dir": direction, "entry_nifty": spot, "entry_time": ts}

    if position is not None:
        spot = float(close[-1])
        d = position["dir"]
        pnl_pts = (position["entry_nifty"] - spot) if d == "PUT" else (spot - position["entry_nifty"])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": df5.index[-1],
            "dir": d, "entry_nifty": position["entry_nifty"], "exit_nifty": spot,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
            "exit_reason": "DATA_END", "hour": 0,
        })

    return trades


def print_results(trades, label):
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")

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

    trades = run_backtest(df)
    print_results(trades, "S1 PSAR — 5m+15m Align SL=60 TP=120 Trail=20 OPEN=10:00")


if __name__ == "__main__":
    main()
