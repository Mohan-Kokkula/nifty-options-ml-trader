"""
backtest_s11.py — Backtest S11 Market Energy Score.
7-dimension scoring: Volume + Force Index + ATR expansion + RSI momentum
+ ADX trending + Structure breakout(+2) + VWAP position.
Score >= 6 to trade. Fixed SL=35, TP=70 (2:1 R:R).
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


def _compute_vwap_daily(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"] if "volume" in df.columns else pd.Series(1000000, index=df.index)
    tp_vol = typical * vol
    vwap = pd.Series(np.nan, index=df.index)
    dates = df.index.date
    for d in sorted(set(dates)):
        mask = dates == d
        cv = vol[mask].cumsum()
        ctv = tp_vol[mask].cumsum()
        vwap[mask] = ctv / cv.replace(0, np.nan)
    return vwap


MIN_SCORE = 6
SWING_LOOKBACK = 20
SL = 35
TP = 70
COOLDOWN_BARS = 5
MAX_PER_DIR_DAY = 1
MAX_PER_DAY = 2
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


def compute_scores(df):
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values if "volume" in df.columns else np.full(n, 1000000.0)

    adx = _compute_adx(df).values
    atr = _compute_atr(df).values
    atr_sma = pd.Series(atr).rolling(20).mean().values
    rsi = _compute_rsi(df["close"]).values
    rsi_prev = np.roll(rsi, 1); rsi_prev[0] = np.nan
    vwap = _compute_vwap_daily(df).values
    vol_avg = pd.Series(vol).rolling(20).mean().values

    swing_high = df["high"].rolling(SWING_LOOKBACK).max().values
    swing_low = df["low"].rolling(SWING_LOOKBACK).min().values
    prev_sh = np.roll(swing_high, 1); prev_sh[0] = np.nan
    prev_sl = np.roll(swing_low, 1); prev_sl[0] = np.nan

    close_change = np.diff(close, prepend=close[0])
    force_index = close_change * vol

    bull = np.zeros(n)
    bear = np.zeros(n)

    for i in range(60, n):
        bs = brs = 0

        if not np.isnan(vol_avg[i]) and vol_avg[i] > 0:
            if vol[i] > 1.5 * vol_avg[i]:
                bs += 1; brs += 1

        if force_index[i] > 0:
            bs += 1
        elif force_index[i] < 0:
            brs += 1

        if not np.isnan(atr[i]) and not np.isnan(atr_sma[i]) and atr_sma[i] > 0:
            if atr[i] > atr_sma[i]:
                bs += 1; brs += 1

        if not np.isnan(rsi[i]) and not np.isnan(rsi_prev[i]):
            if rsi[i] > 50 and rsi[i] > rsi_prev[i]:
                bs += 1
            elif rsi[i] < 50 and rsi[i] < rsi_prev[i]:
                brs += 1

        if not np.isnan(adx[i]) and adx[i] > 20:
            bs += 1; brs += 1

        if not np.isnan(prev_sh[i]) and close[i] > prev_sh[i]:
            bs += 2
        if not np.isnan(prev_sl[i]) and close[i] < prev_sl[i]:
            brs += 2

        if not np.isnan(vwap[i]):
            if close[i] > vwap[i]:
                bs += 1
            elif close[i] < vwap[i]:
                brs += 1

        bull[i] = bs
        bear[i] = brs

    return bull, bear


def run_backtest(df, bull_scores, bear_scores):
    trades = []
    position = None
    min_idx = 60
    last_sl_bar = {"CALL": -999, "PUT": -999}
    daily_dir_count = {}
    daily_total = 0
    last_date = None

    for i in range(min_idx, len(df)):
        bar = df.iloc[i]
        ts = df.index[i]
        spot = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        today = ts.date()

        if today != last_date:
            daily_dir_count = {"CALL": 0, "PUT": 0}
            daily_total = 0
            last_date = today

        # ── Exit ──
        if position is not None:
            hit_sl = hit_tp = eod = False
            if position["dir"] == "PUT":
                if bar_high >= position["entry_nifty"] + SL:
                    hit_sl, pnl_pts = True, -SL
                elif bar_low <= position["entry_nifty"] - TP:
                    hit_tp, pnl_pts = True, TP
                else:
                    pnl_pts = position["entry_nifty"] - spot
            else:
                if bar_low <= position["entry_nifty"] - SL:
                    hit_sl, pnl_pts = True, -SL
                elif bar_high >= position["entry_nifty"] + TP:
                    hit_tp, pnl_pts = True, TP
                else:
                    pnl_pts = spot - position["entry_nifty"]

            hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 0
            if hm >= 1515:
                eod = True

            if hit_sl or hit_tp or eod:
                if hit_sl:
                    last_sl_bar[position["dir"]] = i
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "dir": position["dir"], "entry_nifty": position["entry_nifty"],
                    "exit_nifty": spot,
                    "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "EOD"),
                    "hour": position["entry_time"].hour if hasattr(position["entry_time"], 'hour') else 0,
                    "score": position["score"],
                })
                position = None
            if position is not None:
                continue

        # ── Time ──
        hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 1000
        if hm >= 1500:
            continue

        # ── Score ──
        bs = bull_scores[i]
        brs = bear_scores[i]

        direction = None
        score = 0
        if bs >= MIN_SCORE and bs > brs:
            direction = "CALL"
            score = bs
        elif brs >= MIN_SCORE and brs > bs:
            direction = "PUT"
            score = brs

        if direction is None:
            continue

        # ── Cooldown ──
        if (i - last_sl_bar[direction]) < COOLDOWN_BARS:
            continue

        # ── Max per direction per day ──
        if daily_dir_count[direction] >= MAX_PER_DIR_DAY:
            continue
        if daily_total >= MAX_PER_DAY:
            continue

        daily_dir_count[direction] += 1
        daily_total += 1
        position = {
            "dir": direction, "entry_nifty": spot,
            "entry_time": ts, "score": score,
        }

    if position is not None:
        spot = float(df["close"].iloc[-1])
        pnl_pts = (position["entry_nifty"] - spot) if position["dir"] == "PUT" else (spot - position["entry_nifty"])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": df.index[-1],
            "dir": position["dir"], "entry_nifty": position["entry_nifty"],
            "exit_nifty": spot,
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
            "exit_reason": "DATA_END", "hour": 0, "score": position.get("score", 0),
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

    print(f"  Trades     : {total}")
    print(f"  Winners    : {len(winners)} ({wr:.1f}%)")
    print(f"  Losers     : {len(losers)}")
    print(f"  PF         : {pf:.2f}")
    print(f"  Total PnL  : {total_pnl:+.1f} pts | Rs.{total_rs:+,.0f}")
    print(f"  Avg Win    : {avg_w:+.1f} pts")
    print(f"  Avg Loss   : {avg_l:+.1f} pts")
    print(f"  Max DD     : {max_dd:.1f} pts")

    print(f"\n  Exit breakdown:")
    for reason, grp in tdf.groupby("exit_reason"):
        print(f"    {reason:6s}: {len(grp):3d} trades, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Direction:")
    for d, grp in tdf.groupby("dir"):
        dwr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    {d:5s}: {len(grp):3d} trades, WR={dwr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Score breakdown:")
    for s, grp in tdf.groupby("score"):
        swr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    Score {s:.0f}: {len(grp):3d} trades, WR={swr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Hourly:")
    for h, grp in tdf.groupby("hour"):
        if h == 0:
            continue
        hwr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    {h:02d}:xx : {len(grp):3d} trades, WR={hwr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Trade log:")
    print(f"  {'Entry':20s} {'Dir':5s} {'Entry':>8s} {'Exit':>8s} {'PnL':>8s} {'Rs':>10s} {'Exit':8s} {'Sc':>3s}")
    for _, t in tdf.iterrows():
        print(f"  {str(t['entry_time'])[:16]:20s} {t['dir']:5s} {t['entry_nifty']:8.1f} {t['exit_nifty']:8.1f} "
              f"{t['pnl_pts']:+8.1f} {t['pnl_rs']:+10,.0f} {t['exit_reason']:8s} {t['score']:3.0f}")


def main():
    df = fetch_data()
    dates = sorted(set(df.index.date))
    print(f"Trading days: {len(dates)} | {dates[0]} to {dates[-1]}")

    bull_scores, bear_scores = compute_scores(df)
    trades = run_backtest(df, bull_scores, bear_scores)
    print_results(trades, "S11 Market Energy Score — Swing=20 Score>=6 SL=35 TP=70")


if __name__ == "__main__":
    main()
