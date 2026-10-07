"""
backtest_s9.py — Backtest S9 Range Reversal (ICT edition).
Turtle Soup + RSI(2) exhaustion + ICT concepts:
  Displacement, FVG, Order Blocks, MSS, PDH/PDL, Premium/Discount.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import pandas as pd
import numpy as np


def _compute_rsi(series, period=2):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def _compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _find_swing_highs(highs, lows, left=3, right=1):
    swings = []
    for i in range(left, len(highs) - right):
        is_sw = all(highs[i] > highs[i - j] for j in range(1, left + 1))
        if is_sw:
            is_sw = all(highs[i] > highs[i + j] for j in range(1, right + 1) if i + j < len(highs))
        if is_sw:
            swings.append((i, float(highs[i])))
    return swings


def _find_swing_lows(highs, lows, left=3, right=1):
    swings = []
    for i in range(left, len(lows) - right):
        is_sw = all(lows[i] < lows[i - j] for j in range(1, left + 1))
        if is_sw:
            is_sw = all(lows[i] < lows[i + j] for j in range(1, right + 1) if i + j < len(lows))
        if is_sw:
            swings.append((i, float(lows[i])))
    return swings


def _detect_fvg_bearish(df, idx):
    if idx < 2:
        return None
    c1_low = float(df["low"].iloc[idx - 2])
    c3_high = float(df["high"].iloc[idx])
    if c1_low > c3_high:
        return True
    return None


def _detect_fvg_bullish(df, idx):
    if idx < 2:
        return None
    c1_high = float(df["high"].iloc[idx - 2])
    c3_low = float(df["low"].iloc[idx])
    if c3_low > c1_high:
        return True
    return None


def _find_bearish_ob(df, end_idx, lookback=5):
    for i in range(end_idx - 1, max(end_idx - lookback, 0), -1):
        if float(df["close"].iloc[i]) > float(df["open"].iloc[i]):
            return True
    return None


def _find_bullish_ob(df, end_idx, lookback=5):
    for i in range(end_idx - 1, max(end_idx - lookback, 0), -1):
        if float(df["close"].iloc[i]) < float(df["open"].iloc[i]):
            return True
    return None


ADX_MAX = 25
RSI_OB = 90
RSI_OS = 10
LOOKBACK = 20
SWEEP_MIN = 2
SWEEP_MAX = 50
TRIGGER_WINDOW = 5
BASE_SL = 45
BASE_TP = 90
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
DISPLACEMENT_ATR_MULT = 1.5
LOT_SIZE = 65


def fetch_data():
    print("Fetching 5m Nifty data from TradingView...")
    from core.tv_fetcher import TVFetcher
    tv = TVFetcher()
    df = tv.get_ohlcv("NIFTY", "NSE", interval_minutes=5, n_bars=5000)
    if df is None or df.empty:
        print("ERROR: No data")
        sys.exit(1)
    print(f"Got {len(df)} bars: {df.index[0]} to {df.index[-1]}")
    return df


def run_backtest(df, use_time_filters=True):
    rsi = _compute_rsi(df["close"], 2)
    adx = _compute_adx(df)
    atr = _compute_atr(df)

    # Pre-compute PDH/PDL per day
    dates = sorted(set(df.index.date))
    pdh_pdl = {}
    for idx_d, d in enumerate(dates):
        if idx_d == 0:
            continue
        prev_d = dates[idx_d - 1]
        prev_bars = df[df.index.date == prev_d]
        if len(prev_bars) > 0:
            pdh_pdl[d] = (float(prev_bars["high"].max()), float(prev_bars["low"].min()))

    trades = []
    position = None
    setup = None
    setup_date = None
    min_idx = 35

    for i in range(min_idx, len(df)):
        bar = df.iloc[i]
        ts = df.index[i]
        spot = float(bar["close"])
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        today = ts.date()

        if setup_date != today:
            setup = None
            setup_date = today

        # ── Check exit ──
        if position is not None:
            pnl_pts = (position["entry_nifty"] - spot) if position["dir"] == "PUT" else (spot - position["entry_nifty"])
            hit_sl = hit_tp = eod = False

            if position["dir"] == "PUT":
                if bar_high >= position["entry_nifty"] + position["sl"]:
                    hit_sl, pnl_pts = True, -position["sl"]
                elif bar_low <= position["entry_nifty"] - position["tp"]:
                    hit_tp, pnl_pts = True, position["tp"]
            else:
                if bar_low <= position["entry_nifty"] - position["sl"]:
                    hit_sl, pnl_pts = True, -position["sl"]
                elif bar_high >= position["entry_nifty"] + position["tp"]:
                    hit_tp, pnl_pts = True, position["tp"]

            hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 0
            if hm >= 1515:
                eod = True

            if hit_sl or hit_tp or eod:
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "dir": position["dir"], "entry_nifty": position["entry_nifty"],
                    "exit_nifty": spot, "sl": position["sl"], "tp": position["tp"],
                    "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "EOD"),
                    "rsi2": position.get("rsi2", 0), "adx": position.get("adx", 0),
                    "sweep_pts": position.get("sweep_pts", 0),
                    "trigger": position.get("trigger", ""),
                    "hour": position["entry_time"].hour if hasattr(position["entry_time"], 'hour') else 0,
                    "pdhl": position.get("pdhl", False),
                })
                position = None

            if position is not None:
                continue

        # ── Time filters ──
        hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 1000
        if use_time_filters:
            if hm < 920:
                continue
            if 1200 <= hm <= 1330:
                continue
        if hm >= 1500:
            continue

        # ── Indicators ──
        adx_val = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 30.0
        if adx_val >= ADX_MAX:
            setup = None
            continue

        rsi_val = float(rsi.iloc[i]) if not np.isnan(rsi.iloc[i]) else 50.0
        atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0
        sl = max(SL_ATR_MULT * atr_val, BASE_SL) if atr_val > 0 else BASE_SL
        tp = max(TP_ATR_MULT * atr_val, BASE_TP) if atr_val > 0 else BASE_TP

        is_bearish = spot < bar_open
        is_bullish = spot > bar_open
        bar_body = abs(spot - bar_open)
        is_displacement = atr_val > 0 and bar_body > DISPLACEMENT_ATR_MULT * atr_val

        # ICT: FVG detection
        fvg_bear = _detect_fvg_bearish(df, i)
        fvg_bull = _detect_fvg_bullish(df, i)

        # ICT: swing structure
        highs = df["high"].values[:i + 1]
        lows = df["low"].values[:i + 1]

        # Range
        lookback_slice = df.iloc[i - LOOKBACK:i]
        range_high = float(lookback_slice["high"].max())
        range_low = float(lookback_slice["low"].min())
        range_mid = (range_high + range_low) / 2

        # PDH/PDL
        pdh, pdl = pdh_pdl.get(today, (None, None))

        # ── Phase 2: Trigger on active setup ──
        if setup is not None:
            bars_since = i - setup["bar_idx"]
            if bars_since > TRIGGER_WINDOW:
                setup = None
            elif setup["dir"] == "PUT":
                # ICT triggers
                swing_lows = _find_swing_lows(highs, lows)
                mss = False
                if swing_lows:
                    last_swing_low = swing_lows[-1][1]
                    mss = spot < last_swing_low

                has_displacement = is_bearish and is_displacement
                has_fvg = fvg_bear is not None
                has_ob = _find_bearish_ob(df, i)
                reversal_inside = is_bearish and spot < setup["range_high"]
                in_premium = spot > range_mid

                trigger = None
                if mss and (has_displacement or has_fvg):
                    trigger = "MSS+disp"
                elif mss:
                    trigger = "MSS"
                elif has_displacement and has_fvg:
                    trigger = "disp+FVG"
                elif has_displacement and has_ob:
                    trigger = "disp+OB"
                elif has_displacement and reversal_inside:
                    trigger = "disp_rev"
                elif reversal_inside:
                    trigger = "rev_inside"

                if trigger:
                    position = {
                        "dir": "PUT", "entry_nifty": spot,
                        "sl": round(sl, 1), "tp": round(tp, 1),
                        "entry_time": ts, "rsi2": setup["rsi"],
                        "adx": adx_val, "sweep_pts": setup["sweep_pts"],
                        "trigger": trigger,
                        "pdhl": setup.get("pdh_swept", False),
                    }
                    setup = None
                    continue

            elif setup["dir"] == "CALL":
                swing_highs = _find_swing_highs(highs, lows)
                mss = False
                if swing_highs:
                    last_swing_high = swing_highs[-1][1]
                    mss = spot > last_swing_high

                has_displacement = is_bullish and is_displacement
                has_fvg = fvg_bull is not None
                has_ob = _find_bullish_ob(df, i)
                reversal_inside = is_bullish and spot > setup["range_low"]
                in_discount = spot < range_mid

                trigger = None
                if mss and (has_displacement or has_fvg):
                    trigger = "MSS+disp"
                elif mss:
                    trigger = "MSS"
                elif has_displacement and has_fvg:
                    trigger = "disp+FVG"
                elif has_displacement and has_ob:
                    trigger = "disp+OB"
                elif has_displacement and reversal_inside:
                    trigger = "disp_rev"
                elif reversal_inside:
                    trigger = "rev_inside"

                if trigger:
                    position = {
                        "dir": "CALL", "entry_nifty": spot,
                        "sl": round(sl, 1), "tp": round(tp, 1),
                        "entry_time": ts, "rsi2": setup["rsi"],
                        "adx": adx_val, "sweep_pts": setup["sweep_pts"],
                        "trigger": trigger,
                        "pdhl": setup.get("pdl_swept", False),
                    }
                    setup = None
                    continue

        # ── Phase 1: Setup ──
        sweep_above = bar_high - range_high
        sweep_below = range_low - bar_low

        pdh_swept = pdh is not None and bar_high > pdh and SWEEP_MIN <= (bar_high - pdh) <= SWEEP_MAX
        pdl_swept = pdl is not None and bar_low < pdl and SWEEP_MIN <= (pdl - bar_low) <= SWEEP_MAX

        if SWEEP_MIN <= sweep_above <= SWEEP_MAX and rsi_val >= RSI_OB:
            setup = {
                "dir": "PUT", "bar_idx": i, "sweep_pts": sweep_above,
                "rsi": rsi_val, "range_high": range_high, "range_low": range_low,
                "pdh_swept": pdh_swept,
            }
        elif SWEEP_MIN <= sweep_below <= SWEEP_MAX and rsi_val <= RSI_OS:
            setup = {
                "dir": "CALL", "bar_idx": i, "sweep_pts": sweep_below,
                "rsi": rsi_val, "range_high": range_high, "range_low": range_low,
                "pdl_swept": pdl_swept,
            }

    if position is not None:
        spot = float(df["close"].iloc[-1])
        pnl_pts = (position["entry_nifty"] - spot) if position["dir"] == "PUT" else (spot - position["entry_nifty"])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": df.index[-1],
            "dir": position["dir"], "entry_nifty": position["entry_nifty"],
            "exit_nifty": spot, "sl": position["sl"], "tp": position["tp"],
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
            "exit_reason": "DATA_END", "rsi2": position.get("rsi2", 0),
            "adx": position.get("adx", 0), "sweep_pts": position.get("sweep_pts", 0),
            "trigger": position.get("trigger", ""), "hour": 0,
            "pdhl": position.get("pdhl", False),
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

    if "trigger" in tdf.columns:
        print(f"\n  ICT Trigger breakdown:")
        for t, grp in tdf.groupby("trigger"):
            if not t:
                continue
            twr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
            print(f"    {t:15s}: {len(grp):3d} trades, WR={twr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    if "pdhl" in tdf.columns:
        pdhl_trades = tdf[tdf["pdhl"] == True]
        non_pdhl = tdf[tdf["pdhl"] != True]
        if len(pdhl_trades) > 0:
            pw = len(pdhl_trades[pdhl_trades["pnl_pts"] > 0]) / len(pdhl_trades) * 100
            print(f"\n  PDH/PDL confluence:")
            print(f"    With PDH/PDL : {len(pdhl_trades):3d} trades, WR={pw:.1f}%, PnL={pdhl_trades['pnl_pts'].sum():+.1f} pts")
            if len(non_pdhl) > 0:
                nw = len(non_pdhl[non_pdhl["pnl_pts"] > 0]) / len(non_pdhl) * 100
                print(f"    Without      : {len(non_pdhl):3d} trades, WR={nw:.1f}%, PnL={non_pdhl['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Hourly:")
    for h, grp in tdf.groupby("hour"):
        if h == 0:
            continue
        hwr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    {h:02d}:xx : {len(grp):3d} trades, WR={hwr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Trade log:")
    print(f"  {'Entry':20s} {'Dir':5s} {'Entry':>8s} {'Exit':>8s} {'PnL':>8s} {'Rs':>10s} {'Exit':8s} {'RSI2':>6s} {'ADX':>5s} {'Trigger':15s} {'PDH/L':5s}")
    for _, t in tdf.iterrows():
        print(f"  {str(t['entry_time'])[:16]:20s} {t['dir']:5s} {t['entry_nifty']:8.1f} {t['exit_nifty']:8.1f} "
              f"{t['pnl_pts']:+8.1f} {t['pnl_rs']:+10,.0f} {t['exit_reason']:8s} {t['rsi2']:6.1f} {t['adx']:5.1f} "
              f"{t.get('trigger',''):15s} {'Y' if t.get('pdhl') else '':5s}")


def main():
    df = fetch_data()
    dates = sorted(set(df.index.date))
    print(f"Trading days: {len(dates)} | {dates[0]} to {dates[-1]}")

    r_with = run_backtest(df, use_time_filters=True)
    print_results(r_with, "S9 ICT Range Reversal — WITH time filters")

    r_without = run_backtest(df, use_time_filters=False)
    print_results(r_without, "S9 ICT Range Reversal — WITHOUT time filters")

    print(f"\n{'='*70}")
    print(f" FILTER IMPACT")
    print(f"{'='*70}")
    n1, n2 = len(r_with), len(r_without)
    p1 = sum(t["pnl_pts"] for t in r_with)
    p2 = sum(t["pnl_pts"] for t in r_without)
    w1 = sum(1 for t in r_with if t["pnl_pts"] > 0) / n1 * 100 if n1 else 0
    w2 = sum(1 for t in r_without if t["pnl_pts"] > 0) / n2 * 100 if n2 else 0
    print(f"  With filters    : {n1:3d} trades, WR={w1:.1f}%, PnL={p1:+.1f} pts")
    print(f"  Without filters : {n2:3d} trades, WR={w2:.1f}%, PnL={p2:+.1f} pts")

    if n2 > n1:
        times_with = {str(t["entry_time"]) for t in r_with}
        filtered = [t for t in r_without if str(t["entry_time"]) not in times_with]
        if filtered:
            fp = sum(t["pnl_pts"] for t in filtered)
            fw = sum(1 for t in filtered if t["pnl_pts"] > 0) / len(filtered) * 100
            early = [t for t in filtered if hasattr(t["entry_time"], 'hour') and t["entry_time"].hour * 100 + t["entry_time"].minute < 920]
            lunch = [t for t in filtered if hasattr(t["entry_time"], 'hour') and 1200 <= t["entry_time"].hour * 100 + t["entry_time"].minute <= 1330]
            print(f"\n  Filtered-out trades:")
            print(f"    Total: {len(filtered)}, WR={fw:.1f}%, PnL={fp:+.1f} pts")
            print(f"    Open settle: {len(early)}, PnL={sum(t['pnl_pts'] for t in early):+.1f}")
            print(f"    Lunch chop:  {len(lunch)}, PnL={sum(t['pnl_pts'] for t in lunch):+.1f}")

    if n2 > 0 and p2 > p1 and w2 >= w1:
        print(f"\n  >>> REMOVE time filters")
    else:
        print(f"\n  >>> KEEP time filters")


if __name__ == "__main__":
    main()
