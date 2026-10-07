"""
backtest_s10.py — Backtest S10 Momentum Confluence.
Chaikin Oscillator + PMO dual consensus, ADX > 20 trending filter.
Cooldown 5 bars after SL, max 2 trades per direction per day.
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


def _compute_chaikin_osc(df, fast=3, slow=10):
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, np.nan)
    clv = clv.fillna(0)
    vol = df["volume"] if "volume" in df.columns else pd.Series(1000000, index=df.index)
    ad = (clv * vol).cumsum()
    return ad.ewm(span=fast, adjust=False).mean() - ad.ewm(span=slow, adjust=False).mean()


def _compute_pmo(series, smooth1=20, smooth2=10, signal_period=35):
    roc = ((series / series.shift(1)) - 1) * 100
    pmo_raw = roc.ewm(span=smooth1, adjust=False).mean()
    pmo_line = pmo_raw.ewm(span=smooth2, adjust=False).mean()
    pmo_signal = pmo_line.ewm(span=signal_period, adjust=False).mean()
    return pmo_line, pmo_signal


ADX_MIN = 20
BASE_SL = 45
BASE_TP = 90
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
COOLDOWN_BARS = 5
MAX_PER_DIR_DAY = 2
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


def run_backtest(df, use_time_filters=False):
    adx = _compute_adx(df)
    atr = _compute_atr(df)
    chaikin = _compute_chaikin_osc(df)
    pmo_line, pmo_sig = _compute_pmo(df["close"])

    trades = []
    position = None
    min_idx = 60
    last_sl_bar = {"CALL": -999, "PUT": -999}
    daily_dir_count = {}
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
            last_date = today

        # ── Exit ──
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
                if hit_sl:
                    last_sl_bar[position["dir"]] = i
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": ts,
                    "dir": position["dir"], "entry_nifty": position["entry_nifty"],
                    "exit_nifty": spot, "sl": position["sl"], "tp": position["tp"],
                    "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "EOD"),
                    "hour": position["entry_time"].hour if hasattr(position["entry_time"], 'hour') else 0,
                    "adx": position.get("adx", 0),
                })
                position = None
            if position is not None:
                continue

        # ── Time ──
        hm = ts.hour * 100 + ts.minute if hasattr(ts, 'hour') else 1000
        if use_time_filters:
            if hm < 920 or (1200 <= hm <= 1330):
                continue
        if hm >= 1500:
            continue

        # ── ADX ──
        adx_val = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 15.0
        if adx_val < ADX_MIN:
            continue

        atr_val = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0
        sl = max(SL_ATR_MULT * atr_val, BASE_SL) if atr_val > 0 else BASE_SL
        tp = max(TP_ATR_MULT * atr_val, BASE_TP) if atr_val > 0 else BASE_TP

        # ── Oscillator consensus ──
        cv = float(chaikin.iloc[i]) if not np.isnan(chaikin.iloc[i]) else 0
        pl = float(pmo_line.iloc[i]) if not np.isnan(pmo_line.iloc[i]) else 0
        ps = float(pmo_sig.iloc[i]) if not np.isnan(pmo_sig.iloc[i]) else 0

        direction = None
        if cv > 0 and pl > ps:
            direction = "CALL"
        elif cv < 0 and pl < ps:
            direction = "PUT"
        if direction is None:
            continue

        # ── Cooldown ──
        if (i - last_sl_bar[direction]) < COOLDOWN_BARS:
            continue

        # ── Max per direction per day ──
        if daily_dir_count[direction] >= MAX_PER_DIR_DAY:
            continue

        daily_dir_count[direction] += 1
        position = {
            "dir": direction, "entry_nifty": spot,
            "sl": round(sl, 1), "tp": round(tp, 1),
            "entry_time": ts, "adx": adx_val,
        }

    if position is not None:
        spot = float(df["close"].iloc[-1])
        pnl_pts = (position["entry_nifty"] - spot) if position["dir"] == "PUT" else (spot - position["entry_nifty"])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": df.index[-1],
            "dir": position["dir"], "entry_nifty": position["entry_nifty"],
            "exit_nifty": spot, "sl": position["sl"], "tp": position["tp"],
            "pnl_pts": round(pnl_pts, 2), "pnl_rs": round(pnl_pts * LOT_SIZE, 2),
            "exit_reason": "DATA_END", "hour": 0, "adx": position.get("adx", 0),
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

    print(f"\n  Hourly:")
    for h, grp in tdf.groupby("hour"):
        if h == 0:
            continue
        hwr = len(grp[grp["pnl_pts"] > 0]) / len(grp) * 100 if len(grp) else 0
        print(f"    {h:02d}:xx : {len(grp):3d} trades, WR={hwr:.1f}%, PnL={grp['pnl_pts'].sum():+.1f} pts")

    print(f"\n  Trade log:")
    print(f"  {'Entry':20s} {'Dir':5s} {'Entry':>8s} {'Exit':>8s} {'PnL':>8s} {'Rs':>10s} {'Exit':8s} {'ADX':>5s}")
    for _, t in tdf.iterrows():
        print(f"  {str(t['entry_time'])[:16]:20s} {t['dir']:5s} {t['entry_nifty']:8.1f} {t['exit_nifty']:8.1f} "
              f"{t['pnl_pts']:+8.1f} {t['pnl_rs']:+10,.0f} {t['exit_reason']:8s} {t['adx']:5.1f}")


def main():
    df = fetch_data()
    dates = sorted(set(df.index.date))
    print(f"Trading days: {len(dates)} | {dates[0]} to {dates[-1]}")

    r_no_tf = run_backtest(df, use_time_filters=False)
    print_results(r_no_tf, "S10 Momentum Confluence — NO time filters (recommended)")

    r_with_tf = run_backtest(df, use_time_filters=True)
    print_results(r_with_tf, "S10 Momentum Confluence — WITH time filters")

    print(f"\n{'='*70}")
    print(f" FILTER IMPACT")
    print(f"{'='*70}")
    n1, n2 = len(r_no_tf), len(r_with_tf)
    p1 = sum(t["pnl_pts"] for t in r_no_tf)
    p2 = sum(t["pnl_pts"] for t in r_with_tf)
    w1 = sum(1 for t in r_no_tf if t["pnl_pts"] > 0) / n1 * 100 if n1 else 0
    w2 = sum(1 for t in r_with_tf if t["pnl_pts"] > 0) / n2 * 100 if n2 else 0
    print(f"  No time filters : {n1:3d} trades, WR={w1:.1f}%, PnL={p1:+.1f} pts")
    print(f"  With filters    : {n2:3d} trades, WR={w2:.1f}%, PnL={p2:+.1f} pts")
    if p1 >= p2:
        print(f"\n  >>> NO time filters (momentum works through lunch)")
    else:
        print(f"\n  >>> KEEP time filters")


if __name__ == "__main__":
    main()
