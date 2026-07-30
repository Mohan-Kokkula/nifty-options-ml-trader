"""
trade_detail.py — Show detailed trades for recent days with SL/TP/entry/exit
"""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(".")))
from core import ml_engine
from core.ml_engine import load_model
from core.vix_regime import classify_vix, apply_vix_to_sl_tp


def _load_csv(path, is_daily=False):
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    dt = next((c for c in df.columns if "date" in c or "time" in c), None)
    if not dt:
        return pd.DataFrame()
    df[dt] = pd.to_datetime(df[dt], format="mixed", utc=False)
    if hasattr(df[dt].dt, "tz") and df[dt].dt.tz is not None:
        df[dt] = df[dt].dt.tz_localize(None)
    df = df.set_index(dt).sort_index()
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    keep = [c for c in ["open", "high", "low", "close", "vix"] if c in df.columns]
    df = df[keep].astype(float).dropna()
    if not is_daily:
        df = df.between_time("09:15", "15:30")
    df = df[~df.index.duplicated(keep="last")]
    return df


def run():
    load_model(".")

    df5 = _load_csv("data/nifty_5min.csv")
    df15 = _load_csv("data/nifty_15min.csv")
    df30 = _load_csv("data/nifty_30min.csv")
    df60 = _load_csv("data/nifty_60min.csv")
    dfday = _load_csv("data/nifty_day.csv", is_daily=True)
    dfvix = _load_csv("data/india_vix.csv", is_daily=True)

    from scripts.train_model_v8 import (
        features_5min, features_15min, features_30min,
        features_60min, features_daily, features_vix,
        merge_htf, merge_daily_onto_5min,
        merge_vix_onto_5min, add_intraday_context,
    )

    df_feat = features_5min(df5)
    if not df15.empty:
        df_feat = merge_htf(df_feat, features_15min(df15), "15m")
    if not df30.empty:
        df_feat = merge_htf(df_feat, features_30min(df30), "30m")
    if not df60.empty:
        df_feat = merge_htf(df_feat, features_60min(df60), "60m")
    if not dfday.empty:
        dfday = dfday[~dfday.index.duplicated(keep="last")]
        df_feat = merge_daily_onto_5min(df_feat, features_daily(dfday))
    if not dfvix.empty:
        if "close" in dfvix.columns and "vix" not in dfvix.columns:
            dfvix = dfvix.rename(columns={"close": "vix"})
        if "vix" in dfvix.columns:
            df_feat = merge_vix_onto_5min(df_feat, features_vix(dfvix[["vix"]]))

    df_feat = add_intraday_context(df_feat)
    df_feat.dropna(inplace=True)

    # Only last 15 trading days (more context for feature warmup)
    unique_dates = sorted(set(df_feat.index.date))
    cutoff = unique_dates[-15] if len(unique_dates) >= 15 else unique_dates[0]
    df_feat = df_feat[df_feat.index.date >= cutoff]

    print(f"  Trade simulation: {df_feat.index[0].date()} -> {df_feat.index[-1].date()} ({len(unique_dates)} days)")


    _fcols = ml_engine._fcols
    _scaler = ml_engine._scaler
    _models = ml_engine._models

    for col in [c for c in _fcols if c not in df_feat.columns]:
        df_feat[col] = 0.0

    X = _scaler.transform(df_feat[_fcols].values)
    probas = np.mean([m.predict_proba(X) for m in _models.values()], axis=0)

    CONFIDENCE = 0.30
    MIN_EDGE = 0.05
    SKIP_CEIL = 0.60
    signals = np.full(len(probas), 2)
    signals[
        (probas[:, 0] >= CONFIDENCE)
        & (probas[:, 0] - probas[:, 1] >= MIN_EDGE)
        & (probas[:, 2] < SKIP_CEIL)
    ] = 0
    signals[
        (probas[:, 1] >= CONFIDENCE)
        & (probas[:, 1] - probas[:, 0] >= MIN_EDGE)
        & (probas[:, 2] < SKIP_CEIL)
    ] = 1

    # Arrays
    close_arr = df_feat["close"].values
    high_arr = df_feat["high"].values if "high" in df_feat.columns else close_arr * 1.0002
    low_arr = df_feat["low"].values if "low" in df_feat.columns else close_arr * 0.9998

    tr_arr = np.maximum(
        high_arr - low_arr,
        np.maximum(
            np.abs(high_arr - np.roll(close_arr, 1)),
            np.abs(low_arr - np.roll(close_arr, 1)),
        ),
    )
    tr_arr[0] = high_arr[0] - low_arr[0]
    atr_arr = pd.Series(tr_arr).rolling(14).mean().fillna(25.0).values
    vix_arr = df_feat["vix_level"].values if "vix_level" in df_feat.columns else np.zeros(len(df_feat))
    conf_arr = probas.max(axis=1)

    # Config
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 4.0
    MIN_SL = 20.0
    MAX_SL = 60.0
    MIN_TP = 50.0
    MAX_TP = 200.0
    TRAIL_ACT_R = 1.5
    TRAIL_PCT = 0.5
    D = 0.50
    LOT = 65
    DAILY_LOSS_CAP = 3000
    SIGNAL_FLOOD_LIMIT = 4
    MIN_CONF = 0.35
    prev_day_vix = 0.0

    def compute_sl_tp(s, d, a, bar_vix=0.0, prev_vix=0.0):
        sl_pts = a * SL_ATR_MULT
        tp_pts = a * TP_ATR_MULT
        if bar_vix > 0:
            regime = classify_vix(bar_vix, prev_vix)
            sl_pts, tp_pts = apply_vix_to_sl_tp(
                sl_pts, tp_pts, regime,
                min_sl=MIN_SL, max_sl=90.0,
                min_tp=MIN_TP, max_tp=350.0,
            )
        else:
            sl_pts = max(MIN_SL, min(MAX_SL, sl_pts))
            tp_pts = max(MIN_TP, min(MAX_TP, tp_pts))
            if tp_pts < sl_pts * 2:
                tp_pts = sl_pts * 2
        if d == "CALL":
            return s - sl_pts, s + tp_pts, sl_pts
        else:
            return s + sl_pts, s - tp_pts, sl_pts

    # Filters matching backtest
    OPENING_MIN_CONF = 0.60
    REQUIRE_2BAR_CONFIRM = True

    # Simulate
    trades = []
    pos_dir = None
    entry_price = 0.0
    entry_time = None
    sl_level = 0.0
    tp_level = 0.0
    initial_sl_dist = 0.0
    trail_activated = False
    current_day = None
    daily_pnl = 0.0
    daily_halted = False
    pre1300 = 0
    flood_halted = False
    prev_signal = 2

    def close_pos(spot, ts, reason):
        nonlocal pos_dir, daily_pnl, daily_halted, trail_activated
        move = spot - entry_price
        pnl = (D * (move if pos_dir == "CALL" else -move)) * LOT
        trades.append({
            "date": ts.date(),
            "entry_time": entry_time,
            "exit_time": ts,
            "dir": pos_dir,
            "entry": entry_price,
            "exit": spot,
            "sl": sl_level,
            "tp": tp_level,
            "sl_pts": initial_sl_dist,
            "tp_pts": abs(tp_level - entry_price),
            "pnl": pnl,
            "reason": reason,
            "trail": trail_activated,
            "atr": float(atr_arr[min(i, len(atr_arr) - 1)]),
            "vix": float(vix_arr[min(i, len(vix_arr) - 1)]),
        })
        daily_pnl += pnl
        pos_dir = None
        trail_activated = False
        if pnl < 0 and daily_pnl <= -DAILY_LOSS_CAP:
            daily_halted = True

    for i in range(len(df_feat)):
        ts = df_feat.index[i]
        spot = float(close_arr[i])
        sig = signals[i]
        bar_conf = float(conf_arr[i])
        bar_vix = float(vix_arr[i])
        bar_atr = float(atr_arr[i])

        if ts.date() != current_day:
            current_day = ts.date()
            daily_pnl = 0.0
            daily_halted = False
            pre1300 = 0
            flood_halted = False
            prev_signal = 2

        if daily_halted or flood_halted:
            if pos_dir:
                close_pos(spot, ts, "DAILY_CAP" if daily_halted else "FLOOD_EXIT")
            continue

        if ts.hour >= 15 and ts.minute >= 20:
            if pos_dir:
                close_pos(spot, ts, "SQUARE_OFF")
            continue

        if pos_dir:
            unrealized = (spot - entry_price) if pos_dir == "CALL" else (entry_price - spot)
            if not trail_activated and unrealized >= initial_sl_dist * TRAIL_ACT_R:
                trail_activated = True
            if trail_activated:
                lock = unrealized * TRAIL_PCT
                if pos_dir == "CALL":
                    new_sl = entry_price + lock
                    if new_sl > sl_level:
                        sl_level = new_sl
                else:
                    new_sl = entry_price - lock
                    if new_sl < sl_level:
                        sl_level = new_sl

            sl_hit = (pos_dir == "CALL" and spot <= sl_level) or (pos_dir == "PUT" and spot >= sl_level)
            tp_hit = (pos_dir == "CALL" and spot >= tp_level) or (pos_dir == "PUT" and spot <= tp_level)
            if sl_hit or tp_hit:
                reason = ("TRAIL_SL" if trail_activated else "SL") if sl_hit else "TP"
                close_pos(spot, ts, reason)

        if sig == 2:
            prev_signal = 2
            continue
        # Opening session higher conf bar
        effective_conf = MIN_CONF
        if ts.hour == 9 and ts.minute < 30:
            effective_conf = OPENING_MIN_CONF
        if bar_conf < effective_conf:
            prev_signal = sig
            continue
        # Dynamic VIX regime — only hard-block EXTREME+RISING
        if bar_vix > 0:
            regime = classify_vix(bar_vix, prev_day_vix)
            if not regime.tradeable:
                prev_signal = sig; continue
            if bar_conf < regime.min_ml_confidence:
                prev_signal = sig; continue
        if ts.hour >= 15:
            prev_signal = sig; continue
        # 2-bar confirmation
        if REQUIRE_2BAR_CONFIRM and prev_signal != sig:
            prev_signal = sig; continue
        if ts.hour < 13:
            pre1300 += 1
            if pre1300 > SIGNAL_FLOOD_LIMIT:
                flood_halted = True
                if pos_dir:
                    close_pos(spot, ts, "FLOOD_EXIT")
                prev_signal = sig; continue

        sd = "CALL" if sig == 0 else "PUT"
        prev_signal = sig

        if pos_dir is None:
            pos_dir = sd
            entry_price = spot
            entry_time = ts
            sl_level, tp_level, initial_sl_dist = compute_sl_tp(spot, sd, bar_atr, bar_vix, prev_day_vix)
            trail_activated = False
        elif pos_dir != sd:
            close_pos(spot, ts, "REVERSAL")
            pos_dir = sd
            entry_price = spot
            entry_time = ts
            sl_level, tp_level, initial_sl_dist = compute_sl_tp(spot, sd, bar_atr, bar_vix, prev_day_vix)
            trail_activated = False

    if pos_dir:
        spot = float(close_arr[-1])
        close_pos(spot, df_feat.index[-1], "END")

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        print("No trades found!")
        return

    # Print last 7 trading days with trades
    all_dates = sorted(tdf["date"].unique())
    show_dates = all_dates[-7:]

    print("=" * 110)
    print("  DETAILED TRADE LOG -- LAST 5 TRADING DAYS (with Dynamic SL/TP + Trailing Stop)")
    print("=" * 110)

    for d in show_dates:
        day_trades = tdf[tdf["date"] == d]
        day_total = day_trades["pnl"].sum()
        day_wins = (day_trades["pnl"] > 0).sum()
        weekday = pd.Timestamp(d).strftime("%A")

        print()
        print(f"  DATE: {d} ({weekday})")
        print(f"  Day P&L: Rs. {day_total:+,.0f} | Trades: {len(day_trades)} | Wins: {day_wins}/{len(day_trades)}")
        print(f"  {'-' * 106}")
        print(
            f"  {'#':<3} {'Dir':<5} {'Entry@':<7} {'Entry':>9} "
            f"{'SL':>9} {'TP':>9} {'SL pts':>7} {'TP pts':>7} "
            f"{'Exit@':<7} {'Exit':>9} {'P&L':>10} {'Reason':<10} {'Trail':<6} {'ATR':>5}"
        )
        print(f"  {'-' * 106}")

        for idx, (_, t) in enumerate(day_trades.iterrows(), 1):
            et = t["entry_time"].strftime("%H:%M") if pd.notna(t["entry_time"]) else "?"
            xt = t["exit_time"].strftime("%H:%M") if pd.notna(t["exit_time"]) else "?"
            trail_s = "YES" if t["trail"] else "NO"
            print(
                f"  {idx:<3} {t['dir']:<5} {et:<7} {t['entry']:>9,.1f} "
                f"{t['sl']:>9,.1f} {t['tp']:>9,.1f} {t['sl_pts']:>7,.1f} {t['tp_pts']:>7,.1f} "
                f"{xt:<7} {t['exit']:>9,.1f} {t['pnl']:>+10,.0f} {t['reason']:<10} {trail_s:<6} {t['atr']:>5,.1f}"
            )
        print()

    # Summary for the very last day
    last_day = all_dates[-1]
    lt = tdf[tdf["date"] == last_day]
    print("=" * 110)
    print(f"  LATEST DAY ({last_day}) SUMMARY:")
    print(f"    Total Trades:    {len(lt)}")
    print(f"    Winners:         {(lt['pnl'] > 0).sum()}")
    print(f"    Losers:          {(lt['pnl'] <= 0).sum()}")
    print(f"    Day P&L:         Rs. {lt['pnl'].sum():+,.0f}")
    print(f"    Best Trade:      Rs. {lt['pnl'].max():+,.0f}")
    print(f"    Worst Trade:     Rs. {lt['pnl'].min():+,.0f}")
    print(f"    Trail Activations: {int(lt['trail'].sum())}")
    print(f"    Avg SL Used:     {lt['sl_pts'].mean():.1f} pts")
    print(f"    Avg TP Target:   {lt['tp_pts'].mean():.1f} pts")
    print("=" * 110)


if __name__ == "__main__":
    run()
