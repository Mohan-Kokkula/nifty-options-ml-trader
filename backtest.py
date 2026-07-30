"""
backtest.py  -  V8 Multi-Timeframe ML Model Backtest
================================================================
Usage:
    python backtest.py              # All years
    python backtest.py 2024         # Year 2024 only
    python backtest.py 2024 2025    # Range
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core.ml_engine import load_model, predict_batch
from core.vix_regime import classify_vix, apply_vix_to_sl_tp

# -- Config --
LOT_SIZE = 65
DELTA_APPROX = 0.50
DATA_PATH     = "data/nifty_5min.csv"
DATA_15M      = "data/nifty_15min.csv"
DATA_30M      = "data/nifty_30min.csv"
DATA_60M      = "data/nifty_60min.csv"
DATA_DAY      = "data/nifty_day.csv"
DATA_VIX      = "data/india_vix.csv"

DAILY_LOSS_CAP    = 3000
SIGNAL_FLOOD_LIMIT = 6   # Increased from 4 — forced flood exits create losers

# Accuracy filters — tuned for 90%+ win rate
MIN_CONF_THRESHOLD = 0.38   # skip signals below 38% ML confidence
OPENING_MIN_CONF   = 0.60   # higher bar for 09:15-09:35 (opening whipsaw filter)
REQUIRE_2BAR_CONFIRM = True # require 2 consecutive signals in same direction
REQUIRE_3BAR_MORNING = True # require 3 consecutive signals before 10:30 (choppy open)
VIX_EXTREME_CUTOFF = 30.0   # hard block only at VIX > 30 (EXTREME + RISING)
ENTRY_CUTOFF_HOUR  = 15     # no new entries at or after 15:00 (15 min buffer before close)
ENTRY_CUTOFF_MIN   = 0

# Trend alignment filters — keep light, ML already models these
USE_TREND_FILTER     = True   # require EMA alignment for entries
USE_RSI_ZONE_FILTER  = True   # block overbought CALLs / oversold PUTs
RSI_OB_THRESHOLD     = 75     # RSI > 75 = overbought, skip CALLs
RSI_OS_THRESHOLD     = 25     # RSI < 25 = oversold, skip PUTs
USE_VWAP_FILTER      = False  # OFF — ML already handles this

# Anti-reversal filter: don't reverse within N bars of entry (reduces whipsaws)
ANTI_REVERSAL_BARS   = 6      # No reversal within 6 bars (30 min) of entry

# Dynamic SL/TP config (ATR-based) — wider SL + lower TP for high win rate
USE_DYNAMIC_SL_TP    = True
SL_ATR_MULTIPLIER    = 2.2     # SL = 2.2x ATR14 (wide = absorb noise)
TP_ATR_MULTIPLIER    = 2.2     # TP = 2.2x ATR14 (quick take-profit)
MIN_SL_POINTS        = 30.0
MAX_SL_POINTS        = 80.0
MIN_TP_POINTS        = 30.0    # Low min TP for quick wins
MAX_TP_POINTS        = 130.0
FALLBACK_SL_POINTS   = 45
FALLBACK_TP_POINTS   = 70

# Breakeven stop — once profit reaches X pts, move SL to entry (zero-risk trade)
USE_BREAKEVEN_STOP   = True
BREAKEVEN_TRIGGER_PTS = 15.0   # Once +15pts in profit, move SL to entry+3 (very early BE)
BREAKEVEN_LOCK_PTS    = 3.0    # Lock 3 pts profit at breakeven

# Trailing stop config — aggressive profit protection
USE_TRAILING_STOP    = True
TRAIL_ACTIVATION_R   = 0.8     # Activate after 0.8x SL distance profit (very early)
TRAIL_STEP_PCT       = 0.60    # Lock 60% of unrealized profit

# Trailing target — extend TP when momentum is strong
USE_TRAILING_TP         = True
TRAIL_TP_ACTIVATION_PCT = 0.75  # When 75% of TP reached, start trailing TP
TRAIL_TP_EXTEND_PCT     = 0.50  # Extend TP by 50% of peak momentum


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
    keep = [c for c in ["open","high","low","close","vix"] if c in df.columns]
    df = df[keep].astype(float).dropna()
    if not is_daily:
        df = df.between_time("09:15", "15:30")
    df = df[~df.index.duplicated(keep="last")]
    return df


def _build_full_features(df5, year_start=None, year_end=None):
    """Build V8 multi-TF features and return signals for the requested date range."""
    try:
        from scripts.train_model_v8 import (
            features_5min, features_15min, features_30min,
            features_60min, features_daily, features_vix,
            merge_htf, merge_daily_onto_5min,
            merge_vix_onto_5min, add_intraday_context
        )
    except ImportError:
        print("WARNING: train_model_v8 not found — using 5-min only")
        result = predict_batch(df5)
        return result[0], result[1], result[2]

    # Load higher TFs
    df15  = _load_csv(DATA_15M)
    df30  = _load_csv(DATA_30M)
    df60  = _load_csv(DATA_60M)
    dfday = _load_csv(DATA_DAY, is_daily=True)
    dfvix = _load_csv(DATA_VIX, is_daily=True)

    # Compute features on full history
    print("  Computing multi-TF features...")
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
            print("  VIX features merged ✅")
        else:
            print("  WARNING: VIX CSV has no vix column — skipping")
    else:
        print("  WARNING: india_vix.csv not found — run scripts/download_vix.py")

    df_feat = add_intraday_context(df_feat)
    df_feat.dropna(inplace=True)

    if year_start:
        df_feat = df_feat[df_feat.index.year >= int(year_start)]
    if year_end:
        df_feat = df_feat[df_feat.index.year <= int(year_end)]

    from core.ml_engine import _fcols, _scaler, _models

    missing = [c for c in _fcols if c not in df_feat.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing — filling with 0")
        for col in missing:
            df_feat[col] = 0.0

    X      = df_feat[_fcols].values
    Xs     = _scaler.transform(X)
    probas = np.mean([m.predict_proba(Xs) for m in _models.values()], axis=0)

    # Adaptive thresholds for trade-sim trained model
    CONFIDENCE = 0.30
    MIN_EDGE   = 0.05
    SKIP_CEIL  = 0.60

    signals = np.full(len(probas), 2)
    signals[
        (probas[:,0] >= CONFIDENCE) &
        (probas[:,0] - probas[:,1] >= MIN_EDGE) &
        (probas[:,2] < SKIP_CEIL)
    ] = 0
    signals[
        (probas[:,1] >= CONFIDENCE) &
        (probas[:,1] - probas[:,0] >= MIN_EDGE) &
        (probas[:,2] < SKIP_CEIL)
    ] = 1

    return signals, probas, df_feat


def run_backtest(year_start=None, year_end=None):
    if not load_model("."):
        print("ERROR: Model not found in models/")
        return

    # Load full 5-min history (all years — needed for feature warmup)
    df_full = _load_csv(DATA_PATH)

    print(f"\n{'='*65}")
    print(f"  NIFTY V8 ML BACKTEST")
    print(f"  5-min bars loaded: {len(df_full):,} | {df_full.index[0].date()} -> {df_full.index[-1].date()}")
    if USE_DYNAMIC_SL_TP:
        print(f"  SL={SL_ATR_MULTIPLIER}x ATR ({MIN_SL_POINTS}-{MAX_SL_POINTS}pts) | "
              f"TP={TP_ATR_MULTIPLIER}x ATR ({MIN_TP_POINTS}-{MAX_TP_POINTS}pts)")
    else:
        print(f"  SL={FALLBACK_SL_POINTS}pt | TP={FALLBACK_TP_POINTS}pt")
    print(f"  Lot={LOT_SIZE} | Delta={DELTA_APPROX} | "
          f"Trail={'ON' if USE_TRAILING_STOP else 'OFF'} "
          f"(activate={TRAIL_ACTIVATION_R}xSL, lock={TRAIL_STEP_PCT*100:.0f}%)")
    print(f"  Daily cap=Rs{DAILY_LOSS_CAP} | Flood limit={SIGNAL_FLOOD_LIMIT} signals before 13:00")
    print(f"  Filters: MinConf>={MIN_CONF_THRESHOLD} (open>={OPENING_MIN_CONF}) | "
          f"2bar={'ON' if REQUIRE_2BAR_CONFIRM else 'OFF'} 3bar-AM={'ON' if REQUIRE_3BAR_MORNING else 'OFF'} | "
          f"EMA={'ON' if USE_TREND_FILTER else 'OFF'} RSI={'ON' if USE_RSI_ZONE_FILTER else 'OFF'} VWAP={'ON' if USE_VWAP_FILTER else 'OFF'} | "
          f"VIX=DYNAMIC | Before {ENTRY_CUTOFF_HOUR}:{ENTRY_CUTOFF_MIN:02d}")
    print(f"{'='*65}")

    # Build full V8 features and get signals for requested range
    signals, probas, feat = _build_full_features(df_full, year_start, year_end)

    nc  = (signals == 0).sum()
    np_ = (signals == 1).sum()
    ns  = (signals == 2).sum()
    print(f"  Signals: CALL={nc:,} | PUT={np_:,} | SKIP={ns:,}")
    print(f"  Backtest period: {feat.index[0].date()} -> {feat.index[-1].date()} ({len(feat):,} bars)")


    # Simulate trades with dynamic SL/TP and trailing stop
    LOT = LOT_SIZE
    D = DELTA_APPROX

    # Pre-compute ATR14 for every bar (for dynamic SL/TP)
    close_arr = feat["close"].values
    if "high" in feat.columns and "low" in feat.columns:
        high_arr = feat["high"].values
        low_arr  = feat["low"].values
    else:
        high_arr = close_arr * 1.0002
        low_arr  = close_arr * 0.9998

    # Compute ATR14 array for all bars
    tr_arr = np.maximum(
        high_arr - low_arr,
        np.maximum(
            np.abs(high_arr - np.roll(close_arr, 1)),
            np.abs(low_arr - np.roll(close_arr, 1))
        )
    )
    tr_arr[0] = high_arr[0] - low_arr[0]  # fix first bar
    atr_arr = pd.Series(tr_arr).rolling(14).mean().fillna(25.0).values

    vix_arr = feat["vix_level"].values if "vix_level" in feat.columns else None
    conf_arr = probas.max(axis=1)

    # Pre-compute trend alignment indicators (EMA9 vs EMA21, RSI14, VWAP)
    _close_s = pd.Series(close_arr, index=feat.index)
    ema9_arr = _close_s.ewm(span=9, adjust=False).mean().values
    ema21_arr = _close_s.ewm(span=21, adjust=False).mean().values

    # RSI14
    _delta = _close_s.diff()
    _gain = _delta.clip(lower=0).rolling(14).mean()
    _loss = (-_delta.clip(upper=0)).rolling(14).mean()
    _rs = _gain / _loss.replace(0, np.nan)
    rsi_arr = (100 - (100 / (1 + _rs))).fillna(50).values

    # Session VWAP (reset each day)
    if "volume" in feat.columns:
        vol_arr = feat["volume"].values
    else:
        vol_arr = np.ones(len(feat))  # fallback: equal-weight VWAP
    vwap_arr = np.zeros(len(feat))
    _cum_pv = 0.0
    _cum_v = 0.0
    _prev_date = None
    for _i in range(len(feat)):
        _d = feat.index[_i].date()
        if _d != _prev_date:
            _cum_pv = 0.0
            _cum_v = 0.0
            _prev_date = _d
        _cum_pv += close_arr[_i] * vol_arr[_i]
        _cum_v += vol_arr[_i]
        vwap_arr[_i] = _cum_pv / _cum_v if _cum_v > 0 else close_arr[_i]

    trades = []
    pos_dir = None
    entry_price = 0.0
    sl_level = 0.0
    tp_level = 0.0
    original_tp_level = 0.0   # for trailing TP reference
    initial_sl_dist = 0.0
    trail_activated = False
    breakeven_activated = False
    tp_trailing = False
    peak_unrealized = 0.0     # highest profit seen in current trade
    entry_bar_idx = 0         # track which bar we entered on

    # Per-day state
    current_day = None
    daily_pnl = 0.0
    daily_halted = False
    pre1300_signals = 0
    flood_halted = False
    prev_signal = 2     # for 2-bar confirmation
    prev_prev_signal = 2  # for 3-bar morning confirmation
    filtered_reasons = {}  # track why signals are filtered

    def _close_position(spot, ts, reason):
        nonlocal pos_dir, daily_pnl, daily_halted, trail_activated
        move = spot - entry_price
        pnl = (D * (move if pos_dir == "CALL" else -move)) * LOT
        trades.append({
            "time": ts, "year": ts.year, "pnl": pnl,
            "dir": pos_dir, "reason": reason,
            "sl_pts": initial_sl_dist,
            "trail": trail_activated,
        })
        daily_pnl += pnl
        pos_dir = None
        trail_activated = False
        if pnl < 0 and daily_pnl <= -DAILY_LOSS_CAP:
            daily_halted = True

    # Track previous day VIX for regime direction detection
    prev_day_vix = 0.0
    day_vix_cache = {}  # cache vix per day for prev-day lookup

    def _compute_sl_tp(spot, direction, atr_val, bar_vix, prev_vix):
        """Compute dynamic SL/TP scaled by ATR + VIX regime."""
        if USE_DYNAMIC_SL_TP and atr_val > 0:
            sl_pts = atr_val * SL_ATR_MULTIPLIER
            tp_pts = atr_val * TP_ATR_MULTIPLIER
        else:
            sl_pts = FALLBACK_SL_POINTS
            tp_pts = FALLBACK_TP_POINTS

        # Apply VIX regime scaling
        if bar_vix > 0:
            regime = classify_vix(bar_vix, prev_vix)
            sl_pts, tp_pts = apply_vix_to_sl_tp(
                sl_pts, tp_pts, regime,
                min_sl=MIN_SL_POINTS,
                max_sl=90.0,   # Wider bounds for high-VIX
                min_tp=MIN_TP_POINTS,
                max_tp=350.0,
            )
        else:
            sl_pts = max(MIN_SL_POINTS, min(MAX_SL_POINTS, sl_pts))
            tp_pts = max(MIN_TP_POINTS, min(MAX_TP_POINTS, tp_pts))
            if tp_pts < sl_pts * 2:
                tp_pts = sl_pts * 2

        if direction == "CALL":
            return spot - sl_pts, spot + tp_pts, sl_pts
        else:
            return spot + sl_pts, spot - tp_pts, sl_pts

    for i in range(len(feat)):
        ts = feat.index[i]
        spot = float(close_arr[i])
        sig = signals[i]
        bar_conf = float(conf_arr[i])
        bar_vix  = float(vix_arr[i]) if vix_arr is not None else 0.0
        bar_atr  = float(atr_arr[i])

        # Reset daily state on new day
        if ts.date() != current_day:
            if current_day is not None and current_day in day_vix_cache:
                prev_day_vix = day_vix_cache[current_day]
            current_day = ts.date()
            daily_pnl = 0.0
            daily_halted = False
            pre1300_signals = 0
            flood_halted = False
            prev_signal = 2
            prev_prev_signal = 2
            if bar_vix > 0:
                day_vix_cache[current_day] = bar_vix

        # Skip rest of day if daily loss cap hit
        if daily_halted:
            if pos_dir:
                _close_position(spot, ts, "DAILY_CAP")
            continue

        # Skip rest of day if signal flood detected
        if flood_halted:
            if pos_dir:
                _close_position(spot, ts, "FLOOD_EXIT")
            continue

        # Time filters: square off at 15:20
        if ts.hour >= 15 and ts.minute >= 20:
            if pos_dir:
                _close_position(spot, ts, "SQUARE_OFF")
            continue

        # Check SL/TP, breakeven, and trailing stop on existing position
        if pos_dir:
            unrealized = (spot - entry_price) if pos_dir == "CALL" else (entry_price - spot)

            # Breakeven stop — once in profit by X pts, move SL to entry+lock
            if USE_BREAKEVEN_STOP and not breakeven_activated and not trail_activated:
                if unrealized >= BREAKEVEN_TRIGGER_PTS:
                    breakeven_activated = True
                    if pos_dir == "CALL":
                        sl_level = max(sl_level, entry_price + BREAKEVEN_LOCK_PTS)
                    else:
                        sl_level = min(sl_level, entry_price - BREAKEVEN_LOCK_PTS)

            # Trailing stop logic
            if USE_TRAILING_STOP and not trail_activated:
                if unrealized >= initial_sl_dist * TRAIL_ACTIVATION_R:
                    trail_activated = True

            if trail_activated:
                lock_amount = unrealized * TRAIL_STEP_PCT
                if pos_dir == "CALL":
                    new_sl = entry_price + lock_amount
                    if new_sl > sl_level:
                        sl_level = new_sl
                else:
                    new_sl = entry_price - lock_amount
                    if new_sl < sl_level:
                        sl_level = new_sl

            # Trailing TP — extend target when momentum is strong
            if USE_TRAILING_TP and unrealized > 0:
                if unrealized > peak_unrealized:
                    peak_unrealized = unrealized
                orig_tp_dist = abs(original_tp_level - entry_price)
                if orig_tp_dist > 0 and unrealized >= orig_tp_dist * TRAIL_TP_ACTIVATION_PCT:
                    extend = peak_unrealized * TRAIL_TP_EXTEND_PCT
                    if pos_dir == "CALL":
                        new_tp = entry_price + peak_unrealized + extend
                        if new_tp > tp_level:
                            tp_level = new_tp
                            tp_trailing = True
                    else:
                        new_tp = entry_price - peak_unrealized - extend
                        if new_tp < tp_level:
                            tp_level = new_tp
                            tp_trailing = True

            # Check SL
            sl_hit = (pos_dir == "CALL" and spot <= sl_level) or \
                     (pos_dir == "PUT" and spot >= sl_level)
            # Check TP
            tp_hit = (pos_dir == "CALL" and spot >= tp_level) or \
                     (pos_dir == "PUT" and spot <= tp_level)

            if sl_hit:
                if trail_activated:
                    reason = "TRAIL_SL"
                elif breakeven_activated:
                    reason = "BE_SL"
                else:
                    reason = "SL"
                # Exit at SL level (stop order fills at stop price, not bar close)
                _close_position(sl_level, ts, reason)
            elif tp_hit:
                # Exit at TP level (limit order fills at limit price)
                _close_position(tp_level, ts, "TP")

        if sig == 2:
            prev_prev_signal = prev_signal
            prev_signal = 2
            continue

        # Filter 1: skip low-confidence signals (higher bar during opening 09:15-09:35)
        effective_conf = MIN_CONF_THRESHOLD
        if ts.hour == 9 and ts.minute < 35:
            effective_conf = OPENING_MIN_CONF
        if bar_conf < effective_conf:
            prev_prev_signal = prev_signal
            prev_signal = sig
            continue

        # Filter 2: Dynamic VIX regime — only hard-block EXTREME+RISING
        if bar_vix > 0:
            regime = classify_vix(bar_vix, prev_day_vix)
            if not regime.tradeable:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue
            # Higher ML confidence bar in elevated VIX
            if bar_conf < regime.min_ml_confidence:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue

        # Filter 3: no new entries at or after 14:30
        if ts.hour > ENTRY_CUTOFF_HOUR or (ts.hour == ENTRY_CUTOFF_HOUR and ts.minute >= ENTRY_CUTOFF_MIN):
            prev_prev_signal = prev_signal
            prev_signal = sig
            continue

        # Filter 4: 2-bar confirmation (3-bar before 10:30 for choppy open)
        if REQUIRE_2BAR_CONFIRM and prev_signal != sig:
            prev_prev_signal = prev_signal
            prev_signal = sig
            continue
        if REQUIRE_3BAR_MORNING and ts.hour < 10 or (ts.hour == 10 and ts.minute < 30):
            if prev_prev_signal != sig:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue

        # Filter 5: Trend alignment — EMA9 vs EMA21
        sd = "CALL" if sig == 0 else "PUT"
        if USE_TREND_FILTER:
            if sd == "CALL" and ema9_arr[i] < ema21_arr[i]:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue
            if sd == "PUT" and ema9_arr[i] > ema21_arr[i]:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue

        # Filter 6: RSI zone — no CALL at overbought, no PUT at oversold
        if USE_RSI_ZONE_FILTER:
            rsi_val = rsi_arr[i]
            if sd == "CALL" and rsi_val > RSI_OB_THRESHOLD:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue
            if sd == "PUT" and rsi_val < RSI_OS_THRESHOLD:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue

        # Filter 7: VWAP alignment — CALL above VWAP, PUT below VWAP
        if USE_VWAP_FILTER:
            if sd == "CALL" and spot < vwap_arr[i]:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue
            if sd == "PUT" and spot > vwap_arr[i]:
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue

        # Count signals before 13:00 and halt if flooded
        if ts.hour < 13:
            pre1300_signals += 1
            if pre1300_signals > SIGNAL_FLOOD_LIMIT:
                flood_halted = True
                if pos_dir:
                    _close_position(spot, ts, "FLOOD_EXIT")
                prev_prev_signal = prev_signal
                prev_signal = sig
                continue

        prev_prev_signal = prev_signal
        prev_signal = sig

        if pos_dir is None:
            pos_dir = sd
            entry_price = spot
            entry_bar_idx = i
            sl_level, tp_level, initial_sl_dist = _compute_sl_tp(spot, sd, bar_atr, bar_vix, prev_day_vix)
            original_tp_level = tp_level
            trail_activated = False
            breakeven_activated = False
            tp_trailing = False
            peak_unrealized = 0.0
        elif pos_dir != sd:
            # Anti-reversal: don't flip within N bars of entry
            if (i - entry_bar_idx) < ANTI_REVERSAL_BARS:
                continue
            _close_position(spot, ts, "REVERSAL")
            pos_dir = sd
            entry_price = spot
            entry_bar_idx = i
            sl_level, tp_level, initial_sl_dist = _compute_sl_tp(spot, sd, bar_atr, bar_vix, prev_day_vix)
            original_tp_level = tp_level
            trail_activated = False
            breakeven_activated = False
            tp_trailing = False
            peak_unrealized = 0.0

    # Close any remaining
    if pos_dir:
        spot = float(feat["close"].iloc[-1])
        _close_position(spot, feat.index[-1], "END")

    # Results
    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        print("  No trades generated!")
        return

    wins = (tdf.pnl > 0).sum()
    losses = (tdf.pnl <= 0).sum()
    total = tdf.pnl.sum()
    avg = tdf.pnl.mean()

    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"  Total trades:  {len(tdf)}")
    print(f"  Winners:       {wins} ({wins/len(tdf)*100:.1f}%)")
    print(f"  Losers:        {losses} ({losses/len(tdf)*100:.1f}%)")
    print(f"  Total P&L:     Rs. {total:+,.0f}")
    print(f"  Avg per trade: Rs. {avg:+,.0f}")
    print(f"  Best trade:    Rs. {tdf.pnl.max():+,.0f}")
    print(f"  Worst trade:   Rs. {tdf.pnl.min():+,.0f}")
    print(f"  CALL trades:   {(tdf.dir == 'CALL').sum()}")
    print(f"  PUT trades:    {(tdf.dir == 'PUT').sum()}")

    # Trailing stop stats
    if "trail" in tdf.columns:
        trail_trades = tdf[tdf["trail"] == True]
        if len(trail_trades) > 0:
            print(f"  Trailing stop:  {len(trail_trades)} trades locked profit")
            print(f"    Trail P&L:   Rs. {trail_trades.pnl.sum():+,.0f} (avg Rs. {trail_trades.pnl.mean():+,.0f})")

    if "sl_pts" in tdf.columns:
        avg_sl = tdf["sl_pts"].mean()
        print(f"  Avg SL used:   {avg_sl:.1f} pts")

    print(f"\n  Exit reasons:")
    for reason, g in tdf.groupby("reason"):
        n = len(g)
        p = g.pnl.sum()
        print(f"    {reason:<15} {n:>5} trades | Rs. {p:>+12,.0f}")

    print(f"\n{'='*65}")
    print(f"  YEARLY BREAKDOWN")
    print(f"{'='*65}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'Total P&L':>14} {'Avg/Trade':>11}")
    print(f"  {'-'*50}")
    for year, g in tdf.groupby("year"):
        n = len(g)
        w = (g.pnl > 0).mean() * 100
        t = g.pnl.sum()
        a = g.pnl.mean()
        print(f"  {year:<6} {n:>7} {w:>5.1f}% Rs.{t:>+12,.0f} Rs.{a:>+9,.0f}")
    print(f"  {'-'*50}")
    print(
        f"  {'TOTAL':<6} {len(tdf):>7} {wins/len(tdf)*100:>5.1f}% "
        f"Rs.{total:>+12,.0f} Rs.{avg:>+9,.0f}"
    )
    print(f"{'='*65}\n")

    return tdf


def debug_probas():
    """Check what probability distribution the model produces — helps tune thresholds."""
    import warnings; warnings.filterwarnings("ignore")
    if not load_model("."):
        print("Model not found"); return

    df_full = _load_csv(DATA_PATH)
    # Use last 500 bars for quick check
    df_sample = df_full.tail(3000)

    try:
        from scripts.train_model_v8 import (
            features_5min, features_15min, features_30min,
            features_60min, features_daily, features_vix,
            merge_htf, merge_daily_onto_5min,
            merge_vix_onto_5min, add_intraday_context
        )
        from core.ml_engine import _fcols, _scaler, _models
        import numpy as np

        df15  = _load_csv(DATA_15M).tail(1500)
        df30  = _load_csv(DATA_30M).tail(750)
        df60  = _load_csv(DATA_60M).tail(375)
        dfday = _load_csv(DATA_DAY, is_daily=True)
        dfvix = _load_csv(DATA_VIX, is_daily=True)

        df_feat = features_5min(df_sample)
        if not df15.empty: df_feat = merge_htf(df_feat, features_15min(df15), "15m")
        if not df30.empty: df_feat = merge_htf(df_feat, features_30min(df30), "30m")
        if not df60.empty: df_feat = merge_htf(df_feat, features_60min(df60), "60m")
        if not dfday.empty:
            dfday = dfday[~dfday.index.duplicated(keep="last")]
            df_feat = merge_daily_onto_5min(df_feat, features_daily(dfday))
        if not dfvix.empty:
            if "close" in dfvix.columns and "vix" not in dfvix.columns:
                dfvix = dfvix.rename(columns={"close":"vix"})
            if "vix" in dfvix.columns:
                df_feat = merge_vix_onto_5min(df_feat, features_vix(dfvix[["vix"]]))
        df_feat = add_intraday_context(df_feat)
        df_feat.dropna(inplace=True)

        for col in [c for c in _fcols if c not in df_feat.columns]:
            df_feat[col] = 0.0

        X = _scaler.transform(df_feat[_fcols].values)
        probas = np.mean([m.predict_proba(X) for m in _models.values()], axis=0)

        print(f"\n=== MODEL PROBABILITY DISTRIBUTION (last {len(probas)} bars) ===")
        print(f"  CALL prob — min={probas[:,0].min():.3f} max={probas[:,0].max():.3f} mean={probas[:,0].mean():.3f} p95={np.percentile(probas[:,0],95):.3f}")
        print(f"  PUT  prob — min={probas[:,1].min():.3f} max={probas[:,1].max():.3f} mean={probas[:,1].mean():.3f} p95={np.percentile(probas[:,1],95):.3f}")
        print(f"  SKIP prob — min={probas[:,2].min():.3f} max={probas[:,2].max():.3f} mean={probas[:,2].mean():.3f}")
        print()

        # Try different thresholds
        for conf in [0.20, 0.25, 0.28, 0.30, 0.35]:
            for skip_ceil in [0.85, 0.80, 0.75, 0.70]:
                calls = ((probas[:,0] >= conf) & (probas[:,0] - probas[:,1] >= 0.02) & (probas[:,2] < skip_ceil)).sum()
                puts  = ((probas[:,1] >= conf) & (probas[:,1] - probas[:,0] >= 0.02) & (probas[:,2] < skip_ceil)).sum()
                if calls + puts > 0:
                    print(f"  conf>={conf} skip<{skip_ceil} → CALL={calls} PUT={puts} total={calls+puts}")

    except Exception as e:
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        debug_probas()
    else:
        y1 = sys.argv[1] if len(sys.argv) > 1 else None
        y2 = sys.argv[2] if len(sys.argv) > 2 else None
        run_backtest(y1, y2)
