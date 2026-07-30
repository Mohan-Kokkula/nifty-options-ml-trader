"""
backtest_options.py — Realistic option-premium backtest ENGINE
================================================================
WHY THIS EXISTS
----------------
The original backtest.py prices every trade as:

        pnl = 0.50 * index_move * lot_size

That assumes a constant 0.50 delta and models ZERO theta decay,
ZERO bid-ask spread, ZERO IV change and ZERO transaction costs.
For an intraday option BUYER, theta + spread + STT are the whole
game — so the old "86.6% win rate / Rs.38.6L" number is a
directional index test, not an options result.

WHAT THIS DOES DIFFERENTLY
--------------------------
* Reuses the EXACT same signal pipeline as backtest.py
  (same features, same model, same SL/TP/trailing/filter logic) so
  this is an apples-to-apples test of the SAME strategy.
* Replaces the P&L engine: every trade is priced as a real option
  using Black-Scholes, with implied volatility CALIBRATED to actual
  NSE bhavcopy option closes (data/nse_bhavcopy/).
* Theta is captured naturally — time-to-expiry shrinks from entry
  to exit, so a trade that is flat on the index still LOSES on the
  option.
* Subtracts realistic costs: bid-ask spread, STT, brokerage,
  exchange transaction charges, SEBI fee, stamp duty and GST.

Each trade is reported three ways so you can see where money goes:
   delta_model  -> what the OLD backtest would have shown
   gross_option -> real option P&L before costs  (delta - theta)
   net_option   -> real option P&L after costs   (the honest number)

This file is also an importable ENGINE. Other scripts (e.g.
backtest_walkforward_options.py) call:
   build_iv_map()                          -> IV calibration
   simulate_trades(feat, signals, probas,  -> run the trade sim
                   iv_by_date, expiries)
   report(tdf, label)                      -> print a result block

USAGE
    python backtest_options.py                 # bhavcopy period 2024-2026
    python backtest_options.py 2025            # one year
    python backtest_options.py 2024 2026       # range

NOTE: NSE bhavcopy option data covers 2024-05-21 onward. For any
dates before that, ATM IV falls back to India VIX as a proxy and
the result is less reliable — the headline number to trust is the
bhavcopy-covered period.
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Reuse the EXACT signal pipeline + config from the original backtest.
from backtest import (
    _load_csv, _build_full_features,
    LOT_SIZE, DATA_PATH,
    USE_DYNAMIC_SL_TP, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
    MIN_SL_POINTS, MAX_SL_POINTS, MIN_TP_POINTS, MAX_TP_POINTS,
    FALLBACK_SL_POINTS, FALLBACK_TP_POINTS,
    USE_BREAKEVEN_STOP, BREAKEVEN_TRIGGER_PTS, BREAKEVEN_LOCK_PTS,
    USE_TRAILING_STOP, TRAIL_ACTIVATION_R, TRAIL_STEP_PCT,
    USE_TRAILING_TP, TRAIL_TP_ACTIVATION_PCT, TRAIL_TP_EXTEND_PCT,
    DAILY_LOSS_CAP, SIGNAL_FLOOD_LIMIT,
    MIN_CONF_THRESHOLD, OPENING_MIN_CONF,
    REQUIRE_2BAR_CONFIRM, REQUIRE_3BAR_MORNING,
    ENTRY_CUTOFF_HOUR, ENTRY_CUTOFF_MIN,
    USE_TREND_FILTER, USE_RSI_ZONE_FILTER, USE_VWAP_FILTER,
    RSI_OB_THRESHOLD, RSI_OS_THRESHOLD, ANTI_REVERSAL_BARS,
)
from core.ml_engine import load_model
from core.vix_regime import classify_vix, apply_vix_to_sl_tp

# ----------------------------------------------------------------------
# Market / cost assumptions  (Indian index F&O, ~2024-2026)
# ----------------------------------------------------------------------
RISK_FREE      = 0.065     # ~6.5% Indian risk-free rate
DIV_YIELD      = 0.012     # ~1.2% Nifty dividend yield
STRIKE_STEP    = 50        # Nifty option strike interval
EXPIRY_HOUR    = 15        # options expire at 15:30 IST
EXPIRY_MIN     = 30
# Research instrumentation (2026-06-16): 0 = naked long (production default,
# behavior unchanged). >0 = price the position as a DEBIT SPREAD — long ATM +
# short leg this many points OTM — to study theta/cost reduction vs capped
# upside. Only the backtest reads this; production trading is unaffected.
SPREAD_HEDGE_WIDTH = 0

# Cost model (per round-trip, 1 lot = LOT_SIZE qty)
BROKERAGE_PER_ORDER = 20.0      # discount-broker flat fee
STT_SELL_RATE       = 0.001000  # 0.10% on sell-side premium
EXCH_TXN_RATE       = 0.00035   # NSE option txn charge, on premium turnover
SEBI_RATE           = 0.000001  # SEBI charges
STAMP_BUY_RATE      = 0.00003   # 0.003% stamp duty on buy-side premium
GST_RATE            = 0.18      # GST on (brokerage + txn + sebi)
# Bid-ask spread: round-trip premium points lost crossing the spread.
SPREAD_PCT          = 0.005     # 0.5% of premium, round trip
SPREAD_FLOOR_PTS    = 0.40      # at least 0.40 premium points round trip

BHAV_PATH = "data/nse_bhavcopy/nifty_options_merged.csv"


# ----------------------------------------------------------------------
# Black-Scholes option pricing + implied vol
# ----------------------------------------------------------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, opt, r=RISK_FREE, q=DIV_YIELD):
    """European option price (Black-Scholes with continuous dividend yield)."""
    if T <= 0 or sigma <= 0:
        if opt == "CE":
            return max(0.0, S - K)
        return max(0.0, K - S)
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vt
    d2 = d1 - vt
    if opt == "CE":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def implied_vol(price, S, K, T, opt, r=RISK_FREE, q=DIV_YIELD):
    """Invert Black-Scholes for IV via bisection. Returns None if not solvable."""
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(0.0, S - K) if opt == "CE" else max(0.0, K - S)
    if price <= intrinsic + 0.05:
        return None
    lo, hi = 0.01, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, opt, r, q) > price:
            hi = mid
        else:
            lo = mid
    iv = 0.5 * (lo + hi)
    if iv < 0.02 or iv > 2.5:
        return None
    return iv


def _years_to_expiry(ts, expiry_dt):
    """Calendar time-to-expiry in years (floored to a tiny positive number)."""
    secs = (expiry_dt - ts).total_seconds()
    return max(secs, 60.0) / (365.0 * 24.0 * 3600.0)


# ----------------------------------------------------------------------
# IV calibration from NSE bhavcopy
# ----------------------------------------------------------------------
def build_iv_map():
    """
    Returns:
      iv_by_date : {date -> ATM implied vol (decimal)} from bhavcopy closes
      expiries   : sorted list of available expiry dates
    """
    p = Path(BHAV_PATH)
    if not p.exists():
        print(f"  WARNING: {BHAV_PATH} not found — IV will fall back to VIX")
        return {}, []

    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    expiries = sorted(df["expiry"].unique())

    iv_by_date = {}
    for d, day in df.groupby("date"):
        future_exp = [e for e in sorted(day["expiry"].unique()) if e >= d]
        if not future_exp:
            continue
        exp = future_exp[0]
        chain = day[day["expiry"] == exp]
        if chain.empty:
            continue
        und = chain["underlying"].replace(0.0, np.nan).median()
        if not np.isfinite(und) or und <= 0:
            continue
        atm = round(und / STRIKE_STEP) * STRIKE_STEP
        expiry_dt = datetime(exp.year, exp.month, exp.day, EXPIRY_HOUR, EXPIRY_MIN)
        eod = datetime(d.year, d.month, d.day, 15, 30)
        T = _years_to_expiry(eod, expiry_dt)
        ivs = []
        for opt in ("CE", "PE"):
            row = chain[(chain["strike"] == atm) & (chain["opt_type"] == opt)]
            if row.empty:
                continue
            px = float(row["close"].iloc[0])
            iv = implied_vol(px, und, atm, T, opt)
            if iv is not None:
                ivs.append(iv)
        if ivs:
            iv_by_date[d] = float(np.mean(ivs))

    return iv_by_date, expiries


def _nearest_expiry(d, expiries):
    """Nearest listed expiry >= date d; falls back to next Tuesday if none."""
    for e in expiries:
        if e >= d:
            return e
    nd = d
    while nd.weekday() != 1:   # Tuesday = 1
        nd = nd + timedelta(days=1)
    return nd


# ----------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------
def round_trip_cost(prem_entry, prem_exit, qty):
    """Total round-trip cost in rupees: spread + STT + brokerage + fees + GST."""
    spread_pts = max(SPREAD_FLOOR_PTS, SPREAD_PCT * prem_entry)
    spread_cost = spread_pts * qty
    brokerage = BROKERAGE_PER_ORDER * 2
    stt = STT_SELL_RATE * prem_exit * qty
    turnover = (prem_entry + prem_exit) * qty
    txn = EXCH_TXN_RATE * turnover
    sebi = SEBI_RATE * turnover
    stamp = STAMP_BUY_RATE * prem_entry * qty
    gst = GST_RATE * (brokerage + txn + sebi)
    return spread_cost + brokerage + stt + txn + sebi + stamp + gst


# ----------------------------------------------------------------------
# TRADE SIMULATION ENGINE  (reusable — called by walk-forward too)
# ----------------------------------------------------------------------
def simulate_trades(feat, signals, probas, iv_by_date, expiries, lot=LOT_SIZE):
    """
    Run the strategy's exact SL/TP/trailing/filter logic over `feat`,
    pricing every trade as a real Black-Scholes option.

    feat       : feature DataFrame (datetime index, has 'close','vix_level',
                 ideally 'high'/'low'/'volume')
    signals    : np.array of 0=CALL / 1=PUT / 2=SKIP, aligned to feat
    probas     : Nx3 model probabilities, aligned to feat
    iv_by_date : {date -> ATM IV} from build_iv_map()
    expiries   : sorted expiry-date list from build_iv_map()

    Returns a per-trade DataFrame.
    """
    LOT = lot
    QTY = lot

    close_arr = feat["close"].values
    if "high" in feat.columns and "low" in feat.columns:
        high_arr = feat["high"].values
        low_arr  = feat["low"].values
    else:
        high_arr = close_arr * 1.0002
        low_arr  = close_arr * 0.9998

    tr_arr = np.maximum(
        high_arr - low_arr,
        np.maximum(np.abs(high_arr - np.roll(close_arr, 1)),
                   np.abs(low_arr - np.roll(close_arr, 1))))
    tr_arr[0] = high_arr[0] - low_arr[0]
    atr_arr = pd.Series(tr_arr).rolling(14).mean().fillna(25.0).values

    vix_arr = feat["vix_level"].values if "vix_level" in feat.columns else None
    conf_arr = probas.max(axis=1)

    _close_s = pd.Series(close_arr, index=feat.index)
    ema9_arr  = _close_s.ewm(span=9, adjust=False).mean().values
    ema21_arr = _close_s.ewm(span=21, adjust=False).mean().values
    _delta = _close_s.diff()
    _gain = _delta.clip(lower=0).rolling(14).mean()
    _loss = (-_delta.clip(upper=0)).rolling(14).mean()
    _rs = _gain / _loss.replace(0, np.nan)
    rsi_arr = (100 - (100 / (1 + _rs))).fillna(50).values

    if "volume" in feat.columns:
        vol_arr = feat["volume"].values
    else:
        vol_arr = np.ones(len(feat))
    vwap_arr = np.zeros(len(feat))
    _cpv = _cv = 0.0
    _pd = None
    for _i in range(len(feat)):
        _d = feat.index[_i].date()
        if _d != _pd:
            _cpv = _cv = 0.0
            _pd = _d
        _cpv += close_arr[_i] * vol_arr[_i]
        _cv  += vol_arr[_i]
        vwap_arr[_i] = _cpv / _cv if _cv > 0 else close_arr[_i]

    trades = []
    pos_dir = None
    entry_price = sl_level = tp_level = original_tp_level = 0.0
    initial_sl_dist = 0.0
    trail_activated = breakeven_activated = tp_trailing = False
    peak_unrealized = 0.0
    entry_bar_idx = 0

    e_ts = None
    e_strike = 0.0
    e_opt = "CE"
    e_iv = 0.15
    e_expiry_dt = None
    e_premium = 0.0
    e_strike_short = 0.0      # debit-spread short leg (0 = naked)
    e_premium_short = 0.0

    current_day = None
    daily_pnl = 0.0
    daily_halted = False
    pre1300_signals = 0
    flood_halted = False
    prev_signal = prev_prev_signal = 2
    prev_day_vix = 0.0
    day_vix_cache = {}

    def _iv_for(d, bar_vix):
        iv = iv_by_date.get(d)
        if iv is not None:
            return iv
        v = (bar_vix / 100.0) if bar_vix and bar_vix > 0 else 0.15
        return min(0.60, max(0.06, v))

    def _open(spot, ts, direction, bar_vix):
        nonlocal pos_dir, entry_price, entry_bar_idx
        nonlocal e_ts, e_strike, e_opt, e_iv, e_expiry_dt, e_premium
        nonlocal e_strike_short, e_premium_short
        pos_dir = direction
        entry_price = spot
        e_ts = ts
        e_opt = "CE" if direction == "CALL" else "PE"
        e_strike = round(spot / STRIKE_STEP) * STRIKE_STEP
        d = ts.date()
        exp = _nearest_expiry(d, expiries)
        e_expiry_dt = datetime(exp.year, exp.month, exp.day, EXPIRY_HOUR, EXPIRY_MIN)
        e_iv = _iv_for(d, bar_vix)
        T = _years_to_expiry(ts.to_pydatetime(), e_expiry_dt)
        e_premium = bs_price(spot, e_strike, T, e_iv, e_opt)
        if SPREAD_HEDGE_WIDTH > 0:
            e_strike_short = (e_strike + SPREAD_HEDGE_WIDTH if e_opt == "CE"
                              else e_strike - SPREAD_HEDGE_WIDTH)
            e_premium_short = bs_price(spot, e_strike_short, T, e_iv, e_opt)
        else:
            e_strike_short = 0.0
            e_premium_short = 0.0

    def _close_position(exit_spot, ts, reason):
        nonlocal pos_dir, daily_pnl, daily_halted, trail_activated
        T_exit = _years_to_expiry(ts.to_pydatetime(), e_expiry_dt)
        prem_exit = bs_price(exit_spot, e_strike, T_exit, e_iv, e_opt)

        if SPREAD_HEDGE_WIDTH > 0 and e_strike_short > 0:
            prem_exit_short = bs_price(exit_spot, e_strike_short, T_exit, e_iv, e_opt)
            entry_net = e_premium - e_premium_short
            exit_net  = prem_exit - prem_exit_short
            gross = (exit_net - entry_net) * QTY
            # both legs incur round-trip cost (2x brokerage, but cheaper premia)
            cost  = (round_trip_cost(e_premium, prem_exit, QTY)
                     + round_trip_cost(e_premium_short, prem_exit_short, QTY))
            rec_entry, rec_exit = entry_net, exit_net
        else:
            gross = (prem_exit - e_premium) * QTY
            cost  = round_trip_cost(e_premium, prem_exit, QTY)
            rec_entry, rec_exit = e_premium, prem_exit
        net   = gross - cost

        move = exit_spot - entry_price
        delta_model = (0.50 * (move if pos_dir == "CALL" else -move)) * LOT

        trades.append({
            "time": ts, "year": ts.year, "dir": pos_dir, "reason": reason,
            "net_option": net, "gross_option": gross, "cost": cost,
            "delta_model": delta_model,
            "prem_entry": rec_entry, "prem_exit": rec_exit,
            "iv": e_iv, "sl_pts": initial_sl_dist, "trail": trail_activated,
            "entry_time": e_ts,
            "hold_min": (ts - e_ts).total_seconds() / 60.0,
        })
        daily_pnl += net
        pos_dir = None
        trail_activated = False
        if net < 0 and daily_pnl <= -DAILY_LOSS_CAP:
            daily_halted = True

    def _compute_sl_tp(spot, direction, atr_val, bar_vix, prev_vix):
        if USE_DYNAMIC_SL_TP and atr_val > 0:
            sl_pts = atr_val * SL_ATR_MULTIPLIER
            tp_pts = atr_val * TP_ATR_MULTIPLIER
        else:
            sl_pts = FALLBACK_SL_POINTS
            tp_pts = FALLBACK_TP_POINTS
        if bar_vix > 0:
            regime = classify_vix(bar_vix, prev_vix)
            sl_pts, tp_pts = apply_vix_to_sl_tp(
                sl_pts, tp_pts, regime,
                min_sl=MIN_SL_POINTS, max_sl=90.0,
                min_tp=MIN_TP_POINTS, max_tp=350.0)
        else:
            sl_pts = max(MIN_SL_POINTS, min(MAX_SL_POINTS, sl_pts))
            tp_pts = max(MIN_TP_POINTS, min(MAX_TP_POINTS, tp_pts))
            if tp_pts < sl_pts * 2:
                tp_pts = sl_pts * 2
        if direction == "CALL":
            return spot - sl_pts, spot + tp_pts, sl_pts
        return spot + sl_pts, spot - tp_pts, sl_pts

    for i in range(len(feat)):
        ts = feat.index[i]
        spot = float(close_arr[i])
        sig = signals[i]
        bar_conf = float(conf_arr[i])
        bar_vix  = float(vix_arr[i]) if vix_arr is not None else 0.0
        bar_atr  = float(atr_arr[i])

        if ts.date() != current_day:
            if current_day is not None and current_day in day_vix_cache:
                prev_day_vix = day_vix_cache[current_day]
            current_day = ts.date()
            daily_pnl = 0.0
            daily_halted = False
            pre1300_signals = 0
            flood_halted = False
            prev_signal = prev_prev_signal = 2
            if bar_vix > 0:
                day_vix_cache[current_day] = bar_vix

        if daily_halted:
            if pos_dir:
                _close_position(spot, ts, "DAILY_CAP")
            continue
        if flood_halted:
            if pos_dir:
                _close_position(spot, ts, "FLOOD_EXIT")
            continue
        if ts.hour >= 15 and ts.minute >= 20:
            if pos_dir:
                _close_position(spot, ts, "SQUARE_OFF")
            continue

        if pos_dir:
            bar_hi = float(high_arr[i])
            bar_lo = float(low_arr[i])
            # --- 1) Intrabar SL/TP check against the levels carried in from
            #     the PREVIOUS bar. A real stop/limit order fires intrabar on
            #     the wick, not only on the close. Pessimistic tie-break: if a
            #     bar's range spans BOTH the stop and the target, assume the
            #     STOP filled first.
            if pos_dir == "CALL":
                sl_hit = bar_lo <= sl_level
                tp_hit = bar_hi >= tp_level
            else:
                sl_hit = bar_hi >= sl_level
                tp_hit = bar_lo <= tp_level
            if sl_hit:
                reason = "TRAIL_SL" if trail_activated else ("BE_SL" if breakeven_activated else "SL")
                _close_position(sl_level, ts, reason)
            elif tp_hit:
                _close_position(tp_level, ts, "TP")
            else:
                # --- 2) Still in the trade — ratchet breakeven / trailing
                #     stop / trailing TP using this bar's CLOSE, so the new
                #     levels only apply from the NEXT bar (no within-bar
                #     lookahead: you cannot trail to a high and be stopped at
                #     the low of the same bar).
                unrealized = (spot - entry_price) if pos_dir == "CALL" else (entry_price - spot)
                if USE_BREAKEVEN_STOP and not breakeven_activated and not trail_activated:
                    if unrealized >= BREAKEVEN_TRIGGER_PTS:
                        breakeven_activated = True
                        if pos_dir == "CALL":
                            sl_level = max(sl_level, entry_price + BREAKEVEN_LOCK_PTS)
                        else:
                            sl_level = min(sl_level, entry_price - BREAKEVEN_LOCK_PTS)
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
                if USE_TRAILING_TP and unrealized > 0:
                    if unrealized > peak_unrealized:
                        peak_unrealized = unrealized
                    orig_tp_dist = abs(original_tp_level - entry_price)
                    if orig_tp_dist > 0 and unrealized >= orig_tp_dist * TRAIL_TP_ACTIVATION_PCT:
                        extend = peak_unrealized * TRAIL_TP_EXTEND_PCT
                        if pos_dir == "CALL":
                            new_tp = entry_price + peak_unrealized + extend
                            if new_tp > tp_level:
                                tp_level = new_tp; tp_trailing = True
                        else:
                            new_tp = entry_price - peak_unrealized - extend
                            if new_tp < tp_level:
                                tp_level = new_tp; tp_trailing = True

        if sig == 2:
            prev_prev_signal = prev_signal
            prev_signal = 2
            continue

        effective_conf = MIN_CONF_THRESHOLD
        if ts.hour == 9 and ts.minute < 35:
            effective_conf = OPENING_MIN_CONF
        if bar_conf < effective_conf:
            prev_prev_signal = prev_signal; prev_signal = sig
            continue

        if bar_vix > 0:
            regime = classify_vix(bar_vix, prev_day_vix)
            if not regime.tradeable:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue
            if bar_conf < regime.min_ml_confidence:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue

        if ts.hour > ENTRY_CUTOFF_HOUR or (ts.hour == ENTRY_CUTOFF_HOUR and ts.minute >= ENTRY_CUTOFF_MIN):
            prev_prev_signal = prev_signal; prev_signal = sig
            continue

        if REQUIRE_2BAR_CONFIRM and prev_signal != sig:
            prev_prev_signal = prev_signal; prev_signal = sig
            continue
        if REQUIRE_3BAR_MORNING and ts.hour < 10 or (ts.hour == 10 and ts.minute < 30):
            if prev_prev_signal != sig:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue

        sd = "CALL" if sig == 0 else "PUT"
        if USE_TREND_FILTER:
            if sd == "CALL" and ema9_arr[i] < ema21_arr[i]:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue
            if sd == "PUT" and ema9_arr[i] > ema21_arr[i]:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue
        if USE_RSI_ZONE_FILTER:
            rsi_val = rsi_arr[i]
            if sd == "CALL" and rsi_val > RSI_OB_THRESHOLD:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue
            if sd == "PUT" and rsi_val < RSI_OS_THRESHOLD:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue
        if USE_VWAP_FILTER:
            if sd == "CALL" and spot < vwap_arr[i]:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue
            if sd == "PUT" and spot > vwap_arr[i]:
                prev_prev_signal = prev_signal; prev_signal = sig
                continue

        if ts.hour < 13:
            pre1300_signals += 1
            if pre1300_signals > SIGNAL_FLOOD_LIMIT:
                flood_halted = True
                if pos_dir:
                    _close_position(spot, ts, "FLOOD_EXIT")
                prev_prev_signal = prev_signal; prev_signal = sig
                continue

        prev_prev_signal = prev_signal
        prev_signal = sig

        if pos_dir is None:
            sl_level, tp_level, initial_sl_dist = _compute_sl_tp(spot, sd, bar_atr, bar_vix, prev_day_vix)
            original_tp_level = tp_level
            trail_activated = breakeven_activated = tp_trailing = False
            peak_unrealized = 0.0
            entry_bar_idx = i
            _open(spot, ts, sd, bar_vix)
        elif pos_dir != sd:
            if (i - entry_bar_idx) < ANTI_REVERSAL_BARS:
                continue
            _close_position(spot, ts, "REVERSAL")
            sl_level, tp_level, initial_sl_dist = _compute_sl_tp(spot, sd, bar_atr, bar_vix, prev_day_vix)
            original_tp_level = tp_level
            trail_activated = breakeven_activated = tp_trailing = False
            peak_unrealized = 0.0
            entry_bar_idx = i
            _open(spot, ts, sd, bar_vix)

    if pos_dir:
        _close_position(float(close_arr[-1]), feat.index[-1], "END")

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def report(tdf, label="RESULTS", yearly=True):
    """Print a result block for a per-trade DataFrame from simulate_trades()."""
    if tdf is None or len(tdf) == 0:
        print("  No trades generated!")
        return

    def _rs(x):
        return f"Rs.{x:+,.0f}"

    n = len(tdf)
    net_total   = tdf.net_option.sum()
    gross_total = tdf.gross_option.sum()
    delta_total = tdf.delta_model.sum()
    cost_total  = tdf.cost.sum()
    wins   = int((tdf.net_option > 0).sum())
    losses = n - wins
    wins_delta = int((tdf.delta_model > 0).sum())

    print(f"\n{'='*70}")
    print(f"  {label}  ({n} trades)")
    print(f"{'='*70}")
    print(f"  {'':22}{'WIN%':>8}{'TOTAL P&L':>18}{'AVG/TRADE':>15}")
    print(f"  {'-'*61}")
    print(f"  {'OLD delta-model':22}{wins_delta/n*100:>7.1f}%"
          f"{_rs(delta_total):>18}{_rs(delta_total/n):>15}")
    print(f"  {'Real option (gross)':22}{'':>8}"
          f"{_rs(gross_total):>18}{_rs(gross_total/n):>15}")
    print(f"  {'Real option (NET)':22}{wins/n*100:>7.1f}%"
          f"{_rs(net_total):>18}{_rs(net_total/n):>15}")
    print(f"  {'-'*61}")
    print(f"  Winners: {wins} ({wins/n*100:.1f}%)   Losers: {losses} ({losses/n*100:.1f}%)")
    print(f"  Theta/IV/gamma drag (delta - gross): Rs.{delta_total - gross_total:+,.0f}")
    print(f"  Transaction-cost drag             : Rs.{-cost_total:+,.0f}")
    print(f"  Best NET trade : Rs.{tdf.net_option.max():+,.0f}")
    print(f"  Worst NET trade: Rs.{tdf.net_option.min():+,.0f}")
    print(f"  Avg entry premium: Rs.{tdf.prem_entry.mean():.1f}  "
          f"Avg hold: {tdf.hold_min.mean():.0f} min  "
          f"Avg cost/trade: Rs.{cost_total/n:.0f}")

    profit = tdf[tdf.net_option > 0].net_option.sum()
    loss   = -tdf[tdf.net_option <= 0].net_option.sum()
    pf = profit / loss if loss > 0 else float("inf")
    print(f"  Profit factor (net): {pf:.2f}")

    print(f"\n  Exit reasons (NET P&L):")
    for reason, g in tdf.groupby("reason"):
        print(f"    {reason:<14}{len(g):>5} trades   Rs.{g.net_option.sum():>+13,.0f}")

    # Entry-hour breakdown — does the 09:15-11:00 morning block earn its keep?
    # Grouped by ENTRY time (when the trade was opened), not exit time.
    print(f"\n  Entry hour (NET P&L)  [morning block covers 09:xx + 10:xx]:")
    if "entry_time" in tdf.columns:
        hrs = pd.to_datetime(tdf["entry_time"]).dt.hour
    else:  # fallback for older CSVs: exit time minus holding minutes
        hrs = (pd.to_datetime(tdf["time"])
               - pd.to_timedelta(tdf["hold_min"], unit="m")).dt.hour
    for h in sorted(hrs.unique()):
        g = tdf[hrs == h]
        gw = (g.net_option > 0).mean() * 100
        tag = "  <- blocked live" if h < 11 else ""
        print(f"    {h:02d}:00-{h:02d}:59 {len(g):>5} trades  win {gw:>3.0f}%  "
              f"{_rs(g.net_option.sum()):>14}  avg {_rs(g.net_option.mean())}{tag}")

    if yearly:
        print(f"\n  {'Year':<6}{'Trades':>8}{'Net Win%':>10}{'Net P&L':>16}{'Avg/Trade':>13}")
        print(f"  {'-'*52}")
        for year, g in tdf.groupby("year"):
            gw = (g.net_option > 0).mean() * 100
            print(f"  {year:<6}{len(g):>8}{gw:>9.1f}%"
                  f"{_rs(g.net_option.sum()):>16}"
                  f"{_rs(g.net_option.mean()):>13}")
        print(f"  {'-'*52}")
        print(f"  {'TOTAL':<6}{n:>8}{wins/n*100:>9.1f}%"
              f"{_rs(net_total):>16}{_rs(net_total/n):>13}")
    print(f"{'='*70}\n")


# ----------------------------------------------------------------------
# Standalone backtest (in-sample — uses the already-trained model)
# ----------------------------------------------------------------------
def run_backtest(year_start=None, year_end=None):
    if not load_model("."):
        print("ERROR: Model not found in models/")
        return

    print(f"\n{'='*70}")
    print(f"  NIFTY OPTION-PREMIUM BACKTEST  (Black-Scholes, bhavcopy-calibrated)")
    print(f"  *** IN-SAMPLE: uses the pre-trained model. For a true")
    print(f"  *** out-of-sample test run backtest_walkforward_options.py")
    print(f"{'='*70}")

    print("  Calibrating ATM implied vol from NSE bhavcopy...")
    iv_by_date, expiries = build_iv_map()
    if iv_by_date:
        ivs = list(iv_by_date.values())
        print(f"  Calibrated {len(iv_by_date)} trading days | "
              f"IV range {min(ivs)*100:.1f}%-{max(ivs)*100:.1f}% "
              f"(median {np.median(ivs)*100:.1f}%)")
    else:
        print("  No bhavcopy IV — using VIX fallback for all dates")

    df_full = _load_csv(DATA_PATH)
    signals, probas, feat = _build_full_features(df_full, year_start, year_end)

    nc  = int((signals == 0).sum())
    np_ = int((signals == 1).sum())
    ns  = int((signals == 2).sum())
    print(f"  Signals: CALL={nc:,} | PUT={np_:,} | SKIP={ns:,}")
    print(f"  Period:  {feat.index[0].date()} -> {feat.index[-1].date()} "
          f"({len(feat):,} bars)")
    print(f"  Costs:   spread={SPREAD_PCT*100:.1f}% (floor {SPREAD_FLOOR_PTS}pt) | "
          f"STT 0.10% sell | brokerage Rs.{BROKERAGE_PER_ORDER:.0f}/order | "
          f"txn {EXCH_TXN_RATE*100:.3f}% | GST {GST_RATE*100:.0f}%")
    print("=" * 70)

    tdf = simulate_trades(feat, signals, probas, iv_by_date, expiries)
    report(tdf, label="RESULTS (in-sample)")

    out = "logs/backtest_options_trades.csv"
    try:
        tdf.to_csv(out, index=False)
        print(f"  Per-trade detail written to {out}")
    except Exception as e:
        print(f"  (could not write trade CSV: {e})")
    return tdf


if __name__ == "__main__":
    y1 = sys.argv[1] if len(sys.argv) > 1 else "2024"
    y2 = sys.argv[2] if len(sys.argv) > 2 else "2026"
    run_backtest(y1, y2)
