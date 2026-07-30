"""
honest_backtest.py — Reality-check backtester (v2 — multi-TF aware).

Replays the EXACT live decision pipeline against historical bars. Critically,
it uses the same multi-timeframe feature engineering as production:
  • 5m + 15m + 30m + 60m + daily + VIX
  • Same V9 features_5min/15min/30min/60min/daily functions
  • Same predict_precomputed() call

Then simulates trades with realistic execution + costs.

USAGE:
    python scripts/honest_backtest.py --date 2026-04-27 --diagnose
    python scripts/honest_backtest.py --date 2026-04-25 --days 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("backtest")
log.setLevel(logging.INFO)

# Verbose mode (set via --diagnose)
VERBOSE = False


# ──────────────────────────────────────────────────────────────────────
# Multi-TF data loader
# ──────────────────────────────────────────────────────────────────────

class MultiTfBars:
    """Fetch and cache 5/15/30/60/daily/VIX bars; provide truncated views."""

    def __init__(self):
        from core.tv_fetcher import get_tv_fetcher
        tv = get_tv_fetcher()
        if tv is None:
            raise RuntimeError("TV fetcher unavailable")
        self.tv = tv
        log.info("Fetching multi-TF history from TradingView...")
        self.df5 = self._safe_get(lambda: tv.get_nifty_5min(n_bars=2000))
        self.df15 = self._safe_get(lambda: tv.get_nifty_15min(n_bars=800))
        self.df30 = self._safe_get(lambda: tv.get_nifty_30min(n_bars=500))
        self.df60 = self._safe_get(lambda: tv.get_nifty_60min(n_bars=300))
        self.df_day = self._safe_get(
            lambda: tv.get_ohlcv("NIFTY", "NSE", interval_minutes="D", n_bars=120)
        )
        # VIX (optional)
        self.df_vix = self._safe_get(
            lambda: tv.get_ohlcv("INDIAVIX", "NSE", interval_minutes="D", n_bars=60)
        )
        if self.df_vix is None or self.df_vix.empty:
            self.df_vix = self._safe_get(
                lambda: tv.get_ohlcv("INDIA VIX", "NSE", interval_minutes="D", n_bars=60)
            )
        if self.df_vix is not None and not self.df_vix.empty:
            self.df_vix = self.df_vix.rename(columns={"close": "vix"})[["vix"]]

        log.info(
            f"TV bars loaded: 5m={len(self.df5)}, 15m={len(self.df15)}, "
            f"30m={len(self.df30)}, 60m={len(self.df60)}, "
            f"daily={len(self.df_day)}, vix={len(self.df_vix) if self.df_vix is not None else 0}"
        )

    @staticmethod
    def _safe_get(fn):
        try:
            df = fn()
            if df is None:
                return pd.DataFrame()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            log.warning(f"TV fetch failed: {e}")
            return pd.DataFrame()

    def slice_until(self, ts: pd.Timestamp) -> dict:
        """Return DataFrames truncated at ts (inclusive)."""
        return {
            "df5":  self.df5[self.df5.index <= ts] if not self.df5.empty else pd.DataFrame(),
            "df15": self.df15[self.df15.index <= ts] if not self.df15.empty else pd.DataFrame(),
            "df30": self.df30[self.df30.index <= ts] if not self.df30.empty else pd.DataFrame(),
            "df60": self.df60[self.df60.index <= ts] if not self.df60.empty else pd.DataFrame(),
            "df_day": self.df_day[self.df_day.index <= ts] if not self.df_day.empty else pd.DataFrame(),
            "df_vix": self.df_vix[self.df_vix.index <= ts] if (self.df_vix is not None and not self.df_vix.empty) else pd.DataFrame(),
        }


# ──────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors production)
# ──────────────────────────────────────────────────────────────────────

_FE_FUNCS = None


def _get_fe():
    """Lazy-import the V9 feature engineering functions."""
    global _FE_FUNCS
    if _FE_FUNCS is not None:
        return _FE_FUNCS
    from core.ml_engine import _loaded_version as ver
    if ver == "V9":
        from scripts.train_model_v9 import (
            features_5min, features_15min, features_30min, features_60min,
            features_daily, features_vix, merge_htf, merge_daily_onto_5min,
            merge_vix_onto_5min, add_intraday_context,
        )
    else:
        from scripts.train_model_v8 import (
            features_5min, features_15min, features_30min, features_60min,
            features_daily, features_vix, merge_htf, merge_daily_onto_5min,
            merge_vix_onto_5min, add_intraday_context,
        )
    _FE_FUNCS = dict(
        f5=features_5min, f15=features_15min, f30=features_30min,
        f60=features_60min, fday=features_daily, fvix=features_vix,
        merge_htf=merge_htf, merge_daily=merge_daily_onto_5min,
        merge_vix=merge_vix_onto_5min, intraday=add_intraday_context,
    )
    return _FE_FUNCS


def engineer_features(slices: dict) -> pd.DataFrame:
    """Produce the full feature DataFrame matching production."""
    fe = _get_fe()
    df5 = slices["df5"]
    if df5.empty or len(df5) < 55:
        return pd.DataFrame()

    df_feat = fe["f5"](df5)

    if not slices["df15"].empty:
        df_feat = fe["merge_htf"](df_feat, fe["f15"](slices["df15"]), "15m")
    if not slices["df30"].empty:
        df_feat = fe["merge_htf"](df_feat, fe["f30"](slices["df30"]), "30m")
    if not slices["df60"].empty:
        df_feat = fe["merge_htf"](df_feat, fe["f60"](slices["df60"]), "60m")
    if not slices["df_day"].empty:
        day = slices["df_day"].copy()
        day = day[~day.index.duplicated(keep="last")]
        df_feat = fe["merge_daily"](df_feat, fe["fday"](day))
    if not slices["df_vix"].empty:
        try:
            df_feat = fe["merge_vix"](df_feat, fe["fvix"](slices["df_vix"]))
        except Exception:
            pass

    df_feat = fe["intraday"](df_feat)
    df_feat.ffill(inplace=True)
    df_feat.fillna(0, inplace=True)
    return df_feat


# ──────────────────────────────────────────────────────────────────────
# Decision pipeline (mirrors live pilot)
# ──────────────────────────────────────────────────────────────────────

def make_decision(slices: dict, day_high: float, day_low: float) -> dict:
    from core import ml_engine
    from core.ml_engine import _fcols, predict_precomputed

    if not ml_engine.is_ready():
        return {"decision": "WAIT", "skip_reason": "model not loaded"}

    df_feat = engineer_features(slices)
    if df_feat.empty:
        return {"decision": "WAIT", "skip_reason": "feature engineering empty"}

    # Fill missing features
    if _fcols:
        for col in _fcols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

    spot = float(slices["df5"]["close"].iloc[-1])

    signal, proba, conf, summary = predict_precomputed(df_feat.tail(60))
    sig_name = ["CALL", "PUT", "SKIP"][signal]
    rsi = float(summary.get("rsi14", 50))
    atr = float(summary.get("atr_5m", 30) or summary.get("atr14", 30) or 30)

    result = {
        "time": slices["df5"].index[-1],
        "spot": spot,
        "rsi": rsi,
        "atr": atr,
        "ml_signal": sig_name,
        "ml_proba": [float(p) for p in proba],
        "ml_conf_pct": round(conf * 100, 1),
        "decision": "WAIT",
        "skip_reason": "",
    }

    if signal == 2:
        result["skip_reason"] = "ML returned SKIP"
        return result

    direction = "CALL" if signal == 0 else "PUT"
    option_type = "CE" if signal == 0 else "PE"

    # Trap detector
    try:
        from core.trap_detector import get_trap_detector
        trap = get_trap_detector().is_trap(
            option_type=option_type, spot=spot,
            ml_indicators={"rsi_14": rsi, "atr": atr,
                           "pa_vwap_distance": float(summary.get("pa_vwap_distance", 0))},
        )
        if trap.is_trap:
            result["skip_reason"] = f"TRAP {trap.reason}"
            return result
    except Exception as e:
        log.debug(f"trap check failed: {e}")

    # Regime
    try:
        from core.regime_engine import get_regime_engine
        regime = get_regime_engine().classify(
            slices["df5"], spot, day_high, day_low,
            ref_time=slices["df5"].index[-1].to_pydatetime(),
        )
        result["regime"] = regime.name
        result["regime_floor"] = regime.confidence_floor

        effective_min = 60
        if regime.favored_direction == direction:
            effective_min = max(50, min(effective_min, regime.confidence_floor))
        elif regime.favored_direction in ("CALL", "PUT"):
            effective_min = min(85, effective_min + 10)
        else:
            effective_min = max(effective_min, regime.confidence_floor)
        result["effective_min_pct"] = effective_min

        if result["ml_conf_pct"] < effective_min:
            result["skip_reason"] = (
                f"low_conf {result['ml_conf_pct']}%<{effective_min}% "
                f"(regime={regime.name})"
            )
            return result
    except Exception as e:
        log.debug(f"regime check failed: {e}")
        # Fallback: 60% threshold
        if result["ml_conf_pct"] < 60:
            result["skip_reason"] = f"low_conf {result['ml_conf_pct']}%<60%"
            return result

    result["decision"] = direction
    return result


# ──────────────────────────────────────────────────────────────────────
# Trade simulator
# ──────────────────────────────────────────────────────────────────────

SLIPPAGE_TICKS = 2
TICK_SIZE = 0.05
BROKERAGE_RT = 40.0
LOT_SIZE = 75
SL_ATR_MULT = 1.0
TP_ATR_MULT = 2.0
MAX_HOLD_MIN = 60


def simulate_trade(direction: str, entry_idx: int, day_bars: pd.DataFrame, atr: float) -> dict:
    entry_price = float(day_bars["close"].iloc[entry_idx]) + (
        SLIPPAGE_TICKS * TICK_SIZE if direction == "CALL" else -SLIPPAGE_TICKS * TICK_SIZE
    )
    sl_pts = SL_ATR_MULT * atr
    tp_pts = TP_ATR_MULT * atr

    if direction == "CALL":
        sl_price = entry_price - sl_pts
        tp_price = entry_price + tp_pts
    else:
        sl_price = entry_price + sl_pts
        tp_price = entry_price - tp_pts

    max_bars = MAX_HOLD_MIN // 5
    end_idx = min(entry_idx + max_bars + 1, len(day_bars))
    exit_price = None
    exit_reason = None
    for k in range(entry_idx + 1, end_idx):
        h = float(day_bars["high"].iloc[k])
        l = float(day_bars["low"].iloc[k])
        if direction == "CALL":
            if l <= sl_price:
                exit_price, exit_reason = sl_price, "SL"
                break
            if h >= tp_price:
                exit_price, exit_reason = tp_price, "TP"
                break
        else:
            if h >= sl_price:
                exit_price, exit_reason = sl_price, "SL"
                break
            if l <= tp_price:
                exit_price, exit_reason = tp_price, "TP"
                break
    if exit_price is None:
        exit_price = float(day_bars["close"].iloc[end_idx - 1])
        exit_reason = "TIME"

    exit_price -= SLIPPAGE_TICKS * TICK_SIZE if direction == "CALL" else -SLIPPAGE_TICKS * TICK_SIZE

    pnl_pts = (exit_price - entry_price) if direction == "CALL" else (entry_price - exit_price)
    pnl_rupees = pnl_pts * LOT_SIZE - BROKERAGE_RT

    return {
        "direction": direction,
        "entry_price": round(entry_price, 1),
        "exit_price": round(exit_price, 1),
        "pnl_pts": round(pnl_pts, 1),
        "pnl_rupees": round(pnl_rupees, 0),
        "exit_reason": exit_reason,
        "entry_time": day_bars.index[entry_idx],
    }


# ──────────────────────────────────────────────────────────────────────
# Backtest loop
# ──────────────────────────────────────────────────────────────────────

def run_backtest(date: str, mtf: MultiTfBars, diagnose: bool = False) -> dict:
    target = pd.to_datetime(date).date()

    if mtf.df5.empty:
        return {"date": date, "error": "no 5min data"}

    day_bars = mtf.df5[mtf.df5.index.date == target]
    if day_bars.empty:
        return {"date": date, "error": "no bars for date"}

    market_open = pd.Timestamp(f"{date} 09:15:00")
    market_close = pd.Timestamp(f"{date} 15:15:00")
    day_bars = day_bars[(day_bars.index >= market_open) & (day_bars.index <= market_close)]
    if len(day_bars) < 30:
        return {"date": date, "error": f"only {len(day_bars)} market-hour bars"}

    decisions = []
    trades = []
    cooldown_until = None
    last_trade_dir = None
    last_trade_price = None
    same_dir_cooldown_until = None
    MAX_TRADES_PER_DAY = 5

    for i in range(20, len(day_bars)):
        bar_time = day_bars.index[i]

        # Hard caps
        if len(trades) >= MAX_TRADES_PER_DAY:
            break

        # Global cooldown (any direction)
        if cooldown_until and bar_time < cooldown_until:
            continue

        slices = mtf.slice_until(bar_time)
        day_so_far = day_bars[day_bars.index <= bar_time]
        day_high = float(day_so_far["high"].max())
        day_low = float(day_so_far["low"].min())

        decision = make_decision(slices, day_high, day_low)
        decisions.append(decision)

        # Same-direction guard: skip if last trade was same dir AND
        # current spot is within 0.5% of last entry (same swing)
        if (decision["decision"] in ("CALL", "PUT")
                and decision["decision"] == last_trade_dir
                and last_trade_price
                and abs(decision["spot"] - last_trade_price) / last_trade_price < 0.005):
            decision["skip_reason"] = (
                f"same_dir_same_swing (last {last_trade_dir} @ {last_trade_price:.0f})"
            )
            decision["decision"] = "WAIT"

        # Same-direction cooldown — 45 min minimum
        if (decision["decision"] in ("CALL", "PUT")
                and decision["decision"] == last_trade_dir
                and same_dir_cooldown_until
                and bar_time < same_dir_cooldown_until):
            decision["skip_reason"] = (
                f"same_dir_cooldown ({(same_dir_cooldown_until - bar_time).seconds // 60}m left)"
            )
            decision["decision"] = "WAIT"

        if diagnose:
            print(
                f"{bar_time.strftime('%H:%M')} | spot={decision.get('spot', 0):.0f} "
                f"ML={decision.get('ml_signal', '-'):4s} "
                f"P=[{','.join(f'{p:.2f}' for p in decision.get('ml_proba', []))}] "
                f"conf={decision.get('ml_conf_pct', 0):5.1f}% "
                f"regime={decision.get('regime', '-'):12s} "
                f"floor={decision.get('effective_min_pct', '-')}% "
                f"→ {decision['decision']:5s} "
                f"({decision['skip_reason'][:60]})"
            )

        if decision["decision"] in ("CALL", "PUT"):
            atr = decision.get("atr", 30)
            trade = simulate_trade(decision["decision"], i, day_bars, atr)
            trade["entry_conf_pct"] = decision["ml_conf_pct"]
            trade["entry_regime"] = decision.get("regime", "-")
            trades.append(trade)
            # Cooldowns: 20m global, 45m same-direction
            cooldown_until = bar_time + timedelta(minutes=20)
            same_dir_cooldown_until = bar_time + timedelta(minutes=45)
            last_trade_dir = decision["decision"]
            last_trade_price = decision["spot"]

    n = len(trades)
    if n == 0:
        return {
            "date": date, "trades": 0, "wins": 0, "losses": 0,
            "decisions": len(decisions),
            "skip_reasons": Counter(
                d["skip_reason"].split(":")[0].strip() for d in decisions if d.get("skip_reason")
            ).most_common(10),
        }

    wins = sum(1 for t in trades if t["pnl_rupees"] > 0)
    return {
        "date": date,
        "decisions": len(decisions),
        "trades": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(wins / n * 100, 1),
        "total_pnl_rupees": round(sum(t["pnl_rupees"] for t in trades), 0),
        "avg_pnl_per_trade": round(sum(t["pnl_rupees"] for t in trades) / n, 0),
        "trade_details": trades,
        "skip_reasons": Counter(
            d["skip_reason"].split(":")[0].strip() for d in decisions if d.get("skip_reason")
        ).most_common(10),
    }


def main():
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    VERBOSE = args.diagnose

    log.info("Loading model...")
    from core import ml_engine
    if not ml_engine.load_model():
        log.error("Model load failed")
        return
    log.info(f"Model loaded: {ml_engine._loaded_version}, {len(ml_engine._fcols)} features")

    log.info("Fetching all timeframes (one-time)...")
    mtf = MultiTfBars()

    end = pd.to_datetime(args.date)
    if args.days > 1:
        date_list = pd.bdate_range(end=end, periods=args.days)
    else:
        date_list = [end]

    all_results = []
    for d in date_list:
        try:
            r = run_backtest(d.strftime("%Y-%m-%d"), mtf, diagnose=args.diagnose)
            all_results.append(r)
        except Exception as e:
            log.warning(f"Failed {d}: {e}")

    # Summary
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        print("\nNO VALID BACKTEST DAYS — check 'error' fields:")
        for r in all_results:
            print(f"  {r.get('date')}: {r.get('error')}")
        return

    total_trades = sum(r.get("trades", 0) for r in valid)
    total_wins = sum(r.get("wins", 0) for r in valid)
    total_pnl = sum(r.get("total_pnl_rupees", 0) for r in valid)

    print("\n" + "=" * 70)
    print(f"AGGREGATE — {len(valid)} days")
    print("=" * 70)
    print(f"Total trades:    {total_trades}")
    if total_trades > 0:
        print(f"Win rate:        {total_wins / total_trades * 100:.1f}%")
        print(f"Total P&L:       ₹{total_pnl:+,.0f}")
        print(f"Avg per trade:   ₹{total_pnl / total_trades:+,.0f}")
        print(f"Avg per day:     ₹{total_pnl / len(valid):+,.0f}")
    print("\nPer-day:")
    for r in valid:
        d = r.get("date", "-")
        t = r.get("trades", 0)
        wr = r.get("win_rate", 0)
        pnl = r.get("total_pnl_rupees", 0)
        cyc = r.get("decisions", 0)
        print(f"  {d}: cycles={cyc:3d} trades={t:2d} WR={wr:5.1f}% P&L=₹{pnl:+,.0f}")

    # Aggregate skip reasons
    all_skips = Counter()
    for r in valid:
        for reason, n in r.get("skip_reasons", []):
            all_skips[reason] += n
    if all_skips:
        print("\nSkip reasons (aggregate):")
        for reason, n in all_skips.most_common(15):
            print(f"  {n:4d}× {reason}")

    # Detailed trade dump if single-day diagnose
    if args.days == 1 and valid and valid[0].get("trade_details"):
        print("\nTrade details:")
        for t in valid[0]["trade_details"]:
            print(
                f"  {t['entry_time'].strftime('%H:%M')} {t['direction']:4s} "
                f"entry={t['entry_price']:.0f} exit={t['exit_price']:.0f} "
                f"({t['exit_reason']:4s}) pnl={t['pnl_pts']:+.1f}pts "
                f"₹{t['pnl_rupees']:+,.0f} conf={t['entry_conf_pct']:.0f}% "
                f"regime={t['entry_regime']}"
            )


if __name__ == "__main__":
    main()
