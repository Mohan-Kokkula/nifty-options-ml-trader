"""
backtest_s13.py — Donchian Breakout Backtest + MCPT Validation
==============================================================
Based on neurotrader888's Monte Carlo Permutation Testing framework.

1. Backtest: Donchian breakout on Nifty 5min with SL/TP/Trail
2. Walk-forward optimization: rolling re-optimization of lookback
3. MCPT: bar permutation to validate statistical significance

Usage:
  python backtest_s13.py                      # Basic backtest
  python backtest_s13.py --mcpt               # Full MCPT validation
  python backtest_s13.py --mcpt --perms 500   # Custom permutation count

Requires: 5min Nifty OHLCV CSV with columns: datetime,open,high,low,close,volume
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ──
SL = 45
TP = 70
AUCTION_END = 920       # skip before 09:20
TRAIL_AFTER_TP_STEP = 20
EMA_PERIOD = 40
LB_MIN = 10
LB_MAX = 60
DEFAULT_LOOKBACK = 20
TRAIN_BARS = 75 * 20    # 20 trading days of 5min bars
ADX_PERIOD = 14
ADX_MIN = 15


def load_data(path: str) -> pd.DataFrame:
    """Load 5min OHLCV CSV."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def donchian_signal(closes: pd.Series, lookback: int) -> pd.Series:
    upper = closes.rolling(lookback - 1).max().shift(1)
    lower = closes.rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.nan, index=closes.index)
    signal.loc[closes > upper] = 1
    signal.loc[closes < lower] = -1
    signal = signal.ffill()
    return signal


def optimize_lookback(closes: pd.Series, lb_min: int, lb_max: int) -> int:
    """Find lookback maximizing profit factor."""
    log_r = np.log(closes).diff().shift(-1)
    best_pf = 0.0
    best_lb = lb_min
    for lb in range(lb_min, lb_max + 1):
        sig = donchian_signal(closes, lb)
        rets = sig * log_r
        wins = rets[rets > 0].sum()
        losses = rets[rets < 0].abs().sum()
        pf = wins / losses if losses > 0 else 0.0
        if pf > best_pf:
            best_pf = pf
            best_lb = lb
    return best_lb


def get_bar_permutation(df: pd.DataFrame, start_index: int = 0) -> pd.DataFrame:
    """Monte Carlo bar permutation (neurotrader888 method).
    Shuffles intra-bar relative prices and gaps separately to preserve
    bar shape distribution while destroying temporal structure."""
    log_bars = np.log(df[["open", "high", "low", "close"]].values)
    n = len(df)
    perm_idx = start_index + 1
    perm_n = n - perm_idx

    start_bar = log_bars[start_index].copy()

    r_o = log_bars[perm_idx:, 0] - log_bars[perm_idx - 1 : n - 1, 3]  # open relative to prev close
    r_h = log_bars[perm_idx:, 1] - log_bars[perm_idx:, 0]
    r_l = log_bars[perm_idx:, 2] - log_bars[perm_idx:, 0]
    r_c = log_bars[perm_idx:, 3] - log_bars[perm_idx:, 0]

    # Shuffle intra-bar shape and gaps independently
    perm1 = np.random.permutation(perm_n)
    perm2 = np.random.permutation(perm_n)
    r_h, r_l, r_c = r_h[perm1], r_l[perm1], r_c[perm1]
    r_o = r_o[perm2]

    perm_bars = np.zeros((n, 4))
    perm_bars[:start_index] = log_bars[:start_index]
    perm_bars[start_index] = start_bar

    for i in range(perm_idx, n):
        k = i - perm_idx
        perm_bars[i, 0] = perm_bars[i - 1, 3] + r_o[k]
        perm_bars[i, 1] = perm_bars[i, 0] + r_h[k]
        perm_bars[i, 2] = perm_bars[i, 0] + r_l[k]
        perm_bars[i, 3] = perm_bars[i, 0] + r_c[k]

    perm_bars = np.exp(perm_bars)
    return pd.DataFrame(
        perm_bars, index=df.index, columns=["open", "high", "low", "close"]
    )


def backtest_donchian(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK,
                      walkforward: bool = False) -> dict:
    """Backtest Donchian breakout with SL/TP/Trail on 5min bars.

    Returns dict with trades list and summary stats.
    """
    closes = df["close"]
    ema = closes.ewm(span=EMA_PERIOD, adjust=False).mean()
    adx = compute_adx(df, ADX_PERIOD)

    # Walk-forward: re-optimize lookback every TRAIN_BARS
    if walkforward:
        lb = lookback
        next_opt = TRAIN_BARS
    else:
        lb = lookback
        next_opt = len(df) + 1  # never re-optimize

    sig = donchian_signal(closes, lb)
    trades = []
    in_trade = False
    entry_price = 0.0
    direction = 0  # 1=CALL, -1=PUT
    sl_price = 0.0
    tp_price = 0.0
    trailing = False
    trail_high = 0.0
    trail_low = 999999.0
    prev_sig = 0.0

    for i in range(max(lb + 10, 30), len(df)):
        # Walk-forward re-optimization
        if walkforward and i >= next_opt and i >= TRAIN_BARS:
            train_slice = closes.iloc[i - TRAIN_BARS : i]
            lb = optimize_lookback(train_slice, LB_MIN, LB_MAX)
            sig = donchian_signal(closes, lb)
            next_opt = i + 75 * 5  # re-optimize every 5 trading days

        bar = df.iloc[i]
        price = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])

        # Time filter
        if hasattr(bar.name, "hour"):
            hm = bar.name.hour * 100 + bar.name.minute
        else:
            hm = 1000
        if hm < AUCTION_END:
            continue

        cur_sig = float(sig.iloc[i]) if not np.isnan(sig.iloc[i]) else 0
        ema_val = float(ema.iloc[i])
        adx_val = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 25.0

        # Manage open trade
        if in_trade:
            # Check SL/TP hit on this bar
            if direction == 1:  # CALL
                if low <= sl_price:
                    pnl = sl_price - entry_price
                    trades.append({"entry": entry_price, "exit": sl_price,
                                   "pnl": pnl, "dir": "CALL", "exit_reason": "SL"})
                    in_trade = False
                    continue
                if high >= tp_price:
                    if not trailing:
                        trailing = True
                        trail_high = high
                    else:
                        if high > trail_high:
                            trail_high = high
                        trail_sl = trail_high - TRAIL_AFTER_TP_STEP
                        if low <= trail_sl:
                            pnl = trail_sl - entry_price
                            trades.append({"entry": entry_price, "exit": trail_sl,
                                           "pnl": pnl, "dir": "CALL", "exit_reason": "TRAIL"})
                            in_trade = False
                            continue
                if trailing and high > trail_high:
                    trail_high = high

            elif direction == -1:  # PUT
                if high >= sl_price:
                    pnl = entry_price - sl_price
                    trades.append({"entry": entry_price, "exit": sl_price,
                                   "pnl": pnl, "dir": "PUT", "exit_reason": "SL"})
                    in_trade = False
                    continue
                if low <= tp_price:
                    if not trailing:
                        trailing = True
                        trail_low = low
                    else:
                        if low < trail_low:
                            trail_low = low
                        trail_sl = trail_low + TRAIL_AFTER_TP_STEP
                        if high >= trail_sl:
                            pnl = entry_price - trail_sl
                            trades.append({"entry": entry_price, "exit": trail_sl,
                                           "pnl": pnl, "dir": "PUT", "exit_reason": "TRAIL"})
                            in_trade = False
                            continue
                if trailing and low < trail_low:
                    trail_low = low

            # Signal flip closes position
            if cur_sig != 0 and cur_sig != direction:
                pnl = (price - entry_price) * direction
                trades.append({"entry": entry_price, "exit": price,
                               "pnl": pnl, "dir": "CALL" if direction == 1 else "PUT",
                               "exit_reason": "FLIP"})
                in_trade = False
                # Fall through to potentially open new trade

        # Entry: detect flip
        if not in_trade and cur_sig != 0:
            # Check for flip
            if i > 0:
                p = float(sig.iloc[i - 1]) if not np.isnan(sig.iloc[i - 1]) else 0
            else:
                p = 0
            is_flip = (p != cur_sig and p != 0 and cur_sig != 0)
            if not is_flip:
                prev_sig = cur_sig
                continue

            if adx_val < ADX_MIN:
                continue

            if cur_sig == 1 and price > ema_val:
                entry_price = price
                direction = 1
                sl_price = entry_price - SL
                tp_price = entry_price + TP
                in_trade = True
                trailing = False
                trail_high = price

            elif cur_sig == -1 and price < ema_val:
                entry_price = price
                direction = -1
                sl_price = entry_price + SL
                tp_price = entry_price - TP
                in_trade = True
                trailing = False
                trail_low = price

        prev_sig = cur_sig

    # Close any open trade at end
    if in_trade:
        price = float(df["close"].iloc[-1])
        pnl = (price - entry_price) * direction
        trades.append({"entry": entry_price, "exit": price,
                       "pnl": pnl, "dir": "CALL" if direction == 1 else "PUT",
                       "exit_reason": "EOD"})

    # Summary stats
    if not trades:
        return {"trades": [], "total_pnl": 0, "pf": 0, "wr": 0,
                "n_trades": 0, "max_dd": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
    wr = len(wins) / len(pnls) * 100

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0

    return {
        "trades": trades,
        "total_pnl": round(total_pnl, 2),
        "pf": round(pf, 3),
        "wr": round(wr, 1),
        "n_trades": len(trades),
        "max_dd": round(max_dd, 2),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
    }


def run_mcpt(df: pd.DataFrame, real_pf: float, n_perms: int = 200,
             walkforward: bool = False) -> float:
    """Run Monte Carlo Permutation Test.
    Returns p-value: fraction of permutations that achieved >= real PF."""
    perm_better = 1  # start at 1 (includes the real data)
    perm_pfs = []

    for i in range(1, n_perms):
        perm_df = get_bar_permutation(df, start_index=TRAIN_BARS if walkforward else 0)
        result = backtest_donchian(perm_df, walkforward=walkforward)
        perm_pf = result["pf"]
        perm_pfs.append(perm_pf)
        if perm_pf >= real_pf:
            perm_better += 1
        if (i % 50) == 0:
            print(f"  MCPT progress: {i}/{n_perms-1} permutations...")

    p_value = perm_better / n_perms
    return p_value, perm_pfs


def main():
    parser = argparse.ArgumentParser(description="S13 Donchian Backtest + MCPT")
    parser.add_argument("--data", type=str, default="data/nifty_5min.csv",
                        help="Path to 5min OHLCV CSV")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--walkforward", action="store_true",
                        help="Enable walk-forward lookback optimization")
    parser.add_argument("--mcpt", action="store_true",
                        help="Run MCPT validation after backtest")
    parser.add_argument("--perms", type=int, default=200,
                        help="Number of MCPT permutations")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Expected CSV with columns: datetime,open,high,low,close,volume")
        sys.exit(1)

    df = load_data(str(data_path))
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Config: SL={SL} TP={TP} Trail={TRAIL_AFTER_TP_STEP} "
          f"Lookback={args.lookback} EMA={EMA_PERIOD}")
    print()

    # Run backtest
    result = backtest_donchian(df, lookback=args.lookback,
                                walkforward=args.walkforward)
    print("=" * 50)
    print("S13 DONCHIAN BREAKOUT BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Trades:     {result['n_trades']}")
    print(f"  Total PnL:  {result['total_pnl']:+.0f} pts")
    print(f"  Profit F:   {result['pf']:.3f}")
    print(f"  Win Rate:   {result['wr']:.1f}%")
    print(f"  Avg Win:    {result['avg_win']:+.1f} pts")
    print(f"  Avg Loss:   {result['avg_loss']:.1f} pts")
    print(f"  Max DD:     {result['max_dd']:.0f} pts")
    print()

    if args.mcpt:
        print("=" * 50)
        print(f"MCPT VALIDATION ({args.perms} permutations)")
        print("=" * 50)
        p_value, perm_pfs = run_mcpt(
            df, result["pf"], n_perms=args.perms,
            walkforward=args.walkforward
        )
        print(f"\n  Real PF:    {result['pf']:.3f}")
        print(f"  Perm Mean:  {np.mean(perm_pfs):.3f}")
        print(f"  Perm Max:   {np.max(perm_pfs):.3f}")
        print(f"  P-Value:    {p_value:.4f}")
        if p_value < 0.05:
            print("  RESULT:     SIGNIFICANT (p < 0.05) — strategy has edge")
        elif p_value < 0.10:
            print("  RESULT:     MARGINAL (0.05 < p < 0.10)")
        else:
            print("  RESULT:     NOT SIGNIFICANT (p >= 0.10) — likely curve-fitted")
        print()


if __name__ == "__main__":
    main()
