"""
validate_fixes_sim.py — Step 2 of the 10-year fix validation.

Replays the pilot gate stack over the cached V10 predictions (2015-2026)
under multiple variants and produces the full validation report:
overall metrics, yearly, regime buckets, fix attribution (solo + ablation),
walk-forward split at the model's train boundary, Monte Carlo.

Simulated gates (identical mechanics across variants unless varied):
  Gate1 thresholds, dir-conf blend, reversal filter (FVG/CHoCH/RSI-relax),
  Gate2 base + regime floors/counter-regime, multi-TF agreement,
  MAX_LOSS sizing, morning hard block (45m), single position,
  max 6 trades/day, daily loss cap -5000, EOD 15:15 exit.

NOT simulated (no historical data / flag-off in prod — identical omission
in ALL variants): OI buildup gate & today's OI penalty fix, PCR momentum,
trap gate, momentum/60m overrides, VWAP bias, VIX conf_adj, briefs, greeks.

Fixes under test:
  F_RSI : rsi14 key fix → SMC RSI-extreme CHoCH relaxation alive
  F_SLT : MAX_LOSS SL-tighten (<=15%) instead of binary skip
  F_REV : reversal filter zone-no-CHoCH → +8pp penalty (was hard block)
  F_MTF : multi-TF disagree → +8pp penalty w/ TREND-regime bypass (was veto)
  F_CR  : counter-regime +10/floor70 → +5/floor65
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/validation_predcache.pkl"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
VIXCSV = ROOT / "data/india_vix.csv"
OUTDIR = ROOT / "data/validation_results"
OUTDIR.mkdir(exist_ok=True)

LOT = 65
DELTA = 0.50
COSTS = 100.0          # round-trip brokerage+slippage (premium ₹)
MAX_LOSS_TRADE = 2000.0
MAX_LOSS_DAY = 5000.0
MAX_TRADES_DAY = 6
CAPITAL = 100_000.0

TRAIN_END = pd.Timestamp("2024-08-30")
VAL_END = pd.Timestamp("2025-07-07")


# ──────────────────────────────────────────────────────────────────────
def dir_conf(call_p, put_p, sig):
    """ml_engine 2026-04-27 directional confidence blend (both variants)."""
    if sig == 0:
        d = call_p / max(call_p + put_p, 1e-9)
        return min(0.90, 0.65 * d + 0.35 * min(1.0, call_p * 2.0))
    d = put_p / max(call_p + put_p, 1e-9)
    return min(0.90, 0.65 * d + 0.35 * min(1.0, put_p * 2.0))


def classify_regime(h, l, c, i0, i, day_open):
    """Lightweight replica of core.regime_engine.classify on numpy slices.
    Returns (name, favored, floor_new, strong) — floors per CURRENT file."""
    n = i - i0 + 1
    if n < 20:
        return ("UNKNOWN", "NEITHER", 0, False)
    w = slice(max(i0, i - 19), i + 1)          # last 20 bars
    closes = c[w]
    x = np.arange(len(closes), dtype=float)
    slope, intercept = np.polyfit(x, closes, 1)
    fit = slope * x + intercept
    ss_res = ((closes - fit) ** 2).sum()
    ss_tot = ((closes - closes.mean()) ** 2).sum() + 1e-9
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    slope_pct = slope / closes[-1] * 100
    day_h, day_l = h[i0:i + 1].max(), l[i0:i + 1].min()
    rng = max(day_h - day_l, 1e-9)
    pos = (closes[-1] - day_l) / rng
    bar_min = (i - i0) * 5

    if bar_min < 15:
        return ("OPENING", "NEITHER", 80, False)
    # reversal: at extreme + fading structure
    if pos >= 0.85 and slope_pct < 0.005:
        return ("REVERSAL_DOWN", "PUT", 50, False)
    if pos <= 0.15 and slope_pct > -0.005:
        return ("REVERSAL_UP", "CALL", 75, False)
    if r2 >= 0.55 and abs(slope_pct) >= 0.03:
        if slope > 0:
            return ("TREND_UP", "CALL", 55, True)
        return ("TREND_DOWN", "PUT", 50, True)       # file: 55→50 (Jun-11)
    if r2 >= 0.35 and abs(slope_pct) >= 0.02:
        if slope > 0:
            return ("TREND_UP", "CALL", 60, False)
        return ("TREND_DOWN", "PUT", 55, False)      # file: 60→55 (Jun-11)
    return ("CHOP", "NEITHER", 75 if 0.20 <= r2 else 65, False)


def scan_fvg(h, l, i0, i):
    """FVG zones from bars [i0..i] (last 60), as in claude_pilot Step 3.3."""
    s = max(i0, i - 59)
    bull, bear = [], []
    for k in range(s + 2, i + 1):
        if l[k] > h[k - 2]:
            bull.append((h[k - 2], l[k]))
        if h[k] < l[k - 2]:
            bear.append((h[k], l[k - 2]))
    return bull[-10:], bear[-10:]


def run_variant(df, aux, V):
    """Replay one gate-stack variant. Returns trades DataFrame."""
    ts = df.index
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    call_p = df["call_p"].values; put_p = df["put_p"].values
    skip_p = df["skip_p"].values; rsi = df["rsi14"].values
    atr15 = aux["atr15"].values
    e15b = aux["ema15_bull"].values; e60b = aux["ema60_bull"].values
    dates = aux["date"].values
    minutes = aux["minute_of_day"].values

    # Gate 1 (vectorized)
    cc, cp = V["thr_call"], V["thr_put"]
    ceil_ = V["skip_ceil"]
    sig = np.full(len(df), 2, dtype=np.int8)
    sig[(call_p >= cc) & (call_p - put_p >= 0.05) & (skip_p < ceil_)] = 0
    m = (put_p >= cp) & (put_p - call_p >= 0.05) & (skip_p < ceil_) & (sig == 2)
    sig[m] = 1

    trades = []
    day_starts = aux["day_start_idx"].values

    i = 0
    n = len(df)
    cur_date = None
    day_pnl = 0.0
    day_trades = 0
    pos_exit_i = -1

    while i < n:
        if dates[i] != cur_date:
            cur_date = dates[i]
            day_pnl, day_trades = 0.0, 0
        if (sig[i] == 2 or i <= pos_exit_i or minutes[i] < 600
                or minutes[i] > 900 or day_trades >= MAX_TRADES_DAY
                or day_pnl <= -MAX_LOSS_DAY or i + 1 >= n
                or dates[i + 1] != cur_date):
            i += 1
            continue

        s = int(sig[i])
        i0 = day_starts[i]
        spot = c[i]
        conf = round(dir_conf(call_p[i], put_p[i], s) * 100)

        # ── Reversal filter ──────────────────────────────────────────
        struct_pen = 0
        thr_rev = max(40.0, 1.5 * atr15[i])
        day_h = h[i0:i + 1].max(); day_l = l[i0:i + 1].min()
        rev = (s == 0 and (day_h - spot) >= thr_rev) or \
              (s == 1 and (spot - day_l) >= thr_rev)
        if rev:
            bull_z, bear_z = scan_fvg(h, l, i0, i)
            zones = bull_z if s == 0 else bear_z
            zone = any(zl - 25 <= spot <= zh + 25 for zl, zh in zones)
            choch = False
            if zone and i - i0 >= 4:
                if s == 0 and c[i] > h[i - 3:i].max():
                    choch = True
                if s == 1 and c[i] < l[i - 3:i].min():
                    choch = True
            rsi_x = False
            if V["f_rsi"] and zone and not choch and not np.isnan(rsi[i]):
                if s == 0 and rsi[i] <= 35 and (spot - day_l) / day_l <= 0.005:
                    rsi_x = True
                if s == 1 and rsi[i] >= 65 and (day_h - spot) / day_h <= 0.005:
                    rsi_x = True
            if zone and (choch or rsi_x):
                pass                                   # allowed
            elif zone and V["f_rev"]:
                struct_pen += 8                        # penalty (Fix 4)
            else:
                i += 1
                continue                               # hard block

        # ── Gate 2 ───────────────────────────────────────────────────
        eff = V["g2_call"] if s == 0 else V["g2_put"]
        eff = min(85, eff + struct_pen)
        reg_name, reg_fav, reg_floor, _ = classify_regime(h, l, c, i0, i, o[i0])
        ml_dir = "CALL" if s == 0 else "PUT"
        if not V["new_regime_floors"] and reg_name == "TREND_DOWN":
            reg_floor += 5                             # pre-Jun-10 floors
        if reg_fav == ml_dir:
            eff = max(50, min(eff, reg_floor))
        elif reg_fav in ("CALL", "PUT"):
            eff = min(85, max(eff + V["cr_pen"], V["cr_floor"]))
        elif reg_floor:
            fl = reg_floor
            if V["pe_chop_cap"] and s == 1 and reg_name == "CHOP":
                fl = min(fl, 65)
            eff = max(eff, fl)
        if conf < eff:
            i += 1
            continue

        # ── Multi-TF agreement ───────────────────────────────────────
        against = (s == 0 and not (e15b[i] or e60b[i])) or \
                  (s == 1 and (e15b[i] and e60b[i]))
        if against:
            if V["f_mtf"]:
                bypass = (s == 0 and reg_name == "TREND_UP") or \
                         (s == 1 and reg_name == "TREND_DOWN")
                if not bypass and conf < min(85, eff + 8):
                    i += 1
                    continue
            else:
                i += 1
                continue

        # ── Sizing / MAX_LOSS ────────────────────────────────────────
        sl = min(80.0, max(30.0, 2.2 * atr15[i]))
        loss1 = sl * DELTA * LOT
        if loss1 > MAX_LOSS_TRADE:
            fit = (MAX_LOSS_TRADE / (DELTA * LOT)) * 0.995
            if V["f_slt"] and fit >= sl * 0.85:
                sl = fit
            else:
                i += 1
                continue
        tp = 2.0 * sl

        # ── Execute: enter next bar open, walk to SL/TP/EOD ──────────
        ei = i + 1
        entry = o[ei]
        exit_px, exit_i, hit = None, None, "EOD"
        j = ei
        while j < n and dates[j] == cur_date:
            if s == 0:
                if l[j] <= entry - sl:
                    exit_px, exit_i, hit = entry - sl, j, "SL"; break
                if h[j] >= entry + tp:
                    exit_px, exit_i, hit = entry + tp, j, "TP"; break
            else:
                if h[j] >= entry + sl:
                    exit_px, exit_i, hit = entry + sl, j, "SL"; break
                if l[j] <= entry - tp:
                    exit_px, exit_i, hit = entry - tp, j, "TP"; break
            j += 1
        if exit_px is None:
            exit_i = j - 1
            exit_px = c[exit_i]
        pts = (exit_px - entry) if s == 0 else (entry - exit_px)
        pnl = pts * DELTA * LOT - COSTS
        risk = sl * DELTA * LOT + COSTS
        trades.append({
            "ts": ts[i], "dir": ml_dir, "conf": conf, "entry": entry,
            "exit": exit_px, "hit": hit, "sl_pts": sl, "pnl": pnl,
            "r": pnl / risk, "regime": reg_name,
        })
        day_pnl += pnl
        day_trades += 1
        pos_exit_i = exit_i
        i = exit_i + 1

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────────────────────────────
def metrics(tr, days_span_yrs):
    if tr.empty:
        return {"trades": 0}
    pnl = tr["pnl"]
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    daily = tr.groupby(tr["ts"].dt.date)["pnl"].sum()
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    ret = daily / CAPITAL
    shrp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(252)
    dret = ret[ret < 0]
    srt = ret.mean() / (dret.std() + 1e-12) * np.sqrt(252)
    net = pnl.sum()
    cagr = ((CAPITAL + net) / CAPITAL) ** (1 / max(days_span_yrs, 0.1)) - 1
    return {
        "trades": len(tr), "wr": len(wins) / len(tr),
        "pf": wins.sum() / max(abs(losses.sum()), 1e-9),
        "sharpe": shrp, "sortino": srt, "cagr": cagr,
        "maxdd": dd, "avg_r": tr["r"].mean(),
        "expectancy": pnl.mean(), "net": net,
        "recovery": net / max(abs(dd), 1e-9),
    }


def main():
    raw = pd.read_pickle(CACHE)
    raw = raw[raw.index.time >= pd.Timestamp("09:15").time()]
    # PRIMARY engine: V9 (defined over full 2015-2026, documented purged
    # split: train ≤2024-08-30, val ≤2025-07-07, frozen test after).
    df = raw.rename(columns={"call_v9": "call_p", "put_v9": "put_p",
                             "skip_v9": "skip_p"})
    df = df[["open", "high", "low", "close", "call_p", "put_p", "skip_p",
             "rsi14"]].dropna(subset=["call_p"])

    # aux columns
    aux = pd.DataFrame(index=df.index)
    aux["date"] = df.index.date
    aux["minute_of_day"] = df.index.hour * 60 + df.index.minute
    day_grp = aux.groupby("date")
    first_idx = day_grp.cumcount()
    aux["day_start_idx"] = np.arange(len(df)) - first_idx.values

    # 15m ATR + HTF EMA states (from completed HTF bars)
    df15 = df[["open", "high", "low", "close"]].resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr15 = pd.concat([df15.high - df15.low,
                      (df15.high - df15.close.shift()).abs(),
                      (df15.low - df15.close.shift()).abs()], axis=1).max(axis=1)
    atr15 = tr15.rolling(14).mean()
    e9_15 = df15.close.ewm(span=9, adjust=False).mean()
    e21_15 = df15.close.ewm(span=21, adjust=False).mean()
    df60 = df[["close"]].resample("60min").last().dropna()
    e9_60 = df60.close.ewm(span=9, adjust=False).mean()
    e21_60 = df60.close.ewm(span=21, adjust=False).mean()

    aux["atr15"] = atr15.reindex(df.index, method="ffill").fillna(30.0)
    aux["ema15_bull"] = (e9_15 > e21_15).reindex(df.index, method="ffill").fillna(False)
    aux["ema60_bull"] = (e9_60 > e21_60).reindex(df.index, method="ffill").fillna(False)

    base_new = dict(thr_call=0.32, thr_put=0.25, skip_ceil=0.65,
                    g2_call=72, g2_put=58, new_regime_floors=True,
                    pe_chop_cap=True)
    variants = {
        "PRE_JUN10": dict(thr_call=0.30, thr_put=0.30, skip_ceil=0.60,
                          g2_call=70, g2_put=70, new_regime_floors=False,
                          pe_chop_cap=False, f_rsi=False, f_slt=False,
                          f_rev=False, f_mtf=False, cr_pen=10, cr_floor=70),
        "OLD": dict(**base_new, f_rsi=False, f_slt=False, f_rev=False,
                    f_mtf=False, cr_pen=10, cr_floor=70),
        "NEW": dict(**base_new, f_rsi=True, f_slt=True, f_rev=True,
                    f_mtf=True, cr_pen=5, cr_floor=65),
    }
    solo = {
        "SOLO_RSI": dict(f_rsi=True, f_slt=False, f_rev=False, f_mtf=False, cr_pen=10, cr_floor=70),
        "SOLO_SLT": dict(f_rsi=False, f_slt=True, f_rev=False, f_mtf=False, cr_pen=10, cr_floor=70),
        "SOLO_REV": dict(f_rsi=False, f_slt=False, f_rev=True, f_mtf=False, cr_pen=10, cr_floor=70),
        "SOLO_MTF": dict(f_rsi=False, f_slt=False, f_rev=False, f_mtf=True, cr_pen=10, cr_floor=70),
        "SOLO_CR":  dict(f_rsi=False, f_slt=False, f_rev=False, f_mtf=False, cr_pen=5, cr_floor=65),
    }
    abl = {
        "ABL_RSI": dict(f_rsi=False, f_slt=True, f_rev=True, f_mtf=True, cr_pen=5, cr_floor=65),
        "ABL_SLT": dict(f_rsi=True, f_slt=False, f_rev=True, f_mtf=True, cr_pen=5, cr_floor=65),
        "ABL_REV": dict(f_rsi=True, f_slt=True, f_rev=False, f_mtf=True, cr_pen=5, cr_floor=65),
        "ABL_MTF": dict(f_rsi=True, f_slt=True, f_rev=True, f_mtf=False, cr_pen=5, cr_floor=65),
        "ABL_CR":  dict(f_rsi=True, f_slt=True, f_rev=True, f_mtf=True, cr_pen=10, cr_floor=70),
    }
    for k, v in {**solo, **abl}.items():
        variants[k] = dict(**base_new, **v)

    yrs = (df.index[-1] - df.index[0]).days / 365.25
    all_trades = {}
    print(f"Simulating {len(variants)} variants over {len(df):,} bars "
          f"({df.index[0].date()} → {df.index[-1].date()})...")
    for name, V in variants.items():
        tr = run_variant(df, aux, V)
        all_trades[name] = tr
        tr.to_csv(OUTDIR / f"trades_{name}{SUFFIX}.csv", index=False)
        m = metrics(tr, yrs)
        print(f"  {name:10s}: trades={m.get('trades',0):5d} "
              f"wr={m.get('wr',0):.2%} pf={m.get('pf',0):5.2f} "
              f"net=₹{m.get('net',0):>12,.0f} maxDD=₹{m.get('maxdd',0):>10,.0f}")

    # ── full report data ──────────────────────────────────────────────
    rep = {"span_years": yrs, "bars": len(df),
           "range": [str(df.index[0]), str(df.index[-1])]}

    for name in ("OLD", "NEW"):
        tr = all_trades[name]
        rep[name] = {"overall": metrics(tr, yrs)}
        # yearly
        yearly = {}
        for y, g in tr.groupby(tr["ts"].dt.year):
            yearly[int(y)] = metrics(g, 1.0)
        rep[name]["yearly"] = yearly
        # walk-forward
        rep[name]["wf"] = {
            "train_2015_2024aug": metrics(tr[tr.ts <= TRAIN_END], 9.6),
            "val_2024sep_2025jul": metrics(
                tr[(tr.ts > TRAIN_END) & (tr.ts <= VAL_END)], 0.85),
            "test_2025jul_2026jun": metrics(tr[tr.ts > VAL_END], 0.91),
        }

    # solo / ablation deltas vs OLD / NEW
    rep["solo"] = {k: metrics(all_trades[k], yrs) for k in solo}
    rep["abl"] = {k: metrics(all_trades[k], yrs) for k in abl}
    rep["pre"] = metrics(all_trades["PRE_JUN10"], yrs)

    # regime buckets for OLD/NEW (VIX + ADX + gap)
    vix = pd.read_csv(VIXCSV, parse_dates=["date"]).set_index("date")["vix"]
    vix.index = vix.index.normalize()
    vix = vix[~vix.index.duplicated(keep="last")]
    d5 = df["close"].resample("D").last().dropna()
    day_open = df["open"].resample("D").first().dropna()
    gap = ((day_open - d5.shift()) / d5.shift()).dropna()
    # ADX15 simplified: use atr-based trendiness via 15m r2 proxy — use EMA sep
    for name in ("OLD", "NEW"):
        tr = all_trades[name].copy()
        if tr.empty:
            continue
        td = pd.to_datetime(tr["ts"]).dt.normalize()
        tr["vix"] = td.map(vix).values
        tr["gap"] = td.map(gap).values
        buckets = {
            "trending": tr[tr.regime.isin(["TREND_UP", "TREND_DOWN"])],
            "range_chop": tr[tr.regime.isin(["CHOP", "UNKNOWN", "OPENING"])],
            "reversal": tr[tr.regime.isin(["REVERSAL_UP", "REVERSAL_DOWN"])],
            "high_vol_vix>20": tr[tr.vix > 20],
            "low_vol_vix<14": tr[tr.vix < 14],
            "gap_up>0.3%": tr[tr.gap > 0.003],
            "gap_down<-0.3%": tr[tr.gap < -0.003],
        }
        rep[name]["regimes"] = {k: metrics(g, yrs) for k, g in buckets.items()}

    # Monte Carlo on NEW and OLD (10k shuffles)
    rng = np.random.default_rng(42)
    for name in ("OLD", "NEW"):
        pnl = all_trades[name]["pnl"].values
        if len(pnl) < 10:
            continue
        dds = np.empty(10_000)
        for k in range(10_000):
            eq = np.cumsum(rng.permutation(pnl))
            dds[k] = (eq - np.maximum.accumulate(eq)).min()
        ann = pnl.sum() / yrs
        rep[name]["monte_carlo"] = {
            "dd_p50": float(np.percentile(dds, 50)),
            "dd_p95": float(np.percentile(dds, 5)),   # worse tail
            "dd_p99": float(np.percentile(dds, 1)),
            "prob_ruin_50pct_capital": float((dds <= -0.5 * CAPITAL).mean()),
            "ann_pnl": ann,
        }

    # ── SECONDARY: deployed V10 on its native window (2024-05+) ──────
    if "call_v10" in raw.columns:
        d10 = raw.rename(columns={"call_v10": "call_p", "put_v10": "put_p",
                                  "skip_v10": "skip_p"})
        d10 = d10[["open", "high", "low", "close", "call_p", "put_p",
                   "skip_p", "rsi14"]].dropna(subset=["call_p"])
        if len(d10) > 1000:
            a10 = aux.loc[d10.index].copy()
            # rebuild day_start_idx for the subwindow
            g = pd.Series(d10.index.date, index=d10.index)
            a10["date"] = g.values
            a10["day_start_idx"] = (np.arange(len(d10))
                                    - g.groupby(g.values).cumcount().values)
            y10 = (d10.index[-1] - d10.index[0]).days / 365.25
            rep["v10_window"] = {"range": [str(d10.index[0]), str(d10.index[-1])]}
            print(f"\nSecondary (deployed V10, {d10.index[0].date()} → "
                  f"{d10.index[-1].date()}):")
            for name in ("OLD", "NEW"):
                tr = run_variant(d10, a10, variants[name])
                tr.to_csv(OUTDIR / f"trades_v10_{name}.csv", index=False)
                m = metrics(tr, y10)
                rep["v10_window"][name] = m
                print(f"  V10 {name:4s}: trades={m.get('trades',0):5d} "
                      f"wr={m.get('wr',0):.2%} pf={m.get('pf',0):5.2f} "
                      f"net=₹{m.get('net',0):>11,.0f} maxDD=₹{m.get('maxdd',0):>10,.0f}")

    with open(OUTDIR / f"report{SUFFIX}.json", "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"\nReport → {OUTDIR / f'report{SUFFIX}.json'}")


if __name__ == "__main__":
    main()
