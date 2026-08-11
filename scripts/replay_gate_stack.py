"""
replay_gate_stack.py — measure what every live gate actually costs.

WHY THIS EXISTS
    scripts/honest_backtest.py describes itself as replaying "the EXACT live
    decision pipeline". It implements 2 of the 12 gates claude_pilot.py puts
    between a model signal and an order (COUNTER-REGIME and TRAP). Every PF
    number this project has quoted therefore describes a system nobody runs.

    This script replays the REAL gate objects — RegimeEngine, TrapDetector,
    vix_regime — over the full 5-minute history and measures each gate by
    leave-one-out: run everything, then switch one gate off and see what
    changes. A gate that cannot pay for itself across 11 years should not be
    silently vetoing trades on a Tuesday.

WHAT IT CANNOT MEASURE — stated loudly, never silently skipped
    IV GATE         data/iv_history.jsonl holds 1 record
    OI FILTER       data/oi_archive covers 29 days (2026-06-12 .. 07-22)
    PCR alignment   same 29 days
    BRIEF bias      18 premarket_brief_*.json files
    STRUCTURE PEN.  needs 3-minute bars; the repo only has 5m and coarser
    Those five gate every live trade and no history exists to judge them.
    They are reported as UNMEASURABLE rather than quietly treated as pass.

METHOD
    VAL selects, TEST confirms once — the discipline used everywhere else in
    this project. Same purged split as V9/V11 training (85/7.5/7.5 with a
    78-bar gap) so the numbers line up with the existing V9/V11 table.

    The model is loaded from models/ (the promoted V9), so TRAIN bars are
    in-sample and are never reported — only VAL and TEST.

USAGE
    python scripts/replay_gate_stack.py
    python scripts/replay_gate_stack.py --conf-sweep 45,50,52,55,58,62,65,70
    python scripts/replay_gate_stack.py --exit frozen
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_model_v9 as V9  # noqa: E402
import scripts.train_model_v11 as V11  # noqa: E402
from core.ml_engine import (  # noqa: E402
    CONFIDENCE_CALL, CONFIDENCE_PUT, MIN_EDGE, SKIP_CEIL,
)
from core.regime_engine import RegimeEngine  # noqa: E402
from core.trap_detector import TrapDetector  # noqa: E402
from core.vix_regime import apply_vix_to_sl_tp, classify_vix  # noqa: E402

MODELS = ROOT / "models"

# ── Live constants, read from the same places main.py reads them ──────────
LOT_SIZE = 65                 # DEFAULT_QTY in config/settings.env
MAX_LOSS_CAP = 2000.0         # MAX_LOSS_PER_TRADE in config/settings.env
SL_TIGHTEN_BAND = 0.85        # claude_pilot.py:4255 — tighten only if within 15%
FRICTION_PTS = 5.9            # futures round-trip (EXECUTION_MODE=futures)
BASE_MIN_CONF = 58            # the floor every gate then adjusts
WHIPSAW_COOLDOWN_MIN = 30
EOD_SQUAREOFF = "15:15"
ENTRY_CUTOFF = "15:00"

# Directional SKIP sentinel used by ml_engine
SIG_CALL, SIG_PUT, SIG_SKIP = 0, 1, 2


# ══════════════════════════════════════════════════════════════════════════
# Scoring — vectorised, but bit-for-bit the live rule
# ══════════════════════════════════════════════════════════════════════════
def score_all(df_feat: pd.DataFrame) -> tuple:
    """Reproduce ml_engine.predict_precomputed() over every row at once.

    predict_precomputed scores one row per call; calling it 200k times would
    take hours. The arithmetic below is the same, verified against a live log
    line: CALL=0.097 PUT=0.262 -> conf 0.658, which matches the bot's
    dir_conf=0.658 for cycle #17 on 2026-08-11.
    """
    models = joblib.load(MODELS / "nifty_v9_models.pkl")
    scaler = joblib.load(MODELS / "nifty_v9_scaler.pkl")
    fcols = joblib.load(MODELS / "feature_cols_v9.pkl")

    missing = [c for c in fcols if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0.0
    if missing:
        print(f"  note: {len(missing)} features absent, filled with 0 "
              f"(same as live predict_precomputed)")

    X = (df_feat[fcols]
         .replace([np.inf, -np.inf], 0.0)
         .fillna(0.0)
         .values.astype(np.float64))
    Xs = scaler.transform(X)

    each = np.stack([m.predict_proba(Xs) for m in models.values()])  # (M,n,3)
    proba = each.mean(axis=0)
    call_p, put_p, skip_p = proba[:, 0], proba[:, 1], proba[:, 2]

    # Ensemble disagreement -> forced SKIP (ml_engine.py:518-541)
    disagree = np.maximum(
        each[:, :, 0].max(axis=0) - each[:, :, 0].min(axis=0),
        each[:, :, 1].max(axis=0) - each[:, :, 1].min(axis=0),
    )

    sig = np.full(len(proba), SIG_SKIP, dtype=np.int8)
    sig[(call_p >= CONFIDENCE_CALL) & (call_p - put_p >= MIN_EDGE)
        & (skip_p < SKIP_CEIL)] = SIG_CALL
    sig[(put_p >= CONFIDENCE_PUT) & (put_p - call_p >= MIN_EDGE)
        & (skip_p < SKIP_CEIL)] = SIG_PUT
    sig[(sig != SIG_SKIP) & (disagree > 0.20)] = SIG_SKIP

    # conf = 0.65*dir_prob + 0.35*anchor, capped 0.90 (ml_engine.py:554-563)
    denom = np.maximum(call_p + put_p, 1e-9)
    conf_c = np.minimum(0.90, 0.65 * (call_p / denom)
                        + 0.35 * np.minimum(1.0, call_p * 2.0))
    conf_p = np.minimum(0.90, 0.65 * (put_p / denom)
                        + 0.35 * np.minimum(1.0, put_p * 2.0))
    conf = np.where(sig == SIG_CALL, conf_c,
                    np.where(sig == SIG_PUT, conf_p, skip_p))

    return sig, np.round(conf * 100).astype(int), len(models)


# ══════════════════════════════════════════════════════════════════════════
# Per-bar gate verdicts — computed ONCE, reused by every config
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class BarGates:
    """Everything the gates decided about one candidate bar."""
    regime_name: str = "UNKNOWN"
    regime_favors: str = "NEITHER"
    regime_floor: int = 65
    trap_blocked: bool = False
    trap_reason: str = ""
    vix_conf_adj: int = 0
    vix_sl_mult: float = 1.0
    vix_tp_mult: float = 1.0


def precompute_gates(df: pd.DataFrame, sig: np.ndarray, verbose: bool) -> dict:
    """Walk the history once, driving the real gate objects.

    TrapDetector is stateful (it tracks the session low and when it was made),
    so every bar feeds update_tick even though only candidate bars are asked
    for a verdict.
    """
    regime_engine = RegimeEngine()
    trap = TrapDetector()

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index
    vix_col = next((c for c in ("vix", "vix_close", "vix_level")
                    if c in df.columns), None)
    vix = df[vix_col].values if vix_col else None
    atr = df["a14"].values if "a14" in df.columns else np.full(len(df), 15.0)
    rsi = df["rsi14"].values if "rsi14" in df.columns else np.full(len(df), 50.0)

    out: dict[int, BarGates] = {}
    day_high = day_low = np.nan
    cur_day = None
    prev_vix = 0.0
    n_cand = int((sig != SIG_SKIP).sum())
    done = 0
    t0 = time.time()

    for i in range(len(df)):
        ts = idx[i]
        d = ts.date()
        if d != cur_day:
            cur_day = d
            day_high, day_low = highs[i], lows[i]
            trap = TrapDetector()          # fresh session state
        else:
            day_high = max(day_high, highs[i])
            day_low = min(day_low, lows[i])

        spot = float(closes[i])
        trap.update_tick(spot, ts.to_pydatetime())

        if sig[i] == SIG_SKIP:
            continue

        g = BarGates()
        window = df.iloc[max(0, i - 59): i + 1][["open", "high", "low", "close"]]
        try:
            r = regime_engine.classify(window, spot, float(day_high),
                                       float(day_low), ts.to_pydatetime())
            g.regime_name = r.name
            g.regime_favors = r.favored_direction
            g.regime_floor = int(r.confidence_floor)
        except Exception:
            pass

        option_type = "CE" if sig[i] == SIG_CALL else "PE"
        try:
            v = trap.is_trap(
                option_type=option_type,
                spot=spot,
                ml_indicators={
                    "atr": float(atr[i]),
                    "rsi": float(rsi[i]),
                    "recent_bar_range": float(highs[i] - lows[i]),
                    "pa_vwap_distance": 0.0,   # needs live VWAP feed
                },
                now=ts.to_pydatetime(),
            )
            g.trap_blocked = bool(v.is_trap)
            g.trap_reason = v.reason
        except Exception:
            pass

        if vix is not None and not np.isnan(vix[i]):
            try:
                vr = classify_vix(float(vix[i]), prev_vix)
                g.vix_conf_adj = int(getattr(vr, "confidence_adj", 0))
                g.vix_sl_mult = float(getattr(vr, "sl_multiplier", 1.0))
                g.vix_tp_mult = float(getattr(vr, "tp_multiplier", 1.0))
                prev_vix = float(vix[i])
            except Exception:
                pass

        out[i] = g
        done += 1
        if verbose and done % 5000 == 0:
            print(f"    gates {done}/{n_cand} ({time.time()-t0:.0f}s)")

    return out


# ══════════════════════════════════════════════════════════════════════════
# Gate configuration
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GateSet:
    conf_floor: bool = True
    vix_regime: bool = True
    regime_floor: bool = True
    counter_regime: bool = True
    trap: bool = True
    max_loss: bool = True
    whipsaw: bool = True
    day_halt: bool = True

    @staticmethod
    def none() -> "GateSet":
        return GateSet(False, False, False, False, False, False, False, False)


GATE_NAMES = ("conf_floor", "vix_regime", "regime_floor", "counter_regime",
              "trap", "max_loss", "whipsaw", "day_halt")


# ══════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════
def simulate(df: pd.DataFrame, sig: np.ndarray, conf: np.ndarray,
             gates: dict, gs: GateSet, lo: int, hi: int,
             min_conf: int, exit_model: str) -> dict:
    """Replay [lo, hi) chronologically, one position at a time.

    Returns a dict of performance stats plus a count of why entries died.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index
    atr14 = df["a14"].values if "a14" in df.columns else np.full(len(df), 15.0)

    trades: list[float] = []
    blocked: dict[str, int] = {}
    pos = None            # (dir, entry_px, sl_px, tp_px, entry_i)
    cur_day = None
    day_halted = False
    last_loss_dir = ""
    last_loss_ts = None

    def block(reason: str) -> None:
        blocked[reason] = blocked.get(reason, 0) + 1

    for i in range(lo, hi):
        ts = idx[i]
        d = ts.date()
        if d != cur_day:
            cur_day = d
            day_halted = False
            last_loss_dir, last_loss_ts = "", None
            if pos is not None:                    # safety: never cross days
                pos = None
        hhmm = ts.strftime("%H:%M")

        # ── manage an open position first ────────────────────────────────
        if pos is not None:
            direction, entry_px, sl_px, tp_px, _ = pos
            hit = None
            if direction == "CALL":
                if lows[i] <= sl_px:
                    hit = (sl_px - entry_px, "SL")
                elif highs[i] >= tp_px:
                    hit = (tp_px - entry_px, "TP")
            else:
                if highs[i] >= sl_px:
                    hit = (entry_px - sl_px, "SL")
                elif lows[i] <= tp_px:
                    hit = (entry_px - tp_px, "TP")
            if hit is None and hhmm >= EOD_SQUAREOFF:
                pnl = (closes[i] - entry_px) if direction == "CALL" \
                    else (entry_px - closes[i])
                hit = (pnl, "EOD")
            if hit is not None:
                pnl = hit[0] - FRICTION_PTS
                trades.append(pnl)
                if pnl < 0:
                    if gs.day_halt:
                        day_halted = True
                    last_loss_dir, last_loss_ts = direction, ts
                pos = None
            continue

        # ── entry gates ──────────────────────────────────────────────────
        if sig[i] == SIG_SKIP:
            continue
        if hhmm >= ENTRY_CUTOFF or hhmm < "09:20":
            continue
        if gs.day_halt and day_halted:
            block("day_halt")
            continue

        direction = "CALL" if sig[i] == SIG_CALL else "PUT"
        g = gates.get(i, BarGates())
        c = int(conf[i])

        if gs.whipsaw and last_loss_ts is not None and last_loss_dir:
            age_min = (ts - last_loss_ts).total_seconds() / 60.0
            if age_min < WHIPSAW_COOLDOWN_MIN and direction != last_loss_dir:
                block("whipsaw")
                continue

        eff = min_conf
        if gs.vix_regime:
            c = max(0, min(100, c + g.vix_conf_adj))
        if gs.regime_floor:
            eff = max(eff, g.regime_floor)
        if gs.counter_regime and g.regime_favors in ("CALL", "PUT") \
                and g.regime_favors != direction:
            eff = max(eff, 65)
        if gs.conf_floor and c < eff:
            block("conf_floor")
            continue
        if gs.trap and g.trap_blocked:
            block(f"trap:{g.trap_reason}" if g.trap_reason else "trap")
            continue

        # ── size the stop ────────────────────────────────────────────────
        # train_model_v9.py:432 stores a14 as ATR/close — a ratio, NOT points.
        # Multiplying by the bar's close recovers ATR in index points, which
        # is the unit every stop below is expressed in.
        a = float(atr14[i]) * float(closes[i])
        if not np.isfinite(a) or a <= 0:
            a = 15.0
        if exit_model == "frozen":
            sl_pts, tp_pts = a * 6.0, a * 2.0       # validated research config
        elif exit_model == "eod":
            # The exit that produced V11's PF 1.255: wide stop, no target,
            # ride to the close. ml_eod_brain.py stop_R=2.0 x atr_mult=2.0.
            sl_pts, tp_pts = a * 4.0, 1e9
        else:
            sl_pts = max(20.0, min(90.0, a * 2.2))  # approximates live
            tp_pts = sl_pts * 2.0
        if gs.vix_regime and exit_model != "eod":
            sl_pts, tp_pts = apply_vix_to_sl_tp(
                sl_pts, tp_pts,
                type("R", (), {"sl_multiplier": g.vix_sl_mult,
                               "tp_multiplier": g.vix_tp_mult})(),
            )

        if gs.max_loss:
            fit_sl = (MAX_LOSS_CAP / LOT_SIZE) * 0.995
            if sl_pts * LOT_SIZE > MAX_LOSS_CAP:
                if fit_sl >= sl_pts * SL_TIGHTEN_BAND:
                    sl_pts = round(fit_sl, 1)       # SL-TIGHTEN fallback
                else:
                    block("max_loss")
                    continue

        entry_px = float(closes[i])
        if direction == "CALL":
            pos = (direction, entry_px, entry_px - sl_pts, entry_px + tp_pts, i)
        else:
            pos = (direction, entry_px, entry_px + sl_pts, entry_px - tp_pts, i)

    return _stats(trades, blocked)


def _stats(trades: list, blocked: dict) -> dict:
    a = np.array(trades, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "pf": 0.0, "avg": 0.0, "win": 0.0,
                "total": 0.0, "maxdd": 0.0, "blocked": blocked}
    wins, losses = a[a > 0].sum(), -a[a < 0].sum()
    eq = np.cumsum(a)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if n else 0.0
    return {
        "n": n,
        "pf": float(wins / losses) if losses > 0 else float("inf"),
        "avg": float(a.mean()),
        "win": float((a > 0).mean() * 100),
        "total": float(a.sum()),
        "maxdd": dd,
        "blocked": blocked,
    }


def _row(label: str, s: dict) -> str:
    pf = "  inf " if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    return (f"{label:<26} {pf}  {s['n']:>5}  {s['avg']:+7.2f}  "
            f"{s['win']:5.1f}%  {s['total']:+9.0f}  {s['maxdd']:8.0f}")


HDR = (f"{'config':<26} {'PF':>6}  {'n':>5}  {'avg':>7}  {'win':>6}  "
       f"{'total':>9}  {'maxDD':>8}")


# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the live gate stack")
    ap.add_argument("--conf-sweep", default="45,50,52,55,58,62,65,70",
                    help="confidence floors to test on VAL")
    ap.add_argument("--exit", dest="exit_model", default="dynamic",
                    choices=("dynamic", "frozen", "eod"))
    ap.add_argument("--cache", default="data/.gate_replay_cache.pkl",
                    help="scores+gates cache; delete the file to rebuild")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    verbose = not a.quiet
    # Windows consoles default to cp1252 and die on any non-ASCII output.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    # Scoring + gating cost ~8 minutes and never change between exit models
    # or gate configs, so they are cached. Delete the file to force a rebuild.
    cache = Path(a.cache) if a.cache else None
    payload = None
    if cache and cache.exists():
        print(f"Loading cached scores+gates from {cache.name} ...")
        try:
            payload = joblib.load(cache)
        except Exception as e:
            print(f"  cache unreadable ({e}) — rebuilding")

    if payload is None:
        print("Building dataset (V9 pipeline)...")
        full = V11.build_dataset(V9.CSV_5M, V9.CSV_15M, V9.CSV_30M, V9.CSV_60M,
                                 V9.CSV_DAY, V9.CSV_VIX, V9.CSV_FUT)
        print(f"  rows={len(full):,}  span {full.index[0]:%Y-%m-%d} .. "
              f"{full.index[-1]:%Y-%m-%d}   ({time.time()-t0:.0f}s)")

        print("Scoring with promoted V9...")
        sig, conf, n_models = score_all(full)
        print(f"  {n_models} ensemble members | CALL={int((sig==0).sum()):,} "
              f"PUT={int((sig==1).sum()):,} SKIP={int((sig==2).sum()):,}")

        print("Driving the real gate objects over history...")
        gates = precompute_gates(full, sig, verbose)
        print(f"  {len(gates):,} candidate bars gated   ({time.time()-t0:.0f}s)")

        keep_cols = [c for c in ("open", "high", "low", "close", "a14")
                     if c in full.columns]
        payload = {"df": full[keep_cols].copy(), "sig": sig,
                   "conf": conf, "gates": gates, "n": len(full)}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, cache, compress=3)
            print(f"  cached -> {cache}")

    df = payload["df"]
    sig, conf, gates = payload["sig"], payload["conf"], payload["gates"]
    print(f"  {len(df):,} bars | {len(gates):,} candidate bars gated "
          f"({time.time()-t0:.0f}s)")

    tr_i, va_i, te_i = V9.purged_train_val_test_split(len(df))
    va_lo, va_hi = int(va_i[0]), int(va_i[-1]) + 1
    te_lo, te_hi = int(te_i[0]), int(te_i[-1]) + 1
    print(f"\nsplit: TRAIN {len(tr_i):,} (in-sample, not reported) | "
          f"VAL {len(va_i):,} | TEST {len(te_i):,}")
    print(f"VAL  {df.index[va_lo]:%Y-%m-%d} .. {df.index[va_hi-1]:%Y-%m-%d}")
    print(f"TEST {df.index[te_lo]:%Y-%m-%d} .. {df.index[te_hi-1]:%Y-%m-%d}")

    def run(gs, lo, hi, mc):
        return simulate(df, sig, conf, gates, gs, lo, hi, mc, a.exit_model)

    all_on = GateSet()
    all_off = GateSet.none()

    # ── 1. confidence sweep, on VAL, everything else on ──────────────────
    print(f"\n{'='*92}\n1. CONFIDENCE FLOOR SWEEP (VAL, all other gates ON)"
          f"\n{'='*92}\n{HDR}")
    floors = [int(x) for x in a.conf_sweep.split(",")]
    best_mc, best_pf = BASE_MIN_CONF, -1.0
    for mc in floors:
        s = run(all_on, va_lo, va_hi, mc)
        tag = f"min_conf={mc}%" + ("  <- live" if mc == BASE_MIN_CONF else "")
        print(_row(tag, s))
        if s["n"] >= 25 and s["pf"] > best_pf:
            best_pf, best_mc = s["pf"], mc

    # ── 2. leave-one-out, on VAL, at the live floor ──────────────────────
    print(f"\n{'='*92}\n2. LEAVE-ONE-OUT (VAL, min_conf={BASE_MIN_CONF}%)"
          f"\n{'='*92}\n{HDR}")
    base = run(all_on, va_lo, va_hi, BASE_MIN_CONF)
    print(_row("ALL GATES ON (live)", base))
    print(_row("ALL GATES OFF", run(all_off, va_lo, va_hi, BASE_MIN_CONF)))
    print("-" * 92)
    loo = {}
    for g in GATE_NAMES:
        s = run(replace(all_on, **{g: False}), va_lo, va_hi, BASE_MIN_CONF)
        loo[g] = s
        delta = s["pf"] - base["pf"]
        print(_row(f"without {g}", s) + f"   dPF {delta:+.3f}")

    print("\nA positive ΔPF means removing that gate IMPROVED the result —")
    print("i.e. the gate is costing you money on 11 years of out-of-sample bars.")

    print(f"\nwhy entries died (VAL, all gates on, n_blocked):")
    for k, v in sorted(base["blocked"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v:>6}")

    # ── 3. TEST — confirmed once, no tuning after this ───────────────────
    keep = [g for g in GATE_NAMES if loo[g]["pf"] <= base["pf"]]
    chosen = GateSet(**{g: (g in keep) for g in GATE_NAMES})
    print(f"\n{'='*92}\n3. TEST — CONFIRM ONCE (no further tuning)\n{'='*92}")
    print(f"VAL picked min_conf={best_mc}%  and gates={sorted(keep)}")
    print(HDR)
    print(_row("live config", run(all_on, te_lo, te_hi, BASE_MIN_CONF)))
    print(_row("no gates at all", run(all_off, te_lo, te_hi, BASE_MIN_CONF)))
    print(_row(f"VAL-selected", run(chosen, te_lo, te_hi, best_mc)))

    print(f"\n{'='*92}\nUNMEASURABLE — no history exists for these five gates")
    print(f"{'='*92}")
    for g, why in (("IV GATE", "iv_history.jsonl holds 1 record"),
                   ("OI FILTER", "oi_archive covers 29 days"),
                   ("PCR alignment", "same 29 days"),
                   ("BRIEF bias", "18 premarket_brief files"),
                   ("STRUCTURE PENALTY", "needs 3-minute bars; none stored")):
        print(f"    {g:<20} {why}")
    print("\nThose five veto live trades daily and nothing here judges them.")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
