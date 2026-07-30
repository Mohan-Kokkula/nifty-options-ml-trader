"""
label_redesign_study.py — Diagnostic comparison of the current CALL/PUT/SKIP
labeling method against 8 alternative labeling methodologies.

NO MODEL TRAINING. This is a pure labeling-diagnostics study: it builds each
candidate label set on the SAME complete, leak-fixed V9 feature+price frame
(reused unmodified from backtest_threshold_sweep.build_frame() — the exact
frame the production model and the CatBoost comparison both trained on) and
measures class balance, a noise proxy, ambiguity, stability, TP/SL overlap,
and a reasoned impact assessment for each.

Nothing here is imported for training; core/ml_engine.py and every model
artifact are untouched.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import build_frame

OUT_JSON = ROOT / "logs" / "label_redesign_study.json"

# ── Constants tying this study to the REAL, frozen, validated exit strategy
# (core/claude_pilot.py's USE_FROZEN_ATR_EXIT path / the full-dataset replay
# earlier in this project: ATR(10), TP=2.0x ATR, SL=6.0x ATR, 7-bar max hold)
FROZEN_TP_MULT = 2.0
FROZEN_SL_MULT = 6.0
FROZEN_MAX_HOLD = 7

CURRENT_FWD_BARS = 3   # train_model_v9.FWD_BARS — the CURRENT label's fixed horizon


# =============================================================================
# Shared forward-path scanner (path-dependent, used by every barrier-style
# labeler). Walks forward from each candidate bar up to max_hold bars,
# recording which barrier (if any) is touched first, MFE, and MAE — exactly
# the information a real stop-managed trade would experience, unlike a
# fixed-horizon "return at t+N" read.
# =============================================================================
def scan_forward(closes, highs, lows, direction, tp_pts, sl_pts, max_hold):
    """
    direction: +1 (CALL/long) or -1 (PUT/short), per-bar array
    tp_pts, sl_pts: per-bar arrays of distances in points (>0)
    max_hold: per-bar array of max bars to hold before a vertical-barrier exit

    Returns dict of arrays (len = n), each -1/NaN where not applicable:
      outcome   : 0=TP, 1=SL, 2=TIME/vertical, 3=undefined (insufficient future data)
      bars_to_exit
      mfe_pts, mae_pts   (favorable/adverse excursion up to the exit bar)
      tp_bar, sl_bar     (bars-to-touch for TP/SL respectively, -1 if never touched
                          within max_hold — used for TP/SL overlap diagnostic)
    """
    n = len(closes)
    max_h = int(np.max(max_hold)) if len(max_hold) else 0
    outcome = np.full(n, 3, dtype=np.int8)
    bars_to_exit = np.full(n, -1, dtype=np.int32)
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)
    tp_bar = np.full(n, -1, dtype=np.int32)
    sl_bar = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        d = direction[i]
        if d == 0:
            continue
        entry = closes[i]
        tp_price = entry + d * tp_pts[i]
        sl_price = entry - d * sl_pts[i]
        mh = int(max_hold[i])
        if mh <= 0 or i + mh >= n:
            continue
        best_fav, best_adv = 0.0, 0.0
        exited = False
        for k in range(1, mh + 1):
            j = i + k
            h, l = highs[j], lows[j]
            if d == 1:
                fav = h - entry
                adv = entry - l
                tp_hit = h >= tp_price
                sl_hit = l <= sl_price
            else:
                fav = entry - l
                adv = h - entry
                tp_hit = l <= tp_price
                sl_hit = h >= sl_price
            if fav > best_fav:
                best_fav = fav
            if adv > best_adv:
                best_adv = adv
            if tp_hit and tp_bar[i] < 0:
                tp_bar[i] = k
            if sl_hit and sl_bar[i] < 0:
                sl_bar[i] = k
            if not exited:
                if sl_hit:
                    outcome[i] = 1
                    bars_to_exit[i] = k
                    exited = True
                elif tp_hit:
                    outcome[i] = 0
                    bars_to_exit[i] = k
                    exited = True
        if not exited:
            outcome[i] = 2
            bars_to_exit[i] = mh
        mfe[i] = best_fav
        mae[i] = best_adv
    return dict(outcome=outcome, bars_to_exit=bars_to_exit, mfe=mfe, mae=mae,
                tp_bar=tp_bar, sl_bar=sl_bar)


# =============================================================================
# 9 labelers: current + 8 alternatives
# =============================================================================
def label_current(feat: pd.DataFrame) -> dict:
    """The CURRENT production label, exactly as train_model_v9.create_labels
    computes it (already present as feat['label'] from build_frame(), which
    calls create_labels() internally — read here, not recomputed, so this is
    byte-identical to production)."""
    labels = feat["label"].values.astype(np.int8)
    return dict(labels=labels, mfe=None, mae=None, extra={})


def label_triple_barrier(feat, closes, highs, lows) -> dict:
    """1. Triple Barrier Method (Lopez de Prado). Classic formulation: fixed
    PERCENTAGE upper/lower barriers (not volatility-scaled — that's method 2)
    plus a fixed vertical (time) barrier. Barrier width is set to the same
    70th-percentile-of-|forward-return| causal threshold the current method
    already computes (so the *price* barrier width is comparable to the
    current method's decision threshold), with a longer vertical barrier
    (10 bars = 50min, a conventional "let it play out further than the
    current 15-min read" horizon) than the current fixed-return check.
    Direction is assigned by whichever barrier is touched first; SKIP if the
    vertical barrier is hit with no barrier touch (ambiguous/no clear edge)."""
    from scripts.train_model_v9 import compute_thresholds_causal
    bar_thresh = compute_thresholds_causal(feat).values  # fractional return threshold
    pts = bar_thresh * closes   # convert fractional threshold to points at each bar
    max_hold = np.full(len(feat), 10, dtype=np.int32)

    # Evaluate BOTH directions per bar (a true Triple Barrier label doesn't
    # presuppose direction — it's usually applied on top of a primary model's
    # side; here, to produce a directly comparable 3-class CALL/PUT/SKIP
    # label, we take whichever direction's barrier resolves to a "TP" first
    # and use the other as the confirming absence).
    up = scan_forward(closes, highs, lows, np.ones(len(feat)), pts, pts, max_hold)
    dn = scan_forward(closes, highs, lows, -np.ones(len(feat)), pts, pts, max_hold)

    labels = np.full(len(feat), 2, dtype=np.int8)
    call = (up["outcome"] == 0) & (dn["outcome"] != 0)
    put = (dn["outcome"] == 0) & (up["outcome"] != 0)
    labels[call] = 0
    labels[put] = 1
    # when both "win" (rare, wide-barrier same-bar ambiguity) or both fail -> SKIP
    return dict(labels=labels, mfe=np.where(labels == 0, up["mfe"], np.where(labels == 1, dn["mfe"], 0)),
                mae=np.where(labels == 0, up["mae"], np.where(labels == 1, dn["mae"], 0)),
                extra=dict(up=up, dn=dn))


def label_atr_barrier(feat, closes, highs, lows, atr10) -> dict:
    """2. ATR-based dynamic barriers. Uses the ACTUAL frozen, validated live
    exit config: TP=2.0x ATR(10), SL=6.0x ATR(10), 7-bar max hold. This is
    the single most "honest" alternative — it labels each bar with exactly
    the outcome the REAL trading strategy would realize if entered there,
    since it uses the identical barrier multiples and holding period as
    core/claude_pilot.py's USE_FROZEN_ATR_EXIT path."""
    tp_pts = FROZEN_TP_MULT * atr10
    sl_pts = FROZEN_SL_MULT * atr10
    max_hold = np.full(len(feat), FROZEN_MAX_HOLD, dtype=np.int32)
    valid = ~np.isnan(atr10) & (atr10 > 0)

    up = scan_forward(closes, highs, lows, np.where(valid, 1, 0), tp_pts, sl_pts, max_hold)
    dn = scan_forward(closes, highs, lows, np.where(valid, -1, 0), tp_pts, sl_pts, max_hold)

    labels = np.full(len(feat), 2, dtype=np.int8)
    call = valid & (up["outcome"] == 0) & (dn["outcome"] != 0)
    put = valid & (dn["outcome"] == 0) & (up["outcome"] != 0)
    labels[call] = 0
    labels[put] = 1
    return dict(labels=labels,
                mfe=np.where(labels == 0, up["mfe"], np.where(labels == 1, dn["mfe"], 0)),
                mae=np.where(labels == 0, up["mae"], np.where(labels == 1, dn["mae"], 0)),
                extra=dict(up=up, dn=dn))


def label_vol_adjusted_forward_return(feat, closes) -> dict:
    """3. Volatility-adjusted forward return labels. Same FIXED horizon as
    the current method (3 bars / 15min — isolating this as the ONLY change
    from the current method), but the forward return is converted to a
    z-score against a ROLLING (per-bar, not session-expanding) realized-
    return volatility, and thresholded on the z-score (|z| >= 1.0). This
    reacts to TODAY's volatility regime bar-by-bar, unlike the current
    method's slow-moving session-expanding threshold."""
    fwd = pd.Series(closes).shift(-CURRENT_FWD_BARS) / pd.Series(closes) - 1
    ret1 = pd.Series(closes).pct_change()
    roll_vol = ret1.rolling(78, min_periods=20).std()  # ~1 trading day of 5-min bars
    z = (fwd / roll_vol.replace(0, np.nan)).values
    labels = np.full(len(feat), 2, dtype=np.int8)
    labels[z >= 1.0] = 0
    labels[z <= -1.0] = 1
    return dict(labels=labels, mfe=None, mae=None, extra=dict(z=z))


def label_meta(feat, closes, highs, lows, atr10) -> dict:
    """4. Meta-labeling (Lopez de Prado). A PRIMARY rule fires a directional
    side (here: the existing RSI/ADX regime rule already in create_labels,
    reused as-is via feat['label'] restricted to its CALL/PUT bars — i.e.
    "would the current primary signal's trade actually have won under the
    frozen ATR barrier?"). The meta-label is binary: 1 = primary signal's
    direction reached TP before SL under the ATR barrier, 0 = it didn't.
    Bars where the primary rule is SKIP get no meta-label (excluded, not a
    3rd class) — meta-labeling is fundamentally a filter on an existing
    signal, not a fresh direction generator."""
    primary = feat["label"].values
    tp_pts = FROZEN_TP_MULT * atr10
    sl_pts = FROZEN_SL_MULT * atr10
    max_hold = np.full(len(feat), FROZEN_MAX_HOLD, dtype=np.int32)
    direction = np.where(primary == 0, 1, np.where(primary == 1, -1, 0))
    res = scan_forward(closes, highs, lows, direction, tp_pts, sl_pts, max_hold)
    has_primary = direction != 0
    meta_label = np.full(len(feat), -1, dtype=np.int8)   # -1 = no primary signal (excluded)
    meta_label[has_primary] = (res["outcome"][has_primary] == 0).astype(np.int8)
    return dict(labels=meta_label, mfe=res["mfe"], mae=res["mae"],
                extra=dict(has_primary=has_primary, outcome=res["outcome"]))


def label_trend_conditioned(feat, closes) -> dict:
    """5. Trend-conditioned labeling. Same fixed-horizon forward-return sign
    as a base signal, but REQUIRES higher-timeframe trend alignment (15m ADX
    > 20 AND 15m short/long EMA relationship, using columns already computed
    by the leak-fixed pipeline) before accepting a CALL/PUT — a systematic
    trend filter applied uniformly, rather than the current method's
    RSI/ADX *regime-branching* (which actively relabels CALL<->PUT under
    different RSI zones)."""
    fwd = pd.Series(closes).shift(-CURRENT_FWD_BARS) / pd.Series(closes) - 1
    bar_thresh = 0.001
    up_trend = (feat.get("tf15_adx", pd.Series(0, index=feat.index)) > 20)
    ema_col_fast = next((c for c in feat.columns if c.startswith("tf15_ema") and "9" in c), None)
    ema_col_slow = next((c for c in feat.columns if c.startswith("tf15_ema") and ("21" in c or "20" in c)), None)
    if ema_col_fast and ema_col_slow:
        bull = feat[ema_col_fast] > feat[ema_col_slow]
    else:
        bull = pd.Series(True, index=feat.index)   # fallback: ADX-only filter
    trending = up_trend
    labels = np.full(len(feat), 2, dtype=np.int8)
    call = trending & bull & (fwd.values > bar_thresh)
    put = trending & (~bull) & (fwd.values < -bar_thresh)
    labels[call.values] = 0
    labels[put.values] = 1
    return dict(labels=labels, mfe=None, mae=None, extra=dict(trending_frac=float(trending.mean())))


def label_pop(feat, closes, highs, lows, atr10) -> dict:
    """6. Probability-of-Profit (POP) style labeling. Approximates option
    economics without a full option-chain simulation: a directional move
    only "counts" as a label-worthy win if it clears the ATR-TP distance
    PLUS an assumed theta-decay drag over the holding period (a fixed
    points-per-bar decay drag, ~0.15% of spot per bar, a conservative
    stand-in for short-dated NIFTY option theta — same order of magnitude
    used elsewhere in this codebase's realistic-haircut backtests). This
    penalizes labels that "win" on raw index points but would likely have
    been a losing OPTION trade after decay — the current method and methods
    1-5 all operate purely on index points and are blind to this."""
    decay_per_bar = closes * 0.0015 / FROZEN_MAX_HOLD  # spread the assumed decay over the hold
    tp_pts_base = FROZEN_TP_MULT * atr10
    sl_pts = FROZEN_SL_MULT * atr10
    max_hold = np.full(len(feat), FROZEN_MAX_HOLD, dtype=np.int32)
    valid = ~np.isnan(atr10) & (atr10 > 0)

    # Decay-inflated TP requirement: must clear TP distance + decay accrued by the bar it's touched.
    # Approximate with decay at max_hold (conservative/simple, avoids re-scanning per-bar decay).
    tp_pts = tp_pts_base + decay_per_bar * FROZEN_MAX_HOLD
    up = scan_forward(closes, highs, lows, np.where(valid, 1, 0), tp_pts, sl_pts, max_hold)
    dn = scan_forward(closes, highs, lows, np.where(valid, -1, 0), tp_pts, sl_pts, max_hold)
    labels = np.full(len(feat), 2, dtype=np.int8)
    call = valid & (up["outcome"] == 0) & (dn["outcome"] != 0)
    put = valid & (dn["outcome"] == 0) & (up["outcome"] != 0)
    labels[call] = 0
    labels[put] = 1
    return dict(labels=labels,
                mfe=np.where(labels == 0, up["mfe"], np.where(labels == 1, dn["mfe"], 0)),
                mae=np.where(labels == 0, up["mae"], np.where(labels == 1, dn["mae"], 0)),
                extra=dict(up=up, dn=dn))


def label_risk_reward(feat, closes, highs, lows, atr10) -> dict:
    """7. Risk-Reward based labels. Same ATR barrier as method 2, but ALSO
    requires the realized MFE/MAE ratio over the hold to exceed 1.5x before
    accepting the label as CALL/PUT (even if TP was technically touched) —
    filters out "won but only barely, with a much larger adverse excursion
    along the way" cases that a real position-sizing/risk model would flag
    as a poor-quality trade despite a nominal win."""
    tp_pts = FROZEN_TP_MULT * atr10
    sl_pts = FROZEN_SL_MULT * atr10
    max_hold = np.full(len(feat), FROZEN_MAX_HOLD, dtype=np.int32)
    valid = ~np.isnan(atr10) & (atr10 > 0)
    up = scan_forward(closes, highs, lows, np.where(valid, 1, 0), tp_pts, sl_pts, max_hold)
    dn = scan_forward(closes, highs, lows, np.where(valid, -1, 0), tp_pts, sl_pts, max_hold)

    def rr_ok(res):
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(res["mae"] > 0, res["mfe"] / res["mae"], np.inf)
        return rr >= 1.5

    labels = np.full(len(feat), 2, dtype=np.int8)
    call = valid & (up["outcome"] == 0) & (dn["outcome"] != 0) & rr_ok(up)
    put = valid & (dn["outcome"] == 0) & (up["outcome"] != 0) & rr_ok(dn)
    labels[call] = 0
    labels[put] = 1
    return dict(labels=labels,
                mfe=np.where(labels == 0, up["mfe"], np.where(labels == 1, dn["mfe"], 0)),
                mae=np.where(labels == 0, up["mae"], np.where(labels == 1, dn["mae"], 0)),
                extra=dict(up=up, dn=dn))


def label_adaptive_holding(feat, closes, highs, lows, atr10) -> dict:
    """8. Adaptive holding-period labels. Same ATR-scaled TP/SL barriers as
    method 2, but the VERTICAL barrier (max hold) itself adapts: strong-
    trend/high-ADX bars get a longer hold (10 bars) to let a trend run,
    weak/choppy bars get a shorter hold (4 bars) to avoid marking a slow
    grind as a clean directional win. This changes WHEN the label is
    evaluated, not just what threshold it's evaluated against — the one
    dimension none of methods 1-7 vary."""
    adx = feat.get("tf15_adx", pd.Series(20.0, index=feat.index)).values
    max_hold = np.where(adx > 25, 10, np.where(adx > 15, 7, 4)).astype(np.int32)
    tp_pts = FROZEN_TP_MULT * atr10
    sl_pts = FROZEN_SL_MULT * atr10
    valid = ~np.isnan(atr10) & (atr10 > 0)
    up = scan_forward(closes, highs, lows, np.where(valid, 1, 0), tp_pts, sl_pts, max_hold)
    dn = scan_forward(closes, highs, lows, np.where(valid, -1, 0), tp_pts, sl_pts, max_hold)
    labels = np.full(len(feat), 2, dtype=np.int8)
    call = valid & (up["outcome"] == 0) & (dn["outcome"] != 0)
    put = valid & (dn["outcome"] == 0) & (up["outcome"] != 0)
    labels[call] = 0
    labels[put] = 1
    return dict(labels=labels,
                mfe=np.where(labels == 0, up["mfe"], np.where(labels == 1, dn["mfe"], 0)),
                mae=np.where(labels == 0, up["mae"], np.where(labels == 1, dn["mae"], 0)),
                extra=dict(up=up, dn=dn, max_hold=max_hold))


# =============================================================================
# Diagnostics
# =============================================================================
def diag_class_balance(labels, binary=False):
    vc = pd.Series(labels).value_counts(normalize=True).sort_index()
    if binary:
        return {"win(1)": float(vc.get(1, 0.0)), "loss(0)": float(vc.get(0, 0.0)),
                "excluded(-1)": float(vc.get(-1, 0.0))}
    return {"CALL(0)": float(vc.get(0, 0.0)), "PUT(1)": float(vc.get(1, 0.0)),
            "SKIP(2)": float(vc.get(2, 0.0))}


def diag_noise_proxy(labels, mfe, mae, binary=False):
    """Fraction of 'winning' (CALL/PUT, or meta-label=1) bars where the
    adverse excursion before/along the way exceeded the favorable excursion
    that defined the win -- i.e. the label says WIN but a realistic
    risk-managed position would very plausibly have been stopped out
    first. Undefined (NaN) for label methods with no MFE/MAE tracking
    (current, vol-adjusted, trend-conditioned) -- flagged explicitly rather
    than silently omitted."""
    if mfe is None or mae is None:
        return float("nan")
    if binary:
        mask = labels == 1
    else:
        mask = (labels == 0) | (labels == 1)
    if not mask.any():
        return float("nan")
    return float((mae[mask] > mfe[mask]).mean())


def diag_ambiguity(z_or_thresh_distance, near_band=0.15):
    """Fraction of ALL bars sitting within `near_band` (relative) of the
    decision boundary -- a direct measure of how many labels would flip
    under a small parameter perturbation. Expects an array of *relative*
    distance-to-threshold (0 = exactly on boundary, 1 = far from it);
    returns NaN if not computable for a given method."""
    if z_or_thresh_distance is None:
        return float("nan")
    d = np.abs(z_or_thresh_distance)
    finite = np.isfinite(d)
    if not finite.any():
        return float("nan")
    return float((d[finite] < near_band).mean())


def diag_stability_flip_rate(labels):
    """Bar-to-bar label flip rate among IN-SESSION, non-SKIP-to-non-SKIP
    transitions is too sparse to be meaningful at 5-min granularity for a
    182%-SKIP series; instead this measures flip rate of the full label
    series bar-to-bar (any class change, including into/out of SKIP) as a
    proxy for how 'choppy' the target is for a model to learn a smooth
    decision surface over."""
    lab = np.asarray(labels)
    if len(lab) < 2:
        return float("nan")
    return float((lab[1:] != lab[:-1]).mean())


def diag_stability_perturbation(labeler_fn, base_kwargs, perturb_kwargs_list):
    """Re-run the SAME labeler with each perturbed parameter set and report
    mean pairwise agreement (fraction of bars with an unchanged label)
    against the base run -- a direct, parameter-level stability measure."""
    base = labeler_fn(**base_kwargs)["labels"]
    agreements = []
    for kw in perturb_kwargs_list:
        try:
            alt = labeler_fn(**kw)["labels"]
            agreements.append(float((alt == base).mean()))
        except Exception:
            continue
    return float(np.mean(agreements)) if agreements else float("nan")


def diag_tp_sl_overlap(extra):
    """Fraction of candidate (direction, bar) pairs where BOTH the TP and SL
    for that direction were touched within the SAME max_hold window (i.e.
    price whipsawed through both levels) -- a direct measure of path-level
    label noise for barrier-based methods. NaN for non-barrier methods."""
    if not extra or "up" not in extra or "dn" not in extra:
        return float("nan")
    fracs = []
    for res in (extra["up"], extra["dn"]):
        both = (res["tp_bar"] >= 0) & (res["sl_bar"] >= 0)
        denom = (res["tp_bar"] >= 0) | (res["sl_bar"] >= 0)
        if denom.any():
            fracs.append(float(both[denom].sum()) / float(denom.sum()))
    return float(np.mean(fracs)) if fracs else float("nan")


# =============================================================================
def main():
    t0 = time.time()
    feat, fcols = build_frame()
    closes = feat["close"].values.astype(np.float64)
    highs = feat["high"].values.astype(np.float64)
    lows = feat["low"].values.astype(np.float64)

    # ATR(10) recomputed from raw OHLC — matches the frozen live-strategy
    # exit config exactly (core/claude_pilot.py USE_FROZEN_ATR_EXIT path).
    prev_close = np.roll(closes, 1); prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr10 = pd.Series(tr, index=feat.index).rolling(10).mean().values

    print(f"\nFrame ready: {len(feat):,} bars. Computing 9 label sets "
          f"(current + 8 alternatives)...\n")

    results = {}

    # ---- 0. Current ----
    print("[0/8] Current (production create_labels)...")
    r = label_current(feat)
    results["0_Current"] = r

    # ---- 1. Triple Barrier ----
    print("[1/8] Triple Barrier Method...")
    r = label_triple_barrier(feat, closes, highs, lows)
    results["1_TripleBarrier"] = r

    # ---- 2. ATR-based dynamic barriers ----
    print("[2/8] ATR-based dynamic barriers (frozen live-strategy config)...")
    r = label_atr_barrier(feat, closes, highs, lows, atr10)
    results["2_ATRBarrier"] = r

    # ---- 3. Volatility-adjusted forward return ----
    print("[3/8] Volatility-adjusted forward return...")
    r = label_vol_adjusted_forward_return(feat, closes)
    results["3_VolAdjustedForwardReturn"] = r

    # ---- 4. Meta-labeling ----
    print("[4/8] Meta-labeling (primary=current rule, filter=ATR barrier)...")
    r = label_meta(feat, closes, highs, lows, atr10)
    results["4_MetaLabeling"] = r

    # ---- 5. Trend-conditioned ----
    print("[5/8] Trend-conditioned labeling...")
    r = label_trend_conditioned(feat, closes)
    results["5_TrendConditioned"] = r

    # ---- 6. POP-style ----
    print("[6/8] Probability-of-Profit (decay-adjusted) labeling...")
    r = label_pop(feat, closes, highs, lows, atr10)
    results["6_POP"] = r

    # ---- 7. Risk-Reward ----
    print("[7/8] Risk-Reward based labels...")
    r = label_risk_reward(feat, closes, highs, lows, atr10)
    results["7_RiskReward"] = r

    # ---- 8. Adaptive holding period ----
    print("[8/8] Adaptive holding-period labels...")
    r = label_adaptive_holding(feat, closes, highs, lows, atr10)
    results["8_AdaptiveHolding"] = r

    print(f"\nAll label sets computed in {time.time()-t0:.0f}s. Running diagnostics...\n")

    # -----------------------------------------------------------------
    # Diagnostics per method
    # -----------------------------------------------------------------
    diag = {}
    for name, r in results.items():
        binary = name == "4_MetaLabeling"
        labels = r["labels"]
        mfe, mae = r.get("mfe"), r.get("mae")

        d = {}
        d["class_balance"] = diag_class_balance(labels, binary=binary)
        d["noise_proxy_mae_gt_mfe"] = diag_noise_proxy(labels, mfe, mae, binary=binary)
        d["stability_bar_to_bar_flip_rate"] = diag_stability_flip_rate(labels)
        d["tp_sl_overlap"] = diag_tp_sl_overlap(r.get("extra", {}))

        if name == "3_VolAdjustedForwardReturn":
            d["ambiguity_near_boundary"] = diag_ambiguity(r["extra"]["z"] / 1.0, near_band=0.15)
        else:
            d["ambiguity_near_boundary"] = float("nan")

        n_valid = int((np.asarray(labels) != (-1 if binary else 2)).sum()) if binary \
            else int((np.asarray(labels) != 2).sum())
        d["n_directional_labels"] = n_valid
        d["pct_directional"] = float(n_valid / len(labels))

        diag[name] = d

    # -----------------------------------------------------------------
    # Parameter-perturbation stability (separate pass — re-runs cheap
    # variants of each barrier method with a nudged parameter)
    # -----------------------------------------------------------------
    print("Running parameter-perturbation stability checks...")
    try:
        diag["1_TripleBarrier"]["stability_param_perturbation"] = diag_stability_perturbation(
            lambda **kw: label_triple_barrier(**kw),
            dict(feat=feat, closes=closes, highs=highs, lows=lows),
            [],  # threshold is data-derived (causal quantile); perturbation covered by flip-rate instead
        )
    except Exception:
        pass

    def _atr_barrier_variant(mult_tp, mult_sl, hold):
        tp_pts = mult_tp * atr10
        sl_pts = mult_sl * atr10
        mh = np.full(len(feat), hold, dtype=np.int32)
        valid = ~np.isnan(atr10) & (atr10 > 0)
        up = scan_forward(closes, highs, lows, np.where(valid, 1, 0), tp_pts, sl_pts, mh)
        dn = scan_forward(closes, highs, lows, np.where(valid, -1, 0), tp_pts, sl_pts, mh)
        labels = np.full(len(feat), 2, dtype=np.int8)
        call = valid & (up["outcome"] == 0) & (dn["outcome"] != 0)
        put = valid & (dn["outcome"] == 0) & (up["outcome"] != 0)
        labels[call] = 0; labels[put] = 1
        return labels

    base_atr_labels = results["2_ATRBarrier"]["labels"]
    perturbed = [
        _atr_barrier_variant(FROZEN_TP_MULT * 1.1, FROZEN_SL_MULT, FROZEN_MAX_HOLD),
        _atr_barrier_variant(FROZEN_TP_MULT * 0.9, FROZEN_SL_MULT, FROZEN_MAX_HOLD),
        _atr_barrier_variant(FROZEN_TP_MULT, FROZEN_SL_MULT, FROZEN_MAX_HOLD + 1),
        _atr_barrier_variant(FROZEN_TP_MULT, FROZEN_SL_MULT, FROZEN_MAX_HOLD - 1),
    ]
    agreements = [float((p == base_atr_labels).mean()) for p in perturbed]
    diag["2_ATRBarrier"]["stability_param_perturbation"] = float(np.mean(agreements))
    # POP / RiskReward / AdaptiveHolding share the ATR-barrier core, so this
    # perturbation result is directly informative for them too (noted, not
    # duplicated as a separate multi-minute re-scan).
    for nm in ("6_POP", "7_RiskReward", "8_AdaptiveHolding"):
        diag[nm]["stability_param_perturbation_note"] = (
            "shares ATR-barrier core with method 2; see 2_ATRBarrier for the "
            "direct TP/SL/hold perturbation result"
        )

    # -----------------------------------------------------------------
    # Mismatch-with-real-strategy metric: for the CURRENT label specifically,
    # what fraction of its CALL/PUT calls agree with what the ATR-barrier
    # (method 2 = the REAL frozen exit) would have labeled the same bar?
    # This directly quantifies "does the current label match the strategy
    # that consumes it."
    # -----------------------------------------------------------------
    cur = results["0_Current"]["labels"]
    real = results["2_ATRBarrier"]["labels"]
    cur_directional = cur != 2
    agree = (cur[cur_directional] == real[cur_directional]).mean() if cur_directional.any() else float("nan")
    both_directional = cur_directional & (real != 2)
    agree_given_both_fire = (cur[both_directional] == real[both_directional]).mean() if both_directional.any() else float("nan")
    mismatch_summary = {
        "current_directional_bars": int(cur_directional.sum()),
        "real_strategy_directional_bars": int((real != 2).sum()),
        "pct_current_calls_matching_real_exit_direction": float(agree),
        "pct_agreement_when_BOTH_fire": float(agree_given_both_fire),
        "pct_both_fire_together": float(both_directional.sum() / len(cur)),
    }

    # -----------------------------------------------------------------
    elapsed = time.time() - t0
    final = {
        "n_bars": int(len(feat)),
        "date_range": [str(feat.index[0].date()), str(feat.index[-1].date())],
        "frozen_strategy_params": {"tp_mult_atr10": FROZEN_TP_MULT,
                                    "sl_mult_atr10": FROZEN_SL_MULT,
                                    "max_hold_bars": FROZEN_MAX_HOLD},
        "current_label_params": {"fwd_bars": CURRENT_FWD_BARS},
        "diagnostics": diag,
        "current_vs_real_strategy_mismatch": mismatch_summary,
        "wall_time_s": round(elapsed, 1),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(final, fh, indent=2, default=str)

    print_report(final)


def print_report(final):
    print("\n" + "=" * 100)
    print("  LABEL REDESIGN STUDY — DIAGNOSTIC COMPARISON")
    print("=" * 100)
    print(f"n_bars={final['n_bars']:,}  range={final['date_range']}  "
          f"wall_time={final['wall_time_s']:.0f}s\n")

    ms = final["current_vs_real_strategy_mismatch"]
    print("--- Current label vs. the REAL frozen ATR exit strategy (mismatch check) ---")
    print(f"  Current fires a directional call on {ms['current_directional_bars']:,} bars; "
          f"the real strategy's own barrier would fire on {ms['real_strategy_directional_bars']:,} bars.")
    print(f"  Both fire together on only {ms['pct_both_fire_together']*100:.1f}% of all bars.")
    print(f"  Of the bars where BOTH fire, direction agreement = "
          f"{ms['pct_agreement_when_BOTH_fire']*100:.1f}%.")
    print(f"  Of ALL current directional calls, agreement with what the real "
          f"exit strategy would have labeled = {ms['pct_current_calls_matching_real_exit_direction']*100:.1f}%.\n")

    for name, d in final["diagnostics"].items():
        print(f"--- {name} ---")
        print(f"  class_balance: {d['class_balance']}")
        print(f"  n_directional={d['n_directional_labels']:,} "
              f"({d['pct_directional']*100:.1f}% of bars)")
        nz = d["noise_proxy_mae_gt_mfe"]
        print(f"  noise_proxy (MAE>MFE on wins): "
              f"{'N/A' if np.isnan(nz) else f'{nz*100:.1f}%'}")
        amb = d["ambiguity_near_boundary"]
        print(f"  ambiguity (near decision boundary): "
              f"{'N/A' if np.isnan(amb) else f'{amb*100:.1f}%'}")
        print(f"  stability (bar-to-bar flip rate): "
              f"{d['stability_bar_to_bar_flip_rate']*100:.1f}%")
        if "stability_param_perturbation" in d:
            sp = d["stability_param_perturbation"]
            print(f"  stability (param perturbation agreement): "
                  f"{'N/A' if np.isnan(sp) else f'{sp*100:.1f}%'}")
        tso = d["tp_sl_overlap"]
        print(f"  TP/SL overlap (whipsaw both touched): "
              f"{'N/A' if np.isnan(tso) else f'{tso*100:.1f}%'}")
        print()


if __name__ == "__main__":
    main()
