"""
profile_features.py — Market Profile / Volume Profile primitives for NIFTY.

STANDALONE. Imports nothing from the trading pilot and is imported by
nothing in it. Built as a new feature family so it can be tested on its
own merits before any of it is allowed near live code.

WHY TPO AND NOT VOLUME
    NIFTY is an index. It has no traded volume of its own, and
    data/nifty_5min.csv carries volume==0 for 2015-01 .. 2026-03. Real
    volume exists for ~110 trading days only, all of it inside the model's
    TEST window, so a volume profile validated there would have no holdout.

    Steidlmayer's original Market Profile counts TIME at price, not volume:
    each period that trades at a price adds one TPO to that price. POC and
    Value Area have exact time-based definitions. That is what this module
    computes, over the full history.

    build_profile() takes a weights array, so passing real volume gives a
    true Volume Profile wherever volume exists. profile_agreement() measures
    how closely the TPO proxy tracks the volume version on the days where
    both can be computed — run it before trusting the proxy.

DEFINITIONS (standard, no invention)
    POC   price bucket holding the most TPOs
    VA    smallest contiguous band around POC holding >= 70% of all TPOs,
          grown by repeatedly taking the heavier of the two adjacent buckets
    VAH   top of that band          VAL   bottom of that band
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TICK = 5.0          # NIFTY profile bucket, in index points
VALUE_AREA_PCT = 0.70


@dataclass(frozen=True)
class Profile:
    poc: float
    vah: float
    val: float
    total: float
    n_buckets: int

    @property
    def va_width(self) -> float:
        return self.vah - self.val

    @staticmethod
    def empty() -> "Profile":
        return Profile(np.nan, np.nan, np.nan, 0.0, 0)


def build_profile(highs: np.ndarray, lows: np.ndarray,
                  weights: np.ndarray | None = None,
                  tick: float = DEFAULT_TICK,
                  va_pct: float = VALUE_AREA_PCT) -> Profile:
    """Profile for one session.

    Each bar contributes its weight to every bucket its range covers, which
    is the TPO construction when weights are all 1. Pass volume as weights
    to get a Volume Profile instead — the rest of the maths is identical.
    """
    n = len(highs)
    if n == 0:
        return Profile.empty()
    if weights is None:
        weights = np.ones(n, dtype=float)

    lo, hi = float(np.min(lows)), float(np.max(highs))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return Profile.empty()

    base = np.floor(lo / tick) * tick
    nb = int(np.ceil((hi - base) / tick)) + 1
    if nb <= 0 or nb > 20000:
        return Profile.empty()

    hist = np.zeros(nb, dtype=float)
    i0 = np.floor((lows - base) / tick).astype(int)
    i1 = np.floor((highs - base) / tick).astype(int)
    np.clip(i0, 0, nb - 1, out=i0)
    np.clip(i1, 0, nb - 1, out=i1)
    for k in range(n):
        w = weights[k]
        if w <= 0 or not np.isfinite(w):
            continue
        span = i1[k] - i0[k] + 1
        # spread the bar's weight evenly over the levels it actually traded
        hist[i0[k]:i1[k] + 1] += w / span

    total = float(hist.sum())
    if total <= 0:
        return Profile.empty()

    poc_i = int(np.argmax(hist))
    lo_i = hi_i = poc_i
    acc = hist[poc_i]
    need = total * va_pct
    while acc < need and (lo_i > 0 or hi_i < nb - 1):
        below = hist[lo_i - 1] if lo_i > 0 else -1.0
        above = hist[hi_i + 1] if hi_i < nb - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += hist[hi_i]
        else:
            lo_i -= 1
            acc += hist[lo_i]

    return Profile(
        poc=base + (poc_i + 0.5) * tick,
        vah=base + (hi_i + 1) * tick,
        val=base + lo_i * tick,
        total=total,
        n_buckets=nb,
    )


def session_profiles(df: pd.DataFrame, use_volume: bool = False,
                     tick: float = DEFAULT_TICK) -> pd.DataFrame:
    """One completed profile per session. Index = session date."""
    out = {}
    for day, g in df.groupby(df.index.normalize()):
        w = g["volume"].values.astype(float) if use_volume else None
        if use_volume and (w is None or not np.isfinite(w).any() or w.sum() <= 0):
            out[day] = Profile.empty()
            continue
        out[day] = build_profile(g["high"].values, g["low"].values, w, tick)
    return pd.DataFrame(
        [{"day": d, "poc": p.poc, "vah": p.vah, "val": p.val,
          "va_width": p.va_width, "total": p.total} for d, p in out.items()]
    ).set_index("day").sort_index()


def developing_profiles(df: pd.DataFrame, tick: float = DEFAULT_TICK,
                        min_bars: int = 12) -> pd.DataFrame:
    """Profile as it builds through the session, recomputed each bar.

    Strictly causal: bar i sees only bars 0..i of its own session, so this
    can be used as a live feature without look-ahead. Costly but honest —
    the alternative (using the finished profile intraday) is the classic
    look-ahead bug in profile backtests.
    """
    rows = []
    for _, g in df.groupby(df.index.normalize()):
        H, L = g["high"].values, g["low"].values
        for i in range(len(g)):
            if i + 1 < min_bars:
                rows.append((g.index[i], np.nan, np.nan, np.nan))
                continue
            p = build_profile(H[:i + 1], L[:i + 1], None, tick)
            rows.append((g.index[i], p.poc, p.vah, p.val))
    return pd.DataFrame(rows, columns=["date", "d_poc", "d_vah", "d_val"]
                        ).set_index("date")


def profile_agreement(df: pd.DataFrame, tick: float = DEFAULT_TICK) -> dict:
    """How faithful is the TPO proxy to a real Volume Profile?

    Computed only on sessions where volume actually exists. If POC and value
    edges disagree badly, every TPO result in this project is a statement
    about time-at-price and must not be sold as a volume finding.
    """
    tpo = session_profiles(df, use_volume=False, tick=tick)
    vol = session_profiles(df, use_volume=True, tick=tick)
    j = tpo.join(vol, lsuffix="_t", rsuffix="_v").dropna(
        subset=["poc_t", "poc_v"])
    if len(j) < 10:
        return {"n": len(j)}
    rng = (j["vah_v"] - j["val_v"]).replace(0, np.nan)
    return {
        "n": len(j),
        "poc_corr": float(j["poc_t"].corr(j["poc_v"])),
        "poc_mae_pts": float((j["poc_t"] - j["poc_v"]).abs().mean()),
        "poc_mae_va": float(((j["poc_t"] - j["poc_v"]).abs() / rng).mean()),
        "vah_mae_pts": float((j["vah_t"] - j["vah_v"]).abs().mean()),
        "val_mae_pts": float((j["val_t"] - j["val_v"]).abs().mean()),
        "vaw_corr": float(j["va_width_t"].corr(j["va_width_v"])),
    }
