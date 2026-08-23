"""
trend_scanning.py — adaptive-horizon trend labels (Lopez de Prado).

METHOD, not borrowed code. Trend scanning is published in "Advances in
Financial Machine Learning" and SSRN 3257419. This is a clean numpy
implementation of that method; no source was copied from the MQL5 article
that brought it to our attention (art. 23659), whose materials are
explicitly all-rights-reserved by MetaQuotes.

WHAT IT DOES
    For each bar t, fit an OLS of price on time over every candidate
    horizon L in `span`, looking forward across [t, t+L-1]. Keep the
    horizon with the largest |t-statistic| of the slope. The SIGN of that
    t-stat is the trend direction; its MAGNITUDE is how convincing the
    trend is, which doubles as a natural sample weight.

WHY IT MIGHT BEAT A FIXED HORIZON
    train_model_v9.py labels with a fixed 3-bar forward return:
        fwd = close.shift(-FWD_BARS)/close - 1
    Every bar is judged over the same 15 minutes regardless of what the
    market is doing. Trend scanning instead lets each bar be labelled by
    the trend it actually belongs to, and reports how strong that trend
    was. A bar in a clean 40-minute move and a bar in chop no longer get
    graded on the same arbitrary window.

THE TRAP, STATED LOUDLY
    lookforward=True is CORRECT for labels -- a label is allowed to
    describe the future. It is CATASTROPHIC as a feature: the window
    [t, t+L-1] contains bar t+1 whenever L >= 2. On random walks that
    reads ~56-61% next-bar "accuracy" against 50% for the causal
    direction, i.e. pure look-ahead that looks like skill.

    Use lookforward=True to LABEL. Use lookforward=False to build a
    FEATURE. scripts/audit_feature_leakage.py exists to catch the mix-up.

PURGING
    These labels look up to max(span) bars ahead, so any train/test split
    must purge at least max(span) bars -- not FWD_BARS. Callers are
    responsible; get this wrong and the label leaks across the boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["trend_scan", "trend_scan_labels"]


def _window_stats(y: np.ndarray, L: int, lookforward: bool) -> tuple:
    """Closed-form OLS slope and t-stat over every length-L window.

    Returns arrays aligned to the ORIGIN bar of each window: index i holds
    the fit over [i, i+L-1] when lookforward, else over [i-L+1, i].
    Positions without a full window are NaN.
    """
    n = len(y)
    out_t = np.full(n, np.nan)
    out_s = np.full(n, np.nan)
    out_r = np.full(n, np.nan)
    if n < L or L < 3:
        return out_t, out_s, out_r

    w = np.lib.stride_tricks.sliding_window_view(y, L)     # (n-L+1, L)
    t = np.arange(L, dtype=np.float64)
    mean_t = t.mean()
    var_t = ((t - mean_t) ** 2).sum()

    sum_y = w.sum(axis=1)
    sum_y2 = (w ** 2).sum(axis=1)
    s_ty = w @ t
    mean_y = sum_y / L

    slope = (s_ty - L * mean_t * mean_y) / var_t
    beta0 = mean_y - slope * mean_t
    sse = np.maximum(sum_y2 - beta0 * sum_y - slope * s_ty, 0.0)
    sst = np.maximum(sum_y2 - (sum_y ** 2) / L, 1e-12)
    r2 = np.clip(1.0 - sse / sst, 0.0, 1.0)
    sigma2 = sse / max(L - 2, 1)
    se = np.sqrt(np.maximum(sigma2 / var_t, 1e-18))
    tval = slope / se

    if lookforward:
        out_t[: n - L + 1] = tval          # window starts at i
        out_s[: n - L + 1] = slope
        out_r[: n - L + 1] = r2
    else:
        out_t[L - 1:] = tval               # window ends at i
        out_s[L - 1:] = slope
        out_r[L - 1:] = r2
    return out_t, out_s, out_r


def trend_scan(close: pd.Series, span=(5, 21), lookforward: bool = True,
               use_log: bool = True) -> pd.DataFrame:
    """Best-|t| trend fit per bar across the candidate horizons.

    Returns columns: t_value, slope, r_squared, window.
    """
    y = np.log(close.values) if use_log else close.values.astype(float)
    horizons = (list(range(span[0], span[1])) if isinstance(span, tuple)
                else list(span))
    best_t = np.zeros(len(y))
    best_s = np.zeros(len(y))
    best_r = np.zeros(len(y))
    best_L = np.zeros(len(y))
    for L in horizons:
        tv, sl, r2 = _window_stats(y, L, lookforward)
        better = np.abs(np.nan_to_num(tv)) > np.abs(best_t)
        best_t = np.where(better, np.nan_to_num(tv), best_t)
        best_s = np.where(better, np.nan_to_num(sl), best_s)
        best_r = np.where(better, np.nan_to_num(r2), best_r)
        best_L = np.where(better, L, best_L)
    return pd.DataFrame({"t_value": best_t, "slope": best_s,
                         "r_squared": best_r, "window": best_L},
                        index=close.index)


def trend_scan_labels(close: pd.Series, span=(5, 21), t_threshold: float = 2.0,
                      use_log: bool = True) -> pd.DataFrame:
    """3-class labels matching this repo's convention: CALL=0, PUT=1, SKIP=2.

    A bar is directional only when the best fit clears `t_threshold`, so
    SKIP is the honest default rather than a residual.
    """
    ts = trend_scan(close, span=span, lookforward=True, use_log=use_log)
    lab = np.full(len(ts), 2, dtype=np.int8)
    lab[ts["t_value"].values >= t_threshold] = 0        # CALL
    lab[ts["t_value"].values <= -t_threshold] = 1       # PUT
    ts["label"] = lab
    return ts
