"""Calibration metrics: ECE (per-class and top-1), multiclass Brier,
log-loss, and reliability-bin data suitable for plotting.

All metrics accept multi-class probabilities as ``(n, K)`` arrays in the
fixed column order ``[CALL, PUT, SKIP]`` for the trading pipeline; the
functions themselves are class-agnostic and work for any K >= 2.

References
----------
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). "On
calibration of modern neural networks." ICML 2017. — top-1 ECE.

Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., and Tran, D.
(2019). "Measuring calibration in deep learning." — per-class ECE.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
def validate_proba_input(p: Any, name: str) -> np.ndarray:
    """Validate a probability array is (n, K), rows sum to ~1, finite."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name}: expected 2-D probability array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name}: empty probability array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains non-finite values")
    if (arr < -1e-6).any() or (arr > 1 + 1e-6).any():
        raise ValueError(
            f"{name}: probabilities must lie in [0, 1] (with 1e-6 slack)")
    return arr


# ---------------------------------------------------------------------------
def top1_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Top-1 Expected Calibration Error (Guo et al. 2017).

    Uses the max-class predicted probability as the "confidence" and
    whether the max-class prediction was correct as the "accuracy".
    Bins confidences into ``n_bins`` equal-width bins and computes the
    weighted absolute gap between mean confidence and mean accuracy.

    Parameters
    ----------
    y_true : (n,) integer labels
    p : (n, K) probability array
    n_bins : int
        Number of equal-width bins on [0, 1].

    Returns
    -------
    float
        ECE in [0, 1]. Zero means perfect top-1 calibration.
    """
    p = validate_proba_input(p, "p")
    y = np.asarray(y_true)
    if y.ndim != 1 or y.size != p.shape[0]:
        raise ValueError(
            f"y_true shape {y.shape} incompatible with p shape {p.shape}")
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    n = len(y)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        w = mask.sum() / n
        ece += w * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def class_conditional_ece(y_true: np.ndarray, p: np.ndarray,
                            n_bins: int = 10) -> dict[int, float]:
    """Per-class ECE — treats each class as a one-vs-rest binary problem.

    For class k, ``p[:, k]`` is the predicted probability and
    ``(y == k)`` is the true label; compute standard binary-ECE.
    Returns a dict ``{class_index: ece}``. This is what matters for a
    trading pipeline where only CALL / PUT calibration affects P&L.

    Reference: Nixon et al. 2019.
    """
    p = validate_proba_input(p, "p")
    y = np.asarray(y_true)
    out: dict[int, float] = {}
    for k in range(p.shape[1]):
        pk = p[:, k]
        yk = (y == k).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(pk, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        ece = 0.0
        n = len(y)
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            w = mask.sum() / n
            ece += w * abs(pk[mask].mean() - yk[mask].mean())
        out[int(k)] = float(ece)
    return out


def multiclass_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    """Multi-class Brier score.

    ``(1/n) * sum_i || one_hot(y_i) - p_i ||^2``. Lower is better.
    Bounded above by 2 for 3-class probabilities.
    """
    p = validate_proba_input(p, "p")
    y = np.asarray(y_true)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((onehot - p) ** 2, axis=1)))


def multiclass_log_loss(y_true: np.ndarray, p: np.ndarray,
                         eps: float = 1e-12) -> float:
    """Multi-class cross-entropy log loss.

    ``-(1/n) * sum_i log p_i[y_i]`` with probability clipping to avoid
    log(0). Lower is better.
    """
    p = validate_proba_input(p, "p")
    y = np.asarray(y_true)
    pk = np.clip(p[np.arange(len(y)), y], eps, 1.0 - eps)
    return float(-np.mean(np.log(pk)))


def reliability_bins(y_true: np.ndarray, p: np.ndarray,
                      n_bins: int = 10) -> list[dict]:
    """Top-1 reliability-diagram data.

    Returns a list of dicts, one per bin, with keys:
        bin_index, conf_lo, conf_hi, mean_conf, mean_acc, n
    """
    p = validate_proba_input(p, "p")
    y = np.asarray(y_true)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    out: list[dict] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            out.append({
                "bin_index": b, "conf_lo": float(bins[b]),
                "conf_hi": float(bins[b + 1]),
                "mean_conf": None, "mean_acc": None, "n": 0,
            })
        else:
            out.append({
                "bin_index": b, "conf_lo": float(bins[b]),
                "conf_hi": float(bins[b + 1]),
                "mean_conf": float(conf[mask].mean()),
                "mean_acc": float(correct[mask].mean()),
                "n": int(mask.sum()),
            })
    return out


def per_class_reliability_bins(y_true: np.ndarray, p: np.ndarray,
                                 n_bins: int = 10) -> dict[int, list[dict]]:
    """Per-class reliability data for K one-vs-rest reliability curves.

    Returns ``{class_index: [bin_dicts]}``.
    """
    p = validate_proba_input(p, "p")
    y = np.asarray(y_true)
    out: dict[int, list[dict]] = {}
    for k in range(p.shape[1]):
        pk = p[:, k]
        yk = (y == k).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(pk, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                rows.append({"bin_index": b, "conf_lo": float(bins[b]),
                              "conf_hi": float(bins[b + 1]),
                              "mean_pred": None, "mean_actual": None,
                              "n": 0})
            else:
                rows.append({"bin_index": b, "conf_lo": float(bins[b]),
                              "conf_hi": float(bins[b + 1]),
                              "mean_pred": float(pk[mask].mean()),
                              "mean_actual": float(yk[mask].mean()),
                              "n": int(mask.sum())})
        out[int(k)] = rows
    return out
