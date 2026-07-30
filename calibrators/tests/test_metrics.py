"""Tests for calibrators._metrics."""
from __future__ import annotations

import numpy as np
import pytest

from calibrators import (class_conditional_ece, multiclass_brier,
                            multiclass_log_loss, per_class_reliability_bins,
                            reliability_bins, top1_ece)


def test_top1_ece_perfectly_calibrated_is_zero():
    """One-hot predictions with matching labels have zero ECE (all
    confidences at 1.0, accuracy 100 %)."""
    n = 100
    y = np.random.default_rng(0).integers(0, 3, size=n)
    p = np.zeros((n, 3))
    p[np.arange(n), y] = 1.0
    assert top1_ece(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_top1_ece_hand_computed():
    """Two rows, both max-conf=0.9. If one correct → confidence 0.9,
    accuracy 0.5, ECE = 0.4 (all mass in one bin)."""
    p = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    y = np.array([0, 1])
    ece = top1_ece(y, p, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_top1_ece_matches_uniform_baseline():
    """Uniform (1/3, 1/3, 1/3) predictions → argmax = 0 (first tie
    winner), so accuracy = P(y==0). With balanced classes, ~1/3
    accuracy; confidence 1/3; ECE ~ 0."""
    n = 3000
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=n)
    p = np.full((n, 3), 1.0 / 3.0)
    ece = top1_ece(y, p, n_bins=10)
    assert ece < 0.05


def test_class_conditional_ece_zero_for_matched_freq():
    """If p[:, k] equals the empirical rate of class k for every k,
    per-class ECE is 0."""
    n = 900
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=n)
    p = np.zeros((n, 3))
    for k in range(3):
        rate = (y == k).mean()
        p[:, k] = rate
    ece_dict = class_conditional_ece(y, p, n_bins=10)
    for k, val in ece_dict.items():
        assert val < 1e-6


def test_multiclass_brier_hand_computed():
    """One row, p=(0.7, 0.2, 0.1), y=0.
    Brier = (0.3)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    """
    p = np.array([[0.7, 0.2, 0.1]])
    y = np.array([0])
    assert multiclass_brier(y, p) == pytest.approx(0.14, abs=1e-9)


def test_multiclass_brier_bounded_zero_to_two():
    """Multi-class Brier for K=3 has bounds [0, 2]."""
    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(0, 3, size=n)
    p = rng.dirichlet(np.ones(3), size=n)
    b = multiclass_brier(y, p)
    assert 0.0 <= b <= 2.0


def test_multiclass_log_loss_matches_sklearn():
    """Compare our log-loss to sklearn.metrics.log_loss on shared input."""
    from sklearn.metrics import log_loss as sk_log_loss
    rng = np.random.default_rng(4)
    n = 200
    y = rng.integers(0, 3, size=n)
    p = rng.dirichlet(np.ones(3), size=n)
    ours = multiclass_log_loss(y, p)
    theirs = sk_log_loss(y, p, labels=[0, 1, 2])
    assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-9)


def test_reliability_bins_structure():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 3, size=n)
    p = rng.dirichlet(np.ones(3), size=n)
    bins = reliability_bins(y, p, n_bins=10)
    assert len(bins) == 10
    total_n = sum(b["n"] for b in bins)
    assert total_n == n


def test_per_class_reliability_structure():
    rng = np.random.default_rng(6)
    n = 400
    y = rng.integers(0, 3, size=n)
    p = rng.dirichlet(np.ones(3), size=n)
    d = per_class_reliability_bins(y, p, n_bins=10)
    assert set(d) == {0, 1, 2}
    for k, rows in d.items():
        assert len(rows) == 10
        total = sum(r["n"] for r in rows)
        assert total == n


def test_metrics_reject_bad_input():
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="2-D"):
        top1_ece(y, np.array([0.5, 0.5, 0]))    # 1-D
    with pytest.raises(ValueError, match="non-finite"):
        top1_ece(y, np.array([[np.nan, 0.5, 0.5], [0.1, 0.1, 0.8],
                                [0.5, 0.3, 0.2]]))
    with pytest.raises(ValueError, match="incompatible"):
        top1_ece(np.array([0]), np.array([[0.5, 0.3, 0.2],
                                             [0.1, 0.2, 0.7]]))
