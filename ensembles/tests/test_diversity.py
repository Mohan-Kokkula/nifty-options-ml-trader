"""Diversity-metric tests — hand-computed anchors + registry sanity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensembles import (
    EnsembleInputError,
    average_diversity_across_folds,
    disagreement,
    diversity_matrix,
    double_fault,
    prediction_correlation,
    q_statistic,
)


# ---------- Q-statistic ----------
def test_q_statistic_perfectly_redundant_predictions_yields_plus_one():
    """Two classifiers that make the SAME correct AND SAME wrong choices.

    Q-statistic is only well-defined when both N11 and N00 are non-zero,
    so we craft a case with some correct-both and some wrong-both bars.
    Fully-correct-both or fully-wrong-both cases yield the degenerate
    ``0.0`` documented in :func:`q_statistic`.
    """
    a = np.array([0, 1, 0, 1])   # correct on bars 0, 1; wrong on 2, 3
    b = np.array([0, 1, 0, 1])   # same pattern
    y = np.array([0, 1, 2, 0])
    # N11=2 (bars 0, 1), N00=2 (bars 2, 3), N10=N01=0 → Q = +1
    assert q_statistic(a, b, y) == pytest.approx(1.0)


def test_q_statistic_all_correct_returns_degenerate_zero():
    """When both classifiers are always correct (N00=0), Q is undefined
    and our implementation returns 0.0 by convention."""
    a = np.array([0, 1, 2, 0])
    b = np.array([0, 1, 2, 0])
    y = np.array([0, 1, 2, 0])
    assert q_statistic(a, b, y) == 0.0


def test_q_statistic_anti_correlated_predictions_yields_minus_one():
    # both classifiers ever correct/wrong on complementary bars
    a = np.array([0, 1, 2, 0])
    b = np.array([1, 0, 0, 1])
    y = np.array([0, 1, 2, 0])
    # N11=0, N10=4, N01=0, N00=0 → Q formula denominator 0 → returns 0
    # Craft a proper anti-correlated case:
    a2 = np.array([0, 1, 0, 1])
    b2 = np.array([1, 0, 1, 0])
    y2 = np.array([0, 0, 0, 0])
    # a correct on 0,2; b correct on 1,3 → N11=0 N10=2 N01=2 N00=0 → Q=-1
    assert q_statistic(a2, b2, y2) == pytest.approx(-1.0)


def test_q_statistic_zero_denominator_returns_zero():
    a = np.array([0, 0])
    b = np.array([1, 1])
    y = np.array([0, 0])
    # N11=0 N00=0 → denom 0
    assert q_statistic(a, b, y) == 0.0


# ---------- disagreement ----------
def test_disagreement_hand_computed():
    a = np.array([0, 1, 2, 0])
    b = np.array([0, 2, 2, 1])
    assert disagreement(a, b) == pytest.approx(0.5)


def test_disagreement_bounds():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.integers(0, 3, size=50)
        b = rng.integers(0, 3, size=50)
        d = disagreement(a, b)
        assert 0.0 <= d <= 1.0


# ---------- double fault ----------
def test_double_fault_hand_computed():
    # bar 0: a=0(right), b=1(wrong) → N10
    # bar 1: a=1(right), b=0(wrong) → N10
    # bar 2: a=2(right), b=0(wrong) → N10
    # bar 3: a=1(wrong), b=1(wrong) → N00 ← the only double fault
    a = np.array([0, 1, 2, 1])
    b = np.array([1, 0, 0, 1])
    y = np.array([0, 1, 2, 0])
    assert double_fault(a, b, y) == pytest.approx(0.25)


def test_double_fault_below_min_error_rate():
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 3, size=n)
    a = rng.integers(0, 3, size=n)
    b = rng.integers(0, 3, size=n)
    df = double_fault(a, b, y)
    err_a = (a != y).mean()
    err_b = (b != y).mean()
    assert df <= min(err_a, err_b) + 1e-9


# ---------- correlation ----------
def test_correlation_identical_probs_is_plus_one():
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
    assert prediction_correlation(p, p) == pytest.approx(1.0)


def test_correlation_constant_returns_zero():
    """When one series has zero std, Pearson correlation is undefined;
    our implementation returns near-zero rather than NaN by convention.
    Uses ``pytest.approx`` because 1/3 is not exactly representable in
    IEEE-754 so ``a.std()`` is at the noise floor rather than exactly 0.
    """
    a = np.full((5, 3), 1 / 3)
    b = np.arange(15, dtype=float).reshape(5, 3) / 100.0
    assert prediction_correlation(a, b) == pytest.approx(0.0, abs=1e-10)


# ---------- matrix ----------
def test_diversity_matrix_shape_and_index():
    # b0 and b1 share the same pattern (some right, some wrong) so
    # Q(b0, b1) is a well-defined +1.
    preds = {
        "b0": np.array([0, 1, 0, 1]),
        "b1": np.array([0, 1, 0, 1]),
        "b2": np.array([1, 0, 2, 1]),
    }
    y = np.array([0, 1, 2, 0])
    m = diversity_matrix(preds, y, "q_statistic")
    assert m.shape == (3, 3)
    assert list(m.index) == ["b0", "b1", "b2"]
    assert m.loc["b0", "b1"] == pytest.approx(1.0)
    assert m.loc["b0", "b0"] == 0.0


def test_diversity_matrix_correlation_requires_prob_map():
    preds = {"b0": np.array([0]), "b1": np.array([0])}
    y = np.array([0])
    with pytest.raises(EnsembleInputError, match="prob_map"):
        diversity_matrix(preds, y, "correlation")


def test_diversity_matrix_unknown_metric_raises():
    preds = {"b0": np.array([0]), "b1": np.array([0])}
    y = np.array([0])
    with pytest.raises(EnsembleInputError, match="unknown"):
        diversity_matrix(preds, y, "not_a_metric")


def test_average_diversity_across_folds_pass():
    idx = ["b0", "b1"]
    m1 = pd.DataFrame(np.array([[0.0, 0.5], [0.5, 0.0]]), index=idx, columns=idx)
    m2 = pd.DataFrame(np.array([[0.0, 0.7], [0.7, 0.0]]), index=idx, columns=idx)
    mean, std = average_diversity_across_folds([m1, m2])
    assert mean.loc["b0", "b1"] == pytest.approx(0.6)
    assert std.loc["b0", "b1"] == pytest.approx(0.1)


def test_average_diversity_across_folds_inconsistent_axis():
    idx1 = ["b0", "b1"]
    idx2 = ["b0", "b2"]
    m1 = pd.DataFrame(np.zeros((2, 2)), index=idx1, columns=idx1)
    m2 = pd.DataFrame(np.zeros((2, 2)), index=idx2, columns=idx2)
    with pytest.raises(EnsembleInputError, match="inconsistent"):
        average_diversity_across_folds([m1, m2])
