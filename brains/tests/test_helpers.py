"""Tests for shared helpers (as_3class_proba, build_inner_folds)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from brains._base import as_3class_proba
from brains._hpo import build_inner_folds


def test_as_3class_proba_passthrough_when_ordered():
    p = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    out = as_3class_proba(p, [0, 1, 2])
    np.testing.assert_array_equal(out, p)


def test_as_3class_proba_reorders_classes():
    # classifier learned only classes [2, 0] (skip and call, no put)
    p = np.array([[0.6, 0.4], [0.3, 0.7]])
    out = as_3class_proba(p, [2, 0])
    # col 0 (CALL) gets classifier's second column; col 2 (SKIP) gets first
    expected = np.array([[0.4, 0.0, 0.6], [0.7, 0.0, 0.3]])
    np.testing.assert_array_equal(out, expected)


def test_as_3class_proba_handles_catboost_nested_classes():
    p = np.array([[0.2, 0.3, 0.5]])
    out = as_3class_proba(p, [[0, 1, 2]])
    np.testing.assert_array_equal(out, p)


def test_as_3class_proba_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        as_3class_proba(np.array([0.1, 0.9]), [0, 1])


def test_build_inner_folds_produces_expected_count():
    dates = [date(2020, 1, 1) + timedelta(days=d) for d in range(2000)]
    folds = build_inner_folds(dates, k=3)
    assert len(folds) == 3


def test_build_inner_folds_purge_gap_ge_embargo():
    from backtest_threshold_sweep import EMBARGO_DAYS
    dates = [date(2020, 1, 1) + timedelta(days=d) for d in range(2000)]
    for k in (2, 3, 4, 5):
        folds = build_inner_folds(dates, k=k)
        for (train_end, val_start, val_end) in folds:
            gap = (val_start - train_end).days
            assert gap >= EMBARGO_DAYS, (k, train_end, val_start, gap)
            assert val_end > val_start


def test_build_inner_folds_rejects_bad_inputs():
    with pytest.raises(ValueError):
        build_inner_folds([date(2020, 1, 1)], k=0)
    with pytest.raises(ValueError):
        build_inner_folds([], k=3)
