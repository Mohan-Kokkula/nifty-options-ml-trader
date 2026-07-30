"""Tests for PlattScalingCalibrator."""
from __future__ import annotations

import numpy as np
import pytest

from calibrators import (PlattScalingCalibrator, get, multiclass_brier,
                            top1_ece)


def test_platt_reduces_ece_on_overconfident(overconfident_uncal):
    p, y = overconfident_uncal
    ece_before = top1_ece(y, p)
    cal = get("platt").fit(p, y, seed=42)
    p_out = cal.transform(p)
    ece_after = top1_ece(y, p_out)
    assert ece_after < ece_before, f"expected reduction: {ece_before} -> {ece_after}"


def test_platt_rows_sum_to_one(overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get("platt").fit(p, y, seed=42).transform(p)
    row_sums = p_out.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_platt_output_in_unit_interval(overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get("platt").fit(p, y, seed=42).transform(p)
    assert (p_out >= 0).all()
    assert (p_out <= 1).all()


def test_platt_deterministic_given_seed(overconfident_uncal):
    p, y = overconfident_uncal
    a = get("platt").fit(p, y, seed=7).transform(p)
    b = get("platt").fit(p, y, seed=7).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_platt_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        get("platt").transform(np.array([[0.5, 0.3, 0.2]]))


def test_platt_save_load_roundtrip(tmp_path, overconfident_uncal):
    p, y = overconfident_uncal
    cal = get("platt").fit(p, y, seed=42)
    cal.save(tmp_path)
    cal2 = PlattScalingCalibrator().load(tmp_path)
    np.testing.assert_allclose(cal.transform(p), cal2.transform(p),
                                atol=1e-12)


def test_platt_degenerate_class_uses_prior():
    """If a class has no positive labels, we cannot fit a logistic
    regression — the fallback is a constant prior probability."""
    n = 100
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=n)
    # No y == 2 anywhere
    y = rng.integers(0, 2, size=n)
    cal = get("platt").fit(p, y, seed=42)
    p_out = cal.transform(p)
    # class 2 output must be constant (equal to prior = 0.0 here)
    assert np.allclose(p_out[:, 2] / p_out[:, 2].max() if p_out[:, 2].max() > 0
                        else p_out[:, 2], 1.0) or p_out[:, 2].max() < 1e-6
