"""Tests for IsotonicCalibrator."""
from __future__ import annotations

import numpy as np
import pytest

from calibrators import IsotonicCalibrator, get, multiclass_brier, top1_ece


def test_isotonic_reduces_ece_on_overconfident(overconfident_uncal):
    p, y = overconfident_uncal
    ece_before = top1_ece(y, p)
    cal = get("isotonic").fit(p, y)
    p_out = cal.transform(p)
    ece_after = top1_ece(y, p_out)
    assert ece_after < ece_before, (
        f"expected reduction: {ece_before} -> {ece_after}")


def test_isotonic_reduces_brier_on_overconfident(overconfident_uncal):
    p, y = overconfident_uncal
    brier_before = multiclass_brier(y, p)
    p_out = get("isotonic").fit(p, y).transform(p)
    brier_after = multiclass_brier(y, p_out)
    assert brier_after < brier_before


def test_isotonic_rows_sum_to_one(overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get("isotonic").fit(p, y).transform(p)
    np.testing.assert_allclose(p_out.sum(axis=1), 1.0, atol=1e-9)


def test_isotonic_output_in_unit_interval(overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get("isotonic").fit(p, y).transform(p)
    assert (p_out >= 0).all()
    assert (p_out <= 1).all()


def test_isotonic_is_deterministic(overconfident_uncal):
    p, y = overconfident_uncal
    a = get("isotonic").fit(p, y).transform(p)
    b = get("isotonic").fit(p, y).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_isotonic_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        get("isotonic").transform(np.array([[0.5, 0.3, 0.2]]))


def test_isotonic_save_load_roundtrip(tmp_path, overconfident_uncal):
    p, y = overconfident_uncal
    cal = get("isotonic").fit(p, y)
    cal.save(tmp_path)
    cal2 = IsotonicCalibrator().load(tmp_path)
    np.testing.assert_allclose(cal.transform(p), cal2.transform(p),
                                atol=1e-12)


def test_isotonic_monotone_transform_within_class(overconfident_uncal):
    """Isotonic regression is monotone non-decreasing by construction:
    for a given class column, sorting inputs → sorted outputs (up to
    ties, and pre-renormalisation).
    """
    p, y = overconfident_uncal
    cal = get("isotonic").fit(p, y)
    # Query a monotone grid; check per-class output is non-decreasing
    grid = np.linspace(0, 1, 51)
    # Feed grid as (n, 3) uniform-remainder rows and check column 0 monotone
    fake = np.column_stack([grid, (1 - grid) / 2, (1 - grid) / 2])
    fake = fake / fake.sum(axis=1, keepdims=True)
    # We inspect the raw per-class isotonic output before renormalisation
    # by calling the internal fitted regressor.
    kind, obj = cal._per_class[0]
    assert kind == "iso"
    raw = obj.predict(grid)
    diffs = np.diff(raw)
    # allow tiny negative rounding
    assert (diffs >= -1e-12).all()
