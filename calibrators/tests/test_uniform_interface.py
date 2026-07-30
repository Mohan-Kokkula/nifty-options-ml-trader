"""Property tests every calibrator must pass, parametrised over registry."""
from __future__ import annotations

import numpy as np
import pytest

from calibrators import get, list_calibrators


CALIBRATORS = list_calibrators()


@pytest.mark.parametrize("name", CALIBRATORS)
def test_fit_returns_self(name, overconfident_uncal):
    p, y = overconfident_uncal
    cal = get(name)
    out = cal.fit(p, y, seed=1)
    assert out is cal


@pytest.mark.parametrize("name", CALIBRATORS)
def test_transform_preserves_shape(name, overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get(name).fit(p, y, seed=1).transform(p)
    assert p_out.shape == p.shape


@pytest.mark.parametrize("name", CALIBRATORS)
def test_transform_rows_sum_to_one(name, overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get(name).fit(p, y, seed=1).transform(p)
    np.testing.assert_allclose(p_out.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("name", CALIBRATORS)
def test_transform_probabilities_in_unit_interval(name, overconfident_uncal):
    p, y = overconfident_uncal
    p_out = get(name).fit(p, y, seed=1).transform(p)
    assert (p_out >= -1e-9).all() and (p_out <= 1 + 1e-9).all()


@pytest.mark.parametrize("name", CALIBRATORS)
def test_deterministic_given_seed(name, overconfident_uncal):
    p, y = overconfident_uncal
    a = get(name).fit(p, y, seed=99).transform(p)
    b = get(name).fit(p, y, seed=99).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("name", CALIBRATORS)
def test_save_load_roundtrip_preserves_transform(name, tmp_path,
                                                    overconfident_uncal):
    p, y = overconfident_uncal
    cal = get(name).fit(p, y, seed=1)
    p_before = cal.transform(p)
    cal.save(tmp_path)
    cal2 = get(name).load(tmp_path)
    p_after = cal2.transform(p)
    np.testing.assert_allclose(p_before, p_after, atol=1e-9)


@pytest.mark.parametrize("name", CALIBRATORS)
def test_transform_before_fit_raises(name):
    with pytest.raises(RuntimeError, match="before fit"):
        get(name).transform(np.array([[0.5, 0.3, 0.2]]))


@pytest.mark.parametrize("name", CALIBRATORS)
def test_new_instance_between_fits(name, overconfident_uncal, calibrated_uncal):
    """A fresh instance for each fit call — no state leaking between
    different calibration sets."""
    p1, y1 = overconfident_uncal
    p2, y2 = calibrated_uncal
    a = get(name).fit(p1, y1, seed=1)
    b = get(name).fit(p2, y2, seed=1)
    # Different fit data => (in general) different transforms on the
    # same query. NoOp is exempt from this — it never changes the input.
    if name == "noop":
        np.testing.assert_array_equal(a.transform(p1), p1)
        np.testing.assert_array_equal(b.transform(p1), p1)
    else:
        ta = a.transform(p1)
        tb = b.transform(p1)
        assert not np.allclose(ta, tb, atol=1e-6), (
            f"{name}: fresh instance should not carry state between fits")
