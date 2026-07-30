"""Tests for NoOpCalibrator."""
from __future__ import annotations

import numpy as np
import pytest

from calibrators import NoOpCalibrator, get


def test_noop_transform_is_identity(tiny_probs):
    p, y = tiny_probs
    cal = get("noop").fit(p, y)
    p_out = cal.transform(p)
    np.testing.assert_array_equal(p_out, p)


def test_noop_transform_before_fit_raises():
    cal = get("noop")
    with pytest.raises(RuntimeError, match="before fit"):
        cal.transform(np.array([[0.5, 0.3, 0.2]]))


def test_noop_save_load_roundtrip(tmp_path, tiny_probs):
    p, y = tiny_probs
    cal = get("noop").fit(p, y)
    cal.save(tmp_path)
    cal2 = NoOpCalibrator().load(tmp_path)
    np.testing.assert_array_equal(cal.transform(p), cal2.transform(p))
