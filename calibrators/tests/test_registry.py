"""Registry tests."""
from __future__ import annotations

import pytest

from calibrators import (CalibratorAdapter, IsotonicCalibrator,
                            NoOpCalibrator, PlattScalingCalibrator, REGISTRY,
                            get, list_calibrators)


def test_registry_populated():
    assert set(REGISTRY) == {"noop", "platt", "isotonic"}


def test_every_registered_is_calibratoradapter():
    for name, cal in REGISTRY.items():
        assert isinstance(cal, CalibratorAdapter), name
        assert cal.name == name


def test_get_unknown_raises_helpful_error():
    with pytest.raises(KeyError, match="unknown calibrator"):
        get("nope")


def test_list_calibrators_sorted():
    assert list_calibrators() == sorted(list_calibrators())
    assert list_calibrators() == ["isotonic", "noop", "platt"]


def test_get_returns_fresh_instance():
    """Two get() calls must return distinct instances so fit state
    does not leak across folds."""
    a = get("isotonic")
    b = get("isotonic")
    assert a is not b


def test_direct_class_exports():
    assert isinstance(REGISTRY["noop"], NoOpCalibrator)
    assert isinstance(REGISTRY["platt"], PlattScalingCalibrator)
    assert isinstance(REGISTRY["isotonic"], IsotonicCalibrator)
