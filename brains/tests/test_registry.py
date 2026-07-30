"""Tests for the brains registry and public API."""
from __future__ import annotations

import pytest

from brains import (REGISTRY, BrainAdapter, CatAdapter, LGBAdapter,
                      MLPAdapter, XGBAdapter, get, list_brains)


def test_registry_populated():
    assert set(REGISTRY) == {"xgb", "lgb", "cat", "mlp"}


def test_every_registered_is_brainadapter():
    for name, brain in REGISTRY.items():
        assert isinstance(brain, BrainAdapter), name
        assert brain.name == name


def test_get_unknown_raises_helpful_error():
    with pytest.raises(KeyError, match="unknown brain"):
        get("nope")


def test_list_brains_sorted():
    assert list_brains() == sorted(list_brains())
    assert list_brains() == ["cat", "lgb", "mlp", "xgb"]


def test_class_flags_match_expectations():
    assert not get("xgb").needs_scaling
    assert not get("lgb").needs_scaling
    assert not get("cat").needs_scaling
    assert get("mlp").needs_scaling

    assert get("xgb").supports_sample_weight
    assert get("lgb").supports_sample_weight
    assert get("cat").supports_sample_weight
    assert not get("mlp").supports_sample_weight

    for name in ("xgb", "lgb", "cat", "mlp"):
        assert get(name).supports_native_early_stopping


def test_default_params_dict_shape():
    for name in list_brains():
        p = get(name).default_params()
        assert isinstance(p, dict) and p, f"{name} default_params empty"


def test_direct_class_exports_are_registry_types():
    assert isinstance(REGISTRY["xgb"], XGBAdapter)
    assert isinstance(REGISTRY["lgb"], LGBAdapter)
    assert isinstance(REGISTRY["cat"], CatAdapter)
    assert isinstance(REGISTRY["mlp"], MLPAdapter)
