"""Registry tests."""
from __future__ import annotations

import pytest

from ensembles import EnsembleAdapter, REGISTRY, get, list_ensembles


EXPECTED = {"mean", "median", "weighted", "performance",
            "min_variance", "stacking", "confidence", "uncertainty"}


def test_registry_has_all_eight_families():
    assert set(REGISTRY) == EXPECTED


def test_every_registered_is_ensembleadapter_subclass():
    for name, cls in REGISTRY.items():
        assert issubclass(cls, EnsembleAdapter), name


def test_get_returns_fresh_instance():
    """Each get() must return a distinct instance so fit state does not
    leak between callers."""
    a = get("mean")
    b = get("mean")
    assert a is not b


def test_get_unknown_raises_key_error():
    with pytest.raises(KeyError, match="unknown ensemble"):
        get("nonexistent")


def test_list_ensembles_sorted_and_complete():
    assert list_ensembles() == sorted(list_ensembles())
    assert set(list_ensembles()) == EXPECTED


def test_adapter_names_match_registry_keys():
    for name in list_ensembles():
        adapter = get(name)
        assert adapter.name == name
