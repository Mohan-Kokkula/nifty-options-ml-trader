"""brains — pluggable model-family registry for Phase 3+.

Every adapter satisfies the :class:`BrainAdapter` contract.
Adding a new family is a two-line change: subclass and register.

Public API
----------
    REGISTRY : dict[str, BrainAdapter]
    get(name) -> BrainAdapter                    # KeyError on unknown
    list_brains() -> list[str]                   # registered names, sorted

Example
-------
    >>> from brains import get
    >>> brain = get("lgb")
    >>> model = brain.fit(X, y, sample_weight=w, seed=42)
    >>> proba = brain.predict_proba_3class(model, X_test)
"""
from __future__ import annotations

from ._base import BrainAdapter, as_3class_proba
from .cat_adapter import CatAdapter
from .lgb_adapter import LGBAdapter
from .mlp_adapter import MLPAdapter
from .xgb_adapter import XGBAdapter


REGISTRY: dict[str, BrainAdapter] = {
    "xgb": XGBAdapter(),
    "lgb": LGBAdapter(),
    "cat": CatAdapter(),
    "mlp": MLPAdapter(),
}


def get(name: str) -> BrainAdapter:
    """Return the adapter registered under ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not registered. The exception message lists the
        available brain names for a helpful failure.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"unknown brain {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_brains() -> list[str]:
    """Return the sorted list of registered brain names."""
    return sorted(REGISTRY)


__all__ = [
    "BrainAdapter", "as_3class_proba",
    "REGISTRY", "get", "list_brains",
    "XGBAdapter", "LGBAdapter", "CatAdapter", "MLPAdapter",
]

__version__ = "1.0.0"
