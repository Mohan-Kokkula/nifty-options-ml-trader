"""calibrators — pluggable probability-calibration registry for Phase 4+.

Every calibrator satisfies the :class:`CalibratorAdapter` contract:
``fit(p_uncal, y_true, seed) -> self`` then
``transform(p_uncal) -> calibrated_p``. Both take/return ``(n, 3)``
probability arrays in the fixed column order ``[CALL, PUT, SKIP]``.

Public API
----------
    REGISTRY : dict[str, CalibratorAdapter]
    get(name) -> CalibratorAdapter
    list_calibrators() -> list[str]

Metrics helpers exposed for downstream (Phase 4 orchestrator + tests):
    top1_ece, class_conditional_ece, multiclass_brier,
    multiclass_log_loss, reliability_bins, per_class_reliability_bins

Example
-------
    >>> from calibrators import get
    >>> cal = get("isotonic")
    >>> cal.fit(p_oof, y_oof, seed=42)
    >>> p_calibrated = cal.transform(p_outer_test)
"""
from __future__ import annotations

from ._base import CalibratorAdapter, renormalise_rows
from ._metrics import (
    class_conditional_ece, multiclass_brier, multiclass_log_loss,
    per_class_reliability_bins, reliability_bins, top1_ece,
)
from .isotonic import IsotonicCalibrator
from .noop import NoOpCalibrator
from .platt import PlattScalingCalibrator


def _fresh_registry() -> dict[str, CalibratorAdapter]:
    """Rebuild the registry with fresh (unfitted) calibrator instances.

    The orchestrator uses this to get independent calibrators per fold —
    a single shared instance would carry fit state across folds and
    contaminate results.
    """
    return {
        "noop": NoOpCalibrator(),
        "platt": PlattScalingCalibrator(),
        "isotonic": IsotonicCalibrator(),
    }


REGISTRY: dict[str, CalibratorAdapter] = _fresh_registry()


def get(name: str) -> CalibratorAdapter:
    """Return a FRESH (unfitted) calibrator instance registered under ``name``."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown calibrator {name!r}; registered: {sorted(REGISTRY)}")
    # Return a fresh instance so callers do not share fit state.
    return _fresh_registry()[name]


def list_calibrators() -> list[str]:
    return sorted(REGISTRY)


__all__ = [
    "CalibratorAdapter", "renormalise_rows",
    "REGISTRY", "get", "list_calibrators",
    "NoOpCalibrator", "PlattScalingCalibrator", "IsotonicCalibrator",
    "top1_ece", "class_conditional_ece", "multiclass_brier",
    "multiclass_log_loss", "reliability_bins", "per_class_reliability_bins",
]

__version__ = "1.0.0"
