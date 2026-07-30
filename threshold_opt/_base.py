"""Immutable threshold-candidate dataclass + validation.

A candidate is a 4-tuple ``(call_thr, put_thr, skip_ceil, min_edge)``
that fully determines how probabilities are converted into CALL/PUT/SKIP
signals. The dataclass is frozen and hashable so it can key a cache
directly.

Validation:
  * every threshold must lie in [0, 1]
  * ``min_edge`` must be strictly less than both ``call_thr`` and
    ``put_thr`` — otherwise the signal condition ``p_class - p_other >=
    min_edge`` is unsatisfiable
  * ``call_thr == put_thr`` is ALLOWED. The signal logic in
    ``signals_from_probas`` disambiguates via ``p_call - p_put >=
    min_edge`` vs. ``p_put - p_call >= min_edge`` — a bar cannot
    simultaneously satisfy both when ``min_edge > 0``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


class ThresholdOptError(Exception):
    """Base class for threshold_opt exceptions."""


class InvalidThresholdError(ThresholdOptError, ValueError):
    """Raised when a candidate has structurally invalid thresholds."""


@dataclass(frozen=True)
class ThresholdCandidate:
    """A single point in the threshold search space."""

    call_thr: float
    put_thr: float
    skip_ceil: float
    min_edge: float

    def __post_init__(self) -> None:
        for name, val in (
            ("call_thr", self.call_thr),
            ("put_thr", self.put_thr),
            ("skip_ceil", self.skip_ceil),
            ("min_edge", self.min_edge),
        ):
            if not isinstance(val, (int, float)):
                raise InvalidThresholdError(
                    f"{name}: expected float, got {type(val).__name__}")
            v = float(val)
            if not (0.0 <= v <= 1.0):
                raise InvalidThresholdError(
                    f"{name}: must be in [0, 1], got {v}")
        # min_edge must be strictly less than both class thresholds so
        # the class-vs-class differential condition is satisfiable.
        if self.min_edge >= self.call_thr:
            raise InvalidThresholdError(
                f"min_edge ({self.min_edge}) must be < call_thr "
                f"({self.call_thr})")
        if self.min_edge >= self.put_thr:
            raise InvalidThresholdError(
                f"min_edge ({self.min_edge}) must be < put_thr "
                f"({self.put_thr})")
        # call_thr == put_thr allowed: signal logic uses opposing
        # class differentials so a single bar never triggers both.

    def hash8(self) -> str:
        """Deterministic 8-hex identifier for cache directories."""
        key = (f"{self.call_thr:.6f}_{self.put_thr:.6f}_"
                f"{self.skip_ceil:.6f}_{self.min_edge:.6f}")
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {
            "call_thr": float(self.call_thr),
            "put_thr": float(self.put_thr),
            "skip_ceil": float(self.skip_ceil),
            "min_edge": float(self.min_edge),
        }


# The production baseline — used ONLY for comparison. Never eligible to
# win the search per Phase 6 acceptance criteria.
PRODUCTION_BASELINE = ThresholdCandidate(
    call_thr=0.32, put_thr=0.25, skip_ceil=0.65, min_edge=0.05,
)
