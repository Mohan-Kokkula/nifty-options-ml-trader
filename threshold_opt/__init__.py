"""threshold_opt — Phase 6 threshold optimization.

Deterministic grid search over ``(call_thr, put_thr, skip_ceil,
min_edge)`` applied to frozen Phase-5 ensemble probabilities. Every
statistical primitive is imported from ``stat_utils`` — Phase 6
introduces no new statistical methods.
"""
from __future__ import annotations

from ._base import (
    InvalidThresholdError,
    PRODUCTION_BASELINE,
    ThresholdCandidate,
    ThresholdOptError,
)
from ._evaluate import (
    CandidateResult,
    apply_thresholds,
    evaluate_candidate,
)
from ._grid import (
    DEFAULT_CALL_RANGE,
    DEFAULT_EDGE_RANGE,
    DEFAULT_PUT_RANGE,
    DEFAULT_SKIP_RANGE,
    grid_generator,
    grid_size,
)
from ._manifest import (
    CacheMismatchError,
    build_manifest,
    load_manifest,
    sha256_of_file,
    verify_cache,
)
from ._ranking import rank_candidates
from ._stats import compare_to_baseline, top_k_comparison
from ._visualize import build_chart_data, save_chart_data, write_chart_pngs


__version__ = "1.0.0"

__all__ = [
    # Errors
    "ThresholdOptError", "InvalidThresholdError", "CacheMismatchError",
    # Types
    "ThresholdCandidate", "CandidateResult", "PRODUCTION_BASELINE",
    # Grid
    "grid_generator", "grid_size",
    "DEFAULT_CALL_RANGE", "DEFAULT_PUT_RANGE",
    "DEFAULT_SKIP_RANGE", "DEFAULT_EDGE_RANGE",
    # Evaluate
    "apply_thresholds", "evaluate_candidate",
    # Ranking
    "rank_candidates",
    # Manifest
    "build_manifest", "load_manifest", "verify_cache", "sha256_of_file",
    # Stats
    "compare_to_baseline", "top_k_comparison",
    # Viz
    "build_chart_data", "save_chart_data", "write_chart_pngs",
]
