"""Frozen result dataclasses returned by every public procedure.

All dataclasses subclass :class:`_JsonMixin` so they are JSON-serialisable
via ``.to_dict()`` and ``.to_json()``. Numpy arrays are converted to plain
lists; non-finite floats (``inf``, ``-inf``, ``nan``) round-trip as strings
so downstream JSON tooling (Phase 6 aggregator, publication pack) does
not choke on them.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# JSON mixin
# ---------------------------------------------------------------------------
def _jsonify(value: Any) -> Any:
    """Recursively convert *value* to JSON-safe primitives."""
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if dataclasses.is_dataclass(value):
        return _jsonify(dataclasses.asdict(value))
    return value


class _JsonMixin:
    """Adds ``to_dict()`` and ``to_json()`` to a frozen dataclass."""

    def to_dict(self) -> dict:
        return _jsonify(dataclasses.asdict(self))  # type: ignore[arg-type]

    def to_json(self, **json_kwargs: Any) -> str:
        json_kwargs.setdefault("indent", 2)
        json_kwargs.setdefault("sort_keys", True)
        return json.dumps(self.to_dict(), **json_kwargs)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapCI(_JsonMixin):
    """Confidence interval from a bootstrap procedure."""

    stat_name: str
    point_estimate: float
    lower: float
    upper: float
    ci_level: float
    n_resamples: int
    n_valid_resamples: int
    n_blocks: int
    block_unit: str
    method: Literal["percentile"]
    seed: int | None
    paired: bool
    bootstrap_distribution: np.ndarray | None = None

    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DMResult(_JsonMixin):
    """Diebold-Mariano test outcome."""

    statistic: float
    pvalue: float
    alternative: Literal["two-sided", "greater", "less"]
    lag: int
    n: int
    mean_loss_diff: float
    se_loss_diff: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    variance_estimator: str = "newey_west"


# ---------------------------------------------------------------------------
# White's Reality Check
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WhiteRCResult(_JsonMixin):
    """White's Reality Check outcome."""

    pvalue: float
    statistic: float
    n_models: int
    n_bootstrap: int
    block_length: int
    per_model_mean_perf: dict[str, float]
    seed: int | None


# ---------------------------------------------------------------------------
# Hansen SPA
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SPAResult(_JsonMixin):
    """Hansen (2005) Superior Predictive Ability test outcome."""

    pvalue_lower: float
    pvalue_consistent: float
    pvalue_upper: float
    statistic: float
    n_models: int
    n_bootstrap: int
    block_length: int
    per_model_mean_perf: dict[str, float]
    seed: int | None


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DSRResult(_JsonMixin):
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014)."""

    dsr: float
    observed_sharpe: float
    threshold_sharpe: float
    n_samples: int
    n_trials: int
    skewness: float
    kurtosis_excess: float
    reference: str = "Bailey_LopezDePrado_2014"


# ---------------------------------------------------------------------------
# CSCV / PBO
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PBOResult(_JsonMixin):
    """Probability of Backtest Overfitting via combinatorially symmetric CV."""

    pbo: float
    logits: np.ndarray
    n_splits: int
    n_models: int
    performance_degradation_median: float
    selected_model_counts: dict[int, int]
    seed: int | None


# ---------------------------------------------------------------------------
# Multiple hypothesis testing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MHTResult(_JsonMixin):
    """Holm-Bonferroni or Benjamini-Hochberg outcome for a family of tests."""

    method: Literal["holm_bonferroni", "benjamini_hochberg"]
    alpha: float
    decisions: dict[str, dict[str, Any]]

    def rejected(self) -> list[str]:
        return [k for k, v in self.decisions.items() if v.get("reject")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KendallResult(_JsonMixin):
    statistic: float
    pvalue: float
    n: int


@dataclass(frozen=True)
class LeveneResult(_JsonMixin):
    statistic: float
    pvalue: float
    center: Literal["mean", "median", "trimmed"]
    n_groups: int


@dataclass(frozen=True)
class KSResult(_JsonMixin):
    statistic: float
    pvalue: float
    n_x: int
    n_y: int


@dataclass(frozen=True)
class PermutationResult(_JsonMixin):
    statistic: float
    pvalue: float
    alternative: Literal["two-sided", "greater", "less"]
    n_permutations: int
    null_distribution: np.ndarray | None = None
