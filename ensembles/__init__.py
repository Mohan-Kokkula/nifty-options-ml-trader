"""ensembles — pluggable ensemble registry for Phase 5.

Every ensemble satisfies :class:`EnsembleAdapter` and receives brain
probabilities as ``Mapping[str, ndarray]``. No adapter has hardcoded
knowledge of specific brain names — the mapping keys are the brain
names in use for that fit/transform call.

Public API
----------
    REGISTRY : dict[str, type[EnsembleAdapter]]
    get(name) -> EnsembleAdapter           # fresh instance
    list_ensembles() -> list[str]

Discovery + probability-source helpers:
    discover_brains, load_brain_probs, ProbabilitySource

Diversity primitives:
    q_statistic, disagreement, double_fault, prediction_correlation,
    diversity_matrix, average_diversity_across_folds

Meta-learner factory:
    get_meta

OOF helper:
    generate_oof_predictions, OOFResult

Validation:
    validate_probability_array, validate_brain_probs_mapping,
    validate_labels, check_no_duplicate_predictions,
    validate_fold_completeness, verify_manifest_hash,
    sha256_of_file, EnsembleInputError, ManifestMismatchError
"""
from __future__ import annotations

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture
from ._diversity import (
    average_diversity_across_folds,
    disagreement,
    diversity_matrix,
    double_fault,
    prediction_correlation,
    q_statistic,
)
from ._meta import get_meta
from ._oof import OOFResult, generate_oof_predictions
from ._selection import (
    ProbabilitySource,
    discover_brains,
    load_brain_probs,
)
from ._validation import (
    EnsembleInputError,
    ManifestMismatchError,
    check_no_duplicate_predictions,
    check_no_nans,
    check_row_normalisation,
    check_shape_consistency,
    sha256_of_file,
    validate_brain_probs_mapping,
    validate_fold_completeness,
    validate_labels,
    validate_probability_array,
    verify_manifest_hash,
)
from .confidence_weighted import ConfidenceWeightedEnsemble
from .mean_probability import MeanProbabilityEnsemble
from .median_probability import MedianProbabilityEnsemble
from .min_variance import MinVarianceEnsemble
from .performance_weighted import PerformanceWeightedEnsemble
from .stacking import StackingEnsemble
from .uncertainty_weighted import UncertaintyWeightedEnsemble
from .weighted_probability import WeightedProbabilityEnsemble


REGISTRY: dict[str, type[EnsembleAdapter]] = {
    "mean":         MeanProbabilityEnsemble,
    "median":       MedianProbabilityEnsemble,
    "weighted":     WeightedProbabilityEnsemble,
    "performance":  PerformanceWeightedEnsemble,
    "min_variance": MinVarianceEnsemble,
    "stacking":     StackingEnsemble,
    "confidence":   ConfidenceWeightedEnsemble,
    "uncertainty":  UncertaintyWeightedEnsemble,
}


def get(name: str) -> EnsembleAdapter:
    """Return a fresh (unfitted) instance of the ensemble registered as ``name``."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown ensemble {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def list_ensembles() -> list[str]:
    """Return the sorted list of registered ensemble names."""
    return sorted(REGISTRY)


__all__ = [
    # ABC + helpers
    "EnsembleAdapter", "stack_brain_probs", "weighted_mixture",
    # Registry
    "REGISTRY", "get", "list_ensembles",
    # Concrete adapters
    "MeanProbabilityEnsemble", "MedianProbabilityEnsemble",
    "WeightedProbabilityEnsemble", "PerformanceWeightedEnsemble",
    "MinVarianceEnsemble", "StackingEnsemble",
    "ConfidenceWeightedEnsemble", "UncertaintyWeightedEnsemble",
    # Discovery
    "ProbabilitySource", "discover_brains", "load_brain_probs",
    # Diversity
    "q_statistic", "disagreement", "double_fault",
    "prediction_correlation", "diversity_matrix",
    "average_diversity_across_folds",
    # Meta-learner factory
    "get_meta",
    # OOF helper
    "OOFResult", "generate_oof_predictions",
    # Validation
    "EnsembleInputError", "ManifestMismatchError",
    "validate_probability_array", "validate_brain_probs_mapping",
    "validate_labels", "check_no_duplicate_predictions",
    "validate_fold_completeness", "verify_manifest_hash",
    "check_shape_consistency", "check_no_nans", "check_row_normalisation",
    "sha256_of_file",
]

__version__ = "1.0.0"
