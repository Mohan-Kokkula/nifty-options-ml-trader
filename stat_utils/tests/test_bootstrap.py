"""Tests for stat_utils.bootstrap."""
from __future__ import annotations

import numpy as np
import pytest

from stat_utils import (
    BootstrapCI,
    InsufficientDataError,
    InvalidInputError,
    block_bootstrap_ci,
    expectancy,
    paired_block_bootstrap_ci,
    profit_factor,
)


def test_bootstrap_ci_reproducible_with_seed(iid_gaussian_streams):
    a = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                            n_resamples=500, seed=42)
    b = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                            n_resamples=500, seed=42)
    assert a.lower == b.lower and a.upper == b.upper


def test_bootstrap_ci_njobs_invariant(iid_gaussian_streams):
    """Determinism guarantee: output must be independent of n_jobs."""
    a = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                            n_resamples=200, seed=42, n_jobs=1)
    b = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                            n_resamples=200, seed=42, n_jobs=4)
    # Sort not needed — same seed produces same replicate ordering by design.
    if a.bootstrap_distribution is None:
        assert a.lower == b.lower and a.upper == b.upper
    else:
        np.testing.assert_array_equal(a.bootstrap_distribution,
                                       b.bootstrap_distribution)


def test_bootstrap_ci_ordering_and_containment(iid_gaussian_streams):
    ci = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                             n_resamples=2000, seed=1)
    assert ci.lower <= ci.point_estimate <= ci.upper
    assert 0.0 < ci.ci_level < 1.0
    assert ci.n_valid_resamples <= ci.n_resamples


def test_bootstrap_ci_returns_dataclass(iid_gaussian_streams):
    ci = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                             n_resamples=100, seed=0)
    assert isinstance(ci, BootstrapCI)
    d = ci.to_dict()
    assert "lower" in d and "upper" in d and "point_estimate" in d


def test_bootstrap_ci_coverage_close_to_nominal(rng):
    """A properly implemented percentile CI should cover the truth close
    to the nominal rate on iid data. Test with a modest N.

    Truth = mean(loc=0.10). Nominal = 90%.
    """
    truth = 0.10
    hits = 0
    n_trials = 60
    for trial in range(n_trials):
        streams = {k: rng.normal(loc=truth, scale=1.0, size=200)
                   for k in range(1, 9)}
        ci = block_bootstrap_ci(streams, expectancy,
                                 n_resamples=500, seed=trial)
        if ci.lower <= truth <= ci.upper:
            hits += 1
    coverage = hits / n_trials
    # Slack for Monte Carlo variability.
    assert 0.75 <= coverage <= 1.0


def test_bootstrap_ci_rejects_empty_folds():
    with pytest.raises(InsufficientDataError):
        block_bootstrap_ci({0: np.array([]), 1: np.array([])},
                            expectancy, seed=0)


def test_bootstrap_ci_rejects_bad_n_resamples(iid_gaussian_streams):
    with pytest.raises(InvalidInputError):
        block_bootstrap_ci(iid_gaussian_streams, expectancy,
                            n_resamples=0, seed=0)


def test_bootstrap_ci_rejects_non_callable_statistic(iid_gaussian_streams):
    with pytest.raises(InvalidInputError):
        block_bootstrap_ci(iid_gaussian_streams, "not_a_func", seed=0)


def test_bootstrap_ci_rejects_bad_ci_level(iid_gaussian_streams):
    with pytest.raises(InvalidInputError):
        block_bootstrap_ci(iid_gaussian_streams, expectancy,
                            ci_level=1.5, seed=0)


def test_paired_bootstrap_zero_delta_for_identical(iid_gaussian_streams):
    def delta_pf(a, b):
        return profit_factor(a) - profit_factor(b)
    ci = paired_block_bootstrap_ci(iid_gaussian_streams, iid_gaussian_streams,
                                    delta_pf, n_resamples=500, seed=7)
    # When both arms are the same object, every paired replicate yields 0.
    assert ci.point_estimate == pytest.approx(0.0, abs=1e-12)
    assert ci.lower == pytest.approx(0.0, abs=1e-12)
    assert ci.upper == pytest.approx(0.0, abs=1e-12)
    assert ci.paired is True


def test_paired_bootstrap_ci_reproducible(iid_gaussian_streams,
                                            positive_edge_streams):
    def delta_mean(a, b):
        return float(a.mean() - b.mean())
    a = paired_block_bootstrap_ci(positive_edge_streams, iid_gaussian_streams,
                                   delta_mean, n_resamples=500, seed=99)
    b = paired_block_bootstrap_ci(positive_edge_streams, iid_gaussian_streams,
                                   delta_mean, n_resamples=500, seed=99)
    assert a.lower == b.lower and a.upper == b.upper


def test_paired_bootstrap_rejects_mismatched_folds(iid_gaussian_streams):
    def delta_mean(a, b):
        return float(a.mean() - b.mean())
    other = {"X": np.array([1.0, 2.0])}
    with pytest.raises(InvalidInputError):
        paired_block_bootstrap_ci(iid_gaussian_streams, other, delta_mean,
                                   seed=0)


def test_bootstrap_json_roundtrip(iid_gaussian_streams):
    ci = block_bootstrap_ci(iid_gaussian_streams, expectancy,
                             n_resamples=50, seed=0)
    import json
    d = json.loads(ci.to_json())
    assert set(("lower", "upper", "point_estimate", "n_resamples")).issubset(d)


def test_bootstrap_positive_edge_ci_above_zero(positive_edge_streams):
    """Streams with mean 0.20 and low noise → CI on the mean should be
    entirely > 0 at 90%."""
    ci = block_bootstrap_ci(positive_edge_streams, expectancy,
                             n_resamples=2000, seed=3)
    assert ci.lower > 0.0
