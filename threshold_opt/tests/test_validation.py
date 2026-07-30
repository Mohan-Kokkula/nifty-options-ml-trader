"""ThresholdCandidate validation tests."""
from __future__ import annotations

import pytest

from threshold_opt import (InvalidThresholdError, PRODUCTION_BASELINE,
                             ThresholdCandidate)


def test_valid_candidate_ok():
    c = ThresholdCandidate(0.32, 0.25, 0.65, 0.05)
    assert c.call_thr == 0.32
    assert c.put_thr == 0.25


def test_out_of_range_call():
    with pytest.raises(InvalidThresholdError, match="in \\[0, 1\\]"):
        ThresholdCandidate(1.2, 0.25, 0.65, 0.05)


def test_out_of_range_put():
    with pytest.raises(InvalidThresholdError, match="in \\[0, 1\\]"):
        ThresholdCandidate(0.32, -0.1, 0.65, 0.05)


def test_out_of_range_skip():
    with pytest.raises(InvalidThresholdError):
        ThresholdCandidate(0.32, 0.25, 2.0, 0.05)


def test_out_of_range_edge():
    with pytest.raises(InvalidThresholdError):
        ThresholdCandidate(0.32, 0.25, 0.65, -0.01)


def test_min_edge_greater_than_call_rejected():
    with pytest.raises(InvalidThresholdError, match="< call_thr"):
        ThresholdCandidate(0.20, 0.30, 0.65, 0.25)


def test_min_edge_greater_than_put_rejected():
    with pytest.raises(InvalidThresholdError, match="< put_thr"):
        ThresholdCandidate(0.40, 0.15, 0.65, 0.20)


def test_min_edge_equal_to_call_rejected():
    with pytest.raises(InvalidThresholdError, match="< call_thr"):
        ThresholdCandidate(0.10, 0.25, 0.65, 0.10)


def test_call_thr_equal_put_thr_allowed():
    """Signal logic disambiguates via class-vs-class differential."""
    c = ThresholdCandidate(0.30, 0.30, 0.65, 0.05)
    assert c.call_thr == c.put_thr


def test_hash8_deterministic():
    a = ThresholdCandidate(0.32, 0.25, 0.65, 0.05).hash8()
    b = ThresholdCandidate(0.32, 0.25, 0.65, 0.05).hash8()
    assert a == b


def test_hash8_differs_when_inputs_differ():
    a = ThresholdCandidate(0.32, 0.25, 0.65, 0.05).hash8()
    b = ThresholdCandidate(0.30, 0.25, 0.65, 0.05).hash8()
    assert a != b


def test_frozen_dataclass_prevents_mutation():
    c = ThresholdCandidate(0.32, 0.25, 0.65, 0.05)
    with pytest.raises(Exception):
        c.call_thr = 0.99


def test_production_baseline_values():
    assert PRODUCTION_BASELINE.to_dict() == {
        "call_thr": 0.32, "put_thr": 0.25, "skip_ceil": 0.65, "min_edge": 0.05,
    }
