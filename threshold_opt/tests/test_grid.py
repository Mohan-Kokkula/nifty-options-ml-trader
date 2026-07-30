"""Grid generator tests — deterministic + valid outputs only."""
from __future__ import annotations

import pytest

from threshold_opt import (DEFAULT_CALL_RANGE, DEFAULT_EDGE_RANGE,
                             DEFAULT_PUT_RANGE, DEFAULT_SKIP_RANGE,
                             grid_generator, grid_size)


def test_default_grid_size_matches_cartesian_product_minus_invalid():
    # 6 * 5 * 4 * 3 = 360; all should be valid (min_edge < min(call, put))
    total = 6 * 5 * 4 * 3
    # min_edge 0.03/0.05/0.08 vs put ∈ {0.15, 0.20, 0.25, 0.30, 0.35}
    # min_edge=0.08 < 0.15 → OK. So all 360 valid.
    assert grid_size() == total


def test_iteration_order_is_call_put_skip_edge():
    xs = list(grid_generator(
        call=[0.30, 0.40],
        put=[0.20, 0.25],
        skip=[0.65],
        edge=[0.05],
    ))
    # 2 x 2 x 1 x 1 = 4 candidates
    assert len(xs) == 4
    # Expect: (0.30, 0.20), (0.30, 0.25), (0.40, 0.20), (0.40, 0.25)
    got = [(c.call_thr, c.put_thr) for c in xs]
    assert got == [(0.30, 0.20), (0.30, 0.25), (0.40, 0.20), (0.40, 0.25)]


def test_grid_deterministic_across_calls():
    a = [(c.call_thr, c.put_thr, c.skip_ceil, c.min_edge)
         for c in grid_generator()]
    b = [(c.call_thr, c.put_thr, c.skip_ceil, c.min_edge)
         for c in grid_generator()]
    assert a == b


def test_grid_skips_invalid_min_edge_combinations():
    # min_edge = 0.30 would violate put_thr = 0.15
    xs = list(grid_generator(
        call=[0.40], put=[0.15], skip=[0.65], edge=[0.30],
    ))
    assert xs == []


def test_custom_ranges_honored():
    xs = list(grid_generator(
        call=[0.10, 0.15], put=[0.08], skip=[0.70], edge=[0.02],
    ))
    assert len(xs) == 2
    assert all(c.skip_ceil == 0.70 for c in xs)


def test_default_ranges_are_expected_values():
    assert 0.32 in DEFAULT_CALL_RANGE
    assert 0.25 in DEFAULT_PUT_RANGE
    assert 0.65 in DEFAULT_SKIP_RANGE
    assert 0.05 in DEFAULT_EDGE_RANGE


def test_duplicate_values_dedup():
    xs = list(grid_generator(
        call=[0.30, 0.30, 0.30], put=[0.25], skip=[0.65], edge=[0.05],
    ))
    assert len(xs) == 1


def test_grid_ranges_are_sorted_internally():
    xs = list(grid_generator(
        call=[0.40, 0.30, 0.35], put=[0.25],
        skip=[0.65], edge=[0.05],
    ))
    calls = [c.call_thr for c in xs]
    assert calls == sorted(calls)


def test_grid_size_matches_iterator_length():
    for ranges in [
        ({"call": [0.30], "put": [0.25], "skip": [0.65], "edge": [0.05]}, 1),
        ({"call": [0.30, 0.35], "put": [0.20, 0.25], "skip": [0.65], "edge": [0.05]}, 4),
    ]:
        args, expected = ranges
        assert grid_size(**args) == expected
        assert sum(1 for _ in grid_generator(**args)) == expected
