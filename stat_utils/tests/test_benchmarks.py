"""Benchmark tests for stat_utils.

Run with:  pytest stat_utils/tests/test_benchmarks.py -v -m slow

These estimate runtime on production-scale inputs (~3,000 pooled trades
across 8 folds, B=10,000). They are marked ``slow`` and skipped by
default.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from stat_utils import (
    block_bootstrap_ci,
    diebold_mariano,
    hansen_spa,
    holm_bonferroni,
    paired_block_bootstrap_ci,
    probability_backtest_overfitting,
    profit_factor,
    white_reality_check,
)


slow = pytest.mark.slow


@pytest.fixture
def production_streams(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """8 folds x ~350 trades each ≈ 2800 pooled trades."""
    return {k: rng.normal(loc=0.05, scale=1.0, size=350) for k in range(1, 9)}


@slow
def test_bench_block_bootstrap(production_streams):
    t0 = time.perf_counter()
    ci = block_bootstrap_ci(production_streams, profit_factor,
                             n_resamples=10_000, seed=0, n_jobs=1)
    elapsed = time.perf_counter() - t0
    assert ci.n_valid_resamples > 8000
    print(f"\nblock_bootstrap_ci (B=10000, n_jobs=1): {elapsed:.2f}s")


@slow
def test_bench_block_bootstrap_parallel(production_streams):
    t0 = time.perf_counter()
    ci = block_bootstrap_ci(production_streams, profit_factor,
                             n_resamples=10_000, seed=0, n_jobs=4)
    elapsed = time.perf_counter() - t0
    print(f"\nblock_bootstrap_ci (B=10000, n_jobs=4): {elapsed:.2f}s")


@slow
def test_bench_paired_bootstrap(production_streams, rng):
    other = {k: rng.normal(loc=0.05, scale=1.0, size=350) for k in range(1, 9)}
    def delta_pf(a, b):
        return profit_factor(a) - profit_factor(b)
    t0 = time.perf_counter()
    paired_block_bootstrap_ci(production_streams, other, delta_pf,
                                n_resamples=10_000, seed=0)
    elapsed = time.perf_counter() - t0
    print(f"\npaired_block_bootstrap_ci (B=10000): {elapsed:.2f}s")


@slow
def test_bench_dm(rng):
    a = rng.normal(loc=0.05, size=3000)
    b = rng.normal(loc=0.03, size=3000)
    t0 = time.perf_counter()
    for _ in range(100):
        diebold_mariano(a, b, lag=5)
    elapsed = time.perf_counter() - t0
    print(f"\n100 * diebold_mariano (n=3000, lag=5): {elapsed:.3f}s")


@slow
def test_bench_white_rc(rng):
    perf = rng.normal(size=(3000, 10))
    t0 = time.perf_counter()
    white_reality_check(perf, n_bootstrap=10_000, seed=0)
    elapsed = time.perf_counter() - t0
    print(f"\nwhite_reality_check (T=3000, K=10, B=10000): {elapsed:.2f}s")


@slow
def test_bench_spa(rng):
    perf = rng.normal(size=(3000, 10))
    t0 = time.perf_counter()
    hansen_spa(perf, n_bootstrap=10_000, seed=0)
    elapsed = time.perf_counter() - t0
    print(f"\nhansen_spa (T=3000, K=10, B=10000): {elapsed:.2f}s")


@slow
def test_bench_pbo(rng):
    perf = rng.normal(size=(3000, 10))
    t0 = time.perf_counter()
    probability_backtest_overfitting(perf, S=8, n_splits="all", seed=0)
    elapsed = time.perf_counter() - t0
    print(f"\nPBO (T=3000, K=10, S=8): {elapsed:.3f}s")


@slow
def test_bench_holm(rng):
    pvs = {f"h_{i}": float(rng.uniform()) for i in range(1000)}
    t0 = time.perf_counter()
    holm_bonferroni(pvs, alpha=0.10)
    elapsed = time.perf_counter() - t0
    print(f"\nholm_bonferroni (m=1000): {elapsed:.4f}s")
