"""Candidate evaluation tests — signal math + metric aggregation."""
from __future__ import annotations

import numpy as np
import pytest

from threshold_opt import (PRODUCTION_BASELINE, ThresholdCandidate,
                             apply_thresholds)


# ---------- apply_thresholds against reference behaviour ----------
def test_apply_thresholds_matches_backtest_reference():
    """Our re-implementation must equal signals_from_probas + MIN_EDGE."""
    from backtest_threshold_sweep import signals_from_probas, MIN_EDGE
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=200)
    # Production defaults, min_edge = MIN_EDGE (0.05)
    ref = signals_from_probas(p, 0.32, 0.25, 0.65)
    ours = apply_thresholds(p, ThresholdCandidate(0.32, 0.25, 0.65, MIN_EDGE))
    assert np.array_equal(ours, ref)


def test_apply_thresholds_hand_computed(synth_hand_probs):
    p = synth_hand_probs
    c = PRODUCTION_BASELINE   # (0.32, 0.25, 0.65, 0.05)
    sig = apply_thresholds(p, c)
    # bar 0: 0.60 >= 0.32 and 0.60-0.20=0.40 >= 0.05 and 0.20 < 0.65 → CALL(0)
    assert sig[0] == 0
    # bar 1: p_put=0.70 >= 0.25 and 0.70-0.10=0.60 >= 0.05 and 0.20 < 0.65 → PUT(1)
    assert sig[1] == 1
    # bar 2: p_call=0.30 < 0.32 → not CALL; p_put=0.30 >= 0.25 and 0.30-0.30=0 < 0.05 → not PUT → SKIP(2)
    assert sig[2] == 2
    # bar 3: p_call=0.40 >= 0.32, 0.40-0.25=0.15 >= 0.05, 0.35 < 0.65 → CALL(0)
    assert sig[3] == 0
    # bar 4: p_put=0.30 >= 0.25, 0.30-0.20=0.10 >= 0.05, 0.50 < 0.65 → PUT(1)
    assert sig[4] == 1


def test_apply_thresholds_rejects_wrong_shape():
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        apply_thresholds(np.zeros((5,)), PRODUCTION_BASELINE)
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        apply_thresholds(np.zeros((5, 4)), PRODUCTION_BASELINE)


def test_higher_call_thr_reduces_call_signals(synth_probs):
    p, _ = synth_probs
    sig_low = apply_thresholds(p, ThresholdCandidate(0.20, 0.25, 0.65, 0.05))
    sig_high = apply_thresholds(p, ThresholdCandidate(0.40, 0.25, 0.65, 0.05))
    assert (sig_low == 0).sum() >= (sig_high == 0).sum()


def test_higher_min_edge_reduces_trades(synth_probs):
    p, _ = synth_probs
    sig_low = apply_thresholds(p, ThresholdCandidate(0.20, 0.15, 0.75, 0.01))
    sig_high = apply_thresholds(p, ThresholdCandidate(0.20, 0.15, 0.75, 0.08))
    assert (sig_low != 2).sum() >= (sig_high != 2).sum()


def test_lower_skip_ceil_reduces_trades(synth_probs):
    p, _ = synth_probs
    sig_high_ceil = apply_thresholds(p, ThresholdCandidate(0.20, 0.15, 0.90, 0.05))
    sig_low_ceil = apply_thresholds(p, ThresholdCandidate(0.20, 0.15, 0.50, 0.05))
    assert (sig_high_ceil != 2).sum() >= (sig_low_ceil != 2).sum()


# ---------- _fold_metrics ----------
def test_fold_metrics_empty():
    from threshold_opt._evaluate import _fold_metrics
    m = _fold_metrics(np.array([]))
    assert m["n"] == 0
    assert m["net"] == 0.0


def test_fold_metrics_hand_computed():
    from threshold_opt._evaluate import _fold_metrics
    pnl = np.array([+2.0, -1.0, +3.0, -2.0])
    m = _fold_metrics(pnl)
    assert m["n"] == 4
    assert m["net"] == pytest.approx(2.0)
    # PF: gains=5, losses=3 → 5/3
    assert m["pf"] == pytest.approx(5.0 / 3.0)
    # WR: 2/4
    assert m["wr"] == 0.5


def test_fold_metrics_expectancy_hand_computed():
    from threshold_opt._evaluate import _fold_metrics
    # 2 wins (+2, +3, mean=2.5), 2 losses (-1, -2, mean=-1.5)
    # expectancy = 0.5*2.5 + 0.5*(-1.5) = 0.5
    pnl = np.array([+2.0, -1.0, +3.0, -2.0])
    m = _fold_metrics(pnl)
    assert m["expectancy"] == pytest.approx(0.5)


def test_fold_metrics_no_losses_returns_inf_pf():
    from threshold_opt._evaluate import _fold_metrics
    pnl = np.array([1.0, 2.0, 3.0])
    m = _fold_metrics(pnl)
    assert m["pf"] == float("inf")
