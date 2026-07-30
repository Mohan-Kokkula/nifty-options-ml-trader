"""OOF-generator structural tests.

The full OOF path calls into the heavy ``backtest_threshold_sweep`` +
``brains._hpo`` machinery and is exercised by integration tests when
Phase 5 is executed on real data. Here we validate:

  * the ``OOFResult`` dataclass shape + frozen semantics
  * the ``generate_oof_predictions`` function's public signature

We do NOT run a real inner-CV retrain from a unit test — that would
depend on Phase-3/4 assets and heavyweight ML training that this
delivery is explicitly instructed to avoid.
"""
from __future__ import annotations

from datetime import date

import inspect
import numpy as np
import pytest


def test_oof_result_dataclass_shape():
    from ensembles import OOFResult
    r = OOFResult(
        p_oof=np.array([[0.5, 0.3, 0.2]]),
        y_oof=np.array([0]),
        inner_pnl_by_fold=None,
        n_train_per_inner=[10],
        inner_fold_dates=[(date(2020, 1, 1), date(2020, 1, 4),
                             date(2020, 1, 10))],
        elapsed_s=0.1,
    )
    assert r.p_oof.shape == (1, 3)
    assert r.y_oof.shape == (1,)
    assert r.inner_pnl_by_fold is None
    assert r.n_train_per_inner == [10]
    assert r.elapsed_s == 0.1


def test_oof_result_is_frozen():
    from ensembles import OOFResult
    r = OOFResult(
        p_oof=np.zeros((0, 3)),
        y_oof=np.zeros(0, dtype=int),
        inner_pnl_by_fold=None,
        n_train_per_inner=[],
        inner_fold_dates=[],
        elapsed_s=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        r.p_oof = np.ones((1, 3))


def test_generate_oof_predictions_has_expected_signature():
    from ensembles import generate_oof_predictions
    sig = inspect.signature(generate_oof_predictions)
    params = sig.parameters
    assert "brain" in params
    assert "feat" in params
    assert "fcols" in params
    assert "outer_tr_mask" in params
    assert "k_inner" in params
    assert "seed" in params
    assert "collect_inner_pnl" in params
    assert "iv" in params
    assert "exp" in params
    # k_inner default = 3, matches Phase-4 convention
    assert params["k_inner"].default == 3
    assert params["seed"].default == 42
    assert params["collect_inner_pnl"].default is False
