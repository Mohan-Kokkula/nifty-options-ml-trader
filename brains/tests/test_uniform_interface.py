"""Property tests that every registered brain must pass.

Iterating over REGISTRY at parametrisation time means adding a new brain
automatically extends the test suite — no per-family bespoke tests
needed for the basic contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from brains import REGISTRY, as_3class_proba, list_brains


BRAINS = list_brains()


@pytest.fixture
def brain_params(small_params):
    return small_params


@pytest.mark.parametrize("name", BRAINS)
def test_fit_returns_a_model(name, synthetic_3class, brain_params):
    brain = REGISTRY[name]
    d = synthetic_3class
    m = brain.fit(d["X_train"], d["y_train"],
                  X_eval=d["X_eval"], y_eval=d["y_eval"],
                  sample_weight=d["w_train"],
                  params=brain_params[name], seed=7)
    assert m is not None


@pytest.mark.parametrize("name", BRAINS)
def test_predict_proba_shape_and_normalisation(name, synthetic_3class,
                                                 brain_params):
    brain = REGISTRY[name]
    d = synthetic_3class
    m = brain.fit(d["X_train"], d["y_train"],
                  X_eval=d["X_eval"], y_eval=d["y_eval"],
                  sample_weight=d["w_train"],
                  params=brain_params[name], seed=7)
    p = brain.predict_proba_3class(m, d["X_test"])
    assert p.shape == (len(d["X_test"]), 3)
    row_sums = p.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-3)
    assert (p >= 0).all() and (p <= 1).all()


@pytest.mark.parametrize("name", BRAINS)
def test_save_load_roundtrip(name, synthetic_3class, brain_params, tmp_path):
    brain = REGISTRY[name]
    d = synthetic_3class
    m = brain.fit(d["X_train"], d["y_train"],
                  X_eval=d["X_eval"], y_eval=d["y_eval"],
                  sample_weight=d["w_train"],
                  params=brain_params[name], seed=11)
    p_before = brain.predict_proba_3class(m, d["X_test"])
    fold_dir = tmp_path / f"fold_{name}"
    saved = brain.save(m, fold_dir)
    assert saved.exists()
    m2 = brain.load(fold_dir)
    p_after = brain.predict_proba_3class(m2, d["X_test"])
    np.testing.assert_allclose(p_before, p_after, atol=1e-8)


@pytest.mark.parametrize("name", BRAINS)
def test_no_sample_weight_still_fits(name, synthetic_3class, brain_params):
    brain = REGISTRY[name]
    d = synthetic_3class
    m = brain.fit(d["X_train"], d["y_train"],
                  X_eval=d["X_eval"], y_eval=d["y_eval"],
                  sample_weight=None,
                  params=brain_params[name], seed=3)
    p = brain.predict_proba_3class(m, d["X_test"])
    assert p.shape == (len(d["X_test"]), 3)


@pytest.mark.parametrize("name", BRAINS)
def test_optuna_search_space_returns_valid_params(name, synthetic_3class,
                                                    brain_params):
    """Every adapter's optuna_search_space must produce a dict of params
    that can be merged with defaults and used for a real fit."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    brain = REGISTRY[name]

    sampled: dict = {}

    def objective(trial):
        nonlocal sampled
        sampled = brain.optuna_search_space(trial)
        return 0.0  # dummy

    study = optuna.create_study()
    study.optimize(objective, n_trials=1, show_progress_bar=False)
    assert isinstance(sampled, dict)
    # Merge with small_params override and confirm fit succeeds.
    d = synthetic_3class
    params = {**sampled, **brain_params[name]}
    m = brain.fit(d["X_train"], d["y_train"],
                  X_eval=d["X_eval"], y_eval=d["y_eval"],
                  sample_weight=d["w_train"], params=params, seed=1)
    assert m is not None


@pytest.mark.parametrize("name", BRAINS)
def test_deterministic_given_seed(name, synthetic_3class, brain_params):
    brain = REGISTRY[name]
    d = synthetic_3class
    m1 = brain.fit(d["X_train"], d["y_train"],
                    X_eval=d["X_eval"], y_eval=d["y_eval"],
                    sample_weight=d["w_train"],
                    params=brain_params[name], seed=99)
    m2 = brain.fit(d["X_train"], d["y_train"],
                    X_eval=d["X_eval"], y_eval=d["y_eval"],
                    sample_weight=d["w_train"],
                    params=brain_params[name], seed=99)
    p1 = brain.predict_proba_3class(m1, d["X_test"])
    p2 = brain.predict_proba_3class(m2, d["X_test"])
    # Some libraries have very small non-determinism from threading;
    # allow tiny tolerance.
    np.testing.assert_allclose(p1, p2, atol=1e-6)
