"""Property tests every registered ensemble must pass.

Iterating over :data:`REGISTRY` at parametrisation time means adding a
new ensemble automatically extends the test suite — no per-family
bespoke property test is required for the basic contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import REGISTRY, get, list_ensembles


ENSEMBLES = list_ensembles()


def _fit_kwargs_for(name: str, pnl_by_brain_and_fold):
    """Return additional kwargs a particular ensemble needs during .fit()."""
    if name == "min_variance":
        return {"brain_trade_pnl_by_fold": pnl_by_brain_and_fold}
    return {}


@pytest.mark.parametrize("name", ENSEMBLES)
def test_fit_returns_self(name, three_brains_random, pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get(name)
    out = ens.fit(p, y, seed=1, **_fit_kwargs_for(name, pnl_by_brain_and_fold))
    assert out is ens


@pytest.mark.parametrize("name", ENSEMBLES)
def test_transform_shape(name, three_brains_random, pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get(name).fit(p, y, seed=1,
                          **_fit_kwargs_for(name, pnl_by_brain_and_fold))
    out = ens.transform(p)
    assert out.shape == (len(y), 3)


@pytest.mark.parametrize("name", ENSEMBLES)
def test_transform_rows_sum_to_one(name, three_brains_random,
                                     pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get(name).fit(p, y, seed=1,
                          **_fit_kwargs_for(name, pnl_by_brain_and_fold))
    out = ens.transform(p)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", ENSEMBLES)
def test_transform_in_unit_interval(name, three_brains_random,
                                      pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get(name).fit(p, y, seed=1,
                          **_fit_kwargs_for(name, pnl_by_brain_and_fold))
    out = ens.transform(p)
    assert (out >= -1e-9).all()
    assert (out <= 1 + 1e-9).all()


@pytest.mark.parametrize("name", ENSEMBLES)
def test_no_nans_in_output(name, three_brains_random, pnl_by_brain_and_fold):
    p, y = three_brains_random
    ens = get(name).fit(p, y, seed=1,
                          **_fit_kwargs_for(name, pnl_by_brain_and_fold))
    out = ens.transform(p)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("name", ENSEMBLES)
def test_deterministic_given_seed(name, three_brains_random,
                                    pnl_by_brain_and_fold):
    p, y = three_brains_random
    kw = _fit_kwargs_for(name, pnl_by_brain_and_fold)
    a = get(name).fit(p, y, seed=99, **kw).transform(p)
    b = get(name).fit(p, y, seed=99, **kw).transform(p)
    np.testing.assert_allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("name", ENSEMBLES)
def test_save_load_roundtrip(name, tmp_path, three_brains_random,
                                pnl_by_brain_and_fold):
    p, y = three_brains_random
    kw = _fit_kwargs_for(name, pnl_by_brain_and_fold)
    ens = get(name).fit(p, y, seed=1, **kw)
    p_before = ens.transform(p)
    ens.save(tmp_path)
    ens2 = get(name).load(tmp_path)
    p_after = ens2.transform(p)
    np.testing.assert_allclose(p_before, p_after, atol=1e-9)


@pytest.mark.parametrize("name", ENSEMBLES)
def test_transform_before_fit_raises(name):
    p = {"a": np.array([[0.5, 0.3, 0.2]]),
         "b": np.array([[0.4, 0.3, 0.3]])}
    with pytest.raises(RuntimeError, match="before fit"):
        get(name).transform(p)


@pytest.mark.parametrize("name", ENSEMBLES)
def test_arbitrary_brain_names(name, rng, pnl_by_brain_and_fold):
    """Ensembles must not depend on any particular brain naming
    convention (no hardcoded 'xgb'/'lgb'/etc.)."""
    n = 60
    # min_variance needs its own pnl mapping keyed by the exotic brain names.
    if name == "min_variance":
        p = {"quark": rng.dirichlet(np.ones(3), size=n),
             "lepton": rng.dirichlet(np.ones(3), size=n),
             "boson": rng.dirichlet(np.ones(3), size=n)}
        pnl = {b: {1: rng.normal(size=25), 2: rng.normal(size=30),
                    3: rng.normal(size=28)} for b in p}
        kw = {"brain_trade_pnl_by_fold": pnl}
    else:
        p = {"quark": rng.dirichlet(np.ones(3), size=n),
             "lepton": rng.dirichlet(np.ones(3), size=n),
             "boson": rng.dirichlet(np.ones(3), size=n)}
        kw = {}
    y = rng.integers(0, 3, size=n)
    out = get(name).fit(p, y, seed=1, **kw).transform(p)
    assert out.shape == (n, 3)


@pytest.mark.parametrize("name", ENSEMBLES)
def test_transform_missing_brain_raises(name, three_brains_random,
                                          pnl_by_brain_and_fold):
    p, y = three_brains_random
    kw = _fit_kwargs_for(name, pnl_by_brain_and_fold)
    ens = get(name).fit(p, y, seed=1, **kw)
    # Drop one brain at transform time
    partial = {k: v for k, v in list(p.items())[:2]}
    from ensembles import EnsembleInputError
    with pytest.raises(EnsembleInputError):
        ens.transform(partial)
