"""Meta-learner factory tests."""
from __future__ import annotations

import numpy as np
import pytest

from ensembles import get_meta


def test_default_is_logistic():
    m = get_meta()
    assert m.__class__.__name__ == "LogisticRegression"


def test_ridge_variant_available():
    m = get_meta("ridge")
    assert m.__class__.__name__ == "RidgeClassifier"


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown meta"):
        get_meta("gradient_boosted_hologram")


def test_logistic_fits_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 9))
    y = (X[:, 0] > 0).astype(int) + (X[:, 3] > 0).astype(int)   # 3 classes
    a = get_meta("logistic", seed=7).fit(X, y)
    b = get_meta("logistic", seed=7).fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_)


def test_logistic_class_weight_balanced():
    m = get_meta("logistic")
    assert m.get_params().get("class_weight") == "balanced"
