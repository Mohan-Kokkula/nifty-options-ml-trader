"""Meta-learner factory for stacking ensembles.

Isolated in its own module so future ensemble variants (Ridge, small
MLP, others) can be added without touching ``stacking.py``.
"""
from __future__ import annotations

from typing import Any, Literal


def get_meta(kind: Literal["logistic", "ridge"] = "logistic",
              seed: int = 42) -> Any:
    """Return a sklearn-compatible classifier for stacking.

    Parameters
    ----------
    kind : {"logistic", "ridge"}
        Default ``"logistic"`` — multinomial logistic regression with
        moderate regularisation, class-balanced, deterministic given
        ``seed``. Standard for stacking.
    seed : int
    """
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        # NOTE: sklearn >= 1.5 dropped the ``multi_class`` kwarg (the
        # multinomial solver is now auto-selected when ``solver='lbfgs'``
        # is combined with more than two classes). We just pass lbfgs.
        return LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
    if kind == "ridge":
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"unknown meta kind {kind!r}")
