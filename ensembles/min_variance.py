"""Minimum-Variance Ensemble.

Weights are chosen to minimise the variance of the ensemble's per-fold
performance under a Ledoit-Wolf shrunk covariance of per-brain
per-inner-fold trade Sharpe (or mean per-trade P&L when a fold has too
few trades to compute Sharpe reliably).

QP:
    minimise   w' Sigma_hat w
    s.t.       sum(w) = 1
               w_k >= 0

Solved via ``scipy.optimize.minimize(method='SLSQP')``. If the solve
fails or produces non-finite weights, we fall back to equal weights
and record ``weights_fallback = 'qp_did_not_converge'`` in the state.

Requires ``brain_trade_pnl_by_fold`` at fit time — a mapping
``{brain_name -> {inner_fold_id -> per-trade-pnl-array}}``. Absence
raises :class:`EnsembleInputError`.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._base import EnsembleAdapter, stack_brain_probs, weighted_mixture
from ._validation import EnsembleInputError


class MinVarianceEnsemble(EnsembleAdapter):
    name = "min_variance"
    requires_oof = True
    requires_inner_pnl = True
    stateful_across_bars = False

    def _score_per_fold(self, pnl_by_fold: Mapping[int, np.ndarray]
                          ) -> tuple[list[int], np.ndarray]:
        """Return (sorted fold ids, per-fold score vector).

        Score = per-trade mean P&L when the fold has >= 5 trades and
        non-zero std (else 0).
        """
        fold_ids = sorted(pnl_by_fold)
        scores = np.zeros(len(fold_ids), dtype=np.float64)
        for i, k in enumerate(fold_ids):
            pnl = np.asarray(pnl_by_fold[k], dtype=np.float64)
            if pnl.size >= 5 and pnl.std(ddof=1) > 0:
                scores[i] = float(pnl.mean() / pnl.std(ddof=1))
            elif pnl.size >= 1:
                scores[i] = float(pnl.mean())
        return fold_ids, scores

    def fit(self, brain_probs_oof, y_oof, *,
            brain_trade_pnl_by_fold: Mapping[str, Mapping[int, np.ndarray]] | None = None,
            weights=None, seed=42) -> "MinVarianceEnsemble":
        _ = self._record_brains(brain_probs_oof)
        if brain_trade_pnl_by_fold is None:
            raise EnsembleInputError(
                f"{self.name}.fit: requires brain_trade_pnl_by_fold")

        # Build (K, T) score matrix
        rows: list[np.ndarray] = []
        ref_folds: list[int] | None = None
        for b in self._brains:
            if b not in brain_trade_pnl_by_fold:
                raise EnsembleInputError(
                    f"{self.name}: missing pnl_by_fold for brain {b!r}")
            fold_ids, s = self._score_per_fold(brain_trade_pnl_by_fold[b])
            if ref_folds is None:
                ref_folds = fold_ids
            elif fold_ids != ref_folds:
                raise EnsembleInputError(
                    f"{self.name}: inconsistent inner-fold ids across brains "
                    f"({b}: {fold_ids} vs reference {ref_folds})")
            rows.append(s)

        R = np.stack(rows, axis=0)              # (K, T)
        K, T = R.shape
        if T < 2:
            # Not enough folds to estimate covariance — fall back
            w = np.full(K, 1.0 / K)
            self._state = {
                "weights": {b: float(x) for b, x in zip(self._brains, w)},
                "covariance": None,
                "weights_fallback": "insufficient_inner_folds",
                "n_inner_folds": int(T),
            }
            self._fitted = True
            return self

        # Ledoit-Wolf shrinkage on the (T, K) score matrix
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(R.T)               # sklearn expects (n_samples, n_features)
        Sigma = lw.covariance_
        # Numerical guard: symmetrise + add tiny diag
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma += 1e-10 * np.eye(K)

        # Solve the QP
        from scipy.optimize import minimize
        x0 = np.full(K, 1.0 / K)

        def _obj(w):
            return float(w @ Sigma @ w)

        def _grad(w):
            return 2 * Sigma @ w

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                 "jac": lambda w: np.ones_like(w)}]
        bnds = [(0.0, 1.0)] * K
        fallback = False
        try:
            res = minimize(_obj, x0, jac=_grad, method="SLSQP",
                            bounds=bnds, constraints=cons,
                            options={"maxiter": 200, "ftol": 1e-9})
            w = res.x
            if (not res.success) or not np.isfinite(w).all() or w.sum() <= 0:
                fallback = True
        except Exception:
            fallback = True

        if fallback:
            w = np.full(K, 1.0 / K)
            self._state = {
                "weights": {b: float(x) for b, x in zip(self._brains, w)},
                "covariance": Sigma.tolist(),
                "shrinkage": float(lw.shrinkage_),
                "weights_fallback": "qp_did_not_converge",
                "n_inner_folds": int(T),
            }
        else:
            # Clip small negatives from numerical noise, renormalise
            w = np.clip(w, 0.0, None)
            w = w / w.sum()
            self._state = {
                "weights": {b: float(x) for b, x in zip(self._brains, w)},
                "covariance": Sigma.tolist(),
                "shrinkage": float(lw.shrinkage_),
                "n_inner_folds": int(T),
            }
        self._fitted = True
        return self

    def transform(self, brain_probs_test) -> np.ndarray:
        self._require_fitted()
        vt = self._check_transform_brains(brain_probs_test)
        stacked = stack_brain_probs(vt, self._brains)
        w = np.array([self._state["weights"][b] for b in self._brains])
        return weighted_mixture(stacked, w)

    def weights_summary(self) -> dict:
        return {
            "kind": self.name,
            "weights": self._state.get("weights", {}),
            "shrinkage": self._state.get("shrinkage"),
            "n_inner_folds": self._state.get("n_inner_folds"),
            "weights_fallback": self._state.get("weights_fallback"),
        }
