"""Out-of-fold probability generator for Phase 5 (self-contained).

Runs nested purged cross-validation within the outer training window
using the same ``brains._hpo.build_inner_folds`` boundary computer that
Phases 2, 3, and 4 use. Produces a concatenated OOF probability pool
suitable for training stacking / performance-weighted / min-variance
ensembles.

.. note::
   Phase 4 currently contains near-identical logic in
   ``phase4_run_single_fold._train_inner_and_predict``. TODO: after
   Phase 4 has fully completed its overnight run, refactor both call
   sites to share a single implementation (either here or in
   ``brains/_oof.py``). This file is a deliberate temporary duplicate
   so Phase 4 can keep running untouched.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class OOFResult:
    """Result of :func:`generate_oof_predictions`."""

    p_oof: np.ndarray                                        # (n_oof, 3)
    y_oof: np.ndarray                                        # (n_oof,)
    inner_pnl_by_fold: dict[int, np.ndarray] | None          # optional
    n_train_per_inner: list[int]
    inner_fold_dates: list[tuple[date, date, date]]
    elapsed_s: float


def generate_oof_predictions(
    brain,                                # BrainAdapter instance
    feat: pd.DataFrame,
    fcols: list[str],
    outer_tr_mask: np.ndarray,
    *,
    k_inner: int = 3,
    seed: int = 42,
    collect_inner_pnl: bool = False,
    iv: Any = None,
    exp: Any = None,
) -> OOFResult:
    """Train ``brain`` on each inner-CV fold within ``outer_tr_mask`` and
    return the concatenated out-of-fold prediction pool.

    Optional trade P&L collection (``collect_inner_pnl=True``) invokes
    ``backtest_options.simulate_trades`` on each inner val chunk.
    Required for the :class:`MinVarianceEnsemble` adapter's ``fit()``.

    Notes
    -----
    * Uses ``brains._hpo.build_inner_folds`` so inner-CV boundaries are
      identical to Phase 4's calibration OOF pool.
    * Uses ``backtest_threshold_sweep._sample_weights`` so training-time
      class-imbalance handling matches earlier phases exactly.
    * All computation is deterministic given ``seed``.
    """
    from brains._hpo import build_inner_folds
    from backtest_threshold_sweep import _sample_weights, signals_from_probas
    if collect_inner_pnl:
        from backtest_options import simulate_trades  # noqa: F401

    outer_tr_dates = sorted(set(feat.index[outer_tr_mask].date))
    inner_folds = build_inner_folds(outer_tr_dates, k_inner)

    p_pieces: list[np.ndarray] = []
    y_pieces: list[np.ndarray] = []
    inner_pnl: dict[int, np.ndarray] = {}
    n_train_per_inner: list[int] = []
    t0 = time.time()

    for i, (train_end, val_start, val_end) in enumerate(inner_folds, 1):
        inner_tr_mask = outer_tr_mask & (feat.index.date < train_end)
        inner_val_mask = ((feat.index.date >= val_start)
                          & (feat.index.date < val_end))
        inner_tr = feat[inner_tr_mask]
        inner_val = feat[inner_val_mask]
        if len(inner_tr) < 5000 or len(inner_val) < 50:
            continue

        cut = int(len(inner_tr) * 0.95)
        tr_early = inner_tr.iloc[:cut]
        ev_early = inner_tr.iloc[cut:]
        n_train_per_inner.append(len(tr_early))

        w = _sample_weights(tr_early)
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr_early[fcols].values)
        Xev = sc.transform(ev_early[fcols].values)
        Xval = sc.transform(inner_val[fcols].values)
        ytr = tr_early["label"].values
        yev = ev_early["label"].values
        yval = inner_val["label"].values

        model = brain.fit(Xtr, ytr, X_eval=Xev, y_eval=yev,
                          sample_weight=w,
                          params=brain.default_params(), seed=seed)
        p_val = brain.predict_proba_3class(model, Xval)
        p_pieces.append(p_val)
        y_pieces.append(yval)

        if collect_inner_pnl and iv is not None and exp is not None:
            from backtest_options import simulate_trades as _simt
            from backtest_threshold_sweep import (
                signals_from_probas as _sfp,
            )
            CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65
            sig = _sfp(p_val, CALL_THR, PUT_THR, SKIP_CEIL)
            tdf = _simt(inner_val, sig, p_val, iv, exp)
            pnl = (tdf["net_option"].values.astype(np.float64)
                   if len(tdf) and "net_option" in tdf.columns
                   else np.array([], dtype=np.float64))
            inner_pnl[i] = pnl

        del model; gc.collect()

    if not p_pieces:
        raise RuntimeError(
            "generate_oof_predictions produced no OOF predictions; "
            "check inner-CV boundaries and data availability")

    return OOFResult(
        p_oof=np.concatenate(p_pieces, axis=0),
        y_oof=np.concatenate(y_pieces, axis=0),
        inner_pnl_by_fold=inner_pnl if collect_inner_pnl else None,
        n_train_per_inner=n_train_per_inner,
        inner_fold_dates=inner_folds,
        elapsed_s=round(time.time() - t0, 2),
    )
