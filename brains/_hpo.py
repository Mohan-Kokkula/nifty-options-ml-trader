"""HPO helpers shared between Phase 2 and Phase 3.

``build_inner_folds`` is the extracted purged inner-CV boundary computer
originally in ``phase2_run_single_fold.py``. It is imported from there
after the Phase 3 refactor (Phase 2 behaviour unchanged).

``run_hpo`` is a brain-generic Optuna driver used only by Phase 3. It
delegates the actual search-space definition to
``brain.optuna_search_space(trial)``.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_options import simulate_trades
from backtest_threshold_sweep import (
    EMBARGO_DAYS,
    _sample_weights,
    signals_from_probas,
)

CALL_THR, PUT_THR, SKIP_CEIL = 0.32, 0.25, 0.65
SORTINO_TIEBREAK = 1e-3
DEGENERATE_TRIAL_SCORE = -10.0


# ---------------------------------------------------------------------------
def build_inner_folds(outer_tr_dates: list[date], k: int
                       ) -> list[tuple[date, date, date]]:
    """Purged expanding-window inner CV boundaries.

    Given the sorted set of *dates* covered by the outer training
    window and the desired number of inner folds ``k``, return a list
    of ``(train_end_date, val_start_date, val_end_date)`` triples.
    Every triple satisfies
    ``val_start_date - train_end_date >= EMBARGO_DAYS`` by
    construction, i.e. inner-CV is purge-compliant.

    The last ~45% of the outer training window is reserved for the
    ``k`` inner-val chunks; the first ~55% is available to all inner
    folds as train.

    Parameters
    ----------
    outer_tr_dates : list[date]
        Sorted unique dates present in the outer training window.
    k : int
        Number of inner folds. Must be >= 1.

    Returns
    -------
    list of (train_end_date, val_start_date, val_end_date)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not outer_tr_dates:
        raise ValueError("outer_tr_dates is empty")
    start = outer_tr_dates[0]
    end = outer_tr_dates[-1]
    total = (end - start).days
    val_region_start_days = int(total * 0.55)
    val_region_end_days = total
    span = max(1, (val_region_end_days - val_region_start_days) // k)
    folds: list[tuple[date, date, date]] = []
    for i in range(k):
        val_start = start + timedelta(days=val_region_start_days + i * span)
        val_end = start + timedelta(days=val_region_start_days + (i + 1) * span)
        train_end = val_start - timedelta(days=EMBARGO_DAYS)
        folds.append((train_end, val_start, val_end))
    return folds


# ---------------------------------------------------------------------------
def _fold_metrics_lite(pnl: np.ndarray) -> dict:
    """Trimmed metrics used only inside the HPO objective."""
    if len(pnl) == 0:
        return dict(n=0, pf=float("nan"), sortino=float("nan"),
                    sharpe=float("nan"))
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl <= 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf")
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    sh = float(pnl.mean() / sd) if sd > 0 else float("nan")
    downside = pnl[pnl < 0]
    if downside.size and (downside ** 2).mean() > 0:
        sortino = float(pnl.mean() / np.sqrt((downside ** 2).mean()))
    else:
        sortino = float("nan")
    return dict(n=int(len(pnl)), pf=pf, sortino=sortino, sharpe=sh)


def run_hpo(brain, feat: pd.DataFrame, fcols: list[str],
            outer_tr_mask: np.ndarray, iv: dict, exp: dict,
            *, n_trials: int = 30, k_inner: int = 3, seed: int = 42
            ) -> tuple[dict, pd.DataFrame, float]:
    """Optuna nested-purged HPO driver, generic over any BrainAdapter.

    Objective: ``mean(inner_PF) + 1e-3 * mean(inner_Sortino)`` over
    ``k_inner`` purged inner folds. Same convention as Phase 2.

    Parameters
    ----------
    brain : BrainAdapter
        Adapter that supplies ``optuna_search_space`` and ``fit``.
    feat : DataFrame
        Feature frame indexed by timestamp.
    fcols : list[str]
        Feature column names.
    outer_tr_mask : ndarray[bool]
        Boolean mask selecting outer training bars.
    iv, exp : dict
        Options implied-vol and expiry maps from ``build_iv_map``.
    n_trials : int
        Optuna trials. Default 30.
    k_inner : int
        Inner CV folds. Default 3.
    seed : int
        Random seed.

    Returns
    -------
    (best_params, trials_df, elapsed_seconds)
        ``best_params`` is the trial-sampled dict; the orchestrator
        merges with adapter defaults before final retrain.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:  # pragma: no cover - dep should be present
        raise RuntimeError("optuna not installed; pip install optuna")

    outer_tr_dates = sorted(set(feat.index[outer_tr_mask].date))
    inner_folds = build_inner_folds(outer_tr_dates, k_inner)

    def objective(trial: "optuna.Trial") -> float:
        params = brain.optuna_search_space(trial)
        pfs, sortinos, sharpes, counts = [], [], [], []
        t0 = time.time()
        for (train_end, val_start, val_end) in inner_folds:
            inner_tr_mask = outer_tr_mask & (feat.index.date < train_end)
            inner_val_mask = (
                (feat.index.date >= val_start)
                & (feat.index.date < val_end))
            inner_tr = feat[inner_tr_mask]
            inner_val = feat[inner_val_mask]
            if len(inner_tr) < 5000 or len(inner_val) < 50:
                continue
            cut = int(len(inner_tr) * 0.95)
            tr_early = inner_tr.iloc[:cut]
            ev_early = inner_tr.iloc[cut:]
            w = _sample_weights(tr_early)
            sc = StandardScaler()
            Xtr = sc.fit_transform(tr_early[fcols].values)
            Xev = sc.transform(ev_early[fcols].values)
            Xval = sc.transform(inner_val[fcols].values)
            ytr = tr_early["label"].values
            yev = ev_early["label"].values
            m = brain.fit(Xtr, ytr, X_eval=Xev, y_eval=yev,
                          sample_weight=w, params=params, seed=seed)
            p = brain.predict_proba_3class(m, Xval)
            sig = signals_from_probas(p, CALL_THR, PUT_THR, SKIP_CEIL)
            tdf = simulate_trades(inner_val, sig, p, iv, exp)
            pnl = (tdf["net_option"].values.astype(float)
                   if len(tdf) and "net_option" in tdf.columns
                   else np.array([]))
            met = _fold_metrics_lite(pnl)
            pfs.append(met["pf"])
            sortinos.append(met["sortino"])
            sharpes.append(met["sharpe"])
            counts.append(met["n"])
            del m; gc.collect()
        trial.set_user_attr("pfs", pfs)
        trial.set_user_attr("sortinos", sortinos)
        trial.set_user_attr("sharpes", sharpes)
        trial.set_user_attr("trade_counts", counts)
        trial.set_user_attr("runtime_s", time.time() - t0)
        finite_pfs = [x for x in pfs if np.isfinite(x)]
        finite_sortinos = [x for x in sortinos if np.isfinite(x)]
        if not finite_pfs:
            return DEGENERATE_TRIAL_SCORE
        mean_pf = float(np.mean(finite_pfs))
        mean_sortino = (float(np.mean(finite_sortinos))
                        if finite_sortinos else 0.0)
        return mean_pf + SORTINO_TIEBREAK * mean_sortino

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    rows: list[dict] = []
    for t in study.trials:
        row: dict[str, Any] = {"trial": t.number, "state": t.state.name,
                                "value": t.value}
        row.update(t.params)
        for key in ("pfs", "sortinos", "sharpes", "trade_counts"):
            v = t.user_attrs.get(key)
            row[key] = json.dumps(v) if v is not None else None
        row["runtime_s"] = t.user_attrs.get("runtime_s")
        rows.append(row)
    trials_df = pd.DataFrame(rows)
    return study.best_params, trials_df, elapsed
