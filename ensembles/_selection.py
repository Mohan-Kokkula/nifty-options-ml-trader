"""Brain auto-discovery + probability-source resolver.

Phase 5 does NOT hardcode brain names. This module discovers which
brains have Phase-3 outputs on disk, intersects with a CLI request, and
loads probabilities from either raw (Phase 3) or calibrated (Phase 4)
source directories under a uniform interface.

Adding a new brain later requires no code change here: registering an
adapter in ``brains/`` and running Phase 3 for it is enough.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ._validation import EnsembleInputError, validate_probability_array


ProbabilitySource = Literal[
    "raw",
    "calibrated_noop",
    "calibrated_platt",
    "calibrated_isotonic",
]

_CALIBRATED_PREFIX = {
    "calibrated_noop": "noop",
    "calibrated_platt": "platt",
    "calibrated_isotonic": "isotonic",
}


# ---------------------------------------------------------------------------
def _registered_brains() -> list[str]:
    """Return the sorted list of brains registered in the ``brains`` package.

    Delayed import so this module doesn't require ``brains`` to be
    importable at package-import time (relevant for unit tests that mock
    out the registry).
    """
    try:
        from brains import list_brains
        return sorted(list_brains())
    except Exception:
        return []


def _phase3_root(root: Path) -> Path:
    return root / "logs" / "phase3"


def _phase4_root(root: Path) -> Path:
    return root / "logs" / "phase4"


def _phase3_fold_dir(root: Path, brain: str, fold_idx: int) -> Path:
    return _phase3_root(root) / brain / f"fold_{fold_idx}"


def _phase4_fold_dir(root: Path, brain: str, fold_idx: int) -> Path:
    return _phase4_root(root) / brain / f"fold_{fold_idx}"


# ---------------------------------------------------------------------------
def discover_brains(
    project_root: Path,
    fold_idx: int,
    requested: str = "all",
) -> list[str]:
    """Return the sorted list of brains that have Phase-3 output for ``fold_idx``.

    Behaviour
    ---------
    * ``requested == "all"`` (default): intersect the ``brains`` registry
      with brains whose Phase-3 fold directory exists AND contains a
      ``predictions.parquet`` or ``predictions.csv`` file.
    * Otherwise: parse a comma-separated list; every requested brain must
      be BOTH registered AND have Phase-3 output, else
      :class:`EnsembleInputError` is raised describing what's available.
    """
    registered = _registered_brains()
    if not registered:
        # Fall back to filesystem-only discovery — required for tests that
        # deliberately mock the file tree.
        p3 = _phase3_root(project_root)
        registered = sorted(p.name for p in p3.glob("*") if p.is_dir()) if p3.exists() else []

    available: list[str] = []
    for brain in registered:
        d = _phase3_fold_dir(project_root, brain, fold_idx)
        if not d.exists():
            continue
        if (d / "predictions.parquet").exists() or (d / "predictions.csv").exists():
            available.append(brain)
    available.sort()

    if requested == "all":
        result = available
    else:
        wanted = [x.strip() for x in requested.split(",") if x.strip()]
        missing = [b for b in wanted if b not in available]
        if missing:
            raise EnsembleInputError(
                f"requested brains {missing} unavailable for fold {fold_idx}; "
                f"available = {available}, registered = {registered}")
        result = sorted(wanted)

    if len(result) < 2:
        raise EnsembleInputError(
            f"fold {fold_idx}: only {len(result)} brain(s) available "
            f"({result}); ensembles require >= 2")
    return result


# ---------------------------------------------------------------------------
def _read_predictions(dir_path: Path, prefix: str = "") -> pd.DataFrame:
    """Read predictions from ``{prefix}predictions.parquet`` or CSV fallback.

    Raises :class:`FileNotFoundError` when neither variant exists.
    """
    pq = dir_path / f"{prefix}predictions.parquet"
    csv = dir_path / f"{prefix}predictions.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"no predictions file at {dir_path} (prefix={prefix!r}); "
        f"looked for {pq.name} and {csv.name}")


def load_brain_probs(
    project_root: Path,
    brain: str,
    fold_idx: int,
    source: ProbabilitySource = "raw",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load ``(probabilities, y_true, timestamps)`` for ONE brain × fold.

    Parameters
    ----------
    project_root : Path
        Project root containing ``logs/phase3`` and ``logs/phase4``.
    brain : str
    fold_idx : int
    source : {"raw", "calibrated_noop", "calibrated_platt", "calibrated_isotonic"}
        Selects which Phase output to consume. Same call site works for
        every value — the only difference is which file gets read.

    Returns
    -------
    (probs, y_true, timestamps)
        ``probs`` is a validated ``(n, 3)`` float64 array with rows
        summing to 1. ``y_true`` is ``(n,)`` int32. ``timestamps`` is a
        pandas ``DatetimeIndex``.

    Raises
    ------
    FileNotFoundError
        If the source file for this configuration does not exist. The
        error message names the exact path expected.
    EnsembleInputError
        On structural/validation failures (missing columns, non-normalised
        probs, etc.).
    """
    if source == "raw":
        d = _phase3_fold_dir(project_root, brain, fold_idx)
        df = _read_predictions(d, prefix="")
    else:
        d = _phase4_fold_dir(project_root, brain, fold_idx)
        prefix = _CALIBRATED_PREFIX.get(source)
        if prefix is None:
            raise EnsembleInputError(f"unknown probability source {source!r}")
        df = _read_predictions(d, prefix=f"{prefix}_")

    required = ["p_call", "p_put", "p_skip", "y_true", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise EnsembleInputError(
            f"{brain}/fold_{fold_idx} ({source}): predictions missing "
            f"columns {missing}")

    probs = df[["p_call", "p_put", "p_skip"]].values.astype(np.float64)
    probs = validate_probability_array(
        probs, f"{brain}/fold_{fold_idx} ({source}).probs")

    y_true = df["y_true"].values.astype(np.int32)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    if len(ts) != len(probs):
        raise EnsembleInputError(
            f"{brain}/fold_{fold_idx}: len(timestamps)={len(ts)} != "
            f"len(probs)={len(probs)}")
    return probs, y_true, ts
