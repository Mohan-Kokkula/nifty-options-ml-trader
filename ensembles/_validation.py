"""Input validation for Phase 5 ensemble infrastructure.

All checks raise with descriptive error messages. There are no silent
fallbacks in this module — a failed check aborts the pipeline.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
class EnsembleInputError(ValueError):
    """Raised when ensemble input data fails a structural check."""


class ManifestMismatchError(RuntimeError):
    """Raised when an artifact manifest's recorded hashes don't match reality."""


# ---------------------------------------------------------------------------
def validate_probability_array(p: Any, name: str,
                                 n_classes: int = 3,
                                 tol: float = 1e-6) -> np.ndarray:
    """Validate that ``p`` is a well-formed (n, n_classes) probability matrix.

    Checks: 2-D, correct number of columns, dtype-castable to float, no
    NaN/Inf, all values in [-tol, 1 + tol], and each row sums to 1
    within ``tol``.

    Returns
    -------
    ndarray
        Float64 array (may be the input if it already conforms).

    Raises
    ------
    EnsembleInputError
        On any structural failure.
    """
    try:
        arr = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EnsembleInputError(
            f"{name}: cannot coerce to float array ({exc})") from exc
    if arr.ndim != 2:
        raise EnsembleInputError(
            f"{name}: expected 2-D probability array, got shape {arr.shape}")
    if arr.size == 0:
        raise EnsembleInputError(f"{name}: empty probability array")
    if arr.shape[1] != n_classes:
        raise EnsembleInputError(
            f"{name}: expected {n_classes} columns, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise EnsembleInputError(
            f"{name}: contains non-finite values (nan={n_nan}, inf={n_inf})")
    if (arr < -tol).any() or (arr > 1 + tol).any():
        raise EnsembleInputError(
            f"{name}: probabilities out of [0, 1] range "
            f"(min={arr.min():.4g}, max={arr.max():.4g})")
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        bad = int((np.abs(row_sums - 1.0) > tol).sum())
        raise EnsembleInputError(
            f"{name}: {bad} row(s) do not sum to 1 within tolerance {tol}"
            f" (min sum={row_sums.min():.4g}, max sum={row_sums.max():.4g})")
    return arr


def validate_brain_probs_mapping(
    probs: Mapping[str, Any], name: str, n_classes: int = 3,
) -> dict[str, np.ndarray]:
    """Validate a ``{brain_name: probability_array}`` mapping.

    Every array is validated with :func:`validate_probability_array`;
    all arrays must share the same length.

    Returns
    -------
    dict[str, ndarray]
        Validated float64 arrays keyed by the brain names in ``probs``.
    """
    if not isinstance(probs, Mapping):
        raise EnsembleInputError(
            f"{name}: expected Mapping[str, ndarray], got {type(probs)}")
    if len(probs) == 0:
        raise EnsembleInputError(f"{name}: no brains provided")
    if len(probs) < 2:
        raise EnsembleInputError(
            f"{name}: ensembles require >= 2 brains, got {len(probs)}")
    out: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    for brain, p in probs.items():
        if not isinstance(brain, str) or not brain:
            raise EnsembleInputError(
                f"{name}: brain names must be non-empty strings, got {brain!r}")
        arr = validate_probability_array(p, f"{name}[{brain!r}]",
                                          n_classes=n_classes)
        out[brain] = arr
        lengths[brain] = arr.shape[0]
    lens = set(lengths.values())
    if len(lens) != 1:
        raise EnsembleInputError(
            f"{name}: brain probability arrays have inconsistent lengths: "
            f"{lengths}")
    return out


def validate_labels(y: Any, name: str, n_classes: int = 3) -> np.ndarray:
    """Validate that ``y`` is a 1-D integer label array in ``[0, n_classes)``."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise EnsembleInputError(
            f"{name}: expected 1-D labels, got shape {arr.shape}")
    if arr.size == 0:
        raise EnsembleInputError(f"{name}: empty labels array")
    if not np.issubdtype(arr.dtype, np.integer):
        try:
            arr = arr.astype(int)
        except Exception as exc:
            raise EnsembleInputError(
                f"{name}: cannot coerce labels to int ({exc})") from exc
    if arr.min() < 0 or arr.max() >= n_classes:
        raise EnsembleInputError(
            f"{name}: labels must lie in [0, {n_classes}); "
            f"got min={arr.min()}, max={arr.max()}")
    return arr


def check_no_duplicate_predictions(pred_df: pd.DataFrame,
                                     key: str = "timestamp") -> None:
    """Raise if the predictions frame contains duplicate rows at ``key``."""
    if key not in pred_df.columns:
        raise EnsembleInputError(
            f"predictions frame missing required column {key!r}")
    dup = pred_df[key].duplicated()
    if dup.any():
        n_dup = int(dup.sum())
        first_dup = pred_df.loc[dup, key].iloc[0]
        raise EnsembleInputError(
            f"predictions contain {n_dup} duplicate {key} entries "
            f"(first: {first_dup})")


def validate_fold_completeness(
    brains: list[str],
    root: Path,
    fold_idx: int,
    subpath_template: str = "logs/phase3/{brain}/fold_{fold}",
) -> None:
    """Raise with a descriptive message if any brain has no fold artifact.

    Passes silently if every brain in ``brains`` has a directory at
    ``root / subpath_template.format(brain=..., fold=fold_idx)`` that
    exists and is non-empty.
    """
    if not brains:
        raise EnsembleInputError("validate_fold_completeness: empty brain list")
    missing: list[str] = []
    for brain in brains:
        sub = root / subpath_template.format(brain=brain, fold=fold_idx)
        if not sub.exists() or not any(sub.iterdir()):
            missing.append(brain)
    if missing:
        raise EnsembleInputError(
            f"fold {fold_idx}: missing outputs for brains {missing} "
            f"under {subpath_template!r}")


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def verify_manifest_hash(manifest_path: Path,
                          project_root: Path) -> None:
    """Recompute hashes recorded in ``manifest_path`` and raise on drift.

    Only ``code_hash`` entries are re-verified (data hashes reference
    artifacts that may legitimately be regenerated between runs). Missing
    code files or SHA mismatches raise :class:`ManifestMismatchError`.
    """
    doc = json.loads(manifest_path.read_text())
    code_hash = doc.get("code_hash") or {}
    problems: list[str] = []
    for rel, expected in code_hash.items():
        f = project_root / rel
        if not f.exists():
            problems.append(f"{rel}: missing")
            continue
        actual = sha256_of_file(f)
        if actual != expected:
            problems.append(f"{rel}: expected {expected[:10]}..., got {actual[:10]}...")
    if problems:
        raise ManifestMismatchError(
            f"manifest {manifest_path} drifted:\n  " + "\n  ".join(problems))


def check_shape_consistency(brain_probs: Mapping[str, np.ndarray]) -> None:
    """Raise if brain probability arrays have inconsistent (n, K) shapes."""
    if not brain_probs:
        raise EnsembleInputError("check_shape_consistency: empty mapping")
    shapes = {b: np.asarray(p).shape for b, p in brain_probs.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise EnsembleInputError(
            f"brain probability shapes differ: {shapes}")


def check_no_nans(p: np.ndarray, name: str) -> None:
    """Raise if ``p`` contains any NaN."""
    arr = np.asarray(p)
    if arr.size == 0:
        return
    if np.isnan(arr).any():
        n = int(np.isnan(arr).sum())
        raise EnsembleInputError(f"{name}: {n} NaN value(s) present")


def check_row_normalisation(p: np.ndarray, tol: float = 1e-6) -> None:
    """Raise if any row of ``p`` does not sum to 1 within ``tol``."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2:
        raise EnsembleInputError(
            f"expected 2-D array for row normalisation check, got {arr.shape}")
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        bad = int((np.abs(row_sums - 1.0) > tol).sum())
        raise EnsembleInputError(
            f"{bad} row(s) not normalised within tol={tol}")
