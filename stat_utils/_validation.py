"""Shared input validators used across stat_utils modules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._errors import InsufficientDataError, InvalidInputError


def as_1d_float_array(x: Any, name: str) -> np.ndarray:
    """Validate *x* is a 1-D numeric array; return float64 view/copy.

    Raises
    ------
    InvalidInputError
        If *x* cannot be interpreted as a 1-D numeric array, or contains
        no elements.
    """
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"{name}: cannot coerce to 1-D float array ({exc})") from exc
    if arr.ndim != 1:
        raise InvalidInputError(
            f"{name}: expected 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise InsufficientDataError(f"{name}: array is empty")
    return arr


def as_2d_float_array(x: Any, name: str) -> np.ndarray:
    """Validate *x* is a 2-D numeric array; return float64 view/copy."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"{name}: cannot coerce to 2-D float array ({exc})") from exc
    if arr.ndim != 2:
        raise InvalidInputError(
            f"{name}: expected 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise InsufficientDataError(f"{name}: array is empty")
    return arr


def validate_fold_streams(streams: Any, name: str) -> dict[Any, np.ndarray]:
    """Validate a mapping of fold-id -> 1D P&L array.

    Empty per-fold arrays are permitted (they contribute zero to concat)
    but the mapping itself must be non-empty.
    """
    if not isinstance(streams, Mapping):
        raise InvalidInputError(
            f"{name}: expected Mapping[fold_id, array], got {type(streams)}")
    if len(streams) == 0:
        raise InsufficientDataError(f"{name}: no folds provided")
    out: dict[Any, np.ndarray] = {}
    for k, v in streams.items():
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                f"{name}[{k!r}]: cannot coerce to float array ({exc})"
            ) from exc
        if arr.ndim != 1:
            raise InvalidInputError(
                f"{name}[{k!r}]: expected 1-D, got shape {arr.shape}")
        out[k] = arr
    if all(a.size == 0 for a in out.values()):
        raise InsufficientDataError(f"{name}: every fold is empty")
    return out


def validate_positive_int(x: Any, name: str, minimum: int = 1) -> int:
    if not isinstance(x, (int, np.integer)) or isinstance(x, bool):
        raise InvalidInputError(f"{name}: expected int, got {type(x)}")
    if int(x) < minimum:
        raise InvalidInputError(f"{name}: must be >= {minimum}, got {x}")
    return int(x)


def validate_ci_level(ci_level: float) -> float:
    if not isinstance(ci_level, (int, float)):
        raise InvalidInputError(
            f"ci_level: expected float, got {type(ci_level)}")
    ci = float(ci_level)
    if not (0.0 < ci < 1.0):
        raise InvalidInputError(
            f"ci_level: must be in (0, 1), got {ci}")
    return ci


def validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float)):
        raise InvalidInputError(f"alpha: expected float, got {type(alpha)}")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise InvalidInputError(f"alpha: must be in (0, 1), got {a}")
    return a


def validate_alternative(alternative: str) -> str:
    ok = {"two-sided", "greater", "less"}
    if alternative not in ok:
        raise InvalidInputError(
            f"alternative: expected one of {sorted(ok)}, got {alternative!r}")
    return alternative
