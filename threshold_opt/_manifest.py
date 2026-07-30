"""Manifest schema + cache validation for threshold candidates.

Each candidate carries:
    * protocol_version
    * code_hash        — tracked source files
    * data_hash        — Phase-5 prediction files consumed
    * random_seed
    * timestamp_utc
    * target_ensemble
    * input_source
    * threshold_values
    * folds_evaluated
    * n_trades_per_fold
    * min_trades_requirement
    * passes_min_trades_filter
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._base import ThresholdCandidate, ThresholdOptError


class CacheMismatchError(ThresholdOptError):
    """Raised when a cached manifest's hashes don't match current state."""


def sha256_of_file(p: Path, chunk: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of a file's binary content."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def build_manifest(
    cand: ThresholdCandidate,
    target_ensemble: str,
    input_source: str,
    folds_evaluated: list[int],
    n_trades_per_fold: list[int],
    min_trades_requirement: int,
    passes_min_trades_filter: bool,
    code_hash: dict[str, str],
    data_hash: dict[str, str],
    protocol_version: str,
    seed: int = 42,
) -> dict:
    """Build a manifest dict that satisfies Phase-6 schema requirements."""
    return {
        "protocol_version": str(protocol_version),
        "code_hash": dict(code_hash),
        "data_hash": dict(data_hash),
        "random_seed": int(seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "target_ensemble": str(target_ensemble),
        "input_source": str(input_source),
        "threshold_values": cand.to_dict(),
        "folds_evaluated": [int(f) for f in folds_evaluated],
        "n_trades_per_fold": [int(n) for n in n_trades_per_fold],
        "min_trades_requirement": int(min_trades_requirement),
        "passes_min_trades_filter": bool(passes_min_trades_filter),
    }


def verify_cache(
    manifest: dict,
    current_code_hash: dict[str, str],
    current_data_hash: dict[str, str],
) -> bool:
    """Return True iff the manifest's recorded hashes match current state."""
    return (
        manifest.get("code_hash", {}) == current_code_hash
        and manifest.get("data_hash", {}) == current_data_hash
    )


def load_manifest(path: Path) -> dict:
    """Load a manifest JSON. Raises :class:`CacheMismatchError` on missing/broken."""
    if not path.exists():
        raise CacheMismatchError(f"manifest not found: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise CacheMismatchError(f"unreadable manifest {path}: {exc}") from exc
