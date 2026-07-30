"""Manifest schema + hash-based cache validation for Phase 7."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._base import (CacheMismatchError, MANIFEST_SCHEMA_VERSION,
                       PHASE7_VERSION, PROTOCOL_VERSION,
                       sha256_of_file)

# Files whose code hash we lock at experiment time so a downstream edit
# invalidates every cached result.
_CODE_HASH_FILES = (
    "stat_utils/__init__.py",
    "stat_utils/metrics.py",
    "stat_utils/dm.py",
    "stat_utils/spa.py",
    "stat_utils/white_rc.py",
    "stat_utils/bootstrap.py",
    "backtest_options.py",
    "backtest_threshold_sweep.py",
    "threshold_opt/__init__.py",
    "threshold_opt/_base.py",
    "threshold_opt/_evaluate.py",
    "phase7_robustness_v9/_base.py",
    "phase7_robustness_v9/_replay.py",
    "phase7_robustness_v9/_walkforward.py",
    "phase7_robustness_v9/_jackknife.py",
    "phase7_robustness_v9/_slippage.py",
    "phase7_robustness_v9/_tcost.py",
    "phase7_robustness_v9/_delay.py",
    "phase7_robustness_v9/_bootstrap.py",
    "phase7_robustness_v9/_stability.py",
    "phase7_robustness_v9/_stats.py",
    "phase7_robustness_v9/_regime.py",
    "phase7_robustness_v9/_reports.py",
)


def code_hash_dict(root: Path) -> dict[str, str]:
    """Return a {relpath: sha256} map for every registered code file."""
    out: dict[str, str] = {}
    for rel in _CODE_HASH_FILES:
        p = root / rel
        if p.exists():
            out[rel] = sha256_of_file(p)
    return out


def input_hash_for_target(root: Path, target: str) -> dict[str, str]:
    """SHA-256s of the frozen Phase 5/6 artifacts Phase 7 consumes."""
    out: dict[str, str] = {}
    for fold in range(1, 9):
        p = root / "logs" / "phase5" / target / f"fold_{fold}" / "predictions.csv"
        if p.exists():
            out[f"phase5/{target}/fold_{fold}/predictions.csv"] = \
                sha256_of_file(p)
    p6 = root / "logs" / "phase6" / "summary.json"
    if p6.exists():
        out["phase6/summary.json"] = sha256_of_file(p6)
    return out


def build_manifest(*,
                     root: Path,
                     experiment_type: str,
                     target: str,
                     candidate: dict[str, Any],
                     seed: int,
                     extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "phase7_version": PHASE7_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_type": experiment_type,
        "target": target,
        "candidate": candidate,
        "random_seed": int(seed),
        "code_hash": code_hash_dict(root),
        "input_hash": input_hash_for_target(root, target),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "extra": dict(extra or {}),
    }


def save_manifest(manifest: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p


def load_manifest(out_dir: Path) -> dict:
    p = out_dir / "manifest.json"
    if not p.exists():
        raise CacheMismatchError(f"manifest missing: {p}")
    return json.loads(p.read_text())


def verify_cache(out_dir: Path,
                   root: Path,
                   target: str) -> bool:
    """Return True if the cached manifest matches current inputs+code."""
    try:
        m = load_manifest(out_dir)
    except CacheMismatchError:
        return False
    got_input = input_hash_for_target(root, target)
    if m.get("input_hash") != got_input:
        return False
    got_code = code_hash_dict(root)
    if m.get("code_hash") != got_code:
        return False
    return True
