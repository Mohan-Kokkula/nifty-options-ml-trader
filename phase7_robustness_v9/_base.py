"""Phase 7 – shared config, candidates, exceptions.

Phase 7 is additive-only. No Phase 0-6 file is modified; this package
imports Phase 0-6 APIs as read-only clients.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Read-only imports from earlier phases
from threshold_opt import ThresholdCandidate, PRODUCTION_BASELINE  # noqa: F401

PHASE7_VERSION = "1.0"
MANIFEST_SCHEMA_VERSION = "1.0"
PROTOCOL_VERSION = "2.5"       # frozen at Phase 6 supersede

# Fold set: 8 walk-forward folds, identical to Phase 5/6.
FOLDS = tuple(range(1, 9))

# Grid multipliers (pre-registered — locked here so tests can import).
SLIPPAGE_MULTIPLIERS = (1.00, 1.25, 1.50, 1.75, 2.00)
TCOST_MULTIPLIERS    = (1.00, 1.25, 1.50, 1.75, 2.00)
EXEC_DELAYS_BARS     = (0, 1, 2)
ROLLING_WINDOW       = 3       # for rolling walk-forward
BOOTSTRAP_B          = 10_000
DEFAULT_SEED         = 42

# H_rob decision rule constants (per user-approved refinement).
H_ROB_MIN_OUTPERFORM_FRAC = 0.80
H_ROB_MAX_STABILITY_CV    = 0.50
H_ROB_CATASTROPHE_PF      = 0.80
H_ROB_CATASTROPHE_NET_MULT = 2.0
H_ROB_CI_LB_THRESHOLD     = 0.0    # LB90(ΔPF) must be strictly > 0

# Regime classification thresholds (annualised total-return over fold window).
BULL_MIN_ANNUALISED = 0.08
BEAR_MAX_ANNUALISED = -0.08


class Phase7Error(Exception):
    """Base class for Phase 7 errors."""


class InvalidInputError(Phase7Error):
    """A caller passed an object the Phase 7 pipeline cannot use."""


class MissingManifestError(Phase7Error):
    """Required Phase 5/6 manifest not found."""


class CacheMismatchError(Phase7Error):
    """A cached artifact's hashes no longer match expected inputs."""


# ---------------------------------------------------------------------------
def hash8(obj: Any) -> str:
    """Deterministic 8-hex identifier for a JSON-serialisable object."""
    j = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:8]


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NamedCandidate:
    name: str
    candidate: ThresholdCandidate
    origin: str      # short human-readable origin (e.g. "phase6_mean_winner")


@dataclass
class Phase7Config:
    seed: int = DEFAULT_SEED
    targets: tuple[str, ...] = ("mean", "stacking")
    root: Path = field(default_factory=lambda: Path("logs/phase7"))
    skip_charts: bool = False
    force: bool = False
    min_trades_pooled: int = 30      # sanity: report if a variant drops below

    def target_root(self, target: str) -> Path:
        return self.root / target
