#!/usr/bin/env python3
"""
Regression test: PILOT_MIN_CONFIDENCE env var must override PilotConfig's
min_confidence, defaulting to the existing 45 when unset.

Standalone script (repo convention), run with:
    python scripts/test_pilot_min_confidence_env.py

Covers main.py's PilotConfig(...) construction and
core/claude_pilot.py::PilotConfig.min_confidence.

This does not call main() (which needs live broker credentials); it
isolates the exact expression used at the PilotConfig(...) call site.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claude_pilot import PilotConfig


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _resolve_min_confidence() -> int:
    """Mirrors the exact expression at main.py's PilotConfig(...) call site."""
    return int(os.getenv("PILOT_MIN_CONFIDENCE", "45"))


def main():
    all_ok = True

    # ── Unset: preserves the existing hardcoded default (45) ───────────
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PILOT_MIN_CONFIDENCE", None)
        resolved = _resolve_min_confidence()
        all_ok &= check(
            "PILOT_MIN_CONFIDENCE unset -> defaults to 45 (unchanged behavior)",
            resolved == 45,
        )
        all_ok &= check(
            "default matches PilotConfig's own dataclass default",
            resolved == PilotConfig().min_confidence,
        )

    # ── Set: overrides correctly ────────────────────────────────────────
    with patch.dict(os.environ, {"PILOT_MIN_CONFIDENCE": "64"}):
        resolved = _resolve_min_confidence()
        all_ok &= check("PILOT_MIN_CONFIDENCE=64 -> resolves to 64", resolved == 64)

    with patch.dict(os.environ, {"PILOT_MIN_CONFIDENCE": "70"}):
        resolved = _resolve_min_confidence()
        all_ok &= check("PILOT_MIN_CONFIDENCE=70 -> resolves to 70", resolved == 70)

    # ── PilotConfig itself accepts the overridden value unchanged ──────
    cfg = PilotConfig(min_confidence=64)
    all_ok &= check("PilotConfig(min_confidence=64).min_confidence == 64", cfg.min_confidence == 64)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
