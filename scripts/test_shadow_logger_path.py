#!/usr/bin/env python3
"""
Regression test: ShadowLogger must write to a path resolved from the repo
layout, not the process's current working directory, and a write failure
must be logged loudly (not silently swallowed at DEBUG level).

Standalone script (repo convention), run with:
    python scripts/test_shadow_logger_path.py

Covers core/shadow_logger.py::SHADOW_FILE and ShadowLogger.log_skip().

Bug being regression-tested: SHADOW_FILE was `Path("data/shadow_trades.jsonl")`,
resolved relative to os.getcwd() at import time. data/shadow_trades.jsonl had
not received a new entry since 2026-06-05, despite every filter in
claude_pilot.py correctly calling log_skip() -- and log_skip()'s own
`except Exception: logger.debug(...)` meant a write failure (from any cause,
including a CWD mismatch) would never surface above DEBUG level.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.shadow_logger as shadow_logger_module
from core.shadow_logger import ShadowLogger


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True

    # ── SHADOW_FILE resolves under the repo's data/ dir regardless of CWD ──
    repo_root = Path(__file__).parent.parent.resolve()
    expected = repo_root / "data" / "shadow_trades.jsonl"
    all_ok &= check(
        "SHADOW_FILE resolves to <repo>/data/shadow_trades.jsonl",
        shadow_logger_module.SHADOW_FILE == expected,
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(tempfile.gettempdir())
        all_ok &= check(
            "SHADOW_FILE unaffected by changing the process's CWD afterward "
            "(it's a module-level constant, fixed at import time from __file__)",
            shadow_logger_module.SHADOW_FILE == expected,
        )
    finally:
        os.chdir(original_cwd)

    # ── log_skip() actually writes an entry to a real (temp) path ──────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "shadow_trades.jsonl"
        with patch.object(shadow_logger_module, "SHADOW_FILE", tmp_file):
            logger_instance = object.__new__(ShadowLogger)
            logger_instance._lock = shadow_logger_module.threading.Lock()
            logger_instance._pending = []
            logger_instance.log_skip(
                cycle=1, signal="CALL", conf_pct=55.0,
                reason="trap_gate:LATE_CALL", spot=24000.0, ml_proba=[0.3, 0.5, 0.2],
            )
            all_ok &= check("log_skip() creates the file and writes an entry", tmp_file.exists())
            if tmp_file.exists():
                content = tmp_file.read_text(encoding="utf-8")
                all_ok &= check(
                    "written entry contains the reason string",
                    "trap_gate:LATE_CALL" in content,
                )

    # ── A write failure is logged loudly (ERROR), not silently at DEBUG ────
    with tempfile.TemporaryDirectory() as tmpdir:
        # Point SHADOW_FILE at a path whose parent doesn't exist and won't
        # be created -- forces the file-open to raise.
        bad_path = Path(tmpdir) / "does_not_exist" / "shadow_trades.jsonl"
        with patch.object(shadow_logger_module, "SHADOW_FILE", bad_path):
            logger_instance2 = object.__new__(ShadowLogger)
            logger_instance2._lock = shadow_logger_module.threading.Lock()
            logger_instance2._pending = []
            with patch.object(shadow_logger_module.logger, "error") as mock_error, \
                 patch.object(shadow_logger_module.logger, "debug") as mock_debug:
                logger_instance2.log_skip(
                    cycle=1, signal="PUT", conf_pct=55.0,
                    reason="trap_gate:LATE_PUT", spot=24000.0, ml_proba=[0.3, 0.2, 0.5],
                )
                all_ok &= check(
                    "a write failure calls logger.error (visible), not just logger.debug",
                    mock_error.called and not mock_debug.called,
                )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
