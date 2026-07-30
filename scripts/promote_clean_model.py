"""
promote_clean_model.py — one-shot, safe promotion of the clean V9 model.

Resolves, in a single atomic action:
  R1: V10-first load order        → leaked V10 trio renamed *.disabled_leaked
  R2: poisoned champion baseline  → clean model becomes the V9 champion
  +  : leakage-safety preflight, loadability verification, rollback breadcrumbs

DEFAULT IS DRY-RUN. Nothing changes without --execute.

Usage:
  python scripts/promote_clean_model.py            # show plan only
  python scripts/promote_clean_model.py --execute  # do it
After --execute: restart the bot service to load the promoted model
  (sudo systemctl restart openclaw-v9-kotak on the server).

NOTE kept deliberately: the V10 registry champion stays poisoned (0.911) so
nightly V10 challengers KEEP being rejected — that is a desired lock while
V10 has no validated design. BUT registry pruning (keep=14) will delete that
champion entry in ~14 daily runs, after which a V10 challenger auto-promotes
via NO_CHAMPION and outranks V9 in ml_engine's load order. Until a V10 is
deliberately validated, run the cron with --v9-only or re-run this script's
--execute (it re-disables the V10 trio).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                      # model_registry uses relative paths (R4)
sys.path.insert(0, str(ROOT))

import joblib                                                  # noqa: E402

CLEAN = ROOT / "models" / "sandbox_clean"
MODELS = ROOT / "models"
V10_FILES = ["nifty_v10_models.pkl", "nifty_v10_scaler.pkl",
             "feature_cols_v10.pkl"]


def preflight() -> dict:
    print("── PREFLIGHT ──")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_leakage_safety.py")],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"  leakage-safety tests: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(r.stdout[-2000:])
        sys.exit("ABORT: leakage tests failing — fix pipeline before promoting.")
    for f in ("models.pkl", "scaler.pkl", "feature_cols.pkl", "metadata.json"):
        if not (CLEAN / f).exists():
            sys.exit(f"ABORT: {CLEAN / f} missing.")
    meta = json.loads((CLEAN / "metadata.json").read_text())
    test = meta.get("test") or {}
    print(f"  clean model: test dir_acc={test.get('dir_acc'):.4f} "
          f"signals={test.get('n_signals')} (leaked ref 0.9286)")
    joblib.load(CLEAN / "models.pkl")          # loadability
    print("  clean models.pkl loadable: OK")
    return meta


def main():
    execute = "--execute" in sys.argv
    meta = preflight()
    test = meta.get("test") or {}

    from core.model_registry import (save_candidate, rollback,
                                     get_active_metadata)
    old_v9 = get_active_metadata("v9") or {}
    old_v10 = get_active_metadata("v10") or {}

    print("\n── PLAN ──")
    print(f"  1. Register sandbox_clean as v9 registry candidate "
          f"(becomes champion baseline: dir_acc {test.get('dir_acc'):.4f})")
    print(f"  2. Force-promote it to live V9 paths via registry rollback "
          f"(atomic, marks promoted=True)")
    print(f"  3. Disable leaked V10 trio (rename *.disabled_leaked) so "
          f"ml_engine falls through to V9")
    print(f"  4. Verify ml_engine loads [V9] without MISMATCH warning")
    print(f"  Rollback breadcrumbs: previous champions "
          f"v9={old_v9.get('candidate_id')} (acc={old_v9.get('test_dir_acc')}), "
          f"v10={old_v10.get('candidate_id')} (acc={old_v10.get('test_dir_acc')})")

    if not execute:
        print("\nDRY-RUN ONLY — re-run with --execute to apply. "
              "After executing, restart the bot service.")
        return

    print("\n── EXECUTE ──")
    models = joblib.load(CLEAN / "models.pkl")
    scaler = joblib.load(CLEAN / "scaler.pkl")
    fcols = joblib.load(CLEAN / "feature_cols.pkl")
    cid = save_candidate("v9", models, scaler, fcols, {
        "test_dir_acc": float(test.get("dir_acc") or 0),
        "test_signals": int(test.get("n_signals") or 0),
        "val_dir_acc": float((meta.get("val") or {}).get("dir_acc") or 0),
        "n_features": len(fcols),
        "train_date_start": "2015-01-20",
        "train_date_end": str(meta.get("train_end", "2024-08-30"))[:10],
        "test_date_start": str((meta.get("test_range") or ["2025-07-08"])[0])[:10],
        "test_date_end": str((meta.get("test_range") or ["", "2026-06-04"])[1])[:10],
        "clean_retrain": True,
        "note": ("Phase-0 leak-fixed pipeline; replaces leakage-inflated "
                 "champion (0.939). Promoted deliberately on "
                 f"{datetime.now():%Y-%m-%d} to reset the CC baseline."),
    })
    print(f"  registered candidate v9/{cid}")
    if not rollback("v9", cid):
        sys.exit("ABORT: registry rollback (promotion copy) failed.")
    print("  live V9 trio updated atomically, candidate marked promoted")

    for f in V10_FILES:
        src = MODELS / f
        if src.exists():
            dst = MODELS / (f + ".disabled_leaked")
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            print(f"  disabled {f} → {dst.name}")

    # verification: fresh interpreter, must load V9 with no mismatch warning
    code = ("import logging,io; buf=io.StringIO(); "
            "h=logging.StreamHandler(buf); logging.getLogger().addHandler(h); "
            "logging.getLogger().setLevel(logging.INFO); "
            "from core import ml_engine; ok=ml_engine.load_model(); "
            "out=buf.getvalue(); "
            "print('LOADED:', ml_engine._loaded_version, '| ok:', ok); "
            "print('MISMATCH WARNING PRESENT:', 'MISMATCH' in out)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT))
    print("  " + (r.stdout.strip().replace("\n", "\n  ") or r.stderr[-400:]))
    if "LOADED: V9" not in r.stdout or "MISMATCH WARNING PRESENT: False" not in r.stdout:
        print("  ⚠️ VERIFICATION UNEXPECTED — review before restarting the bot!")
    else:
        print("\n✅ Promotion complete. RESTART THE BOT SERVICE to load it:")
        print("   sudo systemctl restart openclaw-v9-kotak")
        print("   (restart also activates: OI archiver, shadow model hook)")
    print(f"\nRollback: python scripts/retrain_weekly.py --rollback "
          f"v9:{old_v9.get('candidate_id')}  + rename the .disabled_leaked "
          f"files back.")


if __name__ == "__main__":
    main()
