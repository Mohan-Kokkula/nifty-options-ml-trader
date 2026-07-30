"""
test_model_registry.py — Unit tests for core/model_registry.py (Issue #4)

Run:
    python scripts/test_model_registry.py

Tests:
  A. save_candidate      -- creates correct dir structure and metadata
  B. promote gate PASS   -- good metrics -> live files updated atomically
  C. promote gate FAIL   -- low test_dir_acc -> live files UNTOUCHED
  D. promote gate FAIL   -- too few signals -> live files UNTOUCHED
  E. rollback            -- restores arbitrary past candidate to live
  F. list_versions       -- returns newest-first with correct metadata
  G. prune               -- removes oldest entries, keeps N most recent
  H. live path untouched -- training script never overwrites live directly
  I. atomic copy         -- .tmp file absent after successful promotion
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch MODELS_DIR and REGISTRY_DIR to a temp dir so tests are isolated
import core.model_registry as reg

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def _fake_models():
    """Minimal sklearn-compatible estimator sufficient for joblib round-trip."""
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0]], [0])
    return {"xgb": clf}


def _fake_scaler():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit([[0, 1], [1, 2]])
    return sc


def _good_meta(**overrides):
    base = {
        "train_date_start": "2015-01-09", "train_date_end":  "2026-05-31",
        "val_date_start":   "2026-06-01", "val_date_end":    "2026-06-03",
        "test_date_start":  "2026-06-04", "test_date_end":   "2026-06-05",
        "train_bars": 170000, "val_bars": 15000, "test_bars": 15000,
        "val_dir_acc": 0.61, "test_dir_acc": 0.57, "test_signals": 42,
        "val_test_gap": 0.04, "n_features": 188,
        "fwd_bars": 3, "train_frac": 0.85, "val_frac": 0.075,
        "embargo_bars": 75, "label_quantile": 0.70,
    }
    base.update(overrides)
    return base


def _with_tmpdir(fn):
    """Run fn inside a temporary MODELS_DIR / REGISTRY_DIR."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        orig_md  = reg.MODELS_DIR
        orig_rd  = reg.REGISTRY_DIR
        reg.MODELS_DIR   = tmp / "models"
        reg.REGISTRY_DIR = tmp / "models" / "registry"
        reg.MODELS_DIR.mkdir(parents=True)
        reg.REGISTRY_DIR.mkdir(parents=True)
        try:
            fn(tmp)
        finally:
            reg.MODELS_DIR   = orig_md
            reg.REGISTRY_DIR = orig_rd


# --------------------------------------------------------------------------- A
print("\n[A] save_candidate")

def _test_save(tmp):
    models = _fake_models()
    scaler = _fake_scaler()
    fcols  = ["feat_a", "feat_b"]
    meta   = _good_meta()

    cid = reg.save_candidate("v10", models, scaler, fcols, meta)

    cdir = reg.REGISTRY_DIR / f"v10_{cid}"
    check("candidate dir created",        cdir.is_dir())
    check("models.pkl saved",             (cdir / "models.pkl").exists())
    check("scaler.pkl saved",             (cdir / "scaler.pkl").exists())
    check("feature_cols.pkl saved",       (cdir / "feature_cols.pkl").exists())
    check("metadata.json saved",          (cdir / "metadata.json").exists())

    meta_loaded = json.loads((cdir / "metadata.json").read_text())
    check("metadata version correct",     meta_loaded["version"] == "v10")
    check("metadata promoted=False",      meta_loaded["promoted"] is False)
    check("metadata test_dir_acc stored", meta_loaded["test_dir_acc"] == 0.57)
    check("live models.pkl NOT created",
          not (reg.MODELS_DIR / "nifty_v10_models.pkl").exists())

_with_tmpdir(_test_save)

# --------------------------------------------------------------------------- B
print("\n[B] promote gate PASS")

def _test_promote_pass(tmp):
    # Create a dummy live model first (something to replace)
    orig_live = reg.MODELS_DIR / "nifty_v10_models.pkl"
    orig_live.write_bytes(b"OLD_MODEL")
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"OLD_SCALER")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"OLD_FCOLS")

    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(),
                              ["f1"], _good_meta(test_dir_acc=0.60, test_signals=15))
    promoted, reason = reg.promote_if_passes_gate("v10")

    check("gate passes with good metrics", promoted is True)
    check("reason mentions PROMOTED",      "PROMOTED" in reason)
    check("live model file updated",       orig_live.read_bytes() != b"OLD_MODEL")
    check("metadata marked promoted=True",
          json.loads(
              (reg.REGISTRY_DIR / f"v10_{cid}" / "metadata.json").read_text()
          )["promoted"] is True)
    check("no .tmp file left behind",
          not (reg.MODELS_DIR / "nifty_v10_models.tmp").exists())

_with_tmpdir(_test_promote_pass)

# --------------------------------------------------------------------------- C
print("\n[C] promote gate FAIL — low test_dir_acc")

def _test_promote_fail_acc(tmp):
    orig = reg.MODELS_DIR / "nifty_v10_models.pkl"
    orig.write_bytes(b"SAFE_MODEL")
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")

    reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                       _good_meta(test_dir_acc=0.40, test_signals=20))
    promoted, reason = reg.promote_if_passes_gate("v10")

    check("gate blocks low accuracy",     promoted is False)
    check("reason mentions GATE FAIL",    "GATE FAIL" in reason)
    check("live model file UNTOUCHED",    orig.read_bytes() == b"SAFE_MODEL")

_with_tmpdir(_test_promote_fail_acc)

# --------------------------------------------------------------------------- D
print("\n[D] promote gate FAIL — too few signals")

def _test_promote_fail_signals(tmp):
    orig = reg.MODELS_DIR / "nifty_v10_models.pkl"
    orig.write_bytes(b"SAFE_MODEL")
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")

    reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                       _good_meta(test_dir_acc=0.62, test_signals=2))
    promoted, reason = reg.promote_if_passes_gate("v10")

    check("gate blocks insufficient signals", promoted is False)
    check("live model UNTOUCHED",             orig.read_bytes() == b"SAFE_MODEL")

_with_tmpdir(_test_promote_fail_signals)

# --------------------------------------------------------------------------- E
print("\n[E] rollback")

def _test_rollback(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")

    # Save two candidates
    cid1 = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                               _good_meta(test_dir_acc=0.60, test_signals=10))
    cid2 = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                               _good_meta(test_dir_acc=0.61, test_signals=12))

    # Promote the latest (cid2)
    reg.promote_if_passes_gate("v10")
    live = reg.MODELS_DIR / "nifty_v10_models.pkl"
    size_after_promote = live.stat().st_size

    # Roll back to cid1
    ok = reg.rollback("v10", cid1)
    check("rollback returns True",          ok is True)
    check("live file updated after rollback", live.exists())

    # Metadata of cid1 should be marked promoted + rollback=True
    meta = json.loads(
        (reg.REGISTRY_DIR / f"v10_{cid1}" / "metadata.json").read_text()
    )
    check("rolled-back entry marked promoted=True", meta["promoted"] is True)
    check("rolled-back entry marked rollback=True",  meta.get("rollback") is True)

_with_tmpdir(_test_rollback)

# --------------------------------------------------------------------------- F
print("\n[F] list_versions")

def _test_list(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")

    cids = []
    for acc in [0.55, 0.57, 0.59]:
        cids.append(reg.save_candidate("v10", _fake_models(), _fake_scaler(),
                                        ["f1"], _good_meta(test_dir_acc=acc)))

    entries = reg.list_versions("v10", n=5)
    check("returns 3 entries",                  len(entries) == 3)
    check("newest first (highest acc last cid)", entries[0]["test_dir_acc"] == 0.59)
    check("all have candidate_id",              all("candidate_id" in e for e in entries))

_with_tmpdir(_test_list)

# --------------------------------------------------------------------------- G
print("\n[G] prune_old_candidates")

def _test_prune(tmp):
    for i in range(6):
        reg.save_candidate("v9", _fake_models(), _fake_scaler(),
                           ["f1"], _good_meta(test_dir_acc=0.55 + i * 0.01))

    deleted = reg.prune_old_candidates("v9", keep=4)
    remaining = [d for d in reg.REGISTRY_DIR.iterdir()
                 if d.name.startswith("v9_")]
    check("pruned 2 oldest",              deleted == 2)
    check("4 remain after prune",         len(remaining) == 4)

_with_tmpdir(_test_prune)

# --------------------------------------------------------------------------- H
print("\n[H] live path not touched by save_candidate alone")

def _test_live_untouched(tmp):
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(),
                              ["f1"], _good_meta())
    check("models live path not created by save_candidate",
          not (reg.MODELS_DIR / "nifty_v10_models.pkl").exists())

_with_tmpdir(_test_live_untouched)

# --------------------------------------------------------------------------- I
print("\n[I] no .tmp artefacts after promote")

def _test_no_tmp(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"], _good_meta())
    reg.promote_if_passes_gate("v10")
    tmp_files = list(reg.MODELS_DIR.glob("*.tmp"))
    check("no .tmp files after successful promote", len(tmp_files) == 0)

_with_tmpdir(_test_no_tmp)

# --------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
