"""Manifest schema tests (structure only — no real evaluation)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ensembles import ManifestMismatchError, sha256_of_file, verify_manifest_hash


# ---------- schema helper ----------
REQUIRED_MANIFEST_FIELDS = {
    "protocol_version",
    "code_hash",
    "data_hash",
    "random_seed",
    "fold_id",
    "test_start",
    "test_end",
    "timestamp_utc",
    "ensemble_type",
    "participating_brains",
    "input_source",
    "weights_summary",
}


def _example_manifest(tmp_path: Path) -> dict:
    src = tmp_path / "src.py"
    src.write_text("# example")
    return {
        "protocol_version": "2.3",
        "code_hash": {"src.py": sha256_of_file(src)},
        "data_hash": {"phase3_predictions_alpha": "abc123"},
        "random_seed": 42,
        "fold_id": 1,
        "test_start": "2024-07-01",
        "test_end": "2024-10-01",
        "timestamp_utc": "2026-07-10T12:34:56+00:00",
        "ensemble_type": "mean",
        "participating_brains": ["alpha", "beta"],
        "input_source": "raw",
        "weights_summary": {"kind": "mean"},
        "n_test_bars": 4800,
        "n_oof_bars": 78503,
        "k_inner": 3,
    }


def test_manifest_has_all_required_fields(tmp_path):
    m = _example_manifest(tmp_path)
    missing = REQUIRED_MANIFEST_FIELDS - set(m)
    assert not missing, f"missing fields: {missing}"


def test_manifest_participating_brains_is_list(tmp_path):
    m = _example_manifest(tmp_path)
    assert isinstance(m["participating_brains"], list)
    assert all(isinstance(b, str) for b in m["participating_brains"])


def test_manifest_json_roundtrip(tmp_path):
    m = _example_manifest(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    reloaded = json.loads(p.read_text())
    assert reloaded == m


def test_verify_manifest_hash_ok(tmp_path):
    m = _example_manifest(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    verify_manifest_hash(p, tmp_path)   # no exception


def test_verify_manifest_hash_drift(tmp_path):
    m = _example_manifest(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    # Mutate source → SHA changes
    (tmp_path / "src.py").write_text("# CHANGED")
    with pytest.raises(ManifestMismatchError):
        verify_manifest_hash(p, tmp_path)


def test_verify_manifest_hash_missing_file(tmp_path):
    m = _example_manifest(tmp_path)
    (tmp_path / "src.py").unlink()
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    with pytest.raises(ManifestMismatchError, match="missing"):
        verify_manifest_hash(p, tmp_path)


def test_manifest_weights_summary_is_dict(tmp_path):
    m = _example_manifest(tmp_path)
    assert isinstance(m["weights_summary"], dict)
    assert "kind" in m["weights_summary"]


def test_manifest_input_source_is_documented_value(tmp_path):
    m = _example_manifest(tmp_path)
    assert m["input_source"] in {
        "raw", "calibrated_noop", "calibrated_platt", "calibrated_isotonic"}


def test_manifest_random_seed_is_int(tmp_path):
    m = _example_manifest(tmp_path)
    assert isinstance(m["random_seed"], int)


def test_manifest_fold_id_is_int(tmp_path):
    m = _example_manifest(tmp_path)
    assert isinstance(m["fold_id"], int)
