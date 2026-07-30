"""Cache-manifest schema + hash validation."""
from __future__ import annotations

import json
from pathlib import Path

from phase7_robustness_v9 import (
    build_manifest, load_manifest, save_manifest, verify_cache,
    CacheMismatchError,
)


def test_build_manifest_contains_required_fields(tmp_path):
    root = Path(".")
    m = build_manifest(
        root=root, experiment_type="unit_test", target="mean",
        candidate={"call_thr": 0.32}, seed=42)
    for k in ("phase7_version", "manifest_schema_version",
              "protocol_version", "experiment_type", "target",
              "candidate", "random_seed", "code_hash",
              "input_hash", "timestamp_utc"):
        assert k in m, k


def test_save_and_load_manifest_roundtrip(tmp_path):
    m = build_manifest(
        root=Path("."), experiment_type="unit_test", target="mean",
        candidate={"call_thr": 0.32}, seed=42)
    save_manifest(m, tmp_path)
    loaded = load_manifest(tmp_path)
    assert loaded["experiment_type"] == "unit_test"


def test_load_manifest_missing_raises(tmp_path):
    import pytest
    with pytest.raises(CacheMismatchError):
        load_manifest(tmp_path / "does_not_exist")


def test_verify_cache_false_when_manifest_missing(tmp_path):
    assert verify_cache(tmp_path, Path("."), "mean") is False


def test_verify_cache_true_after_save(tmp_path):
    m = build_manifest(
        root=Path("."), experiment_type="unit_test", target="mean",
        candidate={"call_thr": 0.32}, seed=42)
    save_manifest(m, tmp_path)
    assert verify_cache(tmp_path, Path("."), "mean") is True


def test_verify_cache_false_after_tampering_manifest(tmp_path):
    m = build_manifest(
        root=Path("."), experiment_type="unit_test", target="mean",
        candidate={"call_thr": 0.32}, seed=42)
    save_manifest(m, tmp_path)
    p = tmp_path / "manifest.json"
    data = json.loads(p.read_text())
    data["input_hash"]["phase6/summary.json"] = "0" * 64
    p.write_text(json.dumps(data))
    assert verify_cache(tmp_path, Path("."), "mean") is False
