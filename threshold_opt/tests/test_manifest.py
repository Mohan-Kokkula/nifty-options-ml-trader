"""Manifest schema + cache validation tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from threshold_opt import (CacheMismatchError, ThresholdCandidate,
                             build_manifest, load_manifest, sha256_of_file,
                             verify_cache)


REQUIRED_FIELDS = {
    "protocol_version", "code_hash", "data_hash", "random_seed",
    "timestamp_utc", "target_ensemble", "input_source",
    "threshold_values", "folds_evaluated", "n_trades_per_fold",
    "min_trades_requirement", "passes_min_trades_filter",
}


def _build(tmp_path):
    src = tmp_path / "src.py"
    src.write_text("# example")
    cand = ThresholdCandidate(0.32, 0.25, 0.65, 0.05)
    return build_manifest(
        cand=cand,
        target_ensemble="mean",
        input_source="calibrated_isotonic",
        folds_evaluated=[1, 2, 3, 4, 5, 6, 7, 8],
        n_trades_per_fold=[10, 15, 20, 25, 30, 5, 12, 8],
        min_trades_requirement=50,
        passes_min_trades_filter=True,
        code_hash={"src.py": sha256_of_file(src)},
        data_hash={"phase5_predictions_mean_fold_1": "abc123"},
        protocol_version="2.5",
        seed=42,
    )


def test_manifest_has_all_required_fields(tmp_path):
    m = _build(tmp_path)
    missing = REQUIRED_FIELDS - set(m)
    assert not missing, f"missing: {missing}"


def test_manifest_serialisable_json(tmp_path):
    m = _build(tmp_path)
    s = json.dumps(m)
    m2 = json.loads(s)
    assert m2["threshold_values"]["call_thr"] == 0.32


def test_verify_cache_clean(tmp_path):
    m = _build(tmp_path)
    assert verify_cache(m, m["code_hash"], m["data_hash"])


def test_verify_cache_detects_code_drift(tmp_path):
    m = _build(tmp_path)
    changed = dict(m["code_hash"])
    changed["src.py"] = "0" * 64
    assert not verify_cache(m, changed, m["data_hash"])


def test_verify_cache_detects_data_drift(tmp_path):
    m = _build(tmp_path)
    changed = dict(m["data_hash"])
    changed["phase5_predictions_mean_fold_1"] = "deadbeef"
    assert not verify_cache(m, m["code_hash"], changed)


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(CacheMismatchError, match="not found"):
        load_manifest(tmp_path / "does_not_exist.json")


def test_load_manifest_unreadable_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not valid json {{")
    with pytest.raises(CacheMismatchError, match="unreadable"):
        load_manifest(p)


def test_sha256_of_file_reproducible(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world")
    h1 = sha256_of_file(p)
    h2 = sha256_of_file(p)
    assert h1 == h2 and len(h1) == 64


def test_manifest_records_min_trades_requirement(tmp_path):
    m = _build(tmp_path)
    assert m["min_trades_requirement"] == 50
    assert isinstance(m["passes_min_trades_filter"], bool)


def test_manifest_records_threshold_values(tmp_path):
    m = _build(tmp_path)
    tv = m["threshold_values"]
    assert tv["call_thr"] == 0.32 and tv["put_thr"] == 0.25
    assert tv["skip_ceil"] == 0.65 and tv["min_edge"] == 0.05
