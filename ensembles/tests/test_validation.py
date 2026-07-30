"""Validation tests — every fail-fast path is exercised."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ensembles import (
    EnsembleInputError,
    ManifestMismatchError,
    check_no_duplicate_predictions,
    check_no_nans,
    check_row_normalisation,
    check_shape_consistency,
    sha256_of_file,
    validate_brain_probs_mapping,
    validate_fold_completeness,
    validate_labels,
    validate_probability_array,
    verify_manifest_hash,
)


# ---------- probability array ----------
def test_valid_probs_pass():
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
    out = validate_probability_array(p, "p")
    assert out.dtype == np.float64


def test_reject_1d_probs():
    with pytest.raises(EnsembleInputError, match="2-D"):
        validate_probability_array(np.array([0.5, 0.3, 0.2]), "p")


def test_reject_wrong_column_count():
    with pytest.raises(EnsembleInputError, match="columns"):
        validate_probability_array(np.array([[0.5, 0.5]]), "p")


def test_reject_nan():
    p = np.array([[np.nan, 0.5, 0.5], [0.3, 0.3, 0.4]])
    with pytest.raises(EnsembleInputError, match="non-finite"):
        validate_probability_array(p, "p")


def test_reject_out_of_range():
    p = np.array([[1.2, -0.2, 0.0]])
    with pytest.raises(EnsembleInputError, match="out of"):
        validate_probability_array(p, "p")


def test_reject_unnormalised_rows():
    p = np.array([[0.5, 0.3, 0.4]])   # sums to 1.2
    with pytest.raises(EnsembleInputError, match="sum to 1"):
        validate_probability_array(p, "p")


def test_reject_empty():
    with pytest.raises(EnsembleInputError, match="empty"):
        validate_probability_array(np.empty((0, 3)), "p")


# ---------- brain-probs mapping ----------
def test_valid_mapping_passes():
    p = {"b0": np.array([[0.5, 0.3, 0.2]]),
         "b1": np.array([[0.4, 0.3, 0.3]])}
    validate_brain_probs_mapping(p, "brain_probs")


def test_reject_less_than_two_brains():
    with pytest.raises(EnsembleInputError, match=">= 2"):
        validate_brain_probs_mapping({"b0": np.array([[0.5, 0.3, 0.2]])},
                                       "brain_probs")


def test_reject_inconsistent_lengths():
    p = {"b0": np.array([[0.5, 0.3, 0.2]]),
         "b1": np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])}
    with pytest.raises(EnsembleInputError, match="inconsistent"):
        validate_brain_probs_mapping(p, "brain_probs")


def test_reject_non_mapping_input():
    with pytest.raises(EnsembleInputError, match="Mapping"):
        validate_brain_probs_mapping([1, 2, 3], "brain_probs")


def test_reject_empty_brain_name():
    p = {"": np.array([[0.5, 0.3, 0.2]]),
         "b1": np.array([[0.4, 0.3, 0.3]])}
    with pytest.raises(EnsembleInputError, match="brain names"):
        validate_brain_probs_mapping(p, "brain_probs")


# ---------- labels ----------
def test_valid_labels_pass():
    y = np.array([0, 1, 2, 1])
    out = validate_labels(y, "y")
    assert out.tolist() == [0, 1, 2, 1]


def test_reject_out_of_range_labels():
    with pytest.raises(EnsembleInputError, match=r"\[0, 3\)"):
        validate_labels(np.array([0, 1, 3]), "y")


# ---------- duplicates + normalisation + no-nans ----------
def test_duplicate_predictions_detected():
    df = pd.DataFrame({"timestamp": ["a", "b", "a"]})
    with pytest.raises(EnsembleInputError, match="duplicate"):
        check_no_duplicate_predictions(df)


def test_no_duplicates_passes():
    df = pd.DataFrame({"timestamp": ["a", "b", "c"]})
    check_no_duplicate_predictions(df)   # no exception


def test_row_normalisation_check():
    check_row_normalisation(np.array([[0.5, 0.3, 0.2]]))
    with pytest.raises(EnsembleInputError, match="normalised"):
        check_row_normalisation(np.array([[0.5, 0.3, 0.4]]))


def test_no_nans_check():
    check_no_nans(np.array([[0.1, 0.5]]), "p")
    with pytest.raises(EnsembleInputError, match="NaN"):
        check_no_nans(np.array([[np.nan, 0.5]]), "p")


# ---------- fold completeness ----------
def test_fold_completeness_missing(tmp_path):
    (tmp_path / "logs/phase3/b0/fold_1").mkdir(parents=True)
    (tmp_path / "logs/phase3/b0/fold_1/predictions.csv").write_text("x")
    with pytest.raises(EnsembleInputError, match="missing outputs"):
        validate_fold_completeness(["b0", "b1"], tmp_path, 1)


def test_fold_completeness_pass(tmp_path):
    for b in ("b0", "b1"):
        d = tmp_path / f"logs/phase3/{b}/fold_1"
        d.mkdir(parents=True)
        (d / "predictions.csv").write_text("x")
    validate_fold_completeness(["b0", "b1"], tmp_path, 1)


# ---------- manifest hash ----------
def test_verify_manifest_hash_roundtrip(tmp_path):
    src = tmp_path / "src.py"
    src.write_text("print('hello')")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "code_hash": {"src.py": sha256_of_file(src)}}))
    # Success case
    verify_manifest_hash(manifest, tmp_path)
    # Now mutate source and expect drift
    src.write_text("print('bye')")
    with pytest.raises(ManifestMismatchError, match="drifted"):
        verify_manifest_hash(manifest, tmp_path)


def test_verify_manifest_hash_missing_file(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "code_hash": {"nonexistent.py": "a" * 64}}))
    with pytest.raises(ManifestMismatchError, match="missing"):
        verify_manifest_hash(manifest, tmp_path)


# ---------- shape consistency ----------
def test_shape_consistency_pass():
    p = {"a": np.zeros((5, 3)), "b": np.zeros((5, 3))}
    check_shape_consistency(p)


def test_shape_consistency_fail():
    p = {"a": np.zeros((5, 3)), "b": np.zeros((6, 3))}
    with pytest.raises(EnsembleInputError, match="shapes differ"):
        check_shape_consistency(p)
