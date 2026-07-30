"""Selection tests — brain discovery from mock file trees + source resolver."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensembles import EnsembleInputError, discover_brains, load_brain_probs


def _write_predictions(dir_path, n=5, prefix=""):
    dir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-07-01", periods=n, freq="5min"),
        "y_true": [0, 1, 2, 0, 1][:n],
        "p_call": [0.5, 0.2, 0.3, 0.6, 0.1][:n],
        "p_put":  [0.3, 0.6, 0.3, 0.3, 0.7][:n],
        "p_skip": [0.2, 0.2, 0.4, 0.1, 0.2][:n],
        "signal": [0, 1, 2, 0, 1][:n],
    })
    df.to_csv(dir_path / f"{prefix}predictions.csv", index=False)


# ---------- discover_brains ----------
def _patch_registry(monkeypatch, brains: list[str]) -> None:
    """Override the ``brains`` registry used by ``discover_brains``."""
    from ensembles import _selection as sel
    monkeypatch.setattr(sel, "_registered_brains", lambda: sorted(brains))


def test_discover_all_from_phase3_tree(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, ["alpha", "beta", "gamma"])
    for b in ("alpha", "beta", "gamma"):
        _write_predictions(tmp_path / f"logs/phase3/{b}/fold_1")
    result = discover_brains(tmp_path, 1, "all")
    assert result == ["alpha", "beta", "gamma"]


def test_discover_respects_subset_request(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, ["alpha", "beta", "gamma"])
    for b in ("alpha", "beta", "gamma"):
        _write_predictions(tmp_path / f"logs/phase3/{b}/fold_1")
    result = discover_brains(tmp_path, 1, "alpha,gamma")
    assert result == ["alpha", "gamma"]


def test_discover_rejects_unknown_brain(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, ["alpha", "beta"])
    for b in ("alpha", "beta"):
        _write_predictions(tmp_path / f"logs/phase3/{b}/fold_1")
    with pytest.raises(EnsembleInputError, match="unavailable"):
        discover_brains(tmp_path, 1, "alpha,does_not_exist")


def test_discover_requires_at_least_two(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, ["alpha"])
    _write_predictions(tmp_path / f"logs/phase3/alpha/fold_1")
    with pytest.raises(EnsembleInputError, match=">= 2"):
        discover_brains(tmp_path, 1, "all")


def test_discover_skips_folds_without_predictions_file(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, ["alpha", "beta", "gamma"])
    # Create dir but no predictions.csv
    (tmp_path / "logs/phase3/alpha/fold_1").mkdir(parents=True)
    _write_predictions(tmp_path / "logs/phase3/beta/fold_1")
    _write_predictions(tmp_path / "logs/phase3/gamma/fold_1")
    result = discover_brains(tmp_path, 1, "all")
    assert result == ["beta", "gamma"]


def test_discover_handles_multiple_folds(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, ["alpha", "beta"])
    _write_predictions(tmp_path / "logs/phase3/alpha/fold_1")
    _write_predictions(tmp_path / "logs/phase3/beta/fold_1")
    _write_predictions(tmp_path / "logs/phase3/alpha/fold_2")
    # fold_2 has only alpha → too few brains
    with pytest.raises(EnsembleInputError, match=">= 2"):
        discover_brains(tmp_path, 2, "all")


# ---------- load_brain_probs — raw source ----------
def test_load_raw_reads_phase3(tmp_path):
    _write_predictions(tmp_path / "logs/phase3/alpha/fold_1", n=3)
    probs, y, ts = load_brain_probs(tmp_path, "alpha", 1, source="raw")
    assert probs.shape == (3, 3)
    assert y.shape == (3,)
    assert len(ts) == 3


def test_load_raw_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no predictions"):
        load_brain_probs(tmp_path, "alpha", 1, source="raw")


# ---------- load_brain_probs — calibrated sources ----------
@pytest.mark.parametrize("source,prefix", [
    ("calibrated_noop",     "noop"),
    ("calibrated_platt",    "platt"),
    ("calibrated_isotonic", "isotonic"),
])
def test_load_calibrated_reads_phase4(tmp_path, source, prefix):
    _write_predictions(tmp_path / "logs/phase4/alpha/fold_1",
                        n=3, prefix=f"{prefix}_")
    probs, y, ts = load_brain_probs(tmp_path, "alpha", 1, source=source)
    assert probs.shape == (3, 3)


def test_load_calibrated_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_brain_probs(tmp_path, "alpha", 1, source="calibrated_isotonic")


def test_load_reads_columns_correctly(tmp_path):
    _write_predictions(tmp_path / "logs/phase3/alpha/fold_1", n=5)
    probs, y, ts = load_brain_probs(tmp_path, "alpha", 1, source="raw")
    # Rows sum to 1 (validated)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_load_unknown_source_raises(tmp_path):
    _write_predictions(tmp_path / "logs/phase3/alpha/fold_1")
    with pytest.raises(EnsembleInputError, match="unknown"):
        load_brain_probs(tmp_path, "alpha", 1, source="calibrated_bogus")


# ---------- source-switching contract ----------
def test_source_switching_reads_different_files(tmp_path):
    """Same code path handles raw vs calibrated — only the source arg changes."""
    _write_predictions(tmp_path / "logs/phase3/alpha/fold_1",
                        n=3, prefix="")
    # Different probabilities under phase4 to detect if the loader
    # mistakenly reads phase3 for calibrated_isotonic
    dir4 = tmp_path / "logs/phase4/alpha/fold_1"
    dir4.mkdir(parents=True)
    df4 = pd.DataFrame({
        "timestamp": pd.date_range("2024-07-01", periods=3, freq="5min"),
        "y_true": [1, 1, 1],
        "p_call": [0.05, 0.05, 0.05],
        "p_put":  [0.90, 0.90, 0.90],
        "p_skip": [0.05, 0.05, 0.05],
        "signal": [1, 1, 1],
    })
    df4.to_csv(dir4 / "isotonic_predictions.csv", index=False)

    raw, _, _ = load_brain_probs(tmp_path, "alpha", 1, source="raw")
    iso, _, _ = load_brain_probs(tmp_path, "alpha", 1,
                                   source="calibrated_isotonic")
    # Different files → different values on the "put" column
    assert raw[0, 1] != iso[0, 1]
    assert iso[0, 1] == pytest.approx(0.90)
