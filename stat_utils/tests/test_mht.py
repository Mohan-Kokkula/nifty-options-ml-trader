"""Tests for stat_utils.mht (Holm-Bonferroni, Benjamini-Hochberg)."""
from __future__ import annotations

import numpy as np
import pytest

from stat_utils import (
    InvalidInputError,
    MHTResult,
    benjamini_hochberg,
    holm_bonferroni,
)


def test_holm_hand_computed():
    """Family: three p-values. alpha=0.10.
        p1=0.005, p2=0.02, p3=0.10
        adjusted alphas: 0.10/3=0.0333, 0.10/2=0.05, 0.10/1=0.10
        step-down: 0.005 <= 0.0333 → reject
                    0.02  <= 0.05   → reject
                    0.10  <= 0.10   → reject
    """
    r = holm_bonferroni({"a": 0.005, "b": 0.02, "c": 0.10}, alpha=0.10)
    assert isinstance(r, MHTResult)
    assert r.decisions["a"]["reject"]
    assert r.decisions["b"]["reject"]
    assert r.decisions["c"]["reject"]


def test_holm_stops_at_first_failure():
    """p1=0.001 reject; p2=0.06 vs 0.05 fail; p3=0.10 must also fail."""
    r = holm_bonferroni({"a": 0.001, "b": 0.06, "c": 0.10}, alpha=0.10)
    assert r.decisions["a"]["reject"]
    assert not r.decisions["b"]["reject"]
    assert not r.decisions["c"]["reject"]


def test_holm_ranks_are_stable():
    r = holm_bonferroni({"a": 0.5, "b": 0.01, "c": 0.05, "d": 0.9},
                        alpha=0.10)
    ranks = {k: v["rank"] for k, v in r.decisions.items()}
    assert ranks["b"] == 1
    assert ranks["c"] == 2
    assert ranks["a"] == 3
    assert ranks["d"] == 4


def test_holm_matches_statsmodels_if_available():
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        pytest.skip("statsmodels not installed")
    pvs = {"a": 0.005, "b": 0.02, "c": 0.10, "d": 0.5}
    r = holm_bonferroni(pvs, alpha=0.10)
    ordered_keys = sorted(pvs, key=lambda k: pvs[k])
    ordered_pvs = [pvs[k] for k in ordered_keys]
    rej, adj, _, _ = multipletests(ordered_pvs, alpha=0.10, method="holm")
    for i, key in enumerate(ordered_keys):
        assert r.decisions[key]["reject"] == bool(rej[i])
        assert r.decisions[key]["adjusted_pvalue"] == pytest.approx(adj[i],
                                                                     rel=1e-9)


def test_bh_matches_statsmodels_if_available():
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        pytest.skip("statsmodels not installed")
    pvs = {"a": 0.005, "b": 0.02, "c": 0.04, "d": 0.06, "e": 0.5}
    r = benjamini_hochberg(pvs, fdr=0.10)
    ordered_keys = sorted(pvs, key=lambda k: pvs[k])
    ordered_pvs = [pvs[k] for k in ordered_keys]
    rej, adj, _, _ = multipletests(ordered_pvs, alpha=0.10, method="fdr_bh")
    for i, key in enumerate(ordered_keys):
        assert r.decisions[key]["reject"] == bool(rej[i])
        assert r.decisions[key]["adjusted_pvalue"] == pytest.approx(adj[i],
                                                                     rel=1e-9)


def test_bh_rejects_more_than_holm_typically():
    pvs = {chr(97 + i): (i + 1) * 0.02 for i in range(5)}
    holm = holm_bonferroni(pvs, alpha=0.10)
    bh = benjamini_hochberg(pvs, fdr=0.10)
    n_holm = sum(1 for v in holm.decisions.values() if v["reject"])
    n_bh = sum(1 for v in bh.decisions.values() if v["reject"])
    assert n_bh >= n_holm


def test_rejects_empty_family():
    with pytest.raises(InvalidInputError):
        holm_bonferroni({}, alpha=0.10)


def test_rejects_out_of_range_pvalue():
    with pytest.raises(InvalidInputError):
        holm_bonferroni({"a": 1.5}, alpha=0.10)


def test_rejects_bad_alpha():
    with pytest.raises(InvalidInputError):
        holm_bonferroni({"a": 0.1}, alpha=1.5)


def test_result_json_serialisable():
    r = holm_bonferroni({"a": 0.01, "b": 0.5}, alpha=0.10)
    import json
    d = json.loads(r.to_json())
    assert d["method"] == "holm_bonferroni"


def test_rejected_helper():
    r = holm_bonferroni({"a": 0.001, "b": 0.5}, alpha=0.10)
    assert r.rejected() == ["a"]
