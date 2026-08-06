#!/usr/bin/env python3
"""
Unit tests for the live-rule evaluation helpers in
scripts/train_model_v9.py (live_rule_signals / live_rule_metrics), plus
the metric_version guard in core/model_registry.py.

Background: training used to score candidates with argmax(proba) while
production (core/ml_engine.py) uses per-class thresholds. SKIP wins the
argmax on nearly every bar, so argmax scoring reported "0 signals / 0.0%
direction accuracy" for models that fire normally in production — which
made the promotion gate reject usable candidates. These tests pin the
corrected behavior.

Standalone script (repo convention):
    python scripts/test_live_rule_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model_v9 import (
    live_rule_signals, live_rule_metrics, METRIC_VERSION,
)
from core.ml_engine import (
    CONFIDENCE_CALL, CONFIDENCE_PUT, MIN_EDGE, SKIP_CEIL,
)

CALL, PUT, SKIP = 0, 1, 2


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return bool(cond)


def main():
    ok = True

    # ── thresholds come from ml_engine, not a local copy ──────────────
    ok &= check(
        "thresholds are imported from core.ml_engine (no drift possible)",
        (CONFIDENCE_CALL, CONFIDENCE_PUT, MIN_EDGE, SKIP_CEIL)
        == (0.32, 0.25, 0.05, 0.65),
    )
    ok &= check("METRIC_VERSION is 2 (live-rule scoring)", METRIC_VERSION == 2)

    # ── live_rule_signals: each gate in isolation ─────────────────────
    p = np.array([
        [0.40, 0.20, 0.40],   # CALL: over conf, edge .20, skip ok
        [0.20, 0.30, 0.50],   # PUT : over conf, edge .10, skip ok
        [0.30, 0.28, 0.42],   # SKIP: edge .02 < MIN_EDGE
        [0.35, 0.20, 0.70],   # SKIP: skip_p .70 >= SKIP_CEIL
        [0.31, 0.10, 0.59],   # SKIP: call_p .31 < CONFIDENCE_CALL .32
        [0.10, 0.24, 0.66],   # SKIP: put_p under conf AND skip over ceil
    ])
    s = live_rule_signals(p)
    ok &= check(f"CALL fires when all 3 conditions met (got {s[0]})", s[0] == CALL)
    ok &= check(f"PUT fires when all 3 conditions met (got {s[1]})", s[1] == PUT)
    ok &= check(f"edge below MIN_EDGE blocks the signal (got {s[2]})", s[2] == SKIP)
    ok &= check(f"skip_p at/above SKIP_CEIL blocks the signal (got {s[3]})", s[3] == SKIP)
    ok &= check(f"call_p just below CONFIDENCE_CALL blocks (got {s[4]})", s[4] == SKIP)
    ok &= check(f"put_p below CONFIDENCE_PUT blocks (got {s[5]})", s[5] == SKIP)

    # ── the regression this whole change exists for ───────────────────
    # A model whose argmax is SKIP on every bar still fires under the
    # live rule. Old argmax scoring reported 0 signals here.
    p_skip_argmax = np.array([
        [0.36, 0.20, 0.44],
        [0.34, 0.22, 0.44],
        [0.18, 0.33, 0.49],
    ])
    ok &= check(
        "argmax says SKIP on every bar (reproduces the old bug's input)",
        all(row.argmax() == SKIP for row in p_skip_argmax),
    )
    s2 = live_rule_signals(p_skip_argmax)
    ok &= check(
        f"...but live rule still fires 3 signals (got {(s2 != SKIP).sum()})",
        int((s2 != SKIP).sum()) == 3,
    )

    # ── live_rule_metrics accounting ──────────────────────────────────
    # bars: 0 CALL-correct, 1 PUT-correct, 2 no-fire, 3 fires on SKIP bar
    pm = np.array([
        [0.40, 0.20, 0.40],   # fires CALL
        [0.20, 0.30, 0.50],   # fires PUT
        [0.30, 0.28, 0.42],   # no fire
        [0.40, 0.20, 0.40],   # fires CALL, but label is SKIP
    ])
    ym = np.array([CALL, PUT, CALL, SKIP])
    m = live_rule_metrics(pm, ym)
    ok &= check(f"n_signals counts only fired bars (got {m['n_signals']})", m["n_signals"] == 3)
    ok &= check(f"n_call/n_put split correct (got {m['n_call']}/{m['n_put']})",
                m["n_call"] == 2 and m["n_put"] == 1)
    ok &= check(f"fired_on_skip counts no-move bars (got {m['fired_on_skip']})",
                m["fired_on_skip"] == 1)
    ok &= check(f"fired_on_dir excludes SKIP-label bars (got {m['fired_on_dir']})",
                m["fired_on_dir"] == 2)
    ok &= check(f"dir_acc scored only over fired_on_dir (got {m['dir_acc']})",
                m["dir_acc"] == 1.0)
    # 3 directional-label bars exist (idx 0,1,2); the model caught 2
    ok &= check(f"recall = caught / real directional bars (got {m['recall']:.3f})",
                abs(m["recall"] - 2 / 3) < 1e-9)

    # a model that fires nothing must not report a misleading 100%
    m_none = live_rule_metrics(np.array([[0.30, 0.28, 0.42]]), np.array([CALL]))
    ok &= check(f"no signals -> dir_acc 0.0 not NaN/1.0 (got {m_none['dir_acc']})",
                m_none["dir_acc"] == 0.0)
    ok &= check(f"no signals -> recall 0.0 (got {m_none['recall']})",
                m_none["recall"] == 0.0)

    # ── registry guard: never compare across metric versions ──────────
    import inspect
    from core import model_registry
    src = inspect.getsource(model_registry.promote_if_passes_gate)
    ok &= check("gate reads metric_version from both champion and challenger",
                'champ_meta.get("metric_version"' in src
                and 'meta.get("metric_version"' in src)
    ok &= check("gate has a comparability guard before the C/C decision",
                "metrics_comparable" in src)

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
