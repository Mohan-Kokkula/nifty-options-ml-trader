"""
test_shadow_framework.py — smoke test for the production shadow framework:
  core/shadow_framework.py     (per-cycle observe(), health, cleanup)
  scripts/shadow_daily_report.py (daily aggregation)
  scripts/shadow_health_check.py (CLI health check)

Run:
    python scripts/test_shadow_framework.py

Tests:
  A. observe() loads the HTF36 candidate and writes a comparison record
  B. record schema: primary/shadow/compare blocks present & well-typed
  C. health.json updated after observe() (model_loaded, n_features, counters)
  D. generate_report() aggregates agreement/trade-count/confidence correctly
  E. cleanup_old_files() removes only files older than retention, keeps recent
  F. shadow_health_check.main() returns 0 (OK) when healthy
  G. observe() never raises even with a feature frame missing all columns
  H. SHADOW_FRAMEWORK_ENABLED=false → observe() is a clean no-op
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


from core import shadow_framework as sf  # noqa: E402


def _reset_state():
    sf._state.update({"loaded": False, "failed": False,
                       "models": None, "scaler": None, "fcols": None, "dir": None})


def _redirect(tmp: Path):
    """Point shadow_framework's output paths at a temp dir for test isolation."""
    sf.SHADOW_DIR = tmp
    sf.PRED_DIR = tmp / "predictions"
    sf.REPORTS_DIR = tmp / "reports"
    sf.HEALTH_FILE = tmp / "health.json"


with tempfile.TemporaryDirectory() as _tmpdir:
    tmp = Path(_tmpdir)
    _redirect(tmp)
    _reset_state()

    # ── A/B/C: basic observe() + record schema + health ────────────────
    print("\n[A/B/C] observe() with real HTF36 artifacts")

    import joblib
    fcols = joblib.load(ROOT / "models" / "htf36" / "feature_cols.pkl")
    rng = np.random.default_rng(7)
    df_feat = pd.DataFrame([{c: float(rng.normal()) for c in fcols}])

    sf.observe(cycle=1, df_feat=df_feat, primary_signal=0,
               primary_proba=[0.40, 0.20, 0.40], primary_conf=0.80, spot=24500.0)

    check("shadow model loaded from models/htf36", sf._state["loaded"] is True)
    check("n_features == 28", sf._state["fcols"] is not None and len(sf._state["fcols"]) == 28)

    day = datetime.now().strftime("%Y-%m-%d")
    pred_file = sf.PRED_DIR / f"shadow_{day}.jsonl"
    check("prediction file created", pred_file.exists())

    lines = pred_file.read_text(encoding="utf-8").strip().splitlines()
    check("one record written", len(lines) == 1)

    rec = json.loads(lines[0])
    check("record has primary/shadow/compare blocks",
          all(k in rec for k in ("primary", "shadow", "compare")))
    check("primary direction == CALL (signal 0)", rec["primary"]["direction"] == "CALL")
    check("primary confidence == 80.0", rec["primary"]["confidence"] == 80.0)
    check("shadow direction is one of CALL/PUT/SKIP",
          rec["shadow"]["direction"] in ("CALL", "PUT", "SKIP"))
    check("shadow proba sums to ~1", abs(sum(rec["shadow"]["proba"]) - 1.0) < 1e-3)
    check("compare.direction_match is bool", isinstance(rec["compare"]["direction_match"], bool))
    check("compare.confidence_diff is numeric", isinstance(rec["compare"]["confidence_diff"], (int, float)))
    check("compare.trade_decision_match is bool", isinstance(rec["compare"]["trade_decision_match"], bool))

    health = sf.health_snapshot()
    check("health.json written", bool(health))
    check("health.model_loaded == True", health.get("model_loaded") is True)
    check("health.n_features == 28", health.get("n_features") == 28)
    check("health.total_observations == 1", health.get("total_observations") == 1)
    check("health.consecutive_errors == 0", health.get("consecutive_errors") == 0)

    # ── D: generate_report() aggregation ────────────────────────────────
    print("\n[D] generate_report() aggregation")

    # Add more cycles with KNOWN primary signal/conf so we can hand-check
    # aggregation. Shadow output for this fixed df_feat is constant.
    shadow_sig = rec["shadow"]["signal"]
    shadow_conf = rec["shadow"]["confidence"]
    shadow_trade = rec["shadow"]["trade_decision"]

    # cycle 2: primary == shadow direction & confidence (perfect agreement)
    sf.observe(cycle=2, df_feat=df_feat, primary_signal=shadow_sig,
               primary_proba=rec["shadow"]["proba"],
               primary_conf=shadow_conf / 100.0, spot=24505.0)
    # cycle 3: primary disagrees on direction (force SKIP vs whatever shadow is)
    forced_primary_sig = 2 if shadow_sig != 2 else 0
    sf.observe(cycle=3, df_feat=df_feat, primary_signal=forced_primary_sig,
               primary_proba=[0.1, 0.1, 0.8] if forced_primary_sig == 2 else [0.5, 0.1, 0.4],
               primary_conf=0.50, spot=24510.0)

    from scripts.shadow_daily_report import generate_report
    report = generate_report()

    check("n_cycles == 3", report["n_cycles"] == 3)
    expect_agree = 2 / 3  # cycle1 agreement depends on shadow_sig, cycle2 always agrees, cycle3 forced disagree
    # Recompute expected agreement directly from the written file for robustness
    recs = [json.loads(l) for l in pred_file.read_text(encoding="utf-8").strip().splitlines()]
    expect_agree = sum(1 for r in recs if r["compare"]["direction_match"]) / len(recs)
    check("agreement_rate matches hand count",
          abs(report["agreement_rate"] - round(expect_agree, 4)) < 1e-9)

    expect_p_trades = sum(1 for r in recs if r["primary"]["trade_decision"])
    expect_s_trades = sum(1 for r in recs if r["shadow"]["trade_decision"])
    check("trade_count.primary matches hand count", report["trade_count"]["primary"] == expect_p_trades)
    check("trade_count.shadow matches hand count", report["trade_count"]["shadow"] == expect_s_trades)
    check("trade_count.diff == shadow - primary",
          report["trade_count"]["diff"] == expect_s_trades - expect_p_trades)

    expect_mean_diff = sum(r["compare"]["confidence_diff"] for r in recs) / len(recs)
    check("confidence.mean_diff matches hand calc",
          abs(report["confidence"]["mean_diff"] - round(expect_mean_diff, 2)) < 1e-2)

    report_file = sf.REPORTS_DIR / f"report_{day}.json"
    check("report file written to disk", report_file.exists())

    # ── E: cleanup_old_files() ───────────────────────────────────────────
    print("\n[E] cleanup_old_files()")

    old_day = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    recent_day = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    old_pred = sf.PRED_DIR / f"shadow_{old_day}.jsonl"
    recent_pred = sf.PRED_DIR / f"shadow_{recent_day}.jsonl"
    old_report = sf.REPORTS_DIR / f"report_{old_day}.json"
    old_pred.write_text("{}\n", encoding="utf-8")
    recent_pred.write_text("{}\n", encoding="utf-8")
    old_report.write_text("{}\n", encoding="utf-8")

    result = sf.cleanup_old_files(retention_days=90)
    check("old prediction file removed", not old_pred.exists())
    check("old report file removed", not old_report.exists())
    check("recent prediction file kept", recent_pred.exists())
    check("today's prediction file kept", pred_file.exists())
    check("cleanup result reports retention_days", result["retention_days"] == 90)

    # ── F: shadow_health_check CLI ───────────────────────────────────────
    print("\n[F] shadow_health_check")
    from scripts import shadow_health_check as shc
    exit_code = shc.main()
    check("health check returns 0 (OK) when healthy", exit_code == 0)

    # ── G: observe() never raises on bad input ───────────────────────────
    print("\n[G] observe() resilience")
    before_errors = sf.health_snapshot().get("consecutive_errors", 0)
    bad_df = pd.DataFrame([{"unrelated_col": 1.0}])
    try:
        sf.observe(cycle=4, df_feat=bad_df, primary_signal=0,
                   primary_proba=[0.4, 0.2, 0.4], primary_conf=0.7, spot=24500.0)
        raised = False
    except Exception:
        raised = True
    check("observe() did not raise on missing-feature df_feat", raised is False)
    # missing columns are filled with 0.0 via reindex, so this should still
    # succeed and write a 4th record (not an error)
    lines_after = pred_file.read_text(encoding="utf-8").strip().splitlines()
    check("4th record written despite missing columns", len(lines_after) == 4)

    # ── H: disabled via env var ───────────────────────────────────────────
    print("\n[H] SHADOW_FRAMEWORK_ENABLED=false")
    os.environ["SHADOW_FRAMEWORK_ENABLED"] = "false"
    _reset_state()
    lines_before_disabled = len(pred_file.read_text(encoding="utf-8").strip().splitlines())
    sf.observe(cycle=5, df_feat=df_feat, primary_signal=0,
               primary_proba=[0.4, 0.2, 0.4], primary_conf=0.7, spot=24500.0)
    lines_after_disabled = len(pred_file.read_text(encoding="utf-8").strip().splitlines())
    check("is_enabled() == False", sf.is_enabled() is False)
    check("no new record written when disabled", lines_after_disabled == lines_before_disabled)
    del os.environ["SHADOW_FRAMEWORK_ENABLED"]
    _reset_state()


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
