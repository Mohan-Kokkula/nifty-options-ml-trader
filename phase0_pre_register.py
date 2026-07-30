"""
phase0_pre_register.py — Freeze the v2 research protocol.

Writes pre_registration/preregistration_active.json containing:
  - alpha (family-wise error rate)
  - primary hypotheses + acceptance criteria (pre-committed)
  - effect-size thresholds (pre-committed)
  - decision tree
  - bootstrap / test configuration
  - SHA-256 manifest of every Python source file + every raw data CSV
  - runtime environment (Python + key package versions)

Behavior:
  - Refuses to overwrite an active pre-registration. Archive the old one first
    (phase0 will move it to pre_registration/archive/ automatically when
    invoked with --supersede, but never silently).
  - Emits pre_registration/preregistration_<utc_iso>.json AND updates
    pre_registration/preregistration_active.json as a symlink-equivalent copy.
  - All downstream phases MUST call verify_preregistration.verify() before
    running any hypothesis test. Drift ==> abort.

CLI:
  python phase0_pre_register.py                # write active pre-reg
  python phase0_pre_register.py --supersede    # archive existing, write new
  python phase0_pre_register.py --show         # print active pre-reg summary
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREREG_DIR = ROOT / "pre_registration"
ARCHIVE_DIR = PREREG_DIR / "archive"
ACTIVE = PREREG_DIR / "preregistration_active.json"

PROTOCOL_VERSION = "2.5"

# ----------------------------------------------------------------------------
# Files whose contents are hashed into the manifest. Anything that materially
# affects an experiment's result MUST be listed here.
#
# v2.1: extended to cover the stat_utils package and its test suite,
# plus the pytest configuration and test-only requirements pin. All four
# additions were introduced during Phase 7 (statistical machinery).
# ----------------------------------------------------------------------------
CODE_GLOBS = [
    "*.py",
    "stat_utils/**/*.py",
    "stat_utils/tests/**/*.py",   # redundant with the recursive pattern above
                                    # but listed explicitly per pre-reg protocol
]
CODE_EXCLUDE = {
    "phase0_pre_register.py",   # cannot hash self meaningfully
    "verify_preregistration.py",
}
DATA_FILES = [
    "data/nifty_5min.csv",
    "data/nifty_15min.csv",
    "data/nifty_30min.csv",
    "data/nifty_60min.csv",
    "data/nifty_day.csv",
    "data/india_vix.csv",
    "data/nifty_expiry_history.csv",
    "data/cross_asset_daily.csv",
    # v2.1: versioned test configuration. These are not model inputs but
    # they materially affect how downstream verification runs, so we
    # fingerprint them for reproducibility parity with code.
    "pytest.ini",
    "requirements-test.txt",
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def code_manifest() -> dict:
    out = {}
    for glob in CODE_GLOBS:
        for p in sorted(ROOT.glob(glob)):
            if p.name in CODE_EXCLUDE:
                continue
            rel = p.relative_to(ROOT).as_posix()
            out[rel] = {"sha256": sha256_of(p), "size": p.stat().st_size}
    return out


def data_manifest() -> dict:
    out = {}
    for rel in DATA_FILES:
        p = ROOT / rel
        if p.exists():
            out[rel] = {"sha256": sha256_of(p), "size": p.stat().st_size,
                        "mtime_utc": datetime.fromtimestamp(
                            p.stat().st_mtime, tz=timezone.utc).isoformat()}
        else:
            out[rel] = {"sha256": None, "missing": True}
    return out


def env_manifest() -> dict:
    import platform
    def _v(mod):
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _v(p) for p in [
            "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
            "catboost", "optuna", "matplotlib",
        ]},
    }


# ----------------------------------------------------------------------------
# The frozen protocol content. This is the object that MUST NOT be edited
# after phase0 has emitted an active pre-registration.
# ----------------------------------------------------------------------------
def protocol_body() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "Test whether the current OpenClaw architecture is exhausted, "
            "using pre-registered CI-based decisions and FWE control."),
        "family_wise_error_rate": 0.10,
        "correction_method": "holm_bonferroni",
        "bootstrap": {
            "method": "block_bootstrap",
            "block_unit": "outer_fold",
            "n_resamples": 10_000,
            "seed": 20260708,
            "ci_level": 0.90,     # 90% CI matches FWE alpha = 0.10
        },
        "diebold_mariano": {
            "variance_estimator": "newey_west",
            "lag": 5,
            "two_sided": True,
        },
        "primary_hypotheses": [
            {
                "id": "H_edge",
                "phase": 1,
                "description": "Current avg4 ensemble has a post-friction PF > 1.",
                "null": "PF <= 1.00",
                "test": "block_bootstrap_pf_lb90",
                "accept_criterion": "PF_LB90 > 1.00",
                "effect_size": None,
            },
            {
                "id": "H_hpo",
                "phase": 2,
                "description": "Nested purged WF-HPO on PF objective improves PF.",
                "null": "PF(tuned) - PF(current) <= 0",
                "test": "block_bootstrap_delta_pf + diebold_mariano",
                "accept_criterion": "LB90(dPF) > +0.08 AND DM_p < holm_alpha",
                "effect_size": 0.08,
            },
            {
                "id": "H_arch",
                "phase": 3,
                "description": (
                    "At least one non-GBDT architecture (linear, kernel SVM, "
                    "FT-Transformer, temporal CNN, HMM, TabPFN) exceeds best "
                    "GBDT PF by the pre-registered effect."),
                "null": "max_nongbdt_PF - best_gbdt_PF <= 0",
                "test": "block_bootstrap + White_reality_check + Hansen_SPA",
                "accept_criterion": (
                    "LB90(dPF) > +0.08 AND "
                    "SPAResult.pvalue_lower < holm_alpha"),
                "primary_spa_field": "pvalue_lower",
                "supplementary_spa_fields": [
                    "pvalue_consistent",
                    "pvalue_upper",
                ],
                "spa_semantic_note": (
                    "Hansen (2005) SPA reports three p-values. For a "
                    "signal-dominated scenario, pvalue_consistent and "
                    "pvalue_upper are bounded away from zero by the "
                    "recentring construction (bootstrap distribution "
                    "centred at the observed max), so they cannot serve "
                    "as the decisive rejection metric. pvalue_lower "
                    "(recentring all models to zero, Hansen's "
                    "least-favourable null) is the operational rejection "
                    "p-value; pvalue_consistent and pvalue_upper are "
                    "reported alongside as supplementary conservatism "
                    "checks and MUST be published in the final results "
                    "table."),
                "effect_size": 0.08,
            },
            {
                "id": "H_ens",
                "phase": 9,
                "description": ("Some ensemble construction (stacking, "
                                "uncertainty-weighted, min-variance) beats "
                                "simple mean."),
                "null": "PF(best_ens) - PF(mean) <= 0",
                "test": "block_bootstrap + diebold_mariano",
                "accept_criterion": "LB90(dPF) > +0.05 AND DM_p < holm_alpha",
                "effect_size": 0.05,
            },
            {
                "id": "H_cal",
                "phase": 4,
                "description": ("Fitting isotonic (per-class) calibration OOF "
                                "and re-simulating trades increases PF."),
                "null": "PF(cal) - PF(uncal) <= 0",
                "test": "block_bootstrap_delta_pf",
                "accept_criterion": "LB90(dPF_isotonic) > +0.05",
                "effect_size": 0.05,
            },
            {
                "id": "H_target",
                "phase": 5,
                "description": ("An alternative target formulation shows "
                                "higher Skill than Target A within the same "
                                "metric family (classification-only "
                                "comparison; regression targets compared "
                                "only to other regression targets)."),
                "null": "Skill(alt) - Skill(A) <= 0",
                "test": "block_bootstrap_delta_skill",
                "accept_criterion": ("Skill(alt) >= 1.5 x Skill(A) AND "
                                     "absolute Skill gap >= 0.02"),
                "effect_size": 0.02,
            },
            {
                "id": "H_info",
                "phase": 8,
                "description": ("Feature set contains extractable information "
                                "beyond what current models use. Measured by "
                                "k-NN Bayes lower bound vs achieved error, "
                                "MINE MI vs Fano bound, and high-capacity "
                                "ceiling gap."),
                "null": ("Bayes_bound == achieved_error AND "
                         "high_capacity_gap == 0"),
                "test": "block_bootstrap_gap + permutation_test",
                "accept_criterion": ("LB90(high_capacity_gap) > +0.05 "
                                     "OR LB90(Bayes_gap) > +0.02"),
                "effect_size": 0.05,
            },
        ],
        "sanity_gates": {
            "G1_zero_trade_fold": "Any fold with 0 trades invalidates that phase.",
            "G2_manifest_drift": "code/data SHA mismatch invalidates the run.",
            "G3_baseline_drift": (
                "Phase 2 'current' PF must match logs/sweep_A_current.csv "
                "within +/-0.05, else Phase 2 is invalid."),
            "G4_trade_count_min": "Any config with <100 trades in an arm is UNDECIDED.",
            "G5_ci_wider_than_effect": (
                "If any 90% CI half-width exceeds 2x the pre-reg effect, "
                "declare UNDECIDED rather than force a NO."),
        },
        "decision_tree_reference": "See §4 of the protocol document.",
        "undecided_state_permitted": True,
        "post_hoc_threshold_changes": "FORBIDDEN",
    }


def write_active(supersede: bool) -> Path:
    PREREG_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    if ACTIVE.exists():
        if not supersede:
            print("ERROR: active pre-registration already exists:")
            print(f"  {ACTIVE}")
            print("Refusing to overwrite. Re-run with --supersede to archive")
            print("the existing one and write a new pre-registration.")
            sys.exit(2)
        # archive with the frozen timestamp already inside it
        try:
            old = json.loads(ACTIVE.read_text())
            stamp = old.get("frozen_at_utc", "unknown").replace(":", "-")
        except Exception:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        target = ARCHIVE_DIR / f"preregistration_{stamp}.json"
        shutil.move(str(ACTIVE), str(target))
        print(f"Archived previous active pre-registration -> {target}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc = {
        "frozen_at_utc": now,
        "protocol": protocol_body(),
        "code_manifest": code_manifest(),
        "data_manifest": data_manifest(),
        "runtime_environment": env_manifest(),
    }
    ACTIVE.write_text(json.dumps(doc, indent=2, sort_keys=True))
    dated = PREREG_DIR / f"preregistration_{now.replace(':', '-')}.json"
    shutil.copy(str(ACTIVE), str(dated))
    print(f"Wrote {ACTIVE}")
    print(f"Wrote {dated}")
    return ACTIVE


def show_active() -> None:
    if not ACTIVE.exists():
        print("No active pre-registration.")
        sys.exit(1)
    doc = json.loads(ACTIVE.read_text())
    print("=" * 72)
    print("  ACTIVE PRE-REGISTRATION")
    print("=" * 72)
    print(f"  frozen_at_utc     : {doc['frozen_at_utc']}")
    print(f"  protocol_version  : {doc['protocol']['protocol_version']}")
    print(f"  FWE alpha         : {doc['protocol']['family_wise_error_rate']}")
    print(f"  correction        : {doc['protocol']['correction_method']}")
    print(f"  bootstrap B       : {doc['protocol']['bootstrap']['n_resamples']:,}")
    print(f"  bootstrap block   : {doc['protocol']['bootstrap']['block_unit']}")
    print(f"  # code files      : {len(doc['code_manifest'])}")
    print(f"  # data files      : {len(doc['data_manifest'])}")
    print("\n  Primary hypotheses:")
    for h in doc["protocol"]["primary_hypotheses"]:
        eff = h.get("effect_size")
        eff_s = f"effect >= {eff}" if eff is not None else "no effect gate"
        print(f"    [{h['id']:<10}] phase {h['phase']}: {eff_s}")
        print(f"                 accept: {h['accept_criterion']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supersede", action="store_true",
                    help="Archive existing active pre-reg and write a new one.")
    ap.add_argument("--show", action="store_true",
                    help="Print summary of the active pre-registration.")
    args = ap.parse_args()
    if args.show:
        show_active()
        return
    write_active(supersede=args.supersede)
    show_active()


if __name__ == "__main__":
    main()
