"""
test_champion_challenger.py — Tests for Champion/Challenger framework (Issue #9).

Run:
    python scripts/test_champion_challenger.py

Tests:
  A. No champion -> first-run promotion passes with cc_mode=True
  B. Challenger beats champion -> PROMOTE
  C. Challenger within CC_TOLERANCE of champion -> PROMOTE
  D. Challenger exceeds CC_TOLERANCE below champion -> REJECT (live model unchanged)
  E. Degraded champion exception: low live win rate overrides C/C rejection
  F. Degraded champion: insufficient live trades (< N//2) -> exception NOT triggered
  G. get_cc_report returns correct verdict for all cases
  H. cc_acc_gap stamped onto promoted metadata
  I. Stage 1 still blocks garbage models even with no champion
"""
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0]], [0])
    return {"xgb": clf}


def _fake_scaler():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit([[0, 1], [1, 2]])
    return sc


def _meta(test_dir_acc=0.60, test_signals=20):
    return {
        "train_date_start": "2015-01-09", "train_date_end": "2026-06-01",
        "val_date_start": "2026-06-02",   "val_date_end": "2026-06-03",
        "test_date_start": "2026-06-04",  "test_date_end": "2026-06-05",
        "train_bars": 170000, "val_bars": 10000, "test_bars": 10000,
        "val_dir_acc": test_dir_acc + 0.02, "test_dir_acc": test_dir_acc,
        "test_signals": test_signals,
        "n_features": 188, "fwd_bars": 3, "train_frac": 0.85,
        "val_frac": 0.075, "embargo_bars": 75, "label_quantile": 0.70,
    }


def _with_tmpdir(fn):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        orig_md = reg.MODELS_DIR
        orig_rd = reg.REGISTRY_DIR
        orig_jf = reg._JOURNAL_FILE
        reg.MODELS_DIR    = tmp / "models"
        reg.REGISTRY_DIR  = tmp / "models" / "registry"
        reg._JOURNAL_FILE = tmp / "logs" / "trade_journal.jsonl"
        reg.MODELS_DIR.mkdir(parents=True)
        reg.REGISTRY_DIR.mkdir(parents=True)
        try:
            fn(tmp)
        finally:
            reg.MODELS_DIR    = orig_md
            reg.REGISTRY_DIR  = orig_rd
            reg._JOURNAL_FILE = orig_jf


def _seed_champion(version, acc):
    """Save + promote a champion candidate with given test accuracy."""
    (reg.MODELS_DIR / f"nifty_{version}_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / f"feature_cols_{version}.pkl").write_bytes(b"x")
    cid = reg.save_candidate(version, _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=acc))
    reg.promote_if_passes_gate(version, cid, cc_mode=False)   # seed without C/C
    return cid


def _write_journal(tmp, trade_pnls):
    """Write fake EXIT records to the test journal."""
    log_dir = tmp / "logs"
    log_dir.mkdir(exist_ok=True)
    with reg._JOURNAL_FILE.open("w") as f:
        for pnl in trade_pnls:
            f.write(json.dumps({
                "event": "EXIT",
                "is_dry_run": False,
                "trade_date": "2026-06-05",
                "pnl_pts": pnl,
            }) + "\n")


# ── A. No champion -> first-run promotion passes ──────────────────────────────
print("\n[A] No champion -> first-run promotion")

def _test_first_run(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.58))
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    check("first-run promoted",        promoted is True)
    check("reason mentions no champion or PROMOTED", "PROMOTED" in reason or "first" in reason.lower())

_with_tmpdir(_test_first_run)


# ── B. Challenger beats champion -> PROMOTE ───────────────────────────────────
print("\n[B] Challenger better than champion -> PROMOTE")

def _test_challenger_wins(tmp):
    _seed_champion("v10", acc=0.58)
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.63))
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    check("challenger beats champion -> promoted",  promoted is True)
    check("reason mentions cc_gap positive",        "cc_gap=+" in reason)

_with_tmpdir(_test_challenger_wins)


# ── C. Challenger within tolerance -> PROMOTE ────────────────────────────────
print("\n[C] Challenger within CC_TOLERANCE -> PROMOTE")

def _test_within_tolerance(tmp):
    _seed_champion("v10", acc=0.61)
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    # 0.61 - 0.59 = 0.02 gap, default tolerance = 0.03 -> should pass
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.59))
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    check("within tolerance -> promoted",    promoted is True)
    check("gap=-0.02 within 0.03 tolerance", "cc_gap=-0.02" in reason)

_with_tmpdir(_test_within_tolerance)


# ── D. Challenger exceeds tolerance -> REJECT, live file unchanged ─────────────
print("\n[D] Challenger exceeds CC_TOLERANCE -> REJECT, live file untouched")

def _test_reject(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    _seed_champion("v10", acc=0.65)
    # Record the champion's model bytes AFTER seeding (before challenger attempt)
    live_model = reg.MODELS_DIR / "nifty_v10_models.pkl"
    champ_bytes = live_model.read_bytes()
    # 0.65 - 0.59 = 0.06 gap, exceeds 0.03 tolerance -> reject
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.59))
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    check("exceeds tolerance -> rejected",      promoted is False)
    check("reason mentions CC REJECT",           "CC REJECT" in reason)
    # Live model must still contain the champion's bytes (not the challenger's)
    check("live model file UNTOUCHED after CC REJECT",
          live_model.read_bytes() == champ_bytes)

_with_tmpdir(_test_reject)


# ── E. Degraded champion exception ────────────────────────────────────────────
print("\n[E] Degraded champion -> Stage 2 overridden")

def _test_degraded_champion(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    _seed_champion("v10", acc=0.65)
    # Write journal showing champion has 2/10 live win rate (20% < CC_DEGRADE_THRESH=45%)
    _write_journal(tmp, [35.0, -70.0, -60.0, -80.0, -50.0,
                         -40.0, 25.0, -65.0, -55.0, -45.0])
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.59))  # 0.06 below champion
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    check("degraded champion -> challenger promoted",  promoted is True)
    check("reason or log mentions degraded champion",  "degraded" in reason.lower()
                                                        or "PROMOTED" in reason)

_with_tmpdir(_test_degraded_champion)


# ── F. Insufficient live trades -> degraded exception NOT triggered ───────────
print("\n[F] Insufficient live trades -> degraded exception NOT triggered")

def _test_insufficient_live(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    _seed_champion("v10", acc=0.65)
    # Only 3 live trades (need at least CC_DEGRADE_N // 2 = 10) -> live_wr = None
    _write_journal(tmp, [-70.0, -60.0, -80.0])
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.59))  # still 0.06 below champion
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    # Should REJECT because insufficient live data and gap exceeds tolerance
    check("insufficient live trades -> CC REJECT applies", promoted is False)

_with_tmpdir(_test_insufficient_live)


# ── G. get_cc_report all verdicts ─────────────────────────────────────────────
print("\n[G] get_cc_report verdicts")

def _test_cc_report(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")

    # Save challenger with no champion
    cid1 = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                               _meta(test_dir_acc=0.60))
    r = reg.get_cc_report("v10", cid1)
    check("no champion -> verdict=NO_CHAMPION", r["cc_verdict"] == "NO_CHAMPION")

    # Promote to make it champion
    reg.promote_if_passes_gate("v10", cid1, cc_mode=False)

    # Challenger that's better
    cid2 = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                               _meta(test_dir_acc=0.63))
    r2 = reg.get_cc_report("v10", cid2)
    check("better challenger -> verdict=PROMOTE", r2["cc_verdict"] == "PROMOTE")
    check("gap is positive",                       r2["gap"] > 0)

    # Challenger that's worse beyond tolerance
    cid3 = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                               _meta(test_dir_acc=0.55))
    r3 = reg.get_cc_report("v10", cid3)
    check("worse challenger -> verdict=REJECT",   r3["cc_verdict"] == "REJECT")
    check("gap is negative",                       r3["gap"] < 0)

    # Degraded champion
    _write_journal(tmp, [-70.0, -60.0, -80.0, -50.0, -45.0,
                         -40.0, -65.0, -55.0, -45.0, -35.0,
                         -70.0, -60.0, -80.0, -50.0, -45.0,
                         -40.0, -65.0, -55.0, -45.0, -35.0])
    r4 = reg.get_cc_report("v10", cid3)
    check("degraded champion -> verdict=DEGRADED_CHAMPION",
          r4["cc_verdict"] == "DEGRADED_CHAMPION")

_with_tmpdir(_test_cc_report)


# ── H. cc_acc_gap stamped onto promoted metadata ──────────────────────────────
print("\n[H] cc_acc_gap stamped onto metadata")

def _test_cc_meta(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    _seed_champion("v10", acc=0.60)
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.62))
    reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    cdir = reg._registry_path("v10", cid)
    meta = json.loads((cdir / "metadata.json").read_text())
    check("cc_acc_gap in metadata",         "cc_acc_gap" in meta)
    check("cc_champion_id in metadata",     "cc_champion_id" in meta)
    check("cc_acc_gap = +0.02",             abs(meta["cc_acc_gap"] - 0.02) < 0.001)

_with_tmpdir(_test_cc_meta)


# ── I. Stage 1 still blocks garbage regardless of champion ────────────────────
print("\n[I] Stage 1 blocks garbage even with no champion")

def _test_stage1_still_active(tmp):
    (reg.MODELS_DIR / "nifty_v10_scaler.pkl").write_bytes(b"x")
    (reg.MODELS_DIR / "feature_cols_v10.pkl").write_bytes(b"x")
    cid = reg.save_candidate("v10", _fake_models(), _fake_scaler(), ["f1"],
                              _meta(test_dir_acc=0.40))  # below 0.52 floor
    promoted, reason = reg.promote_if_passes_gate("v10", cid, cc_mode=True)
    check("garbage below floor -> rejected even without champion", promoted is False)
    check("reason mentions GATE FAIL",  "GATE FAIL" in reason)

_with_tmpdir(_test_stage1_still_active)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
