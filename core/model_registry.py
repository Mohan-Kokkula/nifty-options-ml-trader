"""
model_registry.py — Model Versioning, Promotion Gate, and Rollback
===================================================================

Every training run saves a CANDIDATE to an immutable registry directory.
The live model (models/nifty_v10_models.pkl etc.) is only updated when the
promotion gate passes. If it fails, the previous live model stays untouched.

Directory layout:
    models/
      nifty_v10_models.pkl         <- live model (read by ml_engine.py)
      nifty_v10_scaler.pkl
      feature_cols_v10.pkl
      registry/
        2026-06-05_143022_a1b2c3/  <- one immutable dir per training run
          models.pkl
          scaler.pkl
          feature_cols.pkl
          metadata.json
        2026-06-04_091500_def456/
          ...

Public API:
    save_candidate(version, models, scaler, fcols, metadata)  -> candidate_id
    promote_if_passes_gate(version, gate_cfg)                 -> (promoted, reason)
    rollback(version, candidate_id)                           -> bool
    list_versions(version, n)                                 -> list[dict]
    get_active_metadata(version)                              -> dict | None
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR   = Path("models")
REGISTRY_DIR = MODELS_DIR / "registry"

# Live model filename stems per version (must match ml_engine.py candidates list)
_LIVE_STEMS: dict[str, tuple[str, str, str]] = {
    # v11 added 2026-08-07. It was missing, so save_candidate("v11", ...)
    # succeeded (registry dir is version-agnostic) while
    # promote_if_passes_gate("v11") raised ValueError -- V11 could be
    # trained and stored but never promoted.
    "v11": ("nifty_v11_models.pkl",    "nifty_v11_scaler.pkl",    "feature_cols_v11.pkl"),
    "v10": ("nifty_v10_models.pkl",    "nifty_v10_scaler.pkl",    "feature_cols_v10.pkl"),
    "v9":  ("nifty_v9_models.pkl",     "nifty_v9_scaler.pkl",     "feature_cols_v9.pkl"),
    "v8":  ("nifty_v8_models.pkl",     "nifty_v8_scaler.pkl",     "feature_cols_v8.pkl"),
}

# ── Default promotion gate thresholds ────────────────────────────────────────
# Override via env vars or pass gate_cfg= kwarg to promote_if_passes_gate.
DEFAULT_GATE: dict[str, Any] = {
    "min_test_dir_acc": 0.52,  # absolute floor — worse than random guessing fails
    "min_test_signals": 5,     # need at least 5 directional signals in TEST
    "max_val_test_gap": 0.10,  # WARN (not block) if VAL-TEST gap > 10%
}

# ── Champion/Challenger constants ─────────────────────────────────────────────
# CC_TOLERANCE       : challenger can be at most this much BELOW champion's
#                      test_dir_acc and still be promoted.
#                      0.03 absorbs normal day-to-day regime variation;
#                      anything larger signals a genuinely worse model.
# CC_DEGRADE_THRESH  : if champion's live win rate (last N trades) falls below
#                      this value, the champion is considered "degraded" and any
#                      challenger that passes the absolute floor is accepted,
#                      even if its test_dir_acc is lower than the champion's.
# CC_DEGRADE_N       : number of recent live trades used to measure live win rate.
CC_TOLERANCE      = float(os.getenv("CC_TOLERANCE",       "0.03"))
CC_DEGRADE_THRESH = float(os.getenv("CC_DEGRADE_THRESH",  "0.45"))
CC_DEGRADE_N      = int(  os.getenv("CC_DEGRADE_N",       "20"))

# Journal file path — used to compute champion live win rate
_JOURNAL_FILE = Path("logs/trade_journal.jsonl")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _candidate_id() -> str:
    """Generate a unique, sortable candidate ID: YYYYMMDD_HHMMSS_<hash6>."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    h6  = hashlib.sha1(str(time.time()).encode()).hexdigest()[:6]
    return f"{ts}_{h6}"


def _registry_path(version: str, candidate_id: str) -> Path:
    return REGISTRY_DIR / f"{version}_{candidate_id}"


def _live_paths(version: str) -> tuple[Path, Path, Path]:
    """Return (models, scaler, fcols) live paths for this version."""
    stems = _LIVE_STEMS.get(version.lower())
    if not stems:
        raise ValueError(f"Unknown model version: {version!r}. "
                         f"Known: {list(_LIVE_STEMS)}")
    return tuple(MODELS_DIR / s for s in stems)  # type: ignore[return-value]


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy src → dst atomically (write to .tmp then os.replace).

    os.replace() is atomic on POSIX and near-atomic on Windows (Python 3.3+).
    It overwrites dst if it exists, unlike Path.rename() which raises on Windows.
    """
    import os
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)   # atomic replace; works on both POSIX and Windows


# ── Public API ───────────────────────────────────────────────────────────────

def save_candidate(
    version: str,
    models: Any,
    scaler: Any,
    fcols: Any,
    metadata: dict,
) -> str:
    """
    Save a trained model as an immutable registry candidate.

    Does NOT overwrite the live model — that only happens via
    promote_if_passes_gate() or rollback().

    Parameters
    ----------
    version  : "v10" | "v9" | "v8"
    models   : joblib-serialisable dict of fitted estimators
    scaler   : fitted StandardScaler
    fcols    : list[str] of feature column names
    metadata : dict with at minimum:
                 test_dir_acc, test_signals, val_dir_acc, n_features,
                 train_date_start, train_date_end, test_date_start, test_date_end

    Returns
    -------
    candidate_id : str  (e.g. "20260605_143022_a1b2c3")
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib required: pip install joblib")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    cid  = _candidate_id()
    cdir = _registry_path(version, cid)
    cdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, cdir / "models.pkl")
    joblib.dump(scaler, cdir / "scaler.pkl")
    joblib.dump(fcols,  cdir / "feature_cols.pkl")

    full_meta = {
        "version":      version,
        "candidate_id": cid,
        "created_at":   datetime.now().isoformat(timespec="seconds"),
        "promoted":     False,
        "promoted_at":  None,
        **metadata,
    }
    (cdir / "metadata.json").write_text(
        json.dumps(full_meta, indent=2, default=str)
    )

    logger.info(
        f"Registry: saved candidate {version}/{cid} "
        f"(test_dir_acc={metadata.get('test_dir_acc', '?'):.3f}, "
        f"signals={metadata.get('test_signals', '?')})"
    )
    return cid


def get_live_win_rate(version: str, n: int = CC_DEGRADE_N) -> Optional[float]:
    """
    Return the live win rate of the last `n` real trades from the trade journal.

    Reads logs/trade_journal.jsonl, finds the most recent `n` EXIT records
    where is_dry_run=False, and computes wins / total.

    Returns None if fewer than `n // 2` trades are found (insufficient data).
    This prevents a single bad trade from triggering the degraded-champion
    exception on day 1 of live trading.
    """
    if not _JOURNAL_FILE.exists():
        return None
    try:
        trades = []
        with _JOURNAL_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue
                if t.get("event") != "EXIT":
                    continue
                if t.get("is_dry_run"):
                    continue
                pnl = t.get("pnl_pts")
                if pnl is None:
                    continue
                trades.append(float(pnl))

        if not trades:
            return None
        recent = trades[-n:]
        if len(recent) < max(1, n // 2):
            logger.debug(
                f"get_live_win_rate: only {len(recent)} trades "
                f"(need {n//2} minimum) — returning None"
            )
            return None
        wins = sum(1 for p in recent if p > 0)
        return wins / len(recent)
    except Exception as e:
        logger.debug(f"get_live_win_rate failed: {e}")
        return None


def get_cc_report(version: str, candidate_id: Optional[str] = None) -> dict:
    """
    Build a Champion/Challenger comparison report (read-only, no promotion).

    Returns a dict with:
      champion_id, champion_test_dir_acc, champion_promoted_at,
      challenger_id, challenger_test_dir_acc,
      gap           (challenger - champion, positive = challenger better),
      cc_verdict    ("PROMOTE" | "REJECT" | "NO_CHAMPION" | "DEGRADED_CHAMPION"),
      live_win_rate (champion's recent live performance, or None),
      detail        (human-readable explanation)
    """
    if candidate_id is None:
        candidate_id = _latest_candidate_id(version)
    if candidate_id is None:
        return {"cc_verdict": "NO_DATA", "detail": "No candidates in registry"}

    cdir = _registry_path(version, candidate_id)
    chal_meta_path = cdir / "metadata.json"
    if not chal_meta_path.exists():
        return {"cc_verdict": "NO_DATA", "detail": "Challenger metadata missing"}
    chal_meta = json.loads(chal_meta_path.read_text())

    champ_meta = get_active_metadata(version)
    live_wr    = get_live_win_rate(version)

    report: dict = {
        "challenger_id":            candidate_id,
        "challenger_test_dir_acc":  chal_meta.get("test_dir_acc"),
        "challenger_test_signals":  chal_meta.get("test_signals"),
        "challenger_train_end":     chal_meta.get("train_date_end"),
        "challenger_test_dates":    (
            f"{chal_meta.get('test_date_start')} → {chal_meta.get('test_date_end')}"
        ),
        "champion_id":              None,
        "champion_test_dir_acc":    None,
        "champion_promoted_at":     None,
        "gap":                      None,
        "live_win_rate":            round(live_wr, 3) if live_wr is not None else None,
        "cc_verdict":               "NO_CHAMPION",
        "detail":                   "No previously promoted model — first-run promotion",
    }

    if champ_meta:
        report["champion_id"]           = champ_meta.get("candidate_id")
        report["champion_test_dir_acc"] = champ_meta.get("test_dir_acc")
        report["champion_promoted_at"]  = champ_meta.get("promoted_at")

        chal_acc  = float(chal_meta.get("test_dir_acc", 0))
        champ_acc = float(champ_meta.get("test_dir_acc", 0))
        gap       = chal_acc - champ_acc
        report["gap"] = round(gap, 4)

        if live_wr is not None and live_wr < CC_DEGRADE_THRESH:
            report["cc_verdict"] = "DEGRADED_CHAMPION"
            report["detail"] = (
                f"Champion degraded in live trading "
                f"(live_wr={live_wr:.0%} < threshold {CC_DEGRADE_THRESH:.0%}) — "
                f"accepting challenger regardless of accuracy gap"
            )
        elif gap >= -CC_TOLERANCE:
            report["cc_verdict"] = "PROMOTE"
            report["detail"] = (
                f"Challenger ({chal_acc:.3f}) within {CC_TOLERANCE:.2f} tolerance "
                f"of champion ({champ_acc:.3f}), gap={gap:+.3f}"
            )
        else:
            report["cc_verdict"] = "REJECT"
            report["detail"] = (
                f"Challenger ({chal_acc:.3f}) is {abs(gap):.3f} below champion "
                f"({champ_acc:.3f}), exceeds tolerance {CC_TOLERANCE:.2f}"
            )

    return report


def promote_if_passes_gate(
    version: str,
    candidate_id: Optional[str] = None,
    gate_cfg: Optional[dict] = None,
    cc_mode: bool = True,
) -> tuple[bool, str]:
    """
    Evaluate the promotion gate for the LATEST candidate (or a specific one).

    Stage 1 — Absolute floor (always active):
      1. test_dir_acc >= gate_cfg["min_test_dir_acc"]  (quality floor, default 0.52)
      2. test_signals >= gate_cfg["min_test_signals"]  (enough data, default 5)
      3. All three model files present in the candidate dir
      4. Files loadable by joblib without error

    Stage 2 — Champion/Challenger comparison (when cc_mode=True, default):
      Compares challenger against the currently active (promoted) model.
      challenger.test_dir_acc >= champion.test_dir_acc - CC_TOLERANCE (default 0.03)
      If no champion exists: skip Stage 2 (first model always promoted if Stage 1 passes).

    Stage 3 — Degraded-champion exception (when cc_mode=True):
      If the champion's live win rate (last CC_DEGRADE_N trades) falls below
      CC_DEGRADE_THRESH (default 0.45), Stage 2 rejection is overridden and
      the challenger is accepted if Stage 1 passes.

    Warnings (logged, never block):
      - val_dir_acc - test_dir_acc > gate_cfg["max_val_test_gap"] (overfit)

    On PASS: atomic-copies candidate → live paths, marks metadata promoted=True.
    On FAIL: live model is untouched; returns (False, reason).

    Returns (promoted: bool, reason: str).
    """
    import joblib

    cfg = {**DEFAULT_GATE, **(gate_cfg or {})}

    # Find candidate dir
    if candidate_id is None:
        candidate_id = _latest_candidate_id(version)
        if candidate_id is None:
            return False, f"No candidates in registry for version={version!r}"

    cdir = _registry_path(version, candidate_id)
    if not cdir.exists():
        return False, f"Candidate dir not found: {cdir}"

    # Load metadata
    meta_path = cdir / "metadata.json"
    if not meta_path.exists():
        return False, f"metadata.json missing in {cdir}"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return False, f"metadata.json unreadable: {e}"

    # Gate 1 — quality floor
    test_acc = float(meta.get("test_dir_acc", 0))
    if test_acc < cfg["min_test_dir_acc"]:
        reason = (
            f"GATE FAIL: test_dir_acc={test_acc:.3f} < "
            f"min={cfg['min_test_dir_acc']:.2f} — model not promoted"
        )
        logger.warning(f"Registry: {reason}")
        return False, reason

    # Gate 2 — enough signals
    signals = int(meta.get("test_signals", 0))
    if signals < cfg["min_test_signals"]:
        reason = (
            f"GATE FAIL: test_signals={signals} < "
            f"min={cfg['min_test_signals']} — not enough TEST signals"
        )
        logger.warning(f"Registry: {reason}")
        return False, reason

    # Gate 3 — files present
    for fname in ("models.pkl", "scaler.pkl", "feature_cols.pkl"):
        if not (cdir / fname).exists():
            reason = f"GATE FAIL: {fname} missing in candidate {candidate_id}"
            logger.warning(f"Registry: {reason}")
            return False, reason

    # Gate 4 — files loadable
    try:
        joblib.load(cdir / "models.pkl")
        joblib.load(cdir / "scaler.pkl")
        joblib.load(cdir / "feature_cols.pkl")
    except Exception as e:
        reason = f"GATE FAIL: candidate files not loadable — {e}"
        logger.warning(f"Registry: {reason}")
        return False, reason

    # Warn on overfit signal (non-blocking)
    val_acc = float(meta.get("val_dir_acc", 0))
    gap = val_acc - test_acc
    if gap > cfg["max_val_test_gap"]:
        logger.warning(
            f"Registry: overfit warning — val_dir_acc={val_acc:.3f} "
            f"test_dir_acc={test_acc:.3f} gap={gap:+.3f} "
            f"(threshold={cfg['max_val_test_gap']:.2f}) — promoting anyway"
        )

    # ── Stage 2: Champion/Challenger comparison ───────────────────────────────
    if cc_mode:
        champ_meta = get_active_metadata(version)
        if champ_meta is not None:
            champ_acc  = float(champ_meta.get("test_dir_acc", 0))
            chal_acc   = test_acc
            acc_gap    = chal_acc - champ_acc   # positive = challenger is better
            champ_cid  = champ_meta.get("candidate_id", "unknown")

            # 2026-08-06: test_dir_acc changed meaning in metric_version 2
            # (argmax scoring -> live-rule scoring, see the comment block
            # above live_rule_signals() in scripts/train_model_v9.py).
            # Comparing a v2 challenger against a v1 champion compares two
            # different quantities, so skip C/C entirely rather than make a
            # bogus accept/reject call. The absolute floors (Gates 1-4) still
            # applied above. Once the champion has been retrained under v2
            # this resolves itself and normal C/C resumes.
            champ_mv = int(champ_meta.get("metric_version", 1))
            chal_mv  = int(meta.get("metric_version", 1))
            metrics_comparable = (champ_mv == chal_mv)
            if not metrics_comparable:
                logger.warning(
                    f"Registry: C/C SKIPPED — champion {champ_cid} uses "
                    f"metric_version={champ_mv} but challenger {candidate_id} "
                    f"uses metric_version={chal_mv}; test_dir_acc is not "
                    f"comparable across versions. Absolute quality gates "
                    f"passed, so promoting on those alone."
                )
        else:
            metrics_comparable = False

        if champ_meta is not None and metrics_comparable:

            # Stage 3: degraded-champion exception
            live_wr = get_live_win_rate(version)
            degraded = (live_wr is not None and live_wr < CC_DEGRADE_THRESH)

            if degraded:
                logger.warning(
                    f"Registry: champion degraded in live trading "
                    f"(live_wr={live_wr:.0%} < {CC_DEGRADE_THRESH:.0%}) — "
                    f"accepting challenger {candidate_id} regardless of accuracy gap"
                )
            elif acc_gap < -CC_TOLERANCE:
                # Challenger is significantly worse — reject
                reason = (
                    f"CC REJECT: challenger {version}/{candidate_id} "
                    f"test_dir_acc={chal_acc:.3f} is {abs(acc_gap):.3f} below "
                    f"champion {champ_cid} ({champ_acc:.3f}), "
                    f"exceeds tolerance {CC_TOLERANCE:.2f}"
                )
                logger.warning(f"Registry: {reason}")
                return False, reason
            else:
                logger.info(
                    f"Registry: CC PASS — challenger ({chal_acc:.3f}) "
                    f"vs champion {champ_cid} ({champ_acc:.3f}), "
                    f"gap={acc_gap:+.3f} within tolerance {CC_TOLERANCE:.2f}"
                )
        elif champ_meta is None:
            logger.info(
                f"Registry: no champion found — first-run promotion, "
                f"skipping C/C comparison"
            )

    # All gates passed — atomic promote
    live_m, live_s, live_f = _live_paths(version)
    _atomic_copy(cdir / "models.pkl",       live_m)
    _atomic_copy(cdir / "scaler.pkl",       live_s)
    _atomic_copy(cdir / "feature_cols.pkl", live_f)

    # Update metadata: mark promoted + record C/C context
    meta["promoted"]    = True
    meta["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    if cc_mode:
        champ_meta_snap = get_active_metadata(version)
        if champ_meta_snap:
            meta["cc_champion_id"]       = champ_meta_snap.get("candidate_id")
            meta["cc_champion_test_acc"] = champ_meta_snap.get("test_dir_acc")
            meta["cc_acc_gap"]           = round(test_acc - float(champ_meta_snap.get("test_dir_acc", 0)), 4)
        else:
            meta["cc_champion_id"]       = None   # first model
            meta["cc_acc_gap"]           = None
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    reason = (
        f"PROMOTED {version}/{candidate_id} "
        f"test_dir_acc={test_acc:.3f} signals={signals}"
    )
    if cc_mode and meta.get("cc_acc_gap") is not None:
        reason += f" cc_gap={meta['cc_acc_gap']:+.3f}"
    logger.info(f"Registry: {reason}")
    return True, reason


def rollback(version: str, candidate_id: str) -> bool:
    """
    Restore a specific registry candidate as the live model.

    Usage from retrain_weekly.py CLI:
        python scripts/retrain_weekly.py --rollback v10:20260604_091500_def456

    Returns True on success.
    """
    import joblib

    cdir = _registry_path(version, candidate_id)
    if not cdir.exists():
        logger.error(f"Rollback failed: candidate {version}/{candidate_id} not found")
        return False

    for fname in ("models.pkl", "scaler.pkl", "feature_cols.pkl"):
        if not (cdir / fname).exists():
            logger.error(f"Rollback failed: {fname} missing in {cdir}")
            return False

    # Verify loadable before touching live files
    try:
        joblib.load(cdir / "models.pkl")
    except Exception as e:
        logger.error(f"Rollback failed: models.pkl not loadable — {e}")
        return False

    live_m, live_s, live_f = _live_paths(version)
    _atomic_copy(cdir / "models.pkl",       live_m)
    _atomic_copy(cdir / "scaler.pkl",       live_s)
    _atomic_copy(cdir / "feature_cols.pkl", live_f)

    # Mark as re-promoted in registry metadata
    meta_path = cdir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["promoted"]    = True
            meta["promoted_at"] = datetime.now().isoformat(timespec="seconds")
            meta["rollback"]    = True
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        except Exception:
            pass

    logger.info(f"Registry: rolled back {version} to candidate {candidate_id}")
    return True


def list_versions(version: str, n: int = 10) -> list[dict]:
    """
    Return the n most-recent registry entries for a version, newest first.
    Each entry is the metadata dict plus a 'candidate_id' key.
    """
    if not REGISTRY_DIR.exists():
        return []
    dirs = sorted(
        [d for d in REGISTRY_DIR.iterdir()
         if d.is_dir() and d.name.startswith(f"{version}_")],
        reverse=True,
    )[:n]
    results = []
    for d in dirs:
        cid = d.name[len(version) + 1:]   # strip "v10_" prefix
        meta_path = d / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta.setdefault("candidate_id", cid)
                results.append(meta)
            except Exception:
                results.append({"candidate_id": cid, "error": "metadata unreadable"})
        else:
            results.append({"candidate_id": cid, "error": "no metadata.json"})
    return results


def get_active_metadata(version: str) -> Optional[dict]:
    """
    Return the metadata of the most-recently-promoted candidate for a version.

    Returns None if nothing has been promoted yet, OR if the promoted
    candidate's live model files are missing from disk. The latter is a
    "ghost champion" — a registry record left marked promoted=True after its
    live nifty_v{N}_models.pkl etc. were deleted (e.g. a leaked-pipeline
    model removed during cleanup). Treating it as "no champion" lets the
    next challenger go through first-run promotion (Stage 2 skipped),
    re-creating the missing live files instead of being rejected forever
    against an unbeatable, file-less baseline.
    """
    for entry in list_versions(version, n=50):
        if entry.get("promoted"):
            live_m, live_s, live_f = _live_paths(version)
            if not (live_m.exists() and live_s.exists() and live_f.exists()):
                logger.warning(
                    f"Registry: champion {version}/{entry.get('candidate_id')} "
                    f"is marked promoted (test_dir_acc={entry.get('test_dir_acc')}) "
                    f"but live model files are missing ({live_m.name}) — "
                    f"treating as no champion"
                )
                return None
            return entry
    return None


def _latest_candidate_id(version: str) -> Optional[str]:
    """Return the candidate_id of the most recently saved candidate."""
    if not REGISTRY_DIR.exists():
        return None
    dirs = sorted(
        [d for d in REGISTRY_DIR.iterdir()
         if d.is_dir() and d.name.startswith(f"{version}_")],
        reverse=True,
    )
    if not dirs:
        return None
    return dirs[0].name[len(version) + 1:]


def prune_old_candidates(version: str, keep: int = 14) -> int:
    """
    Delete registry entries older than the `keep` most recent for a version.
    Returns the number of directories deleted.
    """
    if not REGISTRY_DIR.exists():
        return 0
    dirs = sorted(
        [d for d in REGISTRY_DIR.iterdir()
         if d.is_dir() and d.name.startswith(f"{version}_")],
        reverse=True,
    )
    to_delete = dirs[keep:]
    for d in to_delete:
        try:
            shutil.rmtree(d)
            logger.debug(f"Registry: pruned {d.name}")
        except Exception as e:
            logger.warning(f"Registry: prune failed for {d.name}: {e}")
    if to_delete:
        logger.info(f"Registry: pruned {len(to_delete)} old {version} candidates "
                    f"(kept {keep})")
    return len(to_delete)
