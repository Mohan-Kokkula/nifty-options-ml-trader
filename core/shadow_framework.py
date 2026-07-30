"""
shadow_framework.py — Production shadow-model comparison framework.

Every prediction cycle, runs a SECOND ("shadow") model on the exact same
feature frame the primary model just used, and records a side-by-side
comparison — direction, confidence, trade decision — to data/shadow/.

Strictly passive, zero effect on live orders:
  - observe() is called AFTER the primary signal/proba/conf are already
    decided and returned; its result is never read by the caller
  - any exception is caught and swallowed (logged at debug level)
  - no-op if SHADOW_FRAMEWORK_ENABLED=false or candidate artifacts missing
  - adds one predict_proba call (~ms) per cycle

Restart safe:
  - predictions are appended to a per-day JSONL file
    (data/shadow/predictions/shadow_YYYY-MM-DD.jsonl) — a restart simply
    resumes appending to today's file
  - health.json is rewritten atomically (tmp + replace) every cycle, so it
    always reflects the latest state regardless of when the process died

Config (env, all optional):
  SHADOW_FRAMEWORK_ENABLED    default true
  SHADOW_FRAMEWORK_MODEL_DIR  default models/htf36
  SHADOW_RETENTION_DAYS       default 90  (used by cleanup_old_files)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
SHADOW_DIR = _ROOT / "data" / "shadow"
PRED_DIR = SHADOW_DIR / "predictions"
REPORTS_DIR = SHADOW_DIR / "reports"
HEALTH_FILE = SHADOW_DIR / "health.json"

# Gate-1 thresholds — identical to core/ml_engine.py's CONFIDENCE_CALL/PUT,
# MIN_EDGE, SKIP_CEIL (the production "NEW gate" direction filter). Used here
# only to derive each model's own direction signal from its raw
# probabilities — this module never modifies any live gate.
THR_CALL, THR_PUT, MIN_EDGE, SKIP_CEIL = 0.32, 0.25, 0.05, 0.65

# Base Gate-2 confidence floors — the ml_only_mode defaults in
# claude_pilot.py (`effective_min_conf = 72 if option_type == "CE" else 58`).
# Used ONLY to derive the "trade_decision" comparison field below.
#
# The live pilot's actual effective_min_conf is further adjusted every cycle
# by ~15 dynamic, stateful rules (regime, momentum, VIX, risk caps, whipsaw,
# time-of-day session floors, ...) that apply identically regardless of which
# model produced the signal. Replicating that stack here would compare bot
# STATE rather than the MODEL, and would risk drifting out of sync with the
# live gates. trade_decision is therefore a fixed-threshold proxy — "would
# this prediction clear Gate-1 + the base Gate-2 floor" — i.e. a model-only
# comparison. It does NOT predict whether either model would actually fire a
# live order on a given cycle.
G2_CALL, G2_PUT = 72.0, 58.0

_DIR_NAME = {0: "CALL", 1: "PUT", 2: "SKIP"}

_state = {"loaded": False, "failed": False,
          "models": None, "scaler": None, "fcols": None, "dir": None}


def is_enabled() -> bool:
    return os.getenv("SHADOW_FRAMEWORK_ENABLED", "true").lower() not in ("0", "false", "no")


def _load() -> bool:
    if _state["loaded"]:
        return True
    if _state["failed"]:
        return False
    if not is_enabled():
        _state["failed"] = True
        return False
    try:
        import joblib
        d = Path(os.getenv("SHADOW_FRAMEWORK_MODEL_DIR",
                           str(_ROOT / "models" / "htf36")))
        if not (d / "models.pkl").exists():
            _state["failed"] = True
            logger.info(f"Shadow framework: no artifacts at {d} — disabled")
            return False
        _state["models"] = joblib.load(d / "models.pkl")
        _state["scaler"] = joblib.load(d / "scaler.pkl")
        _state["fcols"] = joblib.load(d / "feature_cols.pkl")
        _state["dir"] = str(d)
        _state["loaded"] = True
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"\U0001F465 Shadow framework loaded: {d} "
            f"({len(_state['fcols'])} features) — comparison logging only, "
            f"no decision impact"
        )
        return True
    except Exception as e:
        _state["failed"] = True
        logger.warning(f"Shadow framework load failed (disabled): {e}")
        return False


def _signal_from_proba(proba) -> int:
    call_p, put_p, skip_p = float(proba[0]), float(proba[1]), float(proba[2])
    if call_p >= THR_CALL and call_p - put_p >= MIN_EDGE and skip_p < SKIP_CEIL:
        return 0
    if put_p >= THR_PUT and put_p - call_p >= MIN_EDGE and skip_p < SKIP_CEIL:
        return 1
    return 2


def _conf_from_proba(proba, signal: int) -> float:
    """Replicates ml_engine.predict_precomputed's directional confidence
    blend (0.65 * directional_prob + 0.35 * anchor, capped at 0.90), so the
    shadow model's confidence is computed the same way as the primary's."""
    call_p, put_p, skip_p = float(proba[0]), float(proba[1]), float(proba[2])
    if signal == 0:
        dir_p = call_p / max(call_p + put_p, 1e-9)
        anchor = min(1.0, call_p * 2.0)
        return min(0.90, 0.65 * dir_p + 0.35 * anchor)
    if signal == 1:
        dir_p = put_p / max(call_p + put_p, 1e-9)
        anchor = min(1.0, put_p * 2.0)
        return min(0.90, 0.65 * dir_p + 0.35 * anchor)
    return skip_p


def _trade_decision(signal: int, confidence_pct: float) -> bool:
    if signal == 0:
        return confidence_pct >= G2_CALL
    if signal == 1:
        return confidence_pct >= G2_PUT
    return False


def _write_health(record_ok: bool, error: str | None = None) -> None:
    try:
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        snap = {}
        if HEALTH_FILE.exists():
            try:
                snap = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
            except Exception:
                snap = {}
        now = datetime.now()
        snap["last_update"] = now.isoformat(timespec="seconds")
        snap["model_loaded"] = bool(_state["loaded"])
        snap["model_dir"] = _state["dir"]
        snap["n_features"] = len(_state["fcols"]) if _state["fcols"] else None
        snap["total_observations"] = int(snap.get("total_observations", 0)) + (1 if record_ok else 0)
        if record_ok:
            snap["consecutive_errors"] = 0
        else:
            snap["consecutive_errors"] = int(snap.get("consecutive_errors", 0)) + 1
            snap["last_error"] = error
            snap["last_error_ts"] = now.isoformat(timespec="seconds")
        tmp = HEALTH_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, indent=1), encoding="utf-8")
        tmp.replace(HEALTH_FILE)
    except Exception as e:
        logger.debug(f"Shadow framework health write failed: {e}")


def observe(cycle: int, df_feat, primary_signal: int, primary_proba,
            primary_conf: float, spot: float) -> None:
    """Predict with the shadow model on the same features; record the
    direction / confidence / trade-decision comparison vs the primary model
    to data/shadow/predictions/shadow_<date>.jsonl.

    Strictly passive: never raises, return value unused, zero effect on
    live orders. No-op if disabled or candidate artifacts are missing.
    """
    try:
        if not _load():
            return

        row = df_feat.iloc[[-1]]
        X = row.reindex(columns=_state["fcols"], fill_value=0.0).values
        X = np.nan_to_num(X.astype(float), nan=0.0)
        shadow_proba = np.mean(
            [m.predict_proba(_state["scaler"].transform(X))
             for m in _state["models"].values()], axis=0
        )[0]

        shadow_signal = _signal_from_proba(shadow_proba)
        shadow_conf = _conf_from_proba(shadow_proba, shadow_signal)

        primary_signal = int(primary_signal)
        primary_conf_pct = round(float(primary_conf) * 100, 2)
        shadow_conf_pct = round(float(shadow_conf) * 100, 2)

        primary_trade = _trade_decision(primary_signal, primary_conf_pct)
        shadow_trade = _trade_decision(shadow_signal, shadow_conf_pct)

        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cycle": int(cycle),
            "spot": round(float(spot), 2),
            "primary": {
                "model": "production_champion",
                "signal": primary_signal,
                "direction": _DIR_NAME[primary_signal],
                "proba": [round(float(x), 4) for x in primary_proba],
                "confidence": primary_conf_pct,
                "trade_decision": primary_trade,
            },
            "shadow": {
                "model": "htf36",
                "model_dir": _state["dir"],
                "signal": shadow_signal,
                "direction": _DIR_NAME[shadow_signal],
                "proba": [round(float(x), 4) for x in shadow_proba],
                "confidence": shadow_conf_pct,
                "trade_decision": shadow_trade,
            },
            "compare": {
                "direction_match": primary_signal == shadow_signal,
                "confidence_diff": round(shadow_conf_pct - primary_conf_pct, 2),
                "trade_decision_match": primary_trade == shadow_trade,
            },
        }

        day = datetime.now().strftime("%Y-%m-%d")
        path = PRED_DIR / f"shadow_{day}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        _write_health(record_ok=True)

        if not rec["compare"]["direction_match"]:
            logger.debug(
                f"\U0001F465 Shadow framework DISAGREE cycle #{cycle}: "
                f"primary={rec['primary']['direction']}"
                f"({primary_conf_pct:.0f}%) vs "
                f"shadow={rec['shadow']['direction']}({shadow_conf_pct:.0f}%)"
            )

    except Exception as e:
        logger.debug(f"Shadow framework observe failed (ignored): {e}")
        _write_health(record_ok=False, error=str(e))


def health_snapshot() -> dict:
    """Read data/shadow/health.json. Returns {} if it doesn't exist yet."""
    try:
        if HEALTH_FILE.exists():
            return json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def cleanup_old_files(retention_days: int | None = None) -> dict:
    """Delete prediction/report files older than retention_days (parsed from
    the filename date). Pure filesystem scan — restart-safe, no persistent
    state required beyond the files themselves.
    """
    if retention_days is None:
        retention_days = int(os.getenv("SHADOW_RETENTION_DAYS", "90"))
    cutoff = datetime.now().date() - timedelta(days=retention_days)
    removed = []
    for d, prefix in ((PRED_DIR, "shadow_"), (REPORTS_DIR, "report_")):
        if not d.exists():
            continue
        for f in d.glob(f"{prefix}*.json*"):
            try:
                date_str = f.stem.replace(prefix, "")[:10]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date < cutoff:
                    f.unlink()
                    try:
                        removed.append(str(f.relative_to(_ROOT)))
                    except ValueError:
                        removed.append(str(f))
            except Exception:
                continue
    if removed:
        logger.info(
            f"\U0001F9F9 Shadow framework cleanup: removed {len(removed)} "
            f"file(s) older than {retention_days}d"
        )
    return {"retention_days": retention_days, "removed": removed}
