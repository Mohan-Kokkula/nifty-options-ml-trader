"""
shadow_model.py — Stage-1 shadow deployment for candidate models.

Runs a candidate model (default: models/sandbox_tb/) on the SAME feature
frame as the production model each cycle and logs both predictions side by
side to data/shadow_model_log.jsonl. Strictly passive:
  - never influences any trading decision
  - no-op if candidate artifacts are missing
  - any exception is swallowed after a debug log
  - adds one predict_proba call (~ms) per 5-min cycle

Disable with SHADOW_MODEL_ENABLED=false in the environment.
Point at a different candidate with SHADOW_MODEL_DIR.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_LOG = _ROOT / "data" / "shadow_model_log.jsonl"

# Gate-1 thresholds (current production NEW config — same gates, so the
# shadow comparison isolates the MODEL, not the gates)
_THR_CALL, _THR_PUT, _MIN_EDGE, _SKIP_CEIL = 0.32, 0.25, 0.05, 0.65

_state = {"loaded": False, "failed": False,
          "models": None, "scaler": None, "fcols": None, "dir": None}


def _load() -> bool:
    if _state["loaded"]:
        return True
    if _state["failed"]:
        return False
    if os.getenv("SHADOW_MODEL_ENABLED", "true").lower() in ("0", "false", "no"):
        _state["failed"] = True
        return False
    try:
        import joblib
        d = Path(os.getenv("SHADOW_MODEL_DIR",
                           str(_ROOT / "models" / "sandbox_tb")))
        if not (d / "models.pkl").exists():
            _state["failed"] = True
            logger.info(f"Shadow model: no artifacts at {d} — shadow disabled")
            return False
        _state["models"] = joblib.load(d / "models.pkl")
        _state["scaler"] = joblib.load(d / "scaler.pkl")
        _state["fcols"] = joblib.load(d / "feature_cols.pkl")
        _state["dir"] = str(d)
        _state["loaded"] = True
        logger.info(f"🕶️ Shadow model loaded from {d} "
                    f"({len(_state['fcols'])} features) — logging only, "
                    f"no decision impact")
        return True
    except Exception as e:
        _state["failed"] = True
        logger.warning(f"Shadow model load failed (disabled): {e}")
        return False


def observe(cycle: int, df_feat, primary_signal: int, primary_proba,
            spot: float) -> None:
    """Predict with the shadow model on the same features; log both."""
    try:
        if not _load():
            return
        row = df_feat.iloc[[-1]]
        X = row.reindex(columns=_state["fcols"], fill_value=0.0).values
        X = np.nan_to_num(X.astype(float), nan=0.0)
        p = np.mean([m.predict_proba(_state["scaler"].transform(X))
                     for m in _state["models"].values()], axis=0)[0]
        c, q, s = float(p[0]), float(p[1]), float(p[2])
        if c >= _THR_CALL and c - q >= _MIN_EDGE and s < _SKIP_CEIL:
            sig = 0
        elif q >= _THR_PUT and q - c >= _MIN_EDGE and s < _SKIP_CEIL:
            sig = 1
        else:
            sig = 2
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cycle": cycle, "spot": round(float(spot), 2),
            "primary_signal": int(primary_signal),
            "primary_proba": [round(float(x), 4) for x in primary_proba],
            "shadow_signal": sig,
            "shadow_proba": [round(c, 4), round(q, 4), round(s, 4)],
            "agree": bool(sig == int(primary_signal)),
            "shadow_dir": _state["dir"],
        }
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if sig != 2 and sig != int(primary_signal):
            logger.info(
                f"🕶️ Shadow DISAGREE cycle #{cycle}: shadow="
                f"{['CALL','PUT','SKIP'][sig]} (C={c:.3f} P={q:.3f}) vs "
                f"primary={['CALL','PUT','SKIP'][int(primary_signal)]}"
            )
    except Exception as e:
        logger.debug(f"Shadow model observe failed (ignored): {e}")
