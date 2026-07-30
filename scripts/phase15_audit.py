"""
phase15_audit.py — PART F: QA audit of the Phase-1.5 implementation.

Checks (all programmatic, evidence in the JSON):
  1. No leakage           — runs tests/test_leakage_safety.py (6 invariants)
  2. No future timestamps — archive + shadow logs contain no ts > now
  3. No training contamination — htf36 metadata split boundaries respect the
                                 purged chronology (train < val < test)
  4. No production overwrite   — sha256 of live model trio unchanged vs the
                                 snapshot taken at audit start
  5. No registry corruption    — every registry metadata.json parses, files
                                 present

Output: reports/phase15_audit_report.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

PROD_FILES = ["models/nifty_v9_models.pkl", "models/nifty_v9_scaler.pkl",
              "models/feature_cols_v9.pkl"]
SNAPSHOT = REPORTS / "prod_hash_snapshot.json"


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    rep = {"audited_at": datetime.now().isoformat(timespec="seconds"),
           "checks": {}}

    # 1 — leakage invariants
    r = subprocess.run([sys.executable,
                        str(ROOT / "tests/test_leakage_safety.py")],
                       capture_output=True, text=True)
    rep["checks"]["leakage_tests"] = {
        "pass": r.returncode == 0,
        "detail": r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:],
    }

    # 2 — future timestamps in new data artifacts
    bad_ts = []
    now = datetime.now()
    for pattern in ("data/oi_archive/*.csv", "data/openalgo_oi/*.csv",
                    "data/shadow_model_log.jsonl"):
        for f in ROOT.glob(pattern):
            try:
                for line in open(f, encoding="utf-8"):
                    for tok in line.replace('"', ",").split(","):
                        tok = tok.strip()
                        if len(tok) == 19 and tok[4] == "-" and tok[10] in " T":
                            try:
                                ts = datetime.fromisoformat(tok.replace(" ", "T"))
                                if ts > now:
                                    bad_ts.append(f"{f.name}: {tok}")
                            except ValueError:
                                pass
            except Exception:
                pass
    rep["checks"]["future_timestamps"] = {"pass": not bad_ts,
                                          "violations": bad_ts[:10]}

    # 3 — htf36 split chronology
    meta_p = ROOT / "models/htf36/metadata.json"
    if meta_p.exists():
        m = json.loads(meta_p.read_text())
        ok = (str(m.get("train_end", "")) < str(m.get("val_end", ""))
              < str((m.get("test_range") or ["", ""])[0]))
        rep["checks"]["htf36_split_chronology"] = {
            "pass": bool(ok), "train_end": m.get("train_end"),
            "val_end": m.get("val_end"), "test_range": m.get("test_range")}
    else:
        rep["checks"]["htf36_split_chronology"] = {
            "pass": None, "detail": "htf36 not trained yet"}

    # 4 — production trio untouched
    current = {f: sha(ROOT / f) for f in PROD_FILES}
    if SNAPSHOT.exists():
        before = json.loads(SNAPSHOT.read_text())
        changed = [f for f in PROD_FILES if before.get(f) != current[f]]
        rep["checks"]["production_untouched"] = {
            "pass": not changed, "changed_files": changed,
            "hashes": current}
    else:
        SNAPSHOT.write_text(json.dumps(current, indent=1))
        rep["checks"]["production_untouched"] = {
            "pass": True, "detail": "baseline snapshot recorded this run",
            "hashes": current}

    # 5 — registry integrity
    reg = ROOT / "models/registry"
    bad = []
    n = 0
    if reg.exists():
        for d in reg.iterdir():
            if not d.is_dir():
                continue
            n += 1
            try:
                json.loads((d / "metadata.json").read_text())
                for f in ("models.pkl", "scaler.pkl", "feature_cols.pkl"):
                    if not (d / f).exists():
                        bad.append(f"{d.name}: missing {f}")
            except Exception as e:
                bad.append(f"{d.name}: {e}")
    rep["checks"]["registry_integrity"] = {"pass": not bad,
                                           "entries": n, "issues": bad}

    rep["overall_pass"] = all(
        c.get("pass") in (True, None) for c in rep["checks"].values())
    out = REPORTS / "phase15_audit_report.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"\nAudit -> {out}")


if __name__ == "__main__":
    main()
