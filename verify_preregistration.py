"""
verify_preregistration.py — Runtime integrity check imported by every phase.

Every downstream phase (1, 2, 3, 4, 5, 6, 8, 9, 10, 11) MUST call verify() at
startup. Verify recomputes SHA-256 of every hashed file and compares against
the active pre-registration. Any drift raises PreRegistrationDrift and the
phase must abort. This makes it impossible for a phase to silently run
against a codebase that has changed since the protocol was frozen.

Usage inside a phase:
    from verify_preregistration import verify
    verify()  # or verify(allow=["docs/README.md"]) if a whitelist is needed

Standalone:
    python verify_preregistration.py             # exits 0 if clean, 2 if drift
    python verify_preregistration.py --report    # also prints per-file diff
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "pre_registration" / "preregistration_active.json"


class PreRegistrationDrift(RuntimeError):
    """Raised when code or data hashes differ from the frozen manifest."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _diff(manifest: dict, kind: str, allow: set) -> list:
    problems = []
    for rel, meta in manifest.items():
        if rel in allow:
            continue
        p = ROOT / rel
        expected = meta.get("sha256")
        if expected is None:
            # File was missing at freeze time; still missing is OK.
            if p.exists():
                problems.append((kind, rel, "appeared_since_freeze"))
            continue
        if not p.exists():
            problems.append((kind, rel, "missing"))
            continue
        actual = _sha256(p)
        if actual != expected:
            problems.append((kind, rel, f"sha256 {expected[:10]}... -> {actual[:10]}..."))
    return problems


def verify(allow: list | None = None, raise_on_drift: bool = True) -> dict:
    """Verify current on-disk state against the active pre-registration.

    Returns dict with keys:
      ok: bool
      problems: list of (kind, path, reason)
      frozen_at: iso timestamp of the pre-registration
    Raises PreRegistrationDrift if drift detected and raise_on_drift is True.
    """
    if not ACTIVE.exists():
        raise PreRegistrationDrift(
            "No active pre-registration. Run phase0_pre_register.py first.")
    doc = json.loads(ACTIVE.read_text())
    allow_set = set(allow or [])
    problems = []
    problems += _diff(doc["code_manifest"], "code", allow_set)
    problems += _diff(doc["data_manifest"], "data", allow_set)
    result = {"ok": len(problems) == 0,
              "problems": problems,
              "frozen_at": doc["frozen_at_utc"]}
    if problems and raise_on_drift:
        msg = ["Pre-registration drift detected:"]
        for kind, rel, reason in problems:
            msg.append(f"  [{kind}] {rel}: {reason}")
        msg.append("")
        msg.append("Either revert the change, or run:")
        msg.append("  python phase0_pre_register.py --supersede")
        msg.append("to lock in the new state as a NEW pre-registration.")
        raise PreRegistrationDrift("\n".join(msg))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="Print per-file diff even on success.")
    args = ap.parse_args()
    try:
        res = verify(raise_on_drift=False)
    except PreRegistrationDrift as e:
        print(str(e))
        sys.exit(2)
    if res["ok"]:
        print(f"OK. Pre-registration frozen at {res['frozen_at']}. No drift.")
        sys.exit(0)
    print(f"DRIFT since {res['frozen_at']}:")
    for kind, rel, reason in res["problems"]:
        print(f"  [{kind}] {rel}: {reason}")
    sys.exit(2)


if __name__ == "__main__":
    main()
