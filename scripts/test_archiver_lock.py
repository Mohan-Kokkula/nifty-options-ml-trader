"""Regression tests for the archiver single-writer lockfile.

Reproduces the production failure observed 2026-08-07: under Docker the
app runs as PID 1, atexit (hence _release_lock) does NOT run on SIGKILL,
so an unclean restart leaves "1" in the lockfile. The next container
start read pid 1, confirmed it was alive -- because it WAS the process
doing the checking -- and refused to start. Both archivers were dead:
data/futures_archive empty, data/oi_archive frozen since Aug 5.

Run standalone:  python scripts/test_archiver_lock.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.futures_archiver as FA  # noqa: E402
import scripts.oi_archiver as OA       # noqa: E402

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def dead_pid():
    """A PID that is almost certainly not running."""
    for p in range(999999, 990000, -1):
        try:
            os.kill(p, 0)
        except ProcessLookupError:
            return p
        except (PermissionError, OSError):
            continue
    return 999999


for mod, label in ((FA, "futures_archiver"), (OA, "oi_archiver")):
    print(f"\n-- {label} --")
    tmp = Path(tempfile.mkdtemp()) / ".archiver.lock"
    orig_lock, orig_out = mod._LOCKFILE, mod.OUTDIR
    mod._LOCKFILE, mod.OUTDIR = tmp, tmp.parent
    try:
        # 1. THE PRODUCTION BUG: lockfile holds our own PID (pid 1 in Docker).
        tmp.write_text(str(os.getpid()))
        check("reclaims a lockfile holding OUR OWN pid (the Docker pid-1 bug)",
              mod._claim_lock() is True)
        check("  ...and takes ownership", tmp.read_text().strip() == str(os.getpid()))

        # 2. A genuinely dead PID must also be reclaimed.
        tmp.write_text(str(dead_pid()))
        check("reclaims a stale lockfile from a dead pid", mod._claim_lock() is True)

        # 3. A DIFFERENT live process must still be respected -- the guard
        #    must not become a no-op. Our parent qualifies on POSIX; on
        #    Windows getppid() may not be signalable, so skip there.
        ppid = os.getppid() if hasattr(os, "getppid") else 0
        alive = False
        if ppid and ppid != os.getpid():
            try:
                os.kill(ppid, 0)
                alive = True
            except Exception:
                alive = False
        if alive:
            tmp.write_text(str(ppid))
            check("still REFUSES when a different live process holds the lock",
                  mod._claim_lock() is False)
        else:
            print("  SKIP  different-live-process case (no signalable parent)")

        # 4. Garbage and empty lockfiles must not wedge the archiver.
        tmp.write_text("not-a-pid")
        check("reclaims a corrupt lockfile", mod._claim_lock() is True)
        tmp.write_text("")
        check("reclaims an empty lockfile", mod._claim_lock() is True)

        # 5. No lockfile at all -> clean claim.
        if tmp.exists():
            tmp.unlink()
        check("claims cleanly when no lockfile exists", mod._claim_lock() is True)
    finally:
        mod._LOCKFILE, mod.OUTDIR = orig_lock, orig_out

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
