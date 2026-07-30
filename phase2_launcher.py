r"""
phase2_launcher.py — Production launcher for the full Phase 2 experiment.

Executes all 8 outer folds sequentially with the pre-registered production
settings:
    n_trials = 30
    k_inner  = 3
    seed     = 42

Features
--------
* **Resumable.** A fold is skipped only when its cache is valid
  (source SHA-256s + n_trials + k_inner match), which reuses the check
  already implemented in phase2_xgb_hpo._cache_ok. An interrupted fold
  (partial artifacts on disk) fails the cache test and re-runs from
  scratch.
* **Drift-halt.** Pre-registration manifest is verified at launcher
  start AND before every fold. Any drift aborts immediately.
* **Windows sleep prevention.** Uses SetThreadExecutionState with
  ES_SYSTEM_REQUIRED | ES_CONTINUOUS so the machine will not suspend
  while the launcher process is alive. The display is allowed to blank.
  Linux/VPS should launch under ``systemd-inhibit`` — see the launch
  commands in the module docstring below.
* **Append-only execution log** at ``logs/phase2/execution_log.jsonl``.
  One JSON line per event: ``run_started``, ``fold_started``,
  ``fold_completed``, ``fold_failed``, ``drift_detected``,
  ``run_completed``, ``aggregate_failed``. Survives crashes.
* **Auto-aggregate.** After the fold loop, invokes
  ``phase2_xgb_hpo.aggregate`` which emits the paired block-bootstrap
  CI + Diebold-Mariano + pre-registered H_hpo verdict.

CLI
---
    python phase2_launcher.py                            # standard prod run
    python phase2_launcher.py --dry-run                  # print plan, do nothing
    python phase2_launcher.py --folds 3,4                # subset of folds
    python phase2_launcher.py --aggregate-only           # skip execution
    python phase2_launcher.py --n-trials 30 --k-inner 3 --seed 42

Launch commands
---------------
Windows (foreground with live log tail):
    python phase2_launcher.py 2>&1 | Tee-Object logs\phase2\run.out

Windows (detached, survives shell close):
    Start-Process python phase2_launcher.py `
        -RedirectStandardOutput logs\phase2\run.out `
        -RedirectStandardError  logs\phase2\run.err `
        -WindowStyle Hidden

Linux / VPS (recommended: tmux):
    tmux new -d -s phase2 "python phase2_launcher.py 2>&1 | tee logs/phase2/run.out"
    tmux attach -t phase2                # reattach any time

Linux / VPS (nohup + systemd sleep inhibit):
    nohup systemd-inhibit \\
        --what=sleep:idle \\
        --who=openclaw --why='phase2 hpo' \\
        python phase2_launcher.py > logs/phase2/run.out 2>&1 &

Live monitoring
---------------
    tail -f logs/phase2/run.out                          # human-readable
    tail -f logs/phase2/execution_log.jsonl              # structured events

Resume
------
If a run is killed for any reason, simply re-invoke the same launch
command. Completed folds are detected via cache and skipped; partial
folds are re-run cleanly from scratch.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from phase2_xgb_hpo import (FOLDS, _cache_ok, _current_source,
                              aggregate as run_aggregate, run_one_fold)
from verify_preregistration import PreRegistrationDrift, verify


DEFAULTS = dict(n_trials=30, k_inner=3, seed=42)
EXEC_LOG = ROOT / "logs/phase2/execution_log.jsonl"


# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(event: str, **fields) -> None:
    """Append a single JSON line to the execution log. Atomic on POSIX
    and Windows for the record sizes we emit."""
    rec = {"utc": _now_iso(), "event": event, **fields}
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXEC_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
def _prevent_sleep_windows() -> bool:
    """Prevent system sleep via SetThreadExecutionState.

    ES_SYSTEM_REQUIRED tells Windows the system must stay awake.
    ES_CONTINUOUS makes the state persist until explicitly cleared.
    Display is allowed to blank; the box does not suspend.
    Returns True on success. Silently returns False on non-Windows or
    if the call fails.
    """
    if platform.system() != "Windows":
        return False
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    try:
        r = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return r != 0
    except Exception:
        return False


def _restore_sleep_windows() -> None:
    if platform.system() != "Windows":
        return
    ES_CONTINUOUS = 0x80000000
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


# ---------------------------------------------------------------------------
def _extract_fold_status(fold_idx: int) -> dict:
    """Read summary.json + trials for a completed fold and return the
    per-fold status record persisted to the execution log."""
    fr = ROOT / f"logs/phase2/fold_{fold_idx}"
    out: dict = {"fold": fold_idx}
    try:
        s = json.loads((fr / "summary.json").read_text())
        out["baseline_pf"] = s["baseline"]["pf"]
        out["tuned_pf"] = s["tuned"]["pf"]
        out["delta_pf"] = s["delta_pf"]
        out["delta_net"] = s["delta_net"]
        out["hpo_elapsed_s"] = s["tuned"].get("hpo_elapsed_s")
        out["best_params"] = s["best_params"]
    except Exception as e:
        out["summary_error"] = str(e)

    # best trial + best sortino from trials.parquet (or trials.csv fallback)
    try:
        import numpy as np
        import pandas as pd
        tp = fr / "trials.parquet"
        if not tp.exists():
            tp = fr / "trials.csv"
        if tp.exists():
            if tp.suffix == ".parquet":
                df = pd.read_parquet(tp)
            else:
                df = pd.read_csv(tp)
            if len(df):
                idx_best = df["value"].idxmax()
                best_row = df.loc[idx_best]
                out["best_trial"] = int(best_row["trial"])
                out["best_inner_score"] = float(best_row["value"])
                sortinos = best_row.get("sortinos")
                if isinstance(sortinos, str):
                    sortinos = json.loads(sortinos)
                if isinstance(sortinos, list) and sortinos:
                    finite = [x for x in sortinos if np.isfinite(x)]
                    if finite:
                        out["best_sortino_mean"] = float(np.mean(finite))
                pfs = best_row.get("pfs")
                if isinstance(pfs, str):
                    pfs = json.loads(pfs)
                if isinstance(pfs, list) and pfs:
                    finite = [x for x in pfs if np.isfinite(x)]
                    if finite:
                        out["best_pf_mean"] = float(np.mean(finite))
                out["n_trials_completed"] = int(
                    (df["state"] == "COMPLETE").sum())
    except Exception as e:
        out["trial_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
def _preflight(selected: list[int], n_trials: int, k_inner: int
               ) -> list[tuple[int, str, str]]:
    """Return cache status for every selected fold as (fold, status, reason)."""
    expect = _current_source()
    out = []
    for i in selected:
        fr = ROOT / f"logs/phase2/fold_{i}"
        ok, reason = _cache_ok(fr, expect, n_trials, k_inner)
        out.append((i, "CACHED" if ok else "PENDING", reason))
    return out


def _print_summary_table(results: list[dict]) -> None:
    print("  {:>4} {:>10} {:>11} {:>10} {:>10} {:>13} {:>8}".format(
        "fold", "status", "best_trial", "tuned_PF", "delta_PF",
        "best_Sortino", "wall_s"))
    for r in results:
        st = r.get("status", "OK")
        bt = str(r.get("best_trial", ""))
        pf = r.get("tuned_pf")
        dp = r.get("delta_pf")
        so = r.get("best_sortino_mean")
        wa = r.get("elapsed_wallclock_s") or 0
        pf_s = f"{pf:>10.3f}" if pf is not None else " " * 10
        dp_s = f"{dp:>+10.3f}" if dp is not None else " " * 10
        so_s = f"{so:>13.3f}" if so is not None else " " * 13
        print(f"  {r['fold']:>4} {st:>10} {bt:>11} {pf_s} {dp_s} "
              f"{so_s} {wa:>8.0f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=DEFAULTS["n_trials"])
    ap.add_argument("--k-inner", type=int, default=DEFAULTS["k_inner"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--folds", default="all",
                    help="'all' or comma-separated fold ids (default all).")
    ap.add_argument("--force", action="store_true",
                    help="Ignore fold caches and re-run every fold.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit.")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip execution; only regenerate the aggregate report.")
    args = ap.parse_args()

    if args.aggregate_only:
        print("[aggregate-only] regenerating report ...")
        run_aggregate()
        return

    if args.folds == "all":
        selected = list(range(1, len(FOLDS) + 1))
    else:
        selected = [int(x) for x in args.folds.split(",") if x.strip()]

    run_id = str(uuid.uuid4())[:8]
    (ROOT / "logs/phase2").mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("  Phase 2 launcher (production run)")
    print("=" * 74)
    print(f"  run_id    : {run_id}")
    print(f"  n_trials  : {args.n_trials}")
    print(f"  k_inner   : {args.k_inner}")
    print(f"  seed      : {args.seed}")
    print(f"  folds     : {selected}")
    print(f"  platform  : {platform.platform()}")
    print(f"  exec_log  : {EXEC_LOG}")

    # ---- Pre-flight #1: pre-registration drift check
    print("\n[pre-flight] verifying pre-registration ...")
    try:
        verify()
        print("  OK")
    except PreRegistrationDrift as e:
        print(f"  DRIFT DETECTED:\n{e}")
        _log("drift_detected", run_id=run_id, when="preflight",
             error=str(e))
        sys.exit(2)

    # ---- Pre-flight #2: cache status
    print("\n[pre-flight] cache status ...")
    cache_status = _preflight(selected, args.n_trials, args.k_inner)
    n_cached = sum(1 for _, s, _ in cache_status if s == "CACHED")
    n_pending = len(cache_status) - n_cached
    for i, s, why in cache_status:
        print(f"  fold {i}: {s} ({why})")
    print(f"  -> {n_cached} cached, {n_pending} pending "
          f"({n_pending} will run)")

    # ---- Sleep prevention
    sleep_prevented = _prevent_sleep_windows()
    if platform.system() == "Windows":
        print(f"\n[pre-flight] Windows sleep prevention active: "
              f"{sleep_prevented}")
    elif platform.system() == "Linux":
        print("\n[pre-flight] On Linux, wrap this call with "
              "'systemd-inhibit --what=sleep:idle' (see module docstring).")

    if args.dry_run:
        print("\n(dry-run) exiting without executing folds.")
        return

    # ---- Run
    _log("run_started", run_id=run_id,
         n_trials=args.n_trials, k_inner=args.k_inner, seed=args.seed,
         selected=selected, platform=platform.platform(),
         sleep_prevented=sleep_prevented,
         n_cached_at_start=n_cached, n_pending_at_start=n_pending,
         cache_status=[{"fold": i, "status": s, "reason": r}
                        for i, s, r in cache_status])

    t_run_start = time.time()
    results: list[dict] = []
    try:
        for i in selected:
            a, b = FOLDS[i - 1]

            # Re-check drift before each fold (long runs, human editing risk)
            try:
                verify()
            except PreRegistrationDrift as e:
                print(f"\n[fold {i}] DRIFT DETECTED mid-run; aborting")
                _log("drift_detected", run_id=run_id, when="mid_run",
                     at_fold=i, error=str(e))
                sys.exit(2)

            t_start = time.time()
            print(f"\n[fold {i}] launching {a} -> {b}")
            _log("fold_started", run_id=run_id, fold=i,
                 test_start=str(a), test_end=str(b))

            ran, status = run_one_fold(
                i, a, b,
                n_trials=args.n_trials, k_inner=args.k_inner,
                seed=args.seed, force=args.force, verify_prereg=False,
            )
            elapsed = time.time() - t_start

            if status.startswith("FAILED"):
                rec = {"fold": i, "status": status,
                        "elapsed_wallclock_s": round(elapsed, 1)}
                _log("fold_failed", run_id=run_id, **rec)
                results.append(rec)
                print(f"[fold {i}] FAILED after {elapsed:.0f}s: {status}")
                continue

            fold_status = _extract_fold_status(i)
            fold_status["ran"] = ran
            fold_status["cached_at_check"] = (not ran)
            fold_status["elapsed_wallclock_s"] = round(elapsed, 1)
            fold_status["run_id"] = run_id
            _log("fold_completed", **fold_status)
            results.append(fold_status)
            dp = fold_status.get("delta_pf")
            dp_s = f"{dp:+.3f}" if isinstance(dp, (int, float)) else "nan"
            print(f"[fold {i}] completed in {elapsed:.0f}s "
                  f"(delta_pf={dp_s})")
    finally:
        _restore_sleep_windows()

    total_elapsed = time.time() - t_run_start
    n_ok = sum(1 for r in results if r.get("delta_pf") is not None)
    n_failed = sum(1 for r in results
                    if str(r.get("status", "")).startswith("FAILED"))
    _log("run_completed", run_id=run_id,
         elapsed_s=round(total_elapsed, 1),
         n_folds_attempted=len(selected),
         n_folds_ok=n_ok, n_folds_failed=n_failed)

    print("\n" + "=" * 74)
    print("  Run summary")
    print("=" * 74)
    print(f"  run_id     : {run_id}")
    print(f"  wall-clock : {total_elapsed:.0f}s "
          f"({total_elapsed/3600:.1f}h)")
    print(f"  folds ok   : {n_ok}/{len(selected)}    failed: {n_failed}")
    print()
    _print_summary_table(results)

    # ---- Aggregate
    print("\n[post-run] aggregating ...")
    try:
        run_aggregate()
    except Exception as e:
        print(f"aggregation FAILED: {e}")
        _log("aggregate_failed", run_id=run_id, error=str(e))


if __name__ == "__main__":
    main()
