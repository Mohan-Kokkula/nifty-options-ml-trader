"""
phase1_finish_avg4.py — Orchestrate the 8-fold walk-forward.

Each fold runs in a fresh Python subprocess (subprocess-per-fold pattern
retained) so memory pressure does not accumulate. v2 upgrades:

  * Source-SHA-256-gated cache reuse. A fold is skipped only when
    its ``manifest.json`` matches the current source hashes AND the
    ablation flag. Any code drift => automatic recompute.
  * Per-fold failures surface (exit codes propagated to the summary);
    aggregation clearly labels missing folds instead of silently
    counting fewer.
  * Aggregate stage emits pooled metrics + 90% block-bootstrap CI on
    PF and WR via ``stat_utils``.

CLI:
    python phase1_finish_avg4.py [--no-lookahead-weights]
                                    [--folds all|1,3,7]
                                    [--force] [--skip-aggregate]

Backward compatibility:
    ``python phase1_finish_avg4.py`` with no args reproduces the
    original behaviour: run every missing fold, aggregate at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FOLDS = [
    (date(2024, 7, 1),  date(2024, 10, 1)),
    (date(2024, 10, 1), date(2025, 1, 1)),
    (date(2025, 1, 1),  date(2025, 4, 1)),
    (date(2025, 4, 1),  date(2025, 7, 1)),
    (date(2025, 7, 1),  date(2025, 10, 1)),
    (date(2025, 10, 1), date(2026, 1, 1)),
    (date(2026, 1, 1),  date(2026, 4, 1)),
    (date(2026, 4, 1),  date(2026, 5, 1)),
]

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_multibrain.py",
    "backtest_multibrain_v2.py",
    "backtest_options.py",
    "phase1_run_single_fold.py",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_source_manifest() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED if (ROOT / f).exists()}


def _cache_is_valid(fold_root: Path, expect_source: dict, ablate: bool
                    ) -> tuple[bool, str]:
    m_path = fold_root / "manifest.json"
    t_path = fold_root / "trades.csv"
    if not m_path.exists():
        return False, "no manifest.json"
    if not t_path.exists():
        return False, "no trades.csv"
    try:
        m = json.loads(m_path.read_text())
    except Exception as e:
        return False, f"unreadable manifest ({e})"
    if bool(m.get("ablate_lookahead_weights", False)) != bool(ablate):
        return False, f"ablation flag differs ({m.get('ablate_lookahead_weights')} vs {ablate})"
    src = m.get("source_sha256", {})
    for f, h in expect_source.items():
        if src.get(f) != h:
            return False, f"source SHA changed for {f}"
    return True, "match"


def run_one_fold(fold_idx: int, a_str: str, b_str: str,
                 ablate: bool, force: bool) -> tuple[bool, str]:
    """Returns (ran, status)."""
    fold_root = ROOT / f"logs/phase1/fold_{fold_idx}"
    expect_source = _current_source_manifest()
    if not force:
        ok, reason = _cache_is_valid(fold_root, expect_source, ablate)
        if ok:
            print(f"[Fold {fold_idx}] cache HIT ({reason}); skipping")
            return False, "cached"
        print(f"[Fold {fold_idx}] cache MISS ({reason}); recomputing")
    else:
        print(f"[Fold {fold_idx}] --force; recomputing")

    cmd = [sys.executable, "-u", "phase1_run_single_fold.py",
           str(fold_idx), a_str, b_str, "--verify-prereg"]
    if ablate:
        cmd.append("--no-lookahead-weights")
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        return False, f"FAILED exit={e.returncode}"
    return True, "ran"


def aggregate() -> None:
    import numpy as np
    import pandas as pd

    from stat_utils import (block_bootstrap_ci, max_drawdown, profit_factor,
                              sharpe as sharpe_fn, win_rate)

    print("\n" + "=" * 78)
    print("  Phase 1 v2 — 8-fold walk-forward aggregate")
    print("=" * 78)

    fold_streams: dict[int, np.ndarray] = {}
    per_fold: list[dict] = []
    for i, (a, b) in enumerate(FOLDS, 1):
        p = ROOT / f"logs/phase1/fold_{i}/trades.csv"
        if not p.exists():
            per_fold.append({"fold": i, "status": "MISSING"})
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            per_fold.append({"fold": i, "status": "UNREADABLE"})
            continue
        if df.empty or "net_option" not in df.columns:
            per_fold.append({"fold": i, "status": "EMPTY", "n": 0})
            continue
        pnl = df["net_option"].values.astype(float)
        fold_streams[i] = pnl
        per_fold.append({
            "fold": i, "test_start": str(a), "test_end": str(b),
            "n": int(len(pnl)),
            "pf": float(profit_factor(pnl)) if len(pnl) else float("nan"),
            "net": float(pnl.sum()),
            "wr_pct": float(win_rate(pnl) * 100) if len(pnl) else float("nan"),
            "max_dd": float(max_drawdown(pnl)) if len(pnl) else 0.0,
        })

    if not fold_streams:
        print("  No fold data.")
        return

    all_pnl = np.concatenate(list(fold_streams.values()))
    pooled_pf = float(profit_factor(all_pnl))
    pooled_wr = float(win_rate(all_pnl))
    pooled_dd = float(max_drawdown(all_pnl))
    try:
        pooled_sharpe = float(sharpe_fn(all_pnl))
    except Exception:
        pooled_sharpe = float("nan")

    print(f"  Pooled trades   : {len(all_pnl)}")
    print(f"  Pooled PF       : {pooled_pf:.3f}")
    print(f"  Pooled WR       : {pooled_wr*100:.1f}%")
    print(f"  Pooled Net Rs.  : {all_pnl.sum():+,.0f}")
    print(f"  Pooled MaxDD    : {pooled_dd:+,.0f}")
    print(f"  Sharpe          : {pooled_sharpe:.3f}")

    ci_pf = block_bootstrap_ci(fold_streams, profit_factor,
                                n_resamples=10_000, ci_level=0.90, seed=42,
                                stat_name="profit_factor")
    ci_wr = block_bootstrap_ci(fold_streams, win_rate,
                                n_resamples=10_000, ci_level=0.90, seed=42,
                                stat_name="win_rate")
    verdict = "ABOVE 1.00" if ci_pf.lower > 1.0 else "not above 1.00"
    print(f"\n  90% block-bootstrap CI (block = fold):")
    print(f"    PF : [{ci_pf.lower:.3f}, {ci_pf.upper:.3f}]  "
          f"(LB90 vs 1.00: {verdict})")
    print(f"    WR : [{ci_wr.lower*100:.1f}%, {ci_wr.upper*100:.1f}%]")

    print("\n  Per-fold breakdown:")
    for r in per_fold:
        status = r.get("status")
        if status in ("MISSING", "UNREADABLE", "EMPTY"):
            print(f"    Fold {r['fold']}: {status}")
        else:
            print(f"    Fold {r['fold']}: n={r['n']:>4} PF={r['pf']:.3f} "
                  f"Net=Rs.{r['net']:+,.0f} WR={r['wr_pct']:.1f}%")

    agg = {
        "pooled": {
            "n": int(len(all_pnl)),
            "pf": pooled_pf,
            "wr": pooled_wr,
            "net": float(all_pnl.sum()),
            "max_dd": pooled_dd,
            "sharpe": pooled_sharpe,
        },
        "block_bootstrap_ci_90": {
            "profit_factor": ci_pf.to_dict(),
            "win_rate": ci_wr.to_dict(),
        },
        "per_fold": per_fold,
    }
    (ROOT / "logs/phase1").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs/phase1/aggregate.json", "w") as fh:
        json.dump(agg, fh, indent=2)
    print("\n  wrote logs/phase1/aggregate.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lookahead-weights", action="store_true",
                    dest="ablate")
    ap.add_argument("--folds", default="all",
                    help="'all' or comma-separated 1-indexed fold ids.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore cache and re-run.")
    ap.add_argument("--skip-aggregate", action="store_true")
    args = ap.parse_args()

    (ROOT / "logs/phase1").mkdir(parents=True, exist_ok=True)

    if args.folds == "all":
        selected = list(range(1, len(FOLDS) + 1))
    else:
        selected = [int(x) for x in args.folds.split(",") if x.strip()]

    statuses: list[tuple[int, str]] = []
    for i in selected:
        a, b = FOLDS[i - 1]
        ran, status = run_one_fold(i, a.isoformat(), b.isoformat(),
                                    ablate=args.ablate, force=args.force)
        statuses.append((i, status))

    print("\n  Fold run summary:")
    for i, s in statuses:
        print(f"    Fold {i}: {s}")

    if not args.skip_aggregate:
        aggregate()


if __name__ == "__main__":
    main()
