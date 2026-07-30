"""
phase2_xgb_hpo.py — Orchestrate + aggregate Phase 2 (XGB-only nested HPO).

Runs each of the 8 outer folds in an isolated Python subprocess (same
pattern as Phase 1). After all folds complete, computes the pre-registered
statistical comparison of tuned vs baseline via ``stat_utils``:

    - paired block-bootstrap 90% CI on delta_PF (block = outer fold)
    - Diebold-Mariano test on per-trade P&L differential
      (bars where BOTH configs took a position)

Writes:
    logs/phase2/summary.json        — machine-readable aggregate
    logs/phase2/comparison.txt      — human-readable report

CLI:
    python phase2_xgb_hpo.py                             # all 8 folds, defaults
    python phase2_xgb_hpo.py --folds 1                   # single fold
    python phase2_xgb_hpo.py --n-trials 50 --k-inner 3
    python phase2_xgb_hpo.py --smoke-test                # tiny run (pipeline check)
    python phase2_xgb_hpo.py --skip-aggregate            # for pipeline chaining
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
    "backtest_options.py",
    "phase2_run_single_fold.py",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_source() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED if (ROOT / f).exists()}


def _cache_ok(fold_root: Path, expect: dict, n_trials: int, k_inner: int
              ) -> tuple[bool, str]:
    m_path = fold_root / "manifest.json"
    for req in ("baseline_trades.csv", "tuned_trades.csv",
                "baseline_metrics.json", "tuned_metrics.json",
                "summary.json", "best_params.json"):
        if not (fold_root / req).exists():
            return False, f"missing {req}"
    if not m_path.exists():
        return False, "no manifest.json"
    try:
        m = json.loads(m_path.read_text())
    except Exception as e:
        return False, f"unreadable manifest ({e})"
    if int(m.get("n_trials", 0)) != n_trials:
        return False, f"n_trials differs ({m.get('n_trials')} vs {n_trials})"
    if int(m.get("k_inner", 0)) != k_inner:
        return False, f"k_inner differs ({m.get('k_inner')} vs {k_inner})"
    src = m.get("source_sha256", {})
    for f, h in expect.items():
        if src.get(f) != h:
            return False, f"source SHA changed for {f}"
    return True, "ok"


def run_one_fold(i: int, a: date, b: date, n_trials: int, k_inner: int,
                 seed: int, force: bool, verify_prereg: bool
                 ) -> tuple[bool, str]:
    fold_root = ROOT / f"logs/phase2/fold_{i}"
    expect = _current_source()
    if not force:
        ok, why = _cache_ok(fold_root, expect, n_trials, k_inner)
        if ok:
            print(f"[Fold {i}] cache HIT ({why}); skipping")
            return False, "cached"
        print(f"[Fold {i}] cache MISS ({why}); recomputing")
    else:
        print(f"[Fold {i}] --force; recomputing")

    cmd = [sys.executable, "-u", "phase2_run_single_fold.py",
           str(i), a.isoformat(), b.isoformat(),
           "--n-trials", str(n_trials),
           "--k-inner", str(k_inner),
           "--seed", str(seed)]
    if verify_prereg:
        cmd.append("--verify-prereg")
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        return False, f"FAILED exit={e.returncode}"
    return True, "ran"


# ---------------------------------------------------------------------------
def aggregate() -> None:
    import numpy as np
    import pandas as pd
    from stat_utils import (BootstrapCI, DMResult, diebold_mariano,
                              max_drawdown, paired_block_bootstrap_ci,
                              profit_factor, sharpe as sharpe_fn, sortino,
                              win_rate)

    print("\n" + "=" * 78)
    print("  Phase 2 — XGB HPO aggregate")
    print("=" * 78)

    baseline_streams: dict[int, np.ndarray] = {}
    tuned_streams: dict[int, np.ndarray] = {}
    per_fold: list[dict] = []

    for i, (a, b) in enumerate(FOLDS, 1):
        fr = ROOT / f"logs/phase2/fold_{i}"
        if not (fr / "summary.json").exists():
            per_fold.append({"fold": i, "status": "MISSING"})
            continue
        try:
            summ = json.loads((fr / "summary.json").read_text())
            b_df = pd.read_csv(fr / "baseline_trades.csv")
            t_df = pd.read_csv(fr / "tuned_trades.csv")
        except Exception as e:
            per_fold.append({"fold": i, "status": f"UNREADABLE {e}"})
            continue

        b_pnl = (b_df["net_option"].values.astype(float)
                 if "net_option" in b_df.columns else np.array([]))
        t_pnl = (t_df["net_option"].values.astype(float)
                 if "net_option" in t_df.columns else np.array([]))
        baseline_streams[i] = b_pnl
        tuned_streams[i] = t_pnl
        per_fold.append({
            "fold": i,
            "test_start": str(a), "test_end": str(b),
            "baseline": {"n": int(len(b_pnl)),
                          "pf": summ["baseline"]["pf"],
                          "net": summ["baseline"]["net"]},
            "tuned": {"n": int(len(t_pnl)),
                       "pf": summ["tuned"]["pf"],
                       "net": summ["tuned"]["net"]},
            "delta_pf": summ["delta_pf"],
            "delta_net": summ["delta_net"],
            "best_params": summ["best_params"],
            "hpo_elapsed_s": summ["tuned"].get("hpo_elapsed_s"),
        })

    if not baseline_streams:
        print("  No completed folds found.")
        return

    # ---------- Pooled point estimates ----------
    all_b = np.concatenate(list(baseline_streams.values()))
    all_t = np.concatenate(list(tuned_streams.values()))
    b_pf = float(profit_factor(all_b)) if len(all_b) else float("nan")
    t_pf = float(profit_factor(all_t)) if len(all_t) else float("nan")
    delta_pf_pooled = t_pf - b_pf

    print(f"  Pooled trades   : baseline={len(all_b)}, tuned={len(all_t)}")
    print(f"  Pooled PF       : baseline={b_pf:.3f}   tuned={t_pf:.3f}   "
          f"delta={delta_pf_pooled:+.3f}")
    print(f"  Pooled Net Rs.  : baseline={all_b.sum():+,.0f}   "
          f"tuned={all_t.sum():+,.0f}   "
          f"delta={all_t.sum() - all_b.sum():+,.0f}")

    # ---------- Paired block bootstrap on delta PF ----------
    def delta_pf_stat(a: np.ndarray, b: np.ndarray) -> float:
        pfa = profit_factor(a) if len(a) else float("nan")
        pfb = profit_factor(b) if len(b) else float("nan")
        if not (np.isfinite(pfa) and np.isfinite(pfb)):
            return float("nan")
        return pfa - pfb

    ci: BootstrapCI = paired_block_bootstrap_ci(
        tuned_streams, baseline_streams, delta_pf_stat,
        n_resamples=10_000, ci_level=0.90, seed=42, stat_name="delta_pf",
    )
    print(f"\n  Paired block-bootstrap 90% CI on delta_PF "
          f"(block = outer fold):")
    print(f"    delta_PF point : {ci.point_estimate:+.3f}")
    print(f"    90% CI         : [{ci.lower:+.3f}, {ci.upper:+.3f}]")
    print(f"    CI half-width  : {ci.half_width():.3f}")

    # ---------- Diebold-Mariano ----------
    # We need aligned per-bar P&L for A and B. Merge by entry timestamp
    # per fold; only bars where BOTH configs took a position contribute.
    dm_pieces_a: list[np.ndarray] = []
    dm_pieces_b: list[np.ndarray] = []
    for i in baseline_streams:
        fr = ROOT / f"logs/phase2/fold_{i}"
        try:
            bdf = pd.read_csv(fr / "baseline_trades.csv")
            tdf = pd.read_csv(fr / "tuned_trades.csv")
        except Exception:
            continue
        if "time" not in bdf.columns or "time" not in tdf.columns:
            continue
        b_by = bdf.set_index("time")["net_option"]
        t_by = tdf.set_index("time")["net_option"]
        common = b_by.index.intersection(t_by.index)
        if len(common) < 10:
            continue
        dm_pieces_a.append(t_by.loc[common].values.astype(float))
        dm_pieces_b.append(b_by.loc[common].values.astype(float))

    if dm_pieces_a and dm_pieces_b:
        a_all = np.concatenate(dm_pieces_a)
        b_all = np.concatenate(dm_pieces_b)
        try:
            dm: DMResult = diebold_mariano(
                a_all, b_all, alternative="greater",
                lag=5, ci_level=0.90)
            print(f"\n  Diebold-Mariano (tuned P&L > baseline P&L, "
                  f"HAC-Newey-West lag=5):")
            print(f"    n aligned bars : {dm.n}")
            print(f"    statistic      : {dm.statistic:+.3f}")
            print(f"    p-value        : {dm.pvalue:.4f}")
            print(f"    mean diff      : Rs.{dm.mean_loss_diff:+.2f}   "
                  f"90% CI [Rs.{dm.ci_lower:+.2f}, Rs.{dm.ci_upper:+.2f}]")
        except Exception as e:
            print(f"\n  Diebold-Mariano failed: {e}")
            dm = None
    else:
        print("\n  Diebold-Mariano skipped (no aligned bars).")
        dm = None

    # ---------- Per-fold breakdown ----------
    print("\n  Per-fold breakdown:")
    print(f"    {'fold':>4} {'baseline_PF':>12} {'tuned_PF':>10} "
          f"{'delta_PF':>10} {'delta_net Rs':>16} {'hpo_s':>8}")
    for r in per_fold:
        if r.get("status"):
            print(f"    {r['fold']:>4}  {r['status']}")
            continue
        print(f"    {r['fold']:>4} {r['baseline']['pf']:>12.3f} "
              f"{r['tuned']['pf']:>10.3f} {r['delta_pf']:>+10.3f} "
              f"{r['delta_net']:>+16,.0f} "
              f"{(r.get('hpo_elapsed_s') or 0):>8.0f}")

    # ---------- Persist summary ----------
    summary = {
        "n_folds_completed": len(baseline_streams),
        "pooled": {
            "baseline_n": int(len(all_b)),
            "tuned_n": int(len(all_t)),
            "baseline_pf": b_pf,
            "tuned_pf": t_pf,
            "delta_pf": delta_pf_pooled,
            "baseline_net": float(all_b.sum()),
            "tuned_net": float(all_t.sum()),
            "delta_net": float(all_t.sum() - all_b.sum()),
        },
        "paired_block_bootstrap_ci_90": ci.to_dict(),
        "diebold_mariano": dm.to_dict() if dm is not None else None,
        "per_fold": per_fold,
    }
    with open(ROOT / "logs/phase2/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---------- Human-readable report ----------
    material_effect = 0.08  # H_hpo pre-registered effect size
    accept = ci.lower > material_effect
    dm_accept = (dm is not None and dm.pvalue < 0.02)
    verdict_lines = [
        "",
        "  Verdict against pre-registered H_hpo:",
        f"    effect threshold  : LB90(delta_PF) > +{material_effect}",
        f"    observed LB90     : {ci.lower:+.3f}",
        f"    LB90 > threshold  : {'YES' if accept else 'NO'}",
        f"    DM_p (one-sided)  : {dm.pvalue:.4f}" if dm else "    DM_p              : N/A",
        f"    DM_p < 0.02 (Holm): {'YES' if dm_accept else 'NO'}" if dm else "",
    ]
    print("\n".join(v for v in verdict_lines if v))
    accept_line = ("REJECT H0_hpo (tuned materially better)"
                   if (accept and dm_accept)
                   else "FAIL TO REJECT H0_hpo")
    print(f"    Decision          : {accept_line}")

    with open(ROOT / "logs/phase2/comparison.txt", "w") as fh:
        fh.write("Phase 2 XGB HPO comparison report\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"n_folds_completed : {len(baseline_streams)}/{len(FOLDS)}\n")
        fh.write(f"pooled baseline PF: {b_pf:.3f}   tuned PF: {t_pf:.3f}   "
                 f"delta: {delta_pf_pooled:+.3f}\n")
        fh.write(f"paired 90% CI     : [{ci.lower:+.3f}, {ci.upper:+.3f}]  "
                 f"(point {ci.point_estimate:+.3f})\n")
        if dm is not None:
            fh.write(f"DM p-value (grtr) : {dm.pvalue:.4f}\n")
        fh.write(f"effect threshold  : +{material_effect}\n")
        fh.write(f"decision          : {accept_line}\n\n")
        fh.write("Per-fold breakdown:\n")
        for r in per_fold:
            if r.get("status"):
                fh.write(f"  fold {r['fold']}: {r['status']}\n")
                continue
            fh.write(f"  fold {r['fold']}: baseline PF={r['baseline']['pf']:.3f} "
                     f"tuned PF={r['tuned']['pf']:.3f} "
                     f"delta={r['delta_pf']:+.3f} "
                     f"net_delta=Rs.{r['delta_net']:+,.0f}\n")
            fh.write(f"           best_params={r['best_params']}\n")
    print(f"\n  wrote logs/phase2/summary.json")
    print(f"  wrote logs/phase2/comparison.txt")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", default="all",
                    help="'all' or comma-separated 1-indexed fold ids.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore cache and re-run.")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true",
                    help="Enable pre-registration drift check in child folds.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Very small run (n_trials=3, k_inner=2) for "
                         "pipeline verification.")
    args = ap.parse_args()

    (ROOT / "logs/phase2").mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.n_trials = 3
        args.k_inner = 2

    if args.folds == "all":
        selected = list(range(1, len(FOLDS) + 1))
    else:
        selected = [int(x) for x in args.folds.split(",") if x.strip()]

    statuses: list[tuple[int, str]] = []
    for i in selected:
        a, b = FOLDS[i - 1]
        ran, st = run_one_fold(
            i, a, b,
            n_trials=args.n_trials, k_inner=args.k_inner, seed=args.seed,
            force=args.force, verify_prereg=args.verify_prereg,
        )
        statuses.append((i, st))

    print("\n  Fold run summary:")
    for i, s in statuses:
        print(f"    Fold {i}: {s}")

    if not args.skip_aggregate:
        aggregate()


if __name__ == "__main__":
    main()
