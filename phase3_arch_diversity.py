"""
phase3_arch_diversity.py — Orchestrator + aggregator for Phase 3.

Iterates over (brain × outer_fold), dispatching each to a fresh
subprocess (same isolation pattern as Phase 1/2). After the fold loop,
computes per-brain block-bootstrap CIs, pairwise Diebold-Mariano tests
between brains, and a Hansen SPA across brains (per-fold PF as the
per-period performance measure).

Statistical primitives are reused from ``stat_utils``. This file does
NOT add any new statistical methods.

CLI
---
    python phase3_arch_diversity.py                     # lgb,cat,mlp x 8 folds
    python phase3_arch_diversity.py --brains lgb,cat    # subset
    python phase3_arch_diversity.py --folds 1,2         # subset of folds
    python phase3_arch_diversity.py --hpo               # nested Optuna per (brain, fold)
    python phase3_arch_diversity.py --aggregate-only    # only regenerate report
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Registered non-GBDT families for the H_arch verdict.
# GBDT: xgb, lgb, cat.  Non-GBDT so far: mlp.  Future non-GBDT (kernel
# SVM, FT-Transformer, TabPFN, temporal CNN, HMM) should be added here.
GBDT_BRAINS = {"xgb", "lgb", "cat"}
NON_GBDT_BRAINS = {"mlp"}

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

DEFAULT_BRAINS = ["lgb", "cat", "mlp"]

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase3_run_single_fold.py",
    "brains/__init__.py",
    "brains/_base.py",
    "brains/_hpo.py",
    "brains/xgb_adapter.py",
    "brains/lgb_adapter.py",
    "brains/cat_adapter.py",
    "brains/mlp_adapter.py",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_source() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED
            if (ROOT / f).exists()}


def _cache_ok(fold_root: Path, expect: dict, hpo: bool, n_trials: int,
              k_inner: int) -> tuple[bool, str]:
    required = ["trades.csv", "trade_pnl.csv", "metrics.json",
                "manifest.json", "scaler.pkl", "model.pkl"]
    # predictions file: parquet OR csv
    for req in required:
        if not (fold_root / req).exists():
            return False, f"missing {req}"
    if not ((fold_root / "predictions.parquet").exists()
            or (fold_root / "predictions.csv").exists()):
        return False, "missing predictions"
    try:
        m = json.loads((fold_root / "manifest.json").read_text())
    except Exception as e:
        return False, f"unreadable manifest ({e})"
    if bool(m.get("hpo_enabled", False)) != hpo:
        return False, f"hpo flag differs"
    if hpo:
        if int(m.get("n_trials", 0)) != n_trials:
            return False, f"n_trials differs"
        if int(m.get("k_inner", 0)) != k_inner:
            return False, f"k_inner differs"
    src = m.get("source_sha256", {})
    for f, h in expect.items():
        if src.get(f) != h:
            return False, f"source SHA changed for {f}"
    return True, "ok"


def run_one(brain_name: str, i: int, a: date, b: date, seed: int,
            hpo: bool, n_trials: int, k_inner: int,
            force: bool, verify_prereg: bool) -> tuple[bool, str]:
    fold_root = ROOT / f"logs/phase3/{brain_name}/fold_{i}"
    expect = _current_source()
    if not force:
        ok, why = _cache_ok(fold_root, expect, hpo, n_trials, k_inner)
        if ok:
            print(f"[{brain_name}/fold {i}] cache HIT; skipping ({why})")
            return False, "cached"
        print(f"[{brain_name}/fold {i}] cache MISS ({why}); recomputing")
    cmd = [sys.executable, "-u", "phase3_run_single_fold.py",
           brain_name, str(i), a.isoformat(), b.isoformat(),
           "--seed", str(seed)]
    if hpo:
        cmd += ["--hpo", "--n-trials", str(n_trials),
                "--k-inner", str(k_inner)]
    if verify_prereg:
        cmd.append("--verify-prereg")
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        return False, f"FAILED exit={e.returncode}"
    return True, "ran"


# ---------------------------------------------------------------------------
def aggregate(brains: list[str]) -> None:
    import numpy as np
    import pandas as pd
    from stat_utils import (block_bootstrap_ci, diebold_mariano, hansen_spa,
                              max_drawdown, paired_block_bootstrap_ci,
                              profit_factor, sharpe as sharpe_fn,
                              sortino as sortino_fn, white_reality_check,
                              win_rate)

    print("\n" + "=" * 78)
    print("  Phase 3 — Architecture diversity aggregate")
    print("=" * 78)

    fold_streams: dict[str, dict[int, np.ndarray]] = {b: {} for b in brains}
    per_brain_summary: dict[str, dict] = {}

    for b in brains:
        for i, (a, bb) in enumerate(FOLDS, 1):
            fr = ROOT / f"logs/phase3/{b}/fold_{i}"
            tp = fr / "trade_pnl.csv"
            if not tp.exists():
                continue
            try:
                df = pd.read_csv(tp)
                if not df.empty and "trade_pnl" in df.columns:
                    fold_streams[b][i] = df["trade_pnl"].values.astype(float)
            except Exception:
                continue

    print(f"  {'brain':>6} {'folds':>7} {'pooled_n':>10} {'PF':>7} "
          f"{'90% CI':>18} {'Net Rs.':>15} {'WR%':>6} {'Sharpe':>8} {'Sortino':>8}")
    print("  " + "-" * 96)
    for b in brains:
        streams = fold_streams[b]
        if not streams:
            print(f"  {b:>6}: no completed folds")
            per_brain_summary[b] = {"status": "MISSING"}
            continue
        pooled = np.concatenate(list(streams.values()))
        pf = float(profit_factor(pooled))
        wr = float(win_rate(pooled))
        try:
            sh = float(sharpe_fn(pooled))
            so = float(sortino_fn(pooled))
        except Exception:
            sh, so = float("nan"), float("nan")
        ci = block_bootstrap_ci(streams, profit_factor,
                                 n_resamples=10_000, ci_level=0.90, seed=42,
                                 stat_name=f"{b}_pf")
        per_brain_summary[b] = {
            "n_folds": len(streams),
            "n_pooled_trades": int(len(pooled)),
            "pooled_pf": pf,
            "pf_ci90": [ci.lower, ci.upper],
            "pooled_net": float(pooled.sum()),
            "wr": wr,
            "sharpe": sh,
            "sortino": so,
            "max_dd": float(max_drawdown(pooled)),
        }
        print(f"  {b:>6} {len(streams):>7} {len(pooled):>10} {pf:>7.3f}  "
              f"[{ci.lower:>+6.3f}, {ci.upper:>+6.3f}] "
              f"{pooled.sum():>+15,.0f} {wr*100:>5.1f}% {sh:>8.3f} {so:>8.3f}")

    # ---- Pairwise DM on per-bar P&L differential
    completed = [b for b in brains if fold_streams[b]]
    dm_matrix: dict[str, dict[str, dict]] = {}
    if len(completed) >= 2:
        print("\n  Pairwise Diebold-Mariano (bar-aligned; A better than B, HAC lag=5):")
        print(f"    {'A vs B':>18} {'n_bars':>8} {'stat':>8} {'p':>10} {'mean_diff Rs.':>14}")
        for i_a, a in enumerate(completed):
            dm_matrix[a] = {}
            for b in completed:
                if a == b:
                    continue
                a_pieces, b_pieces = [], []
                for k in fold_streams[a]:
                    if k not in fold_streams[b]:
                        continue
                    fa = ROOT / f"logs/phase3/{a}/fold_{k}/trades.csv"
                    fb = ROOT / f"logs/phase3/{b}/fold_{k}/trades.csv"
                    try:
                        adf = pd.read_csv(fa)
                        bdf = pd.read_csv(fb)
                    except Exception:
                        continue
                    if "time" not in adf.columns or "time" not in bdf.columns:
                        continue
                    a_by = adf.set_index("time")["net_option"]
                    b_by = bdf.set_index("time")["net_option"]
                    common = a_by.index.intersection(b_by.index)
                    if len(common):
                        a_pieces.append(a_by.loc[common].values.astype(float))
                        b_pieces.append(b_by.loc[common].values.astype(float))
                if not a_pieces:
                    dm_matrix[a][b] = {"n": 0, "skipped": True}
                    continue
                aa = np.concatenate(a_pieces)
                ba = np.concatenate(b_pieces)
                try:
                    dm = diebold_mariano(aa, ba, alternative="greater",
                                          lag=5, ci_level=0.90)
                    dm_matrix[a][b] = dm.to_dict()
                    print(f"    {(a+' vs '+b):>18} {dm.n:>8} "
                          f"{dm.statistic:>+8.3f} {dm.pvalue:>10.4f} "
                          f"{dm.mean_loss_diff:>+14.2f}")
                except Exception as e:
                    dm_matrix[a][b] = {"error": str(e)}

    # ---- Hansen SPA + White RC across brains (per-fold PF vs benchmark 1.0)
    spa_result_dict = None
    wrc_result_dict = None
    common_folds: list[int] = []
    if len(completed) >= 2:
        common_folds_set = set.intersection(
            *[set(fold_streams[b].keys()) for b in completed])
        common_folds = sorted(common_folds_set)
        if len(common_folds) >= 2:
            perf = {}
            for b in completed:
                pfs = np.array([profit_factor(fold_streams[b][k])
                                for k in common_folds])
                pfs = np.where(np.isfinite(pfs), pfs, 0.0)
                perf[b] = pfs
            bench = np.ones(len(common_folds))
            try:
                spa = hansen_spa(perf, benchmark=bench,
                                  n_bootstrap=5_000, seed=42)
                spa_result_dict = spa.to_dict()
                print(f"\n  Hansen SPA across brains "
                      f"(per-fold PF vs benchmark 1.0):")
                print(f"    K brains: {spa.n_models}   "
                      f"T folds: {len(common_folds)}")
                print(f"    pvalue_lower       : {spa.pvalue_lower:.4f}  "
                      f"(pre-registered decisive metric)")
                print(f"    pvalue_consistent  : {spa.pvalue_consistent:.4f}  "
                      f"(supplementary)")
                print(f"    pvalue_upper       : {spa.pvalue_upper:.4f}  "
                      f"(supplementary)")
                print(f"    per-brain mean PF over benchmark:")
                for b, mp in spa.per_model_mean_perf.items():
                    print(f"      {b:>6}: {mp:+.3f}")
            except Exception as e:
                print(f"\n  SPA failed: {e}")
            try:
                wrc = white_reality_check(perf, benchmark=bench,
                                            n_bootstrap=5_000, seed=42)
                wrc_result_dict = wrc.to_dict()
                print(f"\n  White's Reality Check "
                      f"(per-fold PF vs benchmark 1.0):")
                print(f"    K brains: {wrc.n_models}   "
                      f"T folds: {len(common_folds)}")
                print(f"    statistic          : {wrc.statistic:+.3f}")
                print(f"    p-value            : {wrc.pvalue:.4f}")
            except Exception as e:
                print(f"\n  White RC failed: {e}")

    # ---- Architecture diversity summary (pairwise signal-disagreement)
    diversity_matrix: dict[str, dict[str, float]] = {}
    if len(completed) >= 2 and len(common_folds) >= 1:
        print(f"\n  Pairwise signal disagreement "
              f"(1 - fraction agreement on {{CALL, PUT, SKIP}}):")
        for a in completed:
            diversity_matrix[a] = {}
        for a in completed:
            for b in completed:
                if a == b:
                    diversity_matrix[a][b] = 0.0
                    continue
                agreements = []
                totals = 0
                for k in common_folds:
                    fa = ROOT / f"logs/phase3/{a}/fold_{k}/predictions.parquet"
                    if not fa.exists():
                        fa = ROOT / f"logs/phase3/{a}/fold_{k}/predictions.csv"
                    fb = ROOT / f"logs/phase3/{b}/fold_{k}/predictions.parquet"
                    if not fb.exists():
                        fb = ROOT / f"logs/phase3/{b}/fold_{k}/predictions.csv"
                    if not (fa.exists() and fb.exists()):
                        continue
                    try:
                        adf = (pd.read_parquet(fa)
                                if fa.suffix == ".parquet" else pd.read_csv(fa))
                        bdf = (pd.read_parquet(fb)
                                if fb.suffix == ".parquet" else pd.read_csv(fb))
                    except Exception:
                        continue
                    if len(adf) != len(bdf):
                        continue
                    agree = (adf["signal"].values == bdf["signal"].values)
                    agreements.append(int(agree.sum()))
                    totals += len(agree)
                if totals:
                    dis = 1.0 - sum(agreements) / totals
                    diversity_matrix[a][b] = float(dis)
        # Print upper triangle
        cols = completed
        header = "        " + " ".join(f"{c:>6}" for c in cols)
        print("    " + header)
        for a in cols:
            row = "    " + f"{a:>6}  "
            for b in cols:
                v = diversity_matrix[a][b]
                row += f"{v:>6.3f} "
            print(row)

    # ---- H_arch verdict against pre-registered acceptance criteria
    h_arch = {"status": "NOT_TESTED"}
    gbdt_here = [b for b in completed if b in GBDT_BRAINS]
    nongbdt_here = [b for b in completed if b in NON_GBDT_BRAINS]
    if gbdt_here and nongbdt_here and len(common_folds) >= 2:
        # Per-fold PF matrix (safe, non-inf) for paired bootstrap
        per_fold_pf = {}
        for b in gbdt_here + nongbdt_here:
            per_fold_pf[b] = np.array(
                [np.clip(profit_factor(fold_streams[b][k]),
                         -10, 10) for k in common_folds])

        # delta = max(non-GBDT per-fold PF) - max(GBDT per-fold PF), fold-wise
        # For paired bootstrap: on each resampled fold set, compute the
        # difference of the pooled-across-fold PF between best non-GBDT
        # and best GBDT (best chosen ON the resample, matching selection
        # bias — this is White-style comparison).
        def _delta_stat(a_stream: np.ndarray,
                        b_stream: np.ndarray) -> float:
            # a is the tuned "arm A" placeholder; paired bootstrap on
            # PF-of-per-fold isn't the right API for max-of-K. Instead
            # compute directly.
            return float(profit_factor(a_stream) - profit_factor(b_stream))

        # Simpler: choose the best-observed non-GBDT and best-observed
        # GBDT, then paired block-bootstrap their delta_PF.
        best_g = max(gbdt_here,
                      key=lambda b: profit_factor(
                          np.concatenate(list(fold_streams[b].values()))))
        best_n = max(nongbdt_here,
                      key=lambda b: profit_factor(
                          np.concatenate(list(fold_streams[b].values()))))
        try:
            ci = paired_block_bootstrap_ci(
                fold_streams[best_n],   # A = non-GBDT
                fold_streams[best_g],   # B = GBDT
                _delta_stat,
                n_resamples=10_000, ci_level=0.90, seed=42,
                stat_name="dPF_nonGBDT_minus_GBDT",
            )
            effect = 0.08
            spa_p_l = (spa_result_dict.get("pvalue_lower")
                        if spa_result_dict else float("nan"))
            wrc_p = (wrc_result_dict.get("pvalue")
                      if wrc_result_dict else float("nan"))
            # Pre-reg FWE alpha 0.10; Phase 3 tests H_arch in isolation.
            # Report both raw and Holm-adjusted (across 7 primary hyps).
            reject_ci = ci.lower > effect
            reject_spa = (isinstance(spa_p_l, float)
                           and np.isfinite(spa_p_l) and spa_p_l < 0.10)
            reject_holm = (isinstance(spa_p_l, float)
                            and np.isfinite(spa_p_l)
                            and spa_p_l < 0.10 / 7)
            h_arch = {
                "status": "TESTED",
                "best_non_gbdt": best_n,
                "best_gbdt": best_g,
                "delta_pf_point": ci.point_estimate,
                "delta_pf_ci_lower_90": ci.lower,
                "delta_pf_ci_upper_90": ci.upper,
                "effect_threshold": effect,
                "spa_pvalue_lower": spa_p_l,
                "white_rc_pvalue": wrc_p,
                "reject_by_ci": reject_ci,
                "reject_by_spa_raw": reject_spa,
                "reject_by_spa_holm_fwe_7": reject_holm,
                "decision": ("REJECT H0_arch (evidence for non-GBDT edge)"
                              if reject_ci and reject_holm
                              else "FAIL TO REJECT H0_arch"),
            }
            print(f"\n  H_arch verdict "
                  f"(pre-registered acceptance criteria):")
            print(f"    Best non-GBDT vs best GBDT: "
                  f"{best_n} vs {best_g}")
            print(f"    delta_PF point            : "
                  f"{ci.point_estimate:+.3f}")
            print(f"    90% paired CI             : "
                  f"[{ci.lower:+.3f}, {ci.upper:+.3f}]")
            print(f"    Effect threshold          : +{effect}")
            print(f"    LB90 > threshold          : "
                  f"{'YES' if reject_ci else 'NO'}")
            print(f"    SPA_l                     : "
                  f"{spa_p_l:.4f}  raw alpha 0.10")
            print(f"    Holm-adjusted alpha (7 hyps): 0.10/7 = 0.0143")
            print(f"    SPA_l < 0.0143            : "
                  f"{'YES' if reject_holm else 'NO'}")
            print(f"    White RC p                : "
                  f"{wrc_p:.4f}")
            print(f"    DECISION                  : {h_arch['decision']}")
        except Exception as e:
            h_arch = {"status": "FAILED", "error": str(e)}
            print(f"\n  H_arch computation failed: {e}")
    else:
        print(f"\n  H_arch NOT_TESTED  "
              f"(gbdt_here={gbdt_here}, nongbdt_here={nongbdt_here}, "
              f"common_folds={len(common_folds)})")

    summary = {
        "brains": brains,
        "per_brain": per_brain_summary,
        "diebold_mariano_pairwise": dm_matrix,
        "hansen_spa": spa_result_dict,
        "white_reality_check": wrc_result_dict,
        "diversity_disagreement_matrix": diversity_matrix,
        "h_arch_verdict": h_arch,
        "common_folds": common_folds,
    }
    (ROOT / "logs/phase3").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs/phase3/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n  wrote logs/phase3/summary.json")


def _prevent_sleep_windows() -> bool:
    """Prevent Windows sleep via SetThreadExecutionState. See phase2_launcher."""
    if platform.system() != "Windows":
        return False
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default=",".join(DEFAULT_BRAINS))
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hpo", action="store_true")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    brains = [b.strip() for b in args.brains.split(",") if b.strip()]

    if args.aggregate_only:
        aggregate(brains)
        return

    if args.folds == "all":
        selected = list(range(1, len(FOLDS) + 1))
    else:
        selected = [int(x) for x in args.folds.split(",") if x.strip()]

    (ROOT / "logs/phase3").mkdir(parents=True, exist_ok=True)

    sleep_prevented = _prevent_sleep_windows()
    if platform.system() == "Windows":
        print(f"[pre-flight] Windows sleep prevention: {sleep_prevented}")

    statuses = []
    for b in brains:
        for i in selected:
            a, bb = FOLDS[i - 1]
            ran, st = run_one(b, i, a, bb, args.seed, args.hpo,
                               args.n_trials, args.k_inner,
                               args.force, args.verify_prereg)
            statuses.append((b, i, st))

    print("\n  Run summary:")
    for b, i, st in statuses:
        print(f"    {b} fold {i}: {st}")

    if not args.skip_aggregate:
        aggregate(brains)


if __name__ == "__main__":
    main()
