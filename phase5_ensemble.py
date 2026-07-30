"""
phase5_ensemble.py - Orchestrator for the Phase 5 ensemble framework.

Iterates (fold), dispatches each to a fresh ``phase5_run_single_fold``
subprocess (same isolation pattern as Phase 3/4). Aggregation is
STUBBED: this delivery does NOT compute Profit Factor, Sharpe, Sortino,
Diebold-Mariano, White Reality Check, Hansen SPA, or an ``H_ens``
verdict. Those calls are marked with ``# TODO: enable after Phase 4
completes and review approves scientific evaluation``.

Structural aggregation (fold counts, brain-participation summary,
diversity-metric mean matrices) is safe and will produce descriptive
JSON only.

NOTE
----
This delivery ONLY compiles the script; it is never executed. When the
Phase-4 overnight run completes and the aggregator's evaluation code
paths are opened up, running::

    python phase5_ensemble.py --input raw

executes the full evaluation without any code change here.
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


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _prevent_sleep_windows() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001))
    except Exception:
        return False


def _cache_ok(ens_root: Path) -> bool:
    for f in ("manifest.json", "metrics.json", "weights.json",
              "probabilities.csv", "trades.csv", "trade_pnl.csv"):
        if not (ens_root / f).exists():
            return False
    # predictions.parquet OR predictions.csv
    if not ((ens_root / "predictions.parquet").exists()
            or (ens_root / "predictions.csv").exists()):
        return False
    return True


def run_one_fold(i: int, a: date, b: date, args) -> tuple[bool, str]:
    """Dispatch a single-fold subprocess. Returns (ran, status)."""
    from ensembles import list_ensembles
    ensembles_to_run = (list_ensembles() if args.ensembles == "all"
                        else [x.strip() for x in args.ensembles.split(",")])
    fold_dir = ROOT / "logs" / "phase5"
    all_cached = all(
        _cache_ok(fold_dir / e / f"fold_{i}") for e in ensembles_to_run)
    if all_cached and not args.force:
        print(f"[fold {i}] all ensembles cached; skipping")
        return False, "cached"
    cmd = [sys.executable, "-u", "phase5_run_single_fold.py",
           str(i), a.isoformat(), b.isoformat(),
           "--brains", args.brains,
           "--ensembles", args.ensembles,
           "--input", args.input,
           "--k-inner", str(args.k_inner),
           "--seed", str(args.seed)]
    if args.verify_prereg:
        cmd.append("--verify-prereg")
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        return False, f"FAILED exit={e.returncode}"
    return True, "ran"


# ---------------------------------------------------------------------------
# Baseline ensemble for H_ens comparisons. Pre-registered as "simple mean".
BASELINE_ENS = "mean"
# H_ens effect threshold (pre-registered)
H_ENS_EFFECT = 0.05
# Holm-adjusted alpha across 7 primary hypotheses (H_edge, H_hpo, H_arch,
# H_ens, H_cal, H_target, H_info) at FWE alpha = 0.10 -> 0.10/7 approx 0.0143
H_ENS_HOLM_ALPHA = 0.10 / 7


def aggregate_structural(args) -> None:
    """Full scientific evaluation of Phase 5 ensembles.

    Wired in per approval to unblock the TODO stub. Computes:
      * per-ensemble pooled metrics (PF, WR, Net, MaxDD, Sharpe, Sortino)
      * block-bootstrap 90% CI on PF, per ensemble
      * paired block-bootstrap CI on delta_PF vs simple-mean baseline
      * Diebold-Mariano tests on bar-aligned trade P&L differential
      * Hansen SPA (l/c/u) and White's Reality Check across all
        ensembles vs mean baseline
      * Holm-Bonferroni on DM p-values across the 7 non-baseline
        ensembles
      * Diversity matrices (Q-stat + disagreement) per fold using
        Phase-4 calibrated brain probabilities
      * Ensemble weight summary aggregated across folds
      * H_ens verdict per pre-registration
    """
    import numpy as np
    import pandas as pd
    from stat_utils import (block_bootstrap_ci, diebold_mariano,
                              hansen_spa, holm_bonferroni,
                              max_drawdown, paired_block_bootstrap_ci,
                              profit_factor, sharpe as sharpe_fn,
                              sortino as sortino_fn,
                              white_reality_check, win_rate)
    from ensembles import (average_diversity_across_folds,
                             diversity_matrix, list_ensembles,
                             load_brain_probs)

    ensembles = (list_ensembles() if args.ensembles == "all"
                 else [x.strip() for x in args.ensembles.split(",")])
    if BASELINE_ENS not in ensembles:
        print(f"WARNING: baseline {BASELINE_ENS!r} not in ensemble set; "
              f"skipping paired comparisons")

    print("\n" + "=" * 82)
    print("  Phase 5 - Full statistical aggregate")
    print("=" * 82)

    # ---- 1. Load per-(ensemble, fold) trade P&L + weights ----
    streams: dict[str, dict[int, np.ndarray]] = {e: {} for e in ensembles}
    weights_per_fold: dict[str, list[dict]] = {e: [] for e in ensembles}
    for e in ensembles:
        for i, (a, b) in enumerate(FOLDS, 1):
            fr = ROOT / "logs/phase5" / e / f"fold_{i}"
            tp = fr / "trade_pnl.csv"
            if not tp.exists():
                continue
            try:
                df = pd.read_csv(tp)
                if not df.empty and "trade_pnl" in df.columns:
                    streams[e][i] = df["trade_pnl"].values.astype(float)
            except Exception:
                continue
            wp = fr / "weights.json"
            if wp.exists():
                try:
                    weights_per_fold[e].append(
                        {"fold": i, **json.loads(wp.read_text())})
                except Exception:
                    pass

    # ---- 2. Per-ensemble pooled metrics + block-bootstrap CI ----
    per_ens_summary: dict[str, dict] = {}
    print(f"\n  {'ensemble':>14} {'folds':>6} {'trades':>7} {'PF':>7} "
          f"{'90% CI':>18} {'Net Rs.':>15} {'WR%':>6} {'Sharpe':>8} "
          f"{'Sortino':>8}")
    print("  " + "-" * 96)
    for e in ensembles:
        s = streams[e]
        if not s:
            per_ens_summary[e] = {"status": "MISSING"}
            print(f"  {e:>14}: no completed folds")
            continue
        pooled = np.concatenate(list(s.values()))
        pf = float(profit_factor(pooled))
        wr = float(win_rate(pooled))
        net = float(pooled.sum())
        try:
            sh = float(sharpe_fn(pooled))
            so = float(sortino_fn(pooled))
        except Exception:
            sh, so = float("nan"), float("nan")
        dd = float(max_drawdown(pooled))
        ci = block_bootstrap_ci(s, profit_factor,
                                 n_resamples=10_000, ci_level=0.90,
                                 seed=42, stat_name=f"{e}_pf")
        per_ens_summary[e] = {
            "n_folds": len(s),
            "n_pooled_trades": int(len(pooled)),
            "pf": pf, "pf_ci90": [ci.lower, ci.upper],
            "net": net, "wr": wr, "sharpe": sh, "sortino": so,
            "max_dd": dd,
            "folds_completed": sorted(s.keys()),
        }
        print(f"  {e:>14} {len(s):>6} {len(pooled):>7} {pf:>7.3f} "
              f"[{ci.lower:>+6.3f}, {ci.upper:>+6.3f}] {net:>+15,.0f} "
              f"{wr*100:>5.1f}% {sh:>+8.3f} {so:>+8.3f}")

    # ---- 3. Paired block-bootstrap delta_PF + Diebold-Mariano vs baseline ----
    delta_summary: dict[str, dict] = {}
    dm_pvals: dict[str, float] = {}
    if BASELINE_ENS in streams and streams[BASELINE_ENS]:
        print(f"\n  delta_PF vs baseline ({BASELINE_ENS}):  paired block-bootstrap "
              f"+ Diebold-Mariano (bar-aligned, HAC lag=5)")
        print(f"  {'ensemble':>14} {'delta_PF':>10} {'90% paired CI':>22} "
              f"{'DM p (grtr)':>13} {'DM n':>6} {'delta_Net Rs.':>15}")
        print("  " + "-" * 82)
        base_streams = streams[BASELINE_ENS]
        for e in ensembles:
            if e == BASELINE_ENS or not streams[e]:
                continue
            common = sorted(set(streams[e]) & set(base_streams))
            if len(common) < 2:
                continue
            a_dict = {k: streams[e][k] for k in common}
            b_dict = {k: base_streams[k] for k in common}

            def _dpf(a_arr, b_arr):
                pfa = profit_factor(a_arr) if len(a_arr) else float("nan")
                pfb = profit_factor(b_arr) if len(b_arr) else float("nan")
                if not (np.isfinite(pfa) and np.isfinite(pfb)):
                    return float("nan")
                return pfa - pfb

            ci = paired_block_bootstrap_ci(
                a_dict, b_dict, _dpf,
                n_resamples=10_000, ci_level=0.90, seed=42,
                stat_name=f"{e}_dPF")

            # DM on per-BAR aligned realised P&L (0 for skipped bars).
            # This gives us a uniformly-aligned series regardless of
            # whether the two ensembles picked the same bars to trade.
            a_pieces, b_pieces = [], []
            for k in common:
                pa_path = ROOT / "logs/phase5" / e / f"fold_{k}" / "predictions.csv"
                pb_path = ROOT / "logs/phase5" / BASELINE_ENS / f"fold_{k}" / "predictions.csv"
                if not (pa_path.exists() and pb_path.exists()):
                    continue
                try:
                    pa = pd.read_csv(pa_path)
                    pb = pd.read_csv(pb_path)
                except Exception:
                    continue
                if "timestamp" not in pa.columns or "timestamp" not in pb.columns:
                    continue
                common_bars = pd.Index(pa["timestamp"]).intersection(
                    pd.Index(pb["timestamp"]))
                if len(common_bars) < 100:
                    continue
                ta_path = ROOT / "logs/phase5" / e / f"fold_{k}" / "trades.csv"
                tb_path = ROOT / "logs/phase5" / BASELINE_ENS / f"fold_{k}" / "trades.csv"
                try:
                    ta = pd.read_csv(ta_path) if ta_path.exists() else pd.DataFrame(columns=["time", "net_option"])
                    tb = pd.read_csv(tb_path) if tb_path.exists() else pd.DataFrame(columns=["time", "net_option"])
                except Exception:
                    ta = tb = pd.DataFrame(columns=["time", "net_option"])
                a_bar = (ta.groupby("time")["net_option"].sum()
                          if "time" in ta.columns and len(ta) else pd.Series(dtype=float))
                b_bar = (tb.groupby("time")["net_option"].sum()
                          if "time" in tb.columns and len(tb) else pd.Series(dtype=float))
                a_pnl = a_bar.reindex(common_bars, fill_value=0.0).values.astype(float)
                b_pnl = b_bar.reindex(common_bars, fill_value=0.0).values.astype(float)
                a_pieces.append(a_pnl)
                b_pieces.append(b_pnl)
            dm_p, dm_stat, dm_n = None, None, 0
            if a_pieces:
                aa = np.concatenate(a_pieces)
                bb = np.concatenate(b_pieces)
                try:
                    dm = diebold_mariano(aa, bb, alternative="greater",
                                          lag=5, ci_level=0.90)
                    dm_p = dm.pvalue
                    dm_stat = dm.statistic
                    dm_n = dm.n
                except Exception:
                    pass

            net_delta = (per_ens_summary[e]["net"]
                         - per_ens_summary[BASELINE_ENS]["net"])
            delta_summary[e] = {
                "delta_pf_point": ci.point_estimate,
                "delta_pf_lb90": ci.lower,
                "delta_pf_ub90": ci.upper,
                "delta_pf_ci_half_width": (ci.upper - ci.lower) / 2,
                "dm_statistic": dm_stat,
                "dm_pvalue": dm_p,
                "dm_n_bars": dm_n,
                "delta_net": net_delta,
                "n_folds": len(common),
            }
            dm_pvals[e] = dm_p if isinstance(dm_p, float) else 1.0
            dm_p_s = f"{dm_p:.4f}" if isinstance(dm_p, float) else "  n/a  "
            print(f"  {e:>14} {ci.point_estimate:>+8.3f} "
                  f"[{ci.lower:>+7.3f}, {ci.upper:>+7.3f}] {dm_p_s:>13} "
                  f"{dm_n:>6} {net_delta:>+15,.0f}")

    # ---- 4. Hansen SPA + White RC across all ensembles vs baseline ----
    spa_result = None
    wrc_result = None
    common_folds = sorted(set.intersection(*[set(streams[e]) for e in ensembles
                                                if streams[e]])) \
        if all(streams[e] for e in ensembles) else []
    if len(common_folds) >= 2 and BASELINE_ENS in streams:
        perf: dict[str, np.ndarray] = {}
        for e in ensembles:
            pfs = np.array([
                profit_factor(streams[e][k]) if streams[e][k].size else 0.0
                for k in common_folds])
            perf[e] = np.where(np.isfinite(pfs), pfs, 0.0)
        bench = perf[BASELINE_ENS]
        try:
            spa = hansen_spa(perf, benchmark=bench, n_bootstrap=5_000, seed=42)
            spa_result = spa.to_dict()
            print(f"\n  Hansen SPA (all ensembles vs {BASELINE_ENS}, T={len(common_folds)} folds):")
            print(f"    pvalue_lower       : {spa.pvalue_lower:.4f}  (pre-reg decisive)")
            print(f"    pvalue_consistent  : {spa.pvalue_consistent:.4f}  (supplementary)")
            print(f"    pvalue_upper       : {spa.pvalue_upper:.4f}  (supplementary)")
        except Exception as ex:
            print(f"  SPA failed: {ex}")
        try:
            wrc = white_reality_check(perf, benchmark=bench,
                                        n_bootstrap=5_000, seed=42)
            wrc_result = wrc.to_dict()
            print(f"\n  White's Reality Check (K={wrc.n_models} ensembles):")
            print(f"    statistic          : {wrc.statistic:+.3f}")
            print(f"    p-value            : {wrc.pvalue:.4f}")
        except Exception as ex:
            print(f"  White RC failed: {ex}")

    # ---- 5. Holm-Bonferroni across the 7 non-baseline ensembles ----
    holm_result = None
    if dm_pvals:
        try:
            hb = holm_bonferroni(dm_pvals, alpha=0.10)
            holm_result = {k: v for k, v in hb.decisions.items()}
        except Exception as ex:
            print(f"  Holm failed: {ex}")

    # ---- 6. Diversity matrices per fold (Q-stat + disagreement) ----
    diversity_summary: dict = {}
    try:
        source = args.input if hasattr(args, "input") else "raw"
        per_fold_q: list = []
        per_fold_dis: list = []
        for f in common_folds:
            # discover brains present for this fold
            brain_argmax: dict[str, np.ndarray] = {}
            y_ref = None
            source_dir = ("logs/phase4" if "calibrated" in source
                          else "logs/phase3")
            for b in sorted(p.name for p in (ROOT / source_dir).glob("*")
                             if p.is_dir()):
                try:
                    probs, y, _ = load_brain_probs(ROOT, b, f, source=source)
                    brain_argmax[b] = probs.argmax(axis=1)
                    if y_ref is None:
                        y_ref = y
                except Exception:
                    continue
            if len(brain_argmax) >= 2 and y_ref is not None:
                per_fold_q.append(
                    diversity_matrix(brain_argmax, y_ref, "q_statistic"))
                per_fold_dis.append(
                    diversity_matrix(brain_argmax, y_ref, "disagreement"))
        if per_fold_q:
            mean_q, std_q = average_diversity_across_folds(per_fold_q)
            mean_dis, std_dis = average_diversity_across_folds(per_fold_dis)
            diversity_summary = {
                "n_folds": len(per_fold_q),
                "mean_q_statistic": mean_q.to_dict(),
                "std_q_statistic": std_q.to_dict(),
                "mean_disagreement": mean_dis.to_dict(),
                "std_disagreement": std_dis.to_dict(),
            }
            print(f"\n  Diversity (mean across {len(per_fold_q)} folds):")
            print(f"    Q-statistic pairwise mean:")
            for a in mean_q.index:
                row_vals = "  ".join(f"{mean_q.loc[a, b]:+.3f}"
                                       for b in mean_q.columns)
                print(f"      {a:>6}: {row_vals}")
    except Exception as ex:
        print(f"  Diversity computation failed: {ex}")

    # ---- 7. Weights summary aggregated across folds ----
    weights_summary_out: dict = {}
    for e, folds_data in weights_per_fold.items():
        if not folds_data:
            continue
        weights_summary_out[e] = {
            "n_folds_with_weights": len(folds_data),
            "example_weights_fold_1": folds_data[0] if folds_data else None,
        }
        # For fixed-weight ensembles, average weights across folds
        first = folds_data[0]
        if "weights" in first and isinstance(first["weights"], dict):
            avg: dict[str, float] = {}
            for brain in first["weights"].keys():
                vals = [d["weights"].get(brain, np.nan) for d in folds_data
                         if "weights" in d]
                if vals:
                    finite = [v for v in vals if isinstance(v, (int, float))
                               and np.isfinite(v)]
                    if finite:
                        avg[brain] = float(np.mean(finite))
            weights_summary_out[e]["mean_weights_across_folds"] = avg

    # ---- 8. H_ens verdict ----
    passing_ensembles: list[str] = []
    for e, row in delta_summary.items():
        lb = row.get("delta_pf_lb90")
        dm_p = row.get("dm_pvalue")
        if (isinstance(lb, (int, float)) and lb > H_ENS_EFFECT
                and isinstance(dm_p, float) and dm_p < H_ENS_HOLM_ALPHA):
            passing_ensembles.append(e)

    if passing_ensembles:
        best = max(passing_ensembles,
                    key=lambda x: delta_summary[x]["delta_pf_point"])
        verdict = f"ACCEPT - reject H0_ens (best: {best})"
    else:
        undecided = False
        for row in delta_summary.values():
            hw = row.get("delta_pf_ci_half_width")
            if isinstance(hw, (int, float)) and hw > 2 * H_ENS_EFFECT:
                undecided = True
                break
        verdict = ("UNDECIDED (paired CI half-width > 2x effect size)"
                    if undecided else "FAIL TO REJECT H0_ens")

    print(f"\n  H_ens verdict (pre-registered acceptance criteria):")
    print(f"    rule: LB90(delta_PF vs {BASELINE_ENS}) > +{H_ENS_EFFECT} "
          f"AND DM_p < Holm alpha ({H_ENS_HOLM_ALPHA:.4f})")
    print(f"    passing ensembles : {passing_ensembles}")
    print(f"    decision          : {verdict}")

    # ---- 9. Best ensemble by pooled PF ----
    valid = {e: v for e, v in per_ens_summary.items()
             if isinstance(v, dict) and "pf" in v}
    best_by_pf = max(valid.items(), key=lambda kv: kv[1]["pf"])[0] if valid else None

    # ---- 10. Write publication-quality summary ----
    out = {
        "phase": 5,
        "protocol_baseline": BASELINE_ENS,
        "h_ens": {
            "effect_threshold": H_ENS_EFFECT,
            "holm_alpha": H_ENS_HOLM_ALPHA,
            "passing_ensembles": passing_ensembles,
            "decision": verdict,
            "best_ensemble_by_pooled_pf": best_by_pf,
        },
        "per_ensemble": per_ens_summary,
        "delta_vs_baseline": delta_summary,
        "holm_bonferroni_dm": holm_result,
        "hansen_spa": spa_result,
        "white_reality_check": wrc_result,
        "diversity": diversity_summary,
        "weights": weights_summary_out,
        "input_source": getattr(args, "input", "raw"),
    }
    (ROOT / "logs/phase5").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs/phase5/summary.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  wrote logs/phase5/summary.json")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default="all",
                    help="'all' or comma-separated brain names.")
    ap.add_argument("--ensembles", default="all",
                    help="'all' or comma-separated ensemble names.")
    ap.add_argument("--folds", default="all",
                    help="'all' or comma-separated 1-indexed fold ids.")
    ap.add_argument("--input", default="raw",
                    choices=["raw", "calibrated_noop",
                              "calibrated_platt", "calibrated_isotonic"])
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate_structural(args)
        return

    if args.folds == "all":
        selected = list(range(1, len(FOLDS) + 1))
    else:
        selected = [int(x) for x in args.folds.split(",") if x.strip()]

    (ROOT / "logs/phase5").mkdir(parents=True, exist_ok=True)
    sleep_prevented = _prevent_sleep_windows()
    if platform.system() == "Windows":
        print(f"[pre-flight] Windows sleep prevention: {sleep_prevented}")

    statuses = []
    for i in selected:
        a, bb = FOLDS[i - 1]
        ran, st = run_one_fold(i, a, bb, args)
        statuses.append((i, st))

    print("\n  Run summary:")
    for i, s in statuses:
        print(f"    fold {i}: {s}")

    if not args.skip_aggregate:
        aggregate_structural(args)


if __name__ == "__main__":
    main()
