"""Phase 7 – Robustness & Stress Testing orchestrator.

Reads frozen Phase 5 predictions and Phase 6 winner thresholds; produces
JSON reports and a publication-quality markdown summary.

Usage
-----

    python phase7_robustness.py \\
        --targets mean,stacking \\
        --seed 42 \\
        [--force] [--skip-charts] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import phase7_robustness_v9 as P7
from phase7_robustness_v9 import (
    Phase7Config, PRODUCTION_BASELINE, ThresholdCandidate,
    apply_cost_stress, block_bootstrap_net, block_bootstrap_pf,
    build_manifest, classify_folds, compute_h_rob_verdict,
    dm_winner_vs_baseline_by_fold, find_break_even_slippage,
    find_break_even_tcost, leave_one_out, load_fold_data,
    paired_bootstrap_delta_pf, per_fold_metrics,
    pooled_metrics_from_replays, render_phase7_md,
    run_delay_stress, run_slippage_curve, run_tcost_curve,
    save_manifest, simulate_candidate,
    spa_and_wrc_over_variants, stability_report,
    tornado_ranking, walkforward_report,
    write_reports, save_chart_data, write_chart_pngs,
)


PHASE6_SUMMARY_PATH = Path("logs/phase6/summary.json")


def _named_candidates_for(target: str, phase6_summary: dict):
    """Return {name -> ThresholdCandidate} for the target."""
    winner = phase6_summary["per_target"][target]["winner_candidate"]
    winner_cand = ThresholdCandidate(
        call_thr=winner["call_thr"], put_thr=winner["put_thr"],
        skip_ceil=winner["skip_ceil"], min_edge=winner["min_edge"])
    return {
        "baseline":       PRODUCTION_BASELINE,
        "phase6_winner":  winner_cand,
    }


def _pooled_and_per_fold_pnl(replays):
    return {f: r.net_pnl for f, r in replays.items()}


def run_target(target: str,
                config: Phase7Config,
                phase6_summary: dict,
                smoke: bool) -> dict:
    print(f"\n[Phase 7 / {target}] loading fold data ...")
    fold_data = load_fold_data(target,
                                 folds=range(1, 3) if smoke else range(1, 9))

    regime = classify_folds(fold_data)
    print(f"  regime counts: {regime['counts']}")

    named = _named_candidates_for(target, phase6_summary)

    # --- baseline replay (production thresholds, production costs) ---
    print("  replaying baseline candidate ...")
    t0 = time.time()
    baseline_replays = simulate_candidate(named["baseline"], fold_data)
    baseline_pnl = _pooled_and_per_fold_pnl(baseline_replays)
    baseline_pooled = pooled_metrics_from_replays(baseline_replays)
    print(f"    baseline pooled: PF={baseline_pooled['pf']:.3f} "
            f"Net=Rs.{baseline_pooled['net']:+,.0f} "
            f"n={baseline_pooled['n']}  ({time.time()-t0:.1f}s)")

    # --- winner replay ---
    print("  replaying Phase-6 winner candidate ...")
    t0 = time.time()
    winner_replays = simulate_candidate(named["phase6_winner"], fold_data)
    winner_pnl = _pooled_and_per_fold_pnl(winner_replays)
    winner_pooled = pooled_metrics_from_replays(winner_replays)
    print(f"    winner   pooled: PF={winner_pooled['pf']:.3f} "
            f"Net=Rs.{winner_pooled['net']:+,.0f} "
            f"n={winner_pooled['n']}  ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------
    # R1 – walk-forward
    # -----------------------------------------------------------------
    print("  R1 walk-forward variants ...")
    wf = walkforward_report(winner_pnl)

    # -----------------------------------------------------------------
    # R2 – jackknife
    # -----------------------------------------------------------------
    print("  R2 jackknife LOO ...")
    jack = leave_one_out(winner_pnl)

    # -----------------------------------------------------------------
    # R3 – slippage
    # -----------------------------------------------------------------
    print("  R3 slippage stress ...")
    winner_trades_by_fold = {f: r.trades for f, r in winner_replays.items()}
    slip = run_slippage_curve(winner_trades_by_fold)

    # -----------------------------------------------------------------
    # R4 – tcost + break-even
    # -----------------------------------------------------------------
    print("  R4 tcost stress + break-even solvers ...")
    tcost_curve = run_tcost_curve(winner_trades_by_fold)
    be_tcost = find_break_even_tcost(winner_trades_by_fold)
    be_slip = find_break_even_slippage(winner_trades_by_fold)
    tcost_block = {
        "curve": tcost_curve,
        "break_even_cost_multiplier": be_tcost.get("break_even_multiplier"),
        "break_even_slippage_multiplier": be_slip.get("break_even_multiplier"),
        "break_even_cost_details": be_tcost,
        "break_even_slippage_details": be_slip,
    }

    # -----------------------------------------------------------------
    # execution delay
    # -----------------------------------------------------------------
    print("  R3b execution-delay stress ...")
    delay = run_delay_stress(named["phase6_winner"], fold_data)

    # -----------------------------------------------------------------
    # R5 – bootstrap
    # -----------------------------------------------------------------
    print("  R5 block/paired bootstrap ...")
    pf_ci  = block_bootstrap_pf(winner_pnl,  seed=config.seed,
                                    n_resamples=P7.BOOTSTRAP_B)
    net_ci = block_bootstrap_net(winner_pnl, seed=config.seed,
                                    n_resamples=P7.BOOTSTRAP_B)
    paired = paired_bootstrap_delta_pf(winner_pnl, baseline_pnl,
                                            seed=config.seed,
                                            n_resamples=P7.BOOTSTRAP_B)

    # -----------------------------------------------------------------
    # R6 – stability
    # -----------------------------------------------------------------
    print("  R6 stability diagnostics ...")
    stab = stability_report(winner_pnl)

    # -----------------------------------------------------------------
    # R7 – statistical tests on new comparisons
    # -----------------------------------------------------------------
    print("  R7 DM/SPA/WRC on delay variants ...")
    delay_variants_pnl = delay["per_fold_pnl_by_delay"]
    stats_block = {
        "dm_winner_vs_baseline_by_fold": dm_winner_vs_baseline_by_fold(
            winner_pnl, baseline_pnl),
        "spa_delay_variants": spa_and_wrc_over_variants(
            {f"delay_{d}": pnl for d, pnl in delay_variants_pnl.items()},
            baseline_pnl, seed=config.seed),
    }

    tornado = tornado_ranking({
        "walkforward": wf,
        "jackknife": jack,
        "slippage": slip,
        "tcost": tcost_block,
        "exec_delay": delay,
    }, baseline_pf=winner_pooled["pf"])

    target_result = {
        "regime": regime,
        "baseline_pooled_pf": baseline_pooled["pf"],
        "baseline_pooled_net": baseline_pooled["net"],
        "baseline_pooled_max_dd": baseline_pooled["dd"],
        "baseline_pooled_trade_count": baseline_pooled["n"],
        "walkforward": wf,
        "jackknife": jack,
        "slippage": slip,
        "tcost": tcost_block,
        "exec_delay": delay,
        "bootstrap": {"pf_ci": pf_ci, "net_ci": net_ci,
                        "paired_delta_pf": paired},
        "stability": stab,
        "stats": stats_block,
        "tornado": tornado,
    }
    target_result["h_rob"] = compute_h_rob_verdict(
        target_result,
        baseline_pf=baseline_pooled["pf"],
        baseline_net=baseline_pooled["net"])

    # persist manifest (per-target root) so downstream can verify cache
    tgt_root = config.target_root(target)
    manifest = build_manifest(
        root=Path("."), experiment_type="phase7_full", target=target,
        candidate=named["phase6_winner"].to_dict(),
        seed=config.seed,
        extra={"h_rob_verdict": target_result["h_rob"]["verdict"]})
    save_manifest(manifest, tgt_root)

    # per-target chart artifacts
    charts_dir = tgt_root / "charts"
    save_chart_data(target_result, charts_dir)
    if not config.skip_charts:
        write_chart_pngs(target_result, charts_dir)

    print(f"  H_rob verdict for {target}: "
            f"{target_result['h_rob']['verdict']}")
    return target_result


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="mean,stacking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-charts", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                     help="run only folds 1-2 for quick verification")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = Phase7Config(
        seed=args.seed,
        targets=tuple(t.strip() for t in args.targets.split(",") if t.strip()),
        skip_charts=args.skip_charts,
        force=args.force,
    )
    config.root.mkdir(parents=True, exist_ok=True)
    print(f"Phase 7 root: {config.root}")
    print(f"Targets: {config.targets}   seed: {config.seed}   "
            f"smoke: {args.smoke}")

    if not PHASE6_SUMMARY_PATH.exists():
        raise SystemExit(
            f"missing Phase 6 summary at {PHASE6_SUMMARY_PATH}. "
            "Phase 7 refuses to run without frozen Phase 6 outputs.")
    phase6_summary = json.loads(PHASE6_SUMMARY_PATH.read_text())

    all_targets = {}
    for t in config.targets:
        all_targets[t] = run_target(t, config, phase6_summary, args.smoke)

    print("\nWriting reports ...")
    paths = write_reports(
        config.root, all_targets,
        {"seed": config.seed, "targets": list(config.targets),
         "smoke": bool(args.smoke)})
    for k, p in paths.items():
        print(f"  {k}: {p}")

    md = render_phase7_md(
        config.root, all_targets,
        {"seed": config.seed, "targets": list(config.targets)})
    print(f"  markdown: {md}")

    # exit code = 0 always; verdict is content, not process signal
    print("\n=========================================================")
    for t, r in all_targets.items():
        v = r["h_rob"]["verdict"]
        print(f"  {t}: {v}")
    print("=========================================================")


if __name__ == "__main__":
    main()
