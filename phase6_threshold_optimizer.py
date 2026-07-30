"""
phase6_threshold_optimizer.py - Threshold-optimization orchestrator.

Reads FROZEN Phase-5 ensemble predictions, applies each grid candidate,
re-runs simulate_trades on the outer test windows, and produces:

  - candidate_results.csv per target ensemble
  - manifest per candidate (with cache validation)
  - statistical comparison of #1 vs production baseline
  - Hansen SPA + White Reality Check + Holm on top-K=10
  - CALL vs PF, PUT vs PF, CALL x PUT PF heatmap, CALL x PUT trade-count heatmap
  - machine-readable summary.json + human-readable comparison.txt

Zero modification of Phase 0-5 code, artifacts, or manifests.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import argparse
import ctypes
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from threshold_opt import (
    CandidateResult, PRODUCTION_BASELINE, ThresholdCandidate,
    build_chart_data, build_manifest, compare_to_baseline,
    evaluate_candidate, grid_generator, grid_size,
    load_manifest, rank_candidates, save_chart_data, sha256_of_file,
    top_k_comparison, verify_cache, write_chart_pngs,
)

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

DEFAULT_TARGETS = ("mean", "stacking")
DEFAULT_INPUT = "calibrated_isotonic"
DEFAULT_MIN_TRADES = 50
TOP_K = 10
PHASE6_VERSION = "1.0"

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase6_threshold_optimizer.py",
    "threshold_opt/__init__.py",
    "threshold_opt/_base.py",
    "threshold_opt/_grid.py",
    "threshold_opt/_evaluate.py",
    "threshold_opt/_ranking.py",
    "threshold_opt/_manifest.py",
    "threshold_opt/_stats.py",
    "threshold_opt/_visualize.py",
)


# ---------------------------------------------------------------------------
def _source_hash() -> dict[str, str]:
    return {f: sha256_of_file(ROOT / f) for f in _MODULES_TRACKED
            if (ROOT / f).exists()}


def _phase5_pred_path(ensemble: str, fold_idx: int) -> Path:
    d = ROOT / "logs/phase5" / ensemble / f"fold_{fold_idx}"
    for name in ("predictions.parquet", "predictions.csv"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Phase-5 predictions missing for "
                             f"{ensemble}/fold_{fold_idx}")


def _data_hash(ensemble: str, folds: list[int]) -> dict[str, str]:
    return {f"phase5_{ensemble}_fold_{f}_predictions":
              sha256_of_file(_phase5_pred_path(ensemble, f))
            for f in folds}


def _load_predictions(pred_path: Path) -> pd.DataFrame:
    if pred_path.suffix == ".parquet":
        try:
            return pd.read_parquet(pred_path)
        except Exception:
            pass
    return pd.read_csv(pred_path)


def _prevent_sleep_windows() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001))
    except Exception:
        return False


# ---------------------------------------------------------------------------
def load_fold_data(ensemble: str, feat: pd.DataFrame,
                     iv: dict, exp: dict,
                     folds: list[int]) -> dict[int, tuple]:
    """Pre-load per-fold (probs, test_df, iv, exp) tuples once per target.

    Reused across all 360 candidates.
    """
    out: dict[int, tuple] = {}
    for i in folds:
        a, b = FOLDS[i - 1]
        te_mask = ((feat.index.date >= a) & (feat.index.date < b))
        test_df = feat[te_mask]
        pred_df = _load_predictions(_phase5_pred_path(ensemble, i))
        if len(pred_df) != len(test_df):
            raise RuntimeError(
                f"length mismatch for {ensemble}/fold_{i}: "
                f"predictions={len(pred_df)} vs test={len(test_df)}")
        probs = pred_df[["p_call", "p_put", "p_skip"]].values.astype(
            np.float64)
        out[i] = (probs, test_df, iv, exp)
    return out


# ---------------------------------------------------------------------------
def _cache_ok_for_candidate(
    cand_dir: Path,
    current_code: dict, current_data: dict, min_trades: int,
) -> tuple[bool, str]:
    """Return (is_cached, reason)."""
    m_path = cand_dir / "manifest.json"
    r_path = cand_dir / "result.json"
    if not m_path.exists() or not r_path.exists():
        return False, "no artifacts"
    try:
        m = load_manifest(m_path)
    except Exception as exc:
        return False, f"unreadable manifest ({exc})"
    if not verify_cache(m, current_code, current_data):
        return False, "hash mismatch"
    if int(m.get("min_trades_requirement", -1)) != int(min_trades):
        return False, "min_trades differs"
    return True, "ok"


def _persist_candidate(
    cand_dir: Path,
    cr: CandidateResult,
    target_ensemble: str,
    input_source: str,
    folds: list[int],
    min_trades: int,
    code_hash: dict,
    data_hash: dict,
    protocol_version: str,
    seed: int,
) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    n_trades_per_fold = [cr.per_fold[f].get("n", 0) for f in folds]
    m = build_manifest(
        cand=cr.candidate,
        target_ensemble=target_ensemble,
        input_source=input_source,
        folds_evaluated=folds,
        n_trades_per_fold=n_trades_per_fold,
        min_trades_requirement=min_trades,
        passes_min_trades_filter=cr.passes_min_trades,
        code_hash=code_hash,
        data_hash=data_hash,
        protocol_version=protocol_version,
        seed=seed,
    )
    with open(cand_dir / "manifest.json", "w") as fh:
        json.dump(m, fh, indent=2, default=str)

    # result.json: per-fold + pooled metrics + trade-pnl (compact)
    result = {
        "threshold_values": cr.candidate.to_dict(),
        "per_fold": {int(k): v for k, v in cr.per_fold.items()},
        "pooled": cr.pooled,
        "passes_min_trades": cr.passes_min_trades,
    }
    with open(cand_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    # per-fold trade pnl
    for f, pnl in cr.trade_pnl_by_fold.items():
        pd.DataFrame({"trade_pnl": pnl}).to_csv(
            cand_dir / f"trade_pnl_fold_{f}.csv", index=False)


# ---------------------------------------------------------------------------
def evaluate_target(
    ensemble: str,
    feat: pd.DataFrame,
    iv: dict, exp: dict,
    args,
    protocol_version: str,
    baseline_cand: ThresholdCandidate,
    include_baseline: bool = True,
) -> tuple[list[CandidateResult], CandidateResult | None]:
    """Evaluate all grid candidates + optionally the baseline for one target.

    Returns (all_candidates, baseline_result). ``baseline_result`` is
    returned separately and is NEVER included in ``all_candidates`` — it
    exists solely for comparison.
    """
    print(f"\n[Phase 6 / {ensemble}] pre-loading fold data ...")
    folds = list(range(1, len(FOLDS) + 1))
    try:
        fold_data = load_fold_data(ensemble, feat, iv, exp, folds)
    except FileNotFoundError as exc:
        raise SystemExit(f"missing Phase-5 predictions: {exc}") from exc

    code_hash = _source_hash()
    data_hash = _data_hash(ensemble, folds)

    n_total = grid_size()
    print(f"  grid size: {n_total} candidates over {len(folds)} folds "
          f"= {n_total * len(folds)} evaluations")

    ens_root = ROOT / "logs/phase6" / ensemble
    ens_root.mkdir(parents=True, exist_ok=True)

    all_results: list[CandidateResult] = []
    t0 = time.time()
    for idx, cand in enumerate(grid_generator(), start=1):
        cand_dir = ens_root / f"thr_{cand.hash8()}"
        cached, reason = _cache_ok_for_candidate(
            cand_dir, code_hash, data_hash, args.min_trades)
        if cached and not args.force:
            # Load cached result
            r = json.loads((cand_dir / "result.json").read_text())
            cr = CandidateResult(
                candidate=cand,
                per_fold={int(k): v for k, v in r["per_fold"].items()},
                pooled=r["pooled"],
                passes_min_trades=r["passes_min_trades"],
                trade_pnl_by_fold={
                    int(f.stem.split("_")[-1]):
                        pd.read_csv(f)["trade_pnl"].values.astype(np.float64)
                    for f in cand_dir.glob("trade_pnl_fold_*.csv")
                },
            )
        else:
            cr = evaluate_candidate(cand, fold_data,
                                       min_trades=args.min_trades)
            _persist_candidate(
                cand_dir, cr, ensemble, args.input,
                folds, args.min_trades,
                code_hash, data_hash, protocol_version, args.seed,
            )
        all_results.append(cr)
        if idx % 30 == 0 or idx == n_total:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            eta = (n_total - idx) / max(rate, 1e-6)
            print(f"  [{idx}/{n_total}]  {elapsed:.1f}s elapsed  "
                  f"ETA {eta:.0f}s")

    # Baseline (comparison-only, NEVER in candidate ranking pool)
    baseline_result = None
    if include_baseline:
        print(f"  evaluating production baseline for comparison ...")
        baseline_result = evaluate_candidate(
            baseline_cand, fold_data, min_trades=args.min_trades)

    return all_results, baseline_result


# ---------------------------------------------------------------------------
def export_candidate_table(results: list[CandidateResult],
                             ensemble: str, out_dir: Path) -> Path:
    """Write candidate_results.csv with full search table."""
    rows = []
    ranked = rank_candidates(results, min_pooled_trades=1)
    rank_map = {id(r): i + 1 for i, r in enumerate(ranked)}
    for r in results:
        c = r.candidate
        per_fold_pfs = []
        for f in sorted(r.per_fold):
            pf = r.per_fold[f].get("pf")
            per_fold_pfs.append(pf if isinstance(pf, (int, float)) else None)
        rows.append({
            "call_thr": c.call_thr,
            "put_thr": c.put_thr,
            "skip_ceil": c.skip_ceil,
            "min_edge": c.min_edge,
            "pooled_PF": r.pooled.get("pf"),
            "pooled_Net": r.pooled.get("net"),
            "pooled_MaxDD": r.pooled.get("dd"),
            "pooled_TradeCount": r.pooled.get("n"),
            "pooled_WR": r.pooled.get("wr"),
            "pooled_Sharpe": r.pooled.get("sharpe"),
            "pooled_Sortino": r.pooled.get("sortino"),
            "per_fold_PF": json.dumps(per_fold_pfs, default=str),
            "passes_min_trades_filter": r.passes_min_trades,
            "rank": rank_map.get(id(r), -1),
        })
    df = pd.DataFrame(rows).sort_values(
        by=["rank"], ascending=True, na_position="last").reset_index(drop=True)
    p = out_dir / "candidate_results.csv"
    df.to_csv(p, index=False)
    try:
        df.to_parquet(out_dir / "candidate_results.parquet", index=False)
    except Exception:
        pass
    return p


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="comma-separated Phase-5 ensembles to optimize")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    choices=["raw", "calibrated_noop", "calibrated_platt",
                              "calibrated_isotonic"],
                    help="probability source (default: calibrated_isotonic)")
    ap.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES,
                    help="pooled trade-count filter (default 50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--skip-charts", action="store_true")
    ap.add_argument("--targets-only", action="store_true",
                    help="skip aggregation, just evaluate")
    args = ap.parse_args()

    if args.verify_prereg:
        from verify_preregistration import verify
        verify()

    (ROOT / "logs/phase6").mkdir(parents=True, exist_ok=True)
    prevented = _prevent_sleep_windows()
    if platform.system() == "Windows":
        print(f"[pre-flight] Windows sleep prevention: {prevented}")

    # Protocol version
    try:
        prereg = json.loads((ROOT / "pre_registration"
                                / "preregistration_active.json").read_text())
        pv = prereg.get("protocol", {}).get("protocol_version", "?")
    except Exception:
        pv = "?"

    print(f"\n{'=' * 72}")
    print(f"  Phase 6 - Threshold Optimization")
    print(f"{'=' * 72}")
    print(f"  targets     : {args.targets}")
    print(f"  input       : {args.input}")
    print(f"  min_trades  : {args.min_trades}")
    print(f"  seed        : {args.seed}")
    print(f"  grid size   : {grid_size()} candidates")
    print(f"  protocol_v  : {pv}")

    # Build frame + iv/exp once
    print(f"\n[pre-flight] build_frame + build_iv_map ...")
    from backtest_threshold_sweep import build_frame
    from backtest_options import build_iv_map
    feat, _ = build_frame()
    iv, exp = build_iv_map()

    # Evaluate each target
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    per_target_summary: dict[str, dict] = {}
    all_top_k_streams: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    baseline_streams_per_target: dict[str, dict[int, np.ndarray]] = {}

    for target in targets:
        all_results, baseline_result = evaluate_target(
            target, feat, iv, exp, args, pv,
            baseline_cand=PRODUCTION_BASELINE, include_baseline=True,
        )
        ens_root = ROOT / "logs/phase6" / target
        table_path = export_candidate_table(all_results, target, ens_root)
        print(f"  wrote {table_path}")

        # Rank + winner
        ranked = rank_candidates(all_results,
                                    min_pooled_trades=args.min_trades)
        if not ranked:
            print(f"  {target}: no eligible candidates "
                  f"(min_trades={args.min_trades})")
            per_target_summary[target] = {"status": "NO_ELIGIBLE"}
            continue
        winner = ranked[0]

        # Compare winner vs baseline
        cmp = compare_to_baseline(
            winner.trade_pnl_by_fold,
            baseline_result.trade_pnl_by_fold if baseline_result else {},
            seed=args.seed,
        )

        # Top-K comparison + SPA/WRC/Holm
        top_k = ranked[:min(TOP_K, len(ranked))]
        top_k_streams = {
            f"cand_{i+1}_thr_{r.candidate.hash8()}": r.trade_pnl_by_fold
            for i, r in enumerate(top_k)
        }
        top_k_result = top_k_comparison(
            top_k_streams,
            baseline_result.trade_pnl_by_fold if baseline_result else {},
            seed=args.seed,
        )

        # Chart data + PNGs
        charts_dir = ens_root / "charts"
        chart_data = build_chart_data(all_results)
        save_chart_data(chart_data, charts_dir)
        if not args.skip_charts:
            written = write_chart_pngs(chart_data, charts_dir,
                                         title_prefix=f"[{target}] ")
            print(f"  wrote {len(written)} chart PNG(s) to {charts_dir}")

        per_target_summary[target] = {
            "grid_size": grid_size(),
            "n_eligible": len(ranked),
            "min_trades_requirement": args.min_trades,
            "winner_candidate": winner.candidate.to_dict(),
            "winner_hash": winner.candidate.hash8(),
            "winner_pooled": winner.pooled,
            "baseline_candidate": PRODUCTION_BASELINE.to_dict(),
            "baseline_hash": PRODUCTION_BASELINE.hash8(),
            "baseline_pooled": (baseline_result.pooled
                                  if baseline_result else None),
            "delta_pf_point": (
                (winner.pooled.get("pf") or 0.0)
                - (baseline_result.pooled.get("pf") or 0.0)
                if baseline_result else None),
            "delta_net": (
                (winner.pooled.get("net") or 0.0)
                - (baseline_result.pooled.get("net") or 0.0)
                if baseline_result else None),
            "delta_max_dd": (
                (winner.pooled.get("dd") or 0.0)
                - (baseline_result.pooled.get("dd") or 0.0)
                if baseline_result else None),
            "delta_trade_count": (
                int(winner.pooled.get("n") or 0)
                - int(baseline_result.pooled.get("n") or 0)
                if baseline_result else None),
            "comparison_vs_baseline": cmp,
            "top_k_vs_baseline": top_k_result,
            "top_k_candidates": [
                {"rank": i + 1,
                 "candidate": r.candidate.to_dict(),
                 "hash": r.candidate.hash8(),
                 "pooled_pf": r.pooled.get("pf"),
                 "pooled_net": r.pooled.get("net"),
                 "pooled_trade_count": r.pooled.get("n"),
                 "pooled_max_dd": r.pooled.get("dd")}
                for i, r in enumerate(top_k)
            ],
            "candidate_table": str(table_path.relative_to(ROOT)),
            "chart_data": str((charts_dir / "chart_data.json").relative_to(ROOT)),
        }

        # Human-readable per-target summary
        print(f"\n{'=' * 72}")
        print(f"  Phase 6 winner for '{target}'")
        print(f"{'=' * 72}")
        w = winner.candidate.to_dict()
        b = PRODUCTION_BASELINE.to_dict()
        print(f"  winner   : call={w['call_thr']:.2f}  put={w['put_thr']:.2f}  "
              f"skip={w['skip_ceil']:.2f}  edge={w['min_edge']:.2f}")
        print(f"  baseline : call={b['call_thr']:.2f}  put={b['put_thr']:.2f}  "
              f"skip={b['skip_ceil']:.2f}  edge={b['min_edge']:.2f}")
        print(f"  winner PF={winner.pooled.get('pf'):.3f}  "
              f"trades={winner.pooled.get('n')}  "
              f"net=Rs.{winner.pooled.get('net'):+,.0f}")
        if baseline_result:
            print(f"  baseline PF={baseline_result.pooled.get('pf'):.3f}  "
                  f"trades={baseline_result.pooled.get('n')}  "
                  f"net=Rs.{baseline_result.pooled.get('net'):+,.0f}")
        ci = cmp.get("paired_ci_90", {})
        print(f"  delta_PF 90% CI: [{ci.get('lower'):+.3f}, "
              f"{ci.get('upper'):+.3f}]  point={ci.get('point_estimate'):+.3f}")
        dm = cmp.get("diebold_mariano")
        if isinstance(dm, dict) and "pvalue" in dm:
            print(f"  Diebold-Mariano p(greater) : {dm['pvalue']:.4f}")
        spa = top_k_result.get("hansen_spa", {})
        if spa:
            print(f"  Hansen SPA p_lower         : "
                  f"{spa.get('pvalue_lower'):.4f}")
        wrc = top_k_result.get("white_reality_check", {})
        if wrc:
            print(f"  White RC p-value           : {wrc.get('pvalue'):.4f}")

    # H_thr verdict
    print(f"\n{'=' * 72}")
    print(f"  Phase 6 H_thr verdict")
    print(f"{'=' * 72}")
    print(f"  rule: LB90(delta_PF) > +0.05 AND DM_p < 0.10/7 = 0.0143 "
          f"(FWE-adjusted)")
    verdicts: dict[str, str] = {}
    for t, s in per_target_summary.items():
        if s.get("status") == "NO_ELIGIBLE":
            verdicts[t] = "NO_ELIGIBLE"
            continue
        ci = s.get("comparison_vs_baseline", {}).get("paired_ci_90", {})
        dm = s.get("comparison_vs_baseline", {}).get("diebold_mariano") or {}
        lb = ci.get("lower")
        dm_p = dm.get("pvalue")
        if isinstance(lb, (int, float)) and lb > 0.05 \
                and isinstance(dm_p, float) and dm_p < 0.10 / 7:
            v = "ACCEPT"
        else:
            hw = None
            if isinstance(ci.get("upper"), (int, float)) and isinstance(lb, (int, float)):
                hw = (ci["upper"] - lb) / 2
            if isinstance(hw, (int, float)) and hw > 2 * 0.05:
                v = "UNDECIDED"
            else:
                v = "FAIL_TO_REJECT"
        verdicts[t] = v
        print(f"    {t:>10}: {v}   LB90={lb}   DM_p={dm_p}")

    summary_path = ROOT / "logs/phase6/summary.json"
    with open(summary_path, "w") as fh:
        json.dump({
            "phase6_version": PHASE6_VERSION,
            "protocol_version": pv,
            "input_source": args.input,
            "min_trades_requirement": args.min_trades,
            "seed": args.seed,
            "grid_size": grid_size(),
            "top_k": TOP_K,
            "targets": targets,
            "h_thr_verdicts": verdicts,
            "per_target": per_target_summary,
        }, fh, indent=2, default=str)
    print(f"\n  wrote {summary_path}")


if __name__ == "__main__":
    main()
