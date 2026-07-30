"""Phase 7 – Robustness & Stress Testing (additive to Phases 0-6)."""
from __future__ import annotations

from ._base import (
    BOOTSTRAP_B, BULL_MIN_ANNUALISED, BEAR_MAX_ANNUALISED,
    CacheMismatchError, DEFAULT_SEED, EXEC_DELAYS_BARS,
    FOLDS, H_ROB_CATASTROPHE_NET_MULT, H_ROB_CATASTROPHE_PF,
    H_ROB_CI_LB_THRESHOLD, H_ROB_MAX_STABILITY_CV,
    H_ROB_MIN_OUTPERFORM_FRAC, InvalidInputError,
    MANIFEST_SCHEMA_VERSION, MissingManifestError,
    NamedCandidate, PHASE7_VERSION, PROTOCOL_VERSION,
    Phase7Config, Phase7Error, PRODUCTION_BASELINE, ROLLING_WINDOW,
    SLIPPAGE_MULTIPLIERS, TCOST_MULTIPLIERS, ThresholdCandidate,
    hash8, sha256_of_bytes, sha256_of_file,
)
from ._bootstrap import (block_bootstrap_net, block_bootstrap_pf,
                            paired_bootstrap_delta_pf)
from ._cache import (build_manifest, code_hash_dict,
                        input_hash_for_target, load_manifest,
                        save_manifest, verify_cache)
from ._delay import run_delay_stress
from ._jackknife import leave_one_out
from ._regime import classify_fold, classify_folds
from ._replay import (FoldReplay, apply_cost_stress, load_fold_data,
                        non_spread_cost_component,
                        pooled_metrics_from_replays,
                        pooled_metrics_from_stressed_pnl,
                        per_fold_metrics, simulate_candidate,
                        spread_cost_component)
from ._reports import (collect_scenarios, compute_h_rob_verdict,
                          dump_json, render_phase7_md, tornado_ranking,
                          write_reports)
from ._slippage import run_slippage_curve
from ._stability import (flag_unstable_folds, fold_variance,
                            rolling_report, stability_report)
from ._stats import (dm_winner_vs_baseline_by_fold,
                        spa_and_wrc_over_variants)
from ._tcost import (find_break_even_slippage, find_break_even_tcost,
                        run_tcost_curve)
from ._visualize import build_chart_data, save_chart_data, write_chart_pngs
from ._walkforward import (expanding_window_variants, fold_shift_variants,
                              rolling_window_variants, walkforward_report)

__all__ = [
    "BOOTSTRAP_B", "BULL_MIN_ANNUALISED", "BEAR_MAX_ANNUALISED",
    "CacheMismatchError", "DEFAULT_SEED", "EXEC_DELAYS_BARS",
    "FoldReplay", "FOLDS", "H_ROB_CATASTROPHE_NET_MULT",
    "H_ROB_CATASTROPHE_PF", "H_ROB_CI_LB_THRESHOLD",
    "H_ROB_MAX_STABILITY_CV", "H_ROB_MIN_OUTPERFORM_FRAC",
    "InvalidInputError", "MANIFEST_SCHEMA_VERSION",
    "MissingManifestError", "NamedCandidate", "PHASE7_VERSION",
    "PROTOCOL_VERSION", "Phase7Config", "Phase7Error",
    "PRODUCTION_BASELINE", "ROLLING_WINDOW",
    "SLIPPAGE_MULTIPLIERS", "TCOST_MULTIPLIERS", "ThresholdCandidate",
    "apply_cost_stress",
    "block_bootstrap_net", "block_bootstrap_pf",
    "build_chart_data", "build_manifest", "classify_fold",
    "classify_folds", "code_hash_dict", "collect_scenarios",
    "compute_h_rob_verdict", "dm_winner_vs_baseline_by_fold",
    "dump_json", "expanding_window_variants",
    "find_break_even_slippage", "find_break_even_tcost",
    "flag_unstable_folds", "fold_shift_variants", "fold_variance",
    "hash8", "input_hash_for_target",
    "leave_one_out", "load_fold_data", "load_manifest",
    "non_spread_cost_component",
    "paired_bootstrap_delta_pf", "per_fold_metrics",
    "pooled_metrics_from_replays", "pooled_metrics_from_stressed_pnl",
    "render_phase7_md", "rolling_report", "rolling_window_variants",
    "run_delay_stress", "run_slippage_curve", "run_tcost_curve",
    "save_chart_data", "save_manifest", "sha256_of_bytes",
    "sha256_of_file", "simulate_candidate",
    "spa_and_wrc_over_variants", "spread_cost_component",
    "stability_report", "tornado_ranking", "verify_cache",
    "walkforward_report", "write_chart_pngs", "write_reports",
]
