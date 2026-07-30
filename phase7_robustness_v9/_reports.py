"""Aggregation, H_rob verdict, and machine-readable JSON writers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._base import (H_ROB_CATASTROPHE_NET_MULT, H_ROB_CATASTROPHE_PF,
                       H_ROB_CI_LB_THRESHOLD, H_ROB_MAX_STABILITY_CV,
                       H_ROB_MIN_OUTPERFORM_FRAC, PHASE7_VERSION,
                       PROTOCOL_VERSION)


# ---------------------------------------------------------------------------
def _finite_pf(x) -> bool:
    try:
        return isinstance(x, (int, float)) and np.isfinite(float(x))
    except Exception:
        return False


def collect_scenarios(target_result: dict) -> list[dict]:
    """Flatten every robustness scenario for a single target into a list.

    Each scenario has ``{name, family, pooled_pf, pooled_net}`` at minimum.
    Used both for the outperform-fraction check and the tornado ranking.
    """
    winner_baseline_all = target_result["walkforward"]["fold_shift"][
        "baseline_all_folds"]["metrics"]
    win_pf_all = winner_baseline_all["pf"]
    win_net_all = winner_baseline_all["net"]

    rows: list[dict] = [
        {"name": "winner_all_folds", "family": "walkforward",
         "pooled_pf": win_pf_all, "pooled_net": win_net_all},
    ]

    # walk-forward variants
    fs = target_result["walkforward"]["fold_shift"]
    for k in ("shift_minus_1_drop_first", "shift_plus_1_drop_last"):
        m = fs[k]["metrics"]
        rows.append({"name": f"walkforward:{k}", "family": "walkforward",
                       "pooled_pf": m["pf"], "pooled_net": m["net"]})
    for row in target_result["walkforward"]["expanding_window"]:
        m = row["metrics"]
        rows.append({
            "name": f"walkforward:expanding_upto_{row['up_to_fold']}",
            "family": "walkforward",
            "pooled_pf": m["pf"], "pooled_net": m["net"]})
    for row in target_result["walkforward"]["rolling_window"]:
        m = row["metrics"]
        rows.append({
            "name": (f"walkforward:rolling_{row['start_fold']}"
                        f"..{row['end_fold']}"),
            "family": "walkforward",
            "pooled_pf": m["pf"], "pooled_net": m["net"]})

    # jackknife LOO
    for f, v in target_result["jackknife"]["per_dropped_fold"].items():
        rows.append({
            "name": f"jackknife:drop_fold_{f}",
            "family": "jackknife",
            "pooled_pf": v["pooled_pf"], "pooled_net": v["pooled_net"]})

    # slippage curve
    for mult, v in target_result["slippage"]["curve"].items():
        rows.append({
            "name": f"slippage:x{float(mult):.2f}",
            "family": "slippage",
            "pooled_pf": v["pooled_pf"], "pooled_net": v["pooled_net"]})

    # tcost curve
    for mult, v in target_result["tcost"]["curve"].items():
        rows.append({
            "name": f"tcost:x{float(mult):.2f}",
            "family": "tcost",
            "pooled_pf": v["pooled_pf"], "pooled_net": v["pooled_net"]})

    # exec delay
    for d, v in target_result["exec_delay"]["curve"].items():
        rows.append({
            "name": f"delay:{d}_bars",
            "family": "delay",
            "pooled_pf": v["pooled_pf"], "pooled_net": v["pooled_net"]})
    return rows


def compute_h_rob_verdict(target_result: dict,
                             baseline_pf: float,
                             baseline_net: float) -> dict:
    """Apply the user-approved H_rob decision rule.

    ACCEPT iff:
      * ≥ 80% of scenarios outperform baseline PF,
      * paired-bootstrap 90% CI LB of ΔPF > 0,
      * no scenario is catastrophic (PF < 0.8 or net-loss > 2× baseline net),
      * per-fold PF CV ≤ 0.5.
    REJECT if two or more of these criteria clearly fail.
    UNDECIDED otherwise.
    """
    scenarios = collect_scenarios(target_result)
    finite_scen = [s for s in scenarios if _finite_pf(s["pooled_pf"])]
    n_scen = len(finite_scen)
    n_out = sum(1 for s in finite_scen if s["pooled_pf"] > baseline_pf)
    out_frac = (n_out / n_scen) if n_scen else float("nan")

    ci_lb = target_result["bootstrap"]["paired_delta_pf"].get("lower")
    ci_lb_ok = (ci_lb is not None) and (ci_lb > H_ROB_CI_LB_THRESHOLD)

    catastrophic = [s for s in finite_scen
                    if (s["pooled_pf"] < H_ROB_CATASTROPHE_PF)
                    or (baseline_net > 0
                        and s["pooled_net"] < -H_ROB_CATASTROPHE_NET_MULT
                        * abs(baseline_net))]

    cv = target_result["stability"]["fold_variance"]["cv_per_fold_pf"]
    cv_ok = np.isfinite(cv) and cv <= H_ROB_MAX_STABILITY_CV

    criteria = {
        "outperform_frac": out_frac,
        "outperform_frac_ok": out_frac >= H_ROB_MIN_OUTPERFORM_FRAC
                              if np.isfinite(out_frac) else False,
        "ci_lb_delta_pf": ci_lb,
        "ci_lb_ok": ci_lb_ok,
        "catastrophic_scenarios": [s["name"] for s in catastrophic],
        "catastrophic_ok": len(catastrophic) == 0,
        "cv_per_fold_pf": cv,
        "cv_ok": cv_ok,
        "n_scenarios_evaluated": n_scen,
        "n_scenarios_outperform_baseline": n_out,
    }
    n_pass = sum(bool(criteria[k]) for k in
                 ("outperform_frac_ok", "ci_lb_ok",
                    "catastrophic_ok", "cv_ok"))
    if n_pass == 4:
        verdict = "ACCEPT"
    elif n_pass <= 1:
        verdict = "REJECT"
    else:
        verdict = "UNDECIDED"
    return {"verdict": verdict, "criteria": criteria}


def tornado_ranking(target_result: dict, baseline_pf: float) -> list[dict]:
    """Rank stress factors by how much they can pull PF away from baseline.

    For each stress family, take the worst PF observed within the family and
    report ``ΔPF = worst - baseline_pf``.
    """
    factors: list[tuple[str, float]] = []
    for k, v in target_result["slippage"]["curve"].items():
        factors.append(
            (f"slippage_x{float(k):.2f}", v["pooled_pf"]))
    for k, v in target_result["tcost"]["curve"].items():
        factors.append(
            (f"tcost_x{float(k):.2f}", v["pooled_pf"]))
    for k, v in target_result["exec_delay"]["curve"].items():
        factors.append((f"delay_{k}_bars", v["pooled_pf"]))
    for f, v in target_result["jackknife"]["per_dropped_fold"].items():
        factors.append((f"drop_fold_{f}", v["pooled_pf"]))

    rows = []
    for name, pf in factors:
        if not _finite_pf(pf):
            continue
        rows.append({
            "factor": name,
            "pooled_pf": float(pf),
            "delta_pf_vs_baseline": float(pf) - float(baseline_pf),
        })
    rows.sort(key=lambda r: r["delta_pf_vs_baseline"])
    return rows


# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dump_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def write_reports(root: Path, all_targets: dict, config_dict: dict) -> dict:
    """Write summary.json, robustness.json, stress_report.json,
    jackknife.json, bootstrap.json under ``root``. Returns the paths."""
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "phase7_version": PHASE7_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_dict,
        "targets": list(all_targets.keys()),
        "per_target": {},
    }

    robustness = {"per_target": {}}
    stress = {"per_target": {}}
    jack = {"per_target": {}}
    boot = {"per_target": {}}

    for t, r in all_targets.items():
        summary["per_target"][t] = {
            "regime": r["regime"]["counts"],
            "verdict": r["h_rob"]["verdict"],
            "criteria": r["h_rob"]["criteria"],
            "winner_pooled_pf": r["walkforward"]["fold_shift"][
                "baseline_all_folds"]["metrics"]["pf"],
            "baseline_pooled_pf": r["baseline_pooled_pf"],
            "break_even_slippage_multiplier": r["tcost"][
                "break_even_slippage_multiplier"],
            "break_even_cost_multiplier": r["tcost"][
                "break_even_cost_multiplier"],
            "top_5_tornado": r["tornado"][:5],
        }
        robustness["per_target"][t] = {
            "regime": r["regime"],
            "walkforward": r["walkforward"],
            "stability": r["stability"],
            "tornado": r["tornado"],
        }
        stress["per_target"][t] = {
            "slippage_curve": r["slippage"]["curve"],
            "tcost_curve": r["tcost"]["curve"],
            "exec_delay_curve": r["exec_delay"]["curve"],
            "break_even_cost_multiplier": r["tcost"][
                "break_even_cost_multiplier"],
            "break_even_slippage_multiplier": r["tcost"][
                "break_even_slippage_multiplier"],
        }
        jack["per_target"][t] = r["jackknife"]
        boot["per_target"][t] = r["bootstrap"]

    paths = {
        "summary":     dump_json(summary,   root / "summary.json"),
        "robustness":  dump_json(robustness, root / "robustness.json"),
        "stress":      dump_json(stress,    root / "stress_report.json"),
        "jackknife":   dump_json(jack,      root / "jackknife.json"),
        "bootstrap":   dump_json(boot,      root / "bootstrap.json"),
    }
    return paths


# ---------------------------------------------------------------------------
def render_phase7_md(root: Path, all_targets: dict, config_dict: dict) -> Path:
    """Publication-quality markdown report."""
    lines: list[str] = []
    lines.append("# Phase 7 – Robustness & Stress Testing Report")
    lines.append("")
    lines.append(f"*Generated {datetime.now(timezone.utc).isoformat()}*")
    lines.append(f"*Phase 7 version: {PHASE7_VERSION}*  |  "
                    f"*Protocol version: {PROTOCOL_VERSION}*")
    lines.append("")
    lines.append("## 1. Methodology")
    lines.append("")
    lines.append(
        "Phase 7 stresses the Phase-6 approved threshold candidates and the "
        "production baseline across walk-forward, jackknife, slippage, "
        "transaction-cost and execution-delay dimensions. It is strictly "
        "read-only with respect to Phases 0-6 — no model is retrained and no "
        "prediction file is regenerated. Trades are re-priced via "
        "`backtest_options.simulate_trades` (byte-identical to Phase 5) with "
        "either shifted signal vectors (execution-delay stress) or per-trade "
        "cost recomputed analytically from `prem_entry`/`prem_exit` using the "
        "same `round_trip_cost` decomposition (slippage / tcost stress).")
    lines.append("")
    lines.append("## 2. Experiments")
    lines.append("")
    lines.append("* R1 walk-forward: ±1 fold shift, expanding window, "
                    "3-fold rolling window.")
    lines.append("* R2 jackknife leave-one-fold-out.")
    lines.append("* R3 slippage stress: multipliers "
                    "1.00 / 1.25 / 1.50 / 1.75 / 2.00 of the bid-ask spread "
                    "portion of round-trip cost, plus bisection to solve "
                    "`break_even_slippage_multiplier`.")
    lines.append("* R4 transaction-cost stress: same multipliers applied to "
                    "the brokerage + STT + txn + SEBI + stamp + GST portion, "
                    "plus `break_even_cost_multiplier`.")
    lines.append("* R5 block bootstrap: pooled PF and pooled Net CIs, plus a "
                    "paired-fold bootstrap on ΔPF winner-vs-baseline.")
    lines.append("* R6 stability: per-fold PF variance, CV, 3-fold rolling.")
    lines.append("* R7 statistical: Diebold-Mariano on new comparisons and "
                    "Hansen SPA + White Reality Check over the union of "
                    "delay-stressed variants.")
    lines.append("* Regime tag on each frozen fold (bull/bear/sideways).")
    lines.append("* Execution-delay stress: 0/1/2-bar delay via signal shift "
                    "(trade-generation logic unchanged).")
    lines.append("")
    lines.append("## 3. Statistical results")
    lines.append("")
    for t, r in all_targets.items():
        lines.append(f"### Target `{t}`")
        lines.append("")
        wa = r["walkforward"]["fold_shift"]["baseline_all_folds"]["metrics"]
        lines.append(
            f"* Winner pooled PF (all folds) : **{wa['pf']:.3f}**  "
            f"| Net Rs. **{wa['net']:+,.0f}**  "
            f"| MaxDD Rs. {wa['dd']:,.0f}  | n={wa['n']}")
        lines.append(
            f"* Baseline pooled PF          : "
            f"**{r['baseline_pooled_pf']:.3f}**  "
            f"| Net Rs. **{r['baseline_pooled_net']:+,.0f}**")
        cib = r["bootstrap"]["paired_delta_pf"]
        lines.append(
            f"* Paired bootstrap 90% CI on ΔPF: "
            f"[{cib['lower']:+.3f}, {cib['upper']:+.3f}] "
            f"point {cib['point_estimate']:+.3f}")
        pf_ci = r["bootstrap"]["pf_ci"]
        net_ci = r["bootstrap"]["net_ci"]
        lines.append(
            f"* Bootstrap PF 90% CI: "
            f"[{pf_ci['lower']:.3f}, {pf_ci['upper']:.3f}]  |  "
            f"Net 90% CI: [Rs.{net_ci['lower']:+,.0f}, "
            f"Rs.{net_ci['upper']:+,.0f}]")
        lines.append(
            f"* Break-even multipliers: slippage "
            f"= {r['tcost']['break_even_slippage_multiplier']}, "
            f"tcost = {r['tcost']['break_even_cost_multiplier']}")
        dm = r["stats"].get("dm_delay_1_vs_baseline")
        if dm:
            lines.append(
                f"* DM winner-vs-baseline (delay=0): "
                f"stat {dm['statistic']:+.3f} p={dm['pvalue']:.4f}")
        spa = r["stats"].get("spa_delay_variants", {}).get("hansen_spa")
        if spa:
            lines.append(
                f"* SPA over delay variants: "
                f"p_lower={spa['pvalue_lower']:.4f}")
        lines.append("")
        lines.append("**Regime distribution across the frozen folds**")
        lines.append("")
        counts = r["regime"]["counts"]
        lines.append("| Regime | Fold count |")
        lines.append("|---|---:|")
        for k in ("BULL", "BEAR", "SIDEWAYS", "UNKNOWN"):
            if k in counts:
                lines.append(f"| {k} | {counts[k]} |")
        lines.append("")
        lines.append("**Top-5 tornado (biggest PF drops):**")
        lines.append("")
        lines.append("| Factor | PF | ΔPF vs winner |")
        lines.append("|---|---:|---:|")
        for row in r["tornado"][:5]:
            lines.append(
                f"| {row['factor']} | {row['pooled_pf']:.3f} | "
                f"{row['delta_pf_vs_baseline']:+.3f} |")
        lines.append("")
    lines.append("## 4. Robustness verdicts (H_rob)")
    lines.append("")
    lines.append("Rule (locked): ACCEPT iff ≥ 80 % of scenarios outperform "
                    "baseline PF **AND** paired-bootstrap 90 % CI lower bound "
                    "on ΔPF > 0 **AND** no scenario is catastrophic (PF < 0.8 "
                    "or Net loss > 2× baseline net) **AND** per-fold PF CV "
                    "≤ 0.5. REJECT if ≥ 3 criteria fail; otherwise UNDECIDED.")
    lines.append("")
    lines.append("| Target | Verdict | Outperform % | CI LB(ΔPF) | Catastrophic | CV |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for t, r in all_targets.items():
        c = r["h_rob"]["criteria"]
        of = f"{c['outperform_frac']*100:.1f}%" \
             if c['outperform_frac'] is not None and np.isfinite(
                 c['outperform_frac']) else "n/a"
        clb = c.get("ci_lb_delta_pf")
        clb_s = f"{clb:+.3f}" if clb is not None else "n/a"
        cats = len(c["catastrophic_scenarios"])
        cv = c["cv_per_fold_pf"]
        cv_s = f"{cv:.3f}" if np.isfinite(cv) else "n/a"
        lines.append(f"| `{t}` | **{r['h_rob']['verdict']}** | "
                        f"{of} | {clb_s} | {cats} | {cv_s} |")
    lines.append("")
    lines.append("## 5. Limitations")
    lines.append("")
    lines.append(
        "* Phase 7 does not retrain models; fold shifts drop edge folds "
        "rather than re-splitting the training window.")
    lines.append(
        "* Slippage and tcost stress reuse the analytical `round_trip_cost` "
        "decomposition. A gap or a jump-diffusion overlay would need a real "
        "re-simulation with a different fill model.")
    lines.append(
        "* Statistical tests operate on N = 8 folds; power is limited and "
        "the paired-bootstrap CI is wide.")
    lines.append("")
    lines.append("## 6. Production recommendation")
    lines.append("")
    for t, r in all_targets.items():
        lines.append(f"* `{t}`: **{r['h_rob']['verdict']}** — "
                        f"see `summary.json` for full criteria.")
    verdicts = {t: r['h_rob']['verdict'] for t, r in all_targets.items()}
    if all(v == "ACCEPT" for v in verdicts.values()):
        overall = "APPROVE for staged rollout"
    elif any(v == "REJECT" for v in verdicts.values()):
        overall = "DO NOT deploy without further evidence"
    else:
        overall = "HOLD — additional live/paper evidence required"
    lines.append("")
    lines.append(f"**Overall recommendation: {overall}.**")
    lines.append("")
    md = root / "Phase7_Report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
