"""
phase6_final_report.py — Read outputs of Phases 1-5 and mechanically answer
the 7 evidence questions. No speculation. Only measured results.

Reads:
  logs/phase1_avg4_full.csv        (Phase 1)
  logs/phase2_summary.json         (Phase 2)
  logs/phase3_summary.json         (Phase 3)
  logs/phase4_calibration.json     (Phase 4)
  logs/phase5_target_audit.json    (Phase 5)
  logs/sweep_A_current.csv         (baseline reference: PF 0.75)

Emits: logs/phase6_final_report.json + printed summary.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BASELINE_PF = 0.75          # measured baseline (XGB+LGB) from prior audit
MATERIAL_PF_GAIN = 0.05     # threshold for "materially improved"


def load_json(name):
    p = ROOT / f"logs/{name}"
    if not p.exists(): return None
    with open(p) as fh: return json.load(fh)


def pf_of_csv(name):
    p = ROOT / f"logs/{name}"
    if not p.exists(): return None
    d = pd.read_csv(p)
    if d.empty or "net_option" not in d.columns: return None
    net = d.net_option.values
    g = net[net > 0].sum(); l = -net[net <= 0].sum()
    return dict(n=int(len(net)), pf=float(g / l) if l > 0 else float("inf"),
                wr=float((net > 0).mean() * 100), net=float(net.sum()))


def main():
    print("=" * 78)
    print("  PHASE 6 — FINAL EVIDENCE REPORT")
    print("=" * 78)

    # ---- Phase 1 ----
    p1 = pf_of_csv("phase1_avg4_full.csv")
    print("\n### Phase 1 — avg4 8-fold walk-forward")
    if p1: print(f"  {p1}")
    else:  print("  Not run.")

    # ---- Phase 2 ----
    p2 = load_json("phase2_summary.json")
    print("\n### Phase 2 — XGB HPO")
    if p2:
        c, t = p2.get("current", {}), p2.get("tuned", {})
        print(f"  current: PF={c.get('pf'):.3f} net=Rs.{c.get('net'):+,.0f}")
        print(f"  tuned:   PF={t.get('pf'):.3f} net=Rs.{t.get('net'):+,.0f}")
    else: print("  Not run.")

    # ---- Phase 3 ----
    p3 = load_json("phase3_summary.json")
    print("\n### Phase 3 — 5-brain comparison")
    if p3:
        for row in p3:
            print(f"  {row['config']:<14} PF={row['pf']:.3f} WR={row['wr']:.0f}% Net=Rs.{row['net']:+,.0f}")
    else: print("  Not run.")

    # ---- Phase 4 ----
    p4 = load_json("phase4_calibration.json")
    print("\n### Phase 4 — Calibration")
    if p4:
        for brain, rec in p4.items():
            print(f"  {brain:<8} ECE={rec['ece']:.4f}  Brier={rec['brier']:.4f}  "
                  f"mean_conf={rec['confidence_hist']['mean_conf']:.3f}")
    else: print("  Not run.")

    # ---- Phase 5 ----
    p5 = load_json("phase5_target_audit.json")
    print("\n### Phase 5 — Target audit")
    if p5:
        for k, v in p5.items(): print(f"  {k}: {v}")
    else: print("  Not run.")

    # ============================================================
    # Answers to the 7 evidence questions
    # ============================================================
    print("\n" + "=" * 78)
    print("  ANSWERS (mechanical, evidence-driven)")
    print("=" * 78)
    answers = {}

    # Q1: hyperparameter optimization materially improved performance?
    if p2 and "current" in p2 and "tuned" in p2:
        delta = p2["tuned"]["pf"] - p2["current"]["pf"]
        q1 = "YES" if delta >= MATERIAL_PF_GAIN else "NO"
        q1_evidence = f"PF change: {p2['current']['pf']:.3f} -> {p2['tuned']['pf']:.3f} (delta {delta:+.3f}); material threshold {MATERIAL_PF_GAIN}"
    else:
        q1 = "Unknown"; q1_evidence = "Phase 2 not completed"
    answers["Q1_hpo_material"] = {"answer": q1, "evidence": q1_evidence}

    # Q2: CatBoost materially outperformed existing?
    if p3:
        best_existing = max((r for r in p3 if r["config"] in ("xgb", "lgb", "rf", "nn")),
                            key=lambda r: r["pf"], default=None)
        cat_row = next((r for r in p3 if r["config"] == "cat"), None)
        if best_existing and cat_row:
            delta = cat_row["pf"] - best_existing["pf"]
            q2 = "YES" if delta >= MATERIAL_PF_GAIN else "NO"
            q2_evidence = f"CatBoost PF={cat_row['pf']:.3f} vs best existing ({best_existing['config']}) PF={best_existing['pf']:.3f}; delta {delta:+.3f}"
        else:
            q2 = "Unknown"; q2_evidence = "Phase 3 incomplete"
    else:
        q2 = "Unknown"; q2_evidence = "Phase 3 not completed"
    answers["Q2_catboost_material"] = {"answer": q2, "evidence": q2_evidence}

    # Q3: poor calibration limiting?
    if p4:
        ece_values = [(b, r["ece"]) for b, r in p4.items()]
        max_ece = max(ece_values, key=lambda x: x[1])
        min_ece = min(ece_values, key=lambda x: x[1])
        # Rule: if any brain has ECE > 0.10 AND another has ECE < 0.05, calibration matters
        if max_ece[1] > 0.10 and min_ece[1] < 0.05:
            q3 = "YES"
            q3_evidence = f"Wide ECE spread: {min_ece[0]}={min_ece[1]:.3f} to {max_ece[0]}={max_ece[1]:.3f}"
        elif max_ece[1] > 0.15:
            q3 = "YES"; q3_evidence = f"Worst brain ({max_ece[0]}) ECE={max_ece[1]:.3f} > 0.15"
        else:
            q3 = "NO"; q3_evidence = f"All brains ECE within reasonable range (max ECE {max_ece[0]}={max_ece[1]:.3f})"
    else:
        q3 = "Unknown"; q3_evidence = "Phase 4 not completed"
    answers["Q3_calibration_limits"] = {"answer": q3, "evidence": q3_evidence}

    # Q4: current target formulation limiting?
    if p5:
        # Target A (current) baseline strength
        a_dir = p5.get("A_direction_15m", {}).get("dir_acc_nonskip")
        # Alternative targets showing stronger signal:
        c_auc = p5.get("C_return_gt_friction", {}).get("auc")
        d_ic = p5.get("D_trade_quality", {}).get("spearman_ic")
        if c_auc is not None and c_auc >= 0.60:
            q4 = "YES"
            q4_evidence = f"Target C (|ret|>friction) AUC={c_auc:.3f} >= 0.60 while current dir_acc_nonskip={a_dir}"
        elif d_ic is not None and d_ic >= 0.10:
            q4 = "YES"
            q4_evidence = f"Target D (trade_quality) Spearman IC={d_ic:.3f} >= 0.10"
        else:
            q4 = "NO"
            q4_evidence = f"No alt target shows materially stronger signal (A dir_acc={a_dir}, C AUC={c_auc}, D IC={d_ic})"
    else:
        q4 = "Unknown"; q4_evidence = "Phase 5 not completed"
    answers["Q4_target_limiting"] = {"answer": q4, "evidence": q4_evidence}

    # Q5: has current architecture been exhausted? (composite)
    # YES only if ALL falsification attempts failed (Q1=NO, Q2=NO, Q3=NO, Q4=NO)
    exhausted_signals = [answers["Q1_hpo_material"]["answer"],
                         answers["Q2_catboost_material"]["answer"],
                         answers["Q3_calibration_limits"]["answer"],
                         answers["Q4_target_limiting"]["answer"]]
    if all(a == "NO" for a in exhausted_signals):
        q5 = "YES"; q5_evidence = "Q1..Q4 all NO — no falsification succeeded"
    elif any(a == "Unknown" for a in exhausted_signals):
        q5 = "Unknown"; q5_evidence = f"Some experiments incomplete: {exhausted_signals}"
    else:
        q5 = "NO"; q5_evidence = f"At least one falsification succeeded: {exhausted_signals}"
    answers["Q5_architecture_exhausted"] = {"answer": q5, "evidence": q5_evidence}

    # Q6: should architecture work stop?
    # If Q5=YES, stop. If Q5=NO, continue in the direction indicated.
    q6 = q5
    q6_evidence = "Mirrors Q5"
    answers["Q6_stop_architecture_work"] = {"answer": q6, "evidence": q6_evidence}

    # Q7: single highest expected-value experiment remaining
    # Ranked ONLY by which falsification succeeded
    ranking = []
    if answers["Q1_hpo_material"]["answer"] == "YES":
        ranking.append(("Extend HPO to LGB/RF/NN/CatBoost", "Q1 succeeded"))
    if answers["Q2_catboost_material"]["answer"] == "YES":
        ranking.append(("Include CatBoost in production ensemble; re-run 8-fold with 5-brain avg", "Q2 succeeded"))
    if answers["Q3_calibration_limits"]["answer"] == "YES":
        ranking.append(("Apply isotonic calibration to worst-calibrated brains, re-simulate", "Q3 succeeded"))
    if answers["Q4_target_limiting"]["answer"] == "YES":
        ranking.append(("Retrain with alternative target formulation showing higher signal", "Q4 succeeded"))
    if not ranking:
        ranking = [("Move to futures execution port (H2) — architecture is exhausted", "All falsification attempts failed")]
    answers["Q7_next_experiment"] = ranking

    print("\n1. Hyperparameter optimization materially improved performance?    ", answers["Q1_hpo_material"]["answer"])
    print("   Evidence:", answers["Q1_hpo_material"]["evidence"])
    print("\n2. CatBoost materially outperformed existing models?               ", answers["Q2_catboost_material"]["answer"])
    print("   Evidence:", answers["Q2_catboost_material"]["evidence"])
    print("\n3. Poor calibration limiting performance?                          ", answers["Q3_calibration_limits"]["answer"])
    print("   Evidence:", answers["Q3_calibration_limits"]["evidence"])
    print("\n4. Current target formulation likely limiting?                     ", answers["Q4_target_limiting"]["answer"])
    print("   Evidence:", answers["Q4_target_limiting"]["evidence"])
    print("\n5. Based on measured evidence, has architecture been exhausted?    ", answers["Q5_architecture_exhausted"]["answer"])
    print("   Evidence:", answers["Q5_architecture_exhausted"]["evidence"])
    print("\n6. Should further architecture work stop?                          ", answers["Q6_stop_architecture_work"]["answer"])
    print("\n7. Single highest expected-value experiment remaining:")
    for r, why in answers["Q7_next_experiment"]:
        print(f"   -> {r}   ({why})")

    with open(ROOT / "logs/phase6_final_report.json", "w") as fh:
        json.dump(answers, fh, indent=2)
    print("\nReport written to logs/phase6_final_report.json")


if __name__ == "__main__":
    main()
