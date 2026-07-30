"""
htf36_decision.py — appends a PROMOTE/SHADOW/REJECT decision block (with
evidence) to data/validation_results/htf36_report.json, based on the
champion-vs-HTF36 comparison already computed by validate_htf36.py.

Pure post-processing of an existing report: no retraining, no new sims.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data/validation_results/htf36_report.json"


def overlap(a, b):
    """True if intervals [a0,a1] and [b0,b1] overlap."""
    if None in a or None in b:
        return None
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return lo <= hi


def main():
    rep = json.loads(PATH.read_text())
    champ, htf = rep["champion"], rep["htf36"]

    cmp = {
        "test_dir_acc": {
            "champion": champ["test"]["dir_acc"], "htf36": htf["test"]["dir_acc"],
            "delta": round(htf["test"]["dir_acc"] - champ["test"]["dir_acc"], 4),
            "champion_ci": champ["test"]["dir_acc_ci"], "htf36_ci": htf["test"]["dir_acc_ci"],
            "ci_overlap": overlap(champ["test"]["dir_acc_ci"], htf["test"]["dir_acc_ci"]),
        },
        "val_dir_acc": {
            "champion": champ["val"]["dir_acc"], "htf36": htf["val"]["dir_acc"],
            "delta": round(htf["val"]["dir_acc"] - champ["val"]["dir_acc"], 4),
            "champion_ci": champ["val"]["dir_acc_ci"], "htf36_ci": htf["val"]["dir_acc_ci"],
            "ci_overlap": overlap(champ["val"]["dir_acc_ci"], htf["val"]["dir_acc_ci"]),
        },
        "test_IC": {
            "champion": champ["test"]["IC"], "htf36": htf["test"]["IC"],
            "champion_p": champ["test"]["IC_p"], "htf36_p": htf["test"]["IC_p"],
            "delta": round(htf["test"]["IC"] - champ["test"]["IC"], 4),
            "champion_ci": champ["test"]["IC_ci"], "htf36_ci": htf["test"]["IC_ci"],
            "ci_overlap": overlap(champ["test"]["IC_ci"], htf["test"]["IC_ci"]),
            "either_significant_p<0.05": (champ["test"]["IC_p"] < 0.05
                                           or htf["test"]["IC_p"] < 0.05),
        },
        "sim_pf": {
            "champion": champ["sim"]["pf"], "htf36": htf["sim"]["pf"],
            "champion_ci": champ["sim_ci"]["pf_ci"], "htf36_ci": htf["sim_ci"]["pf_ci"],
            "ci_overlap": overlap(champ["sim_ci"]["pf_ci"], htf["sim_ci"]["pf_ci"]),
            "champion_ci_excludes_1.0": champ["sim_ci"]["pf_ci"][0] > 1.0,
            "htf36_ci_excludes_1.0": htf["sim_ci"]["pf_ci"][0] > 1.0,
        },
        "sim_expectancy": {
            "champion": champ["sim"]["expectancy"], "htf36": htf["sim"]["expectancy"],
            "champion_ci": champ["sim_ci"]["expectancy_ci"], "htf36_ci": htf["sim_ci"]["expectancy_ci"],
            "ci_overlap": overlap(champ["sim_ci"]["expectancy_ci"], htf["sim_ci"]["expectancy_ci"]),
            "champion_ci_excludes_0": champ["sim_ci"]["expectancy_ci"][0] > 0,
            "htf36_ci_excludes_0": htf["sim_ci"]["expectancy_ci"][0] > 0,
        },
        "sim_sharpe": {
            "champion": champ["sim"]["sharpe"], "htf36": htf["sim"]["sharpe"],
            "champion_ci": champ["sim_ci"]["sharpe_ci"], "htf36_ci": htf["sim_ci"]["sharpe_ci"],
            "ci_overlap": overlap(champ["sim_ci"]["sharpe_ci"], htf["sim_ci"]["sharpe_ci"]),
        },
        "sim_maxdd": {
            "champion": champ["sim"]["maxdd"], "htf36": htf["sim"]["maxdd"],
            "champion_ci": champ["sim_ci"]["maxdd_ci"], "htf36_ci": htf["sim_ci"]["maxdd_ci"],
        },
        "trade_count": {"champion": champ["sim"]["trades"], "htf36": htf["sim"]["trades"]},
    }

    # ── Decision logic ──────────────────────────────────────────────
    # PROMOTE bar: HTF36 must match/exceed champion on the genuinely
    # out-of-sample predictive signal (dir_acc + IC, large-n) with at
    # least directional support, AND trade-sim must not be statistically
    # indistinguishable from champion/breakeven.
    promote_blockers = []
    if cmp["test_dir_acc"]["delta"] < 0:
        promote_blockers.append(
            f"HTF36 test dir_acc ({htf['test']['dir_acc']:.4f}) is below champion "
            f"({champ['test']['dir_acc']:.4f}); same direction on val "
            f"({htf['val']['dir_acc']:.4f} vs {champ['val']['dir_acc']:.4f}).")
    if not cmp["test_IC"]["either_significant_p<0.05"]:
        promote_blockers.append(
            "Neither model's test IC is significant at p<0.05 "
            f"(champion p={champ['test']['IC_p']:.3f}, htf36 p={htf['test']['IC_p']:.3f}); "
            "no demonstrated directional edge to promote on.")
    if cmp["sim_pf"]["ci_overlap"]:
        promote_blockers.append(
            "HTF36 vs champion PF bootstrap CIs overlap heavily "
            f"(champion {cmp['sim_pf']['champion_ci']}, htf36 {cmp['sim_pf']['htf36_ci']}); "
            "the apparent PF/expectancy/Sharpe advantage is not distinguishable from "
            f"noise at n={cmp['trade_count']['htf36']} trades.")
    if not cmp["sim_pf"]["htf36_ci_excludes_1.0"]:
        promote_blockers.append(
            f"HTF36's own PF 95% CI {cmp['sim_pf']['htf36_ci']} includes 1.0 "
            "(breakeven) — its trade-sim edge is not statistically established "
            "in isolation either.")

    reject_reasons = []
    if cmp["test_dir_acc"]["ci_overlap"]:
        reject_reasons.append(
            "test dir_acc CIs overlap substantially "
            f"(champion {cmp['test_dir_acc']['champion_ci']}, htf36 {cmp['test_dir_acc']['htf36_ci']}) "
            "— the dir_acc shortfall is not statistically conclusive evidence of "
            "a worse model, just a point-estimate disadvantage.")
    if cmp["test_IC"]["delta"] > 0:
        reject_reasons.append(
            f"HTF36's test IC point estimate (+{htf['test']['IC']:.4f}) is actually "
            f"directionally HIGHER than champion's (+{champ['test']['IC']:.4f}), "
            "though neither is significant — insufficient to declare the family "
            "valueless.")
    if cmp["sim_expectancy"]["htf36"] > 0 and cmp["sim_pf"]["htf36"] > 1.0:
        reject_reasons.append(
            f"HTF36's frozen-test trade-sim is nominally profitable (PF "
            f"{htf['sim']['pf']:.2f}, expectancy Rs{htf['sim']['expectancy']:.0f}, "
            f"n={htf['sim']['trades']}) — not a clear negative result that would "
            "justify discarding the family outright.")

    # REJECT bar (only if promotion is blocked AND there's strong negative
    # evidence — i.e. HTF36 is significantly worse / sub-breakeven, not just
    # "not proven better"):
    strong_negative_evidence = []
    htf_dacc_ci, champ_dacc_ci = htf["test"]["dir_acc_ci"], champ["test"]["dir_acc_ci"]
    if None not in htf_dacc_ci and None not in champ_dacc_ci and htf_dacc_ci[1] < champ_dacc_ci[0]:
        strong_negative_evidence.append(
            f"HTF36 test dir_acc 95% CI {htf_dacc_ci} is entirely below the "
            f"champion's {champ_dacc_ci} — significantly worse, not just a "
            "point-estimate gap.")
    htf_pf_ci = htf["sim_ci"]["pf_ci"]
    if None not in htf_pf_ci and htf_pf_ci[1] < 1.0:
        strong_negative_evidence.append(
            f"HTF36 trade-sim PF 95% CI {htf_pf_ci} is entirely below 1.0 "
            "(breakeven) — significant net loss under the NEW gate.")
    if htf["test"]["IC_p"] < 0.05 and htf["test"]["IC"] < 0:
        strong_negative_evidence.append(
            f"HTF36 test IC is significantly NEGATIVE "
            f"({htf['test']['IC']:+.4f}, p={htf['test']['IC_p']:.4f}).")

    if not promote_blockers:
        decision = "PROMOTE"
    elif strong_negative_evidence:
        decision = "REJECT"
    else:
        decision = "SHADOW"

    report_decision = {
        "decision": decision,
        "summary": (
            "HTF36 (28 tf15_/tf30_/tf60_ features, clean pipeline, identical "
            "purged split/hyperparameters/NEW gate, CONF=0.22) does NOT clear the "
            "promotion bar: its directional accuracy is below the current V9 "
            "champion on BOTH the validation set (0.5624 vs 0.5719) and the frozen "
            "test (0.5608 vs 0.5868), and neither model's test IC is significant "
            "(p=0.27 vs p=0.66). The frozen-test trade-sim (NEW gate, unmodified) "
            "nominally favors HTF36 (PF 2.02 vs 1.70, expectancy Rs603 vs Rs460, "
            "Sharpe 5.22 vs 4.07) but on only 33 vs 40 trades — bootstrap 95% CIs "
            "for PF/expectancy/Sharpe overlap almost completely between the two "
            "models AND each model's own CI spans breakeven (PF CI lower bound "
            "<1.0, expectancy/Sharpe CIs cross 0). The evidence is therefore "
            "underpowered and directionally mixed: not a basis to promote HTF36 "
            "to champion, but also not strong enough to declare the tf15/30/60 "
            "family valueless (test IC point estimate is even nominally higher "
            "than the champion's, +0.0087 vs +0.0036)."
        ),
        "promote_blockers": promote_blockers,
        "strong_negative_evidence": strong_negative_evidence,
        "reasons_not_to_reject_outright": reject_reasons,
        "recommendation": (
            "SHADOW. Do not promote HTF36 or change the V9 champion/gate. "
            "Continue the existing pre-registered HTF36 shadow-logging "
            "(models/htf36, data/shadow_model_log.jsonl) toward its "
            ">=15 sessions / >=1000 cycles review point, where live data will "
            "sharpen these CIs well beyond the ~33-40 trade / ~2,300-2,500 "
            "signal sample available on the frozen historical test. Re-run this "
            "same validation (validate_htf36.py) at that checkpoint with the "
            "extended data window before any promote/reject call."
        ),
        "comparison": cmp,
        "personas": {
            "jane_street_research_lead": (
                "Signal quality is the deciding factor, not the gate-sim P&L. "
                "HTF36's dir_acc is lower than the champion's on both the "
                "validation fold and the frozen test, and its test IC "
                "(+0.0087, p=0.27) is not significant. A 28-feature model that "
                "underperforms a 170-feature model on out-of-sample accuracy "
                "is not a research win, even if it happens to look better after "
                "a 33-trade gate replay."
            ),
            "citadel_systematic_pm": (
                "From a portfolio-risk lens, n=33 trades over <1 year is far too "
                "thin to size or re-allocate capital on. MaxDD point estimates "
                "(-3714 vs -3832) are close and both carry bootstrap tails past "
                "-14k/-18.6k. PF/Sharpe CIs cross breakeven for both books. No "
                "action on capital allocation is warranted from this evidence; "
                "keep monitoring in shadow at zero capital risk."
            ),
            "xgboost_specialist": (
                "Same hyperparameters (_fit_ensemble), same purged split, no "
                "tuning — clean comparison. A 28-feature ensemble matching "
                "~96% of a 170-feature ensemble's dir_acc would be a strong "
                "parsimony result; here it underperforms by ~2.6pp on the "
                "frozen test, which (given dir_acc CI widths of ~6-7pp) is "
                "plausibly noise but is the wrong direction to claim a win. "
                "No evidence of overfitting in the 170-feature model relative "
                "to HTF36 — val and test dir_acc move together for both."
            ),
            "quant_researcher": (
                "Both models' test IC bootstrap CIs straddle zero "
                "(champion [-0.012, 0.019], htf36 [-0.007, 0.025]); the "
                "'true edge approx zero' finding from Phase 0 still holds for "
                "this family. The apparent gate-sim PF/expectancy edge for "
                "HTF36 is consistent with sampling variation given the heavy "
                "CI overlap with the champion's own gate-sim distribution."
            ),
            "qa_engineer": (
                "Pipeline integrity verified: re-ran train_htf36.py end-to-end "
                "and reproduced identical val/test dir_acc/IC to the existing "
                "models/htf36 artifacts (deterministic). Champion artifacts "
                "(models/sandbox_clean) loaded unmodified — no retraining, no "
                "hyperparameter or gate edits. HTF36 feature set verified as a "
                "strict subset of the champion's 170 features and verified to "
                "contain ONLY tf15_/tf30_/tf60_ columns (28/28), per spec."
            ),
        },
    }

    rep["decision"] = report_decision
    PATH.write_text(json.dumps(rep, indent=1))
    print(f"Decision: {decision}")
    print(f"Wrote -> {PATH}")


if __name__ == "__main__":
    main()
