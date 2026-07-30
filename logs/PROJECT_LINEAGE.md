# OpenClaw – Experiment Lineage (Phase 0 through Phase 7)

**Purpose.** A single self-contained document that lets a new researcher
reconstruct the experimental logic of the OpenClaw NIFTY-options
directional-buying project without reading any source code. Every phase is
described in terms of its objective, frozen inputs, outputs, statistical
tests, hypotheses evaluated, and verdict.

**Codebase root:** `E:/claude-fix-auto/dynamic-stop-loss/openclaw-v9-kotak/`
**Protocol version at freeze:** 2.5 (locked at Phase 6 supersede,
unchanged by Phase 7 correction cycle)
**Overall production recommendation:** *DO NOT deploy without further
evidence.* (Both `mean` and `stacking` ensembles rejected by H_rob.)

---

## Phase 0 — Pre-registration + hashing infrastructure

**Objective.** Establish a tamper-evident record of the falsification
plan before any modelling work begins. Ensure every subsequent phase
locks its code hashes, data hashes, and expected outputs so drift can
be detected.

**Frozen inputs.** None (this is the origin of the lineage).

**Outputs.**
- Pre-registration document with SHA-256 hashes of all data files, code
  files, hypotheses to be tested, and pre-committed decision rules.
- Central `stat_utils` package used from Phase 1 onward.

**Statistical tests defined.** Block bootstrap CI, Diebold–Mariano,
White Reality Check, Hansen SPA, DSR, PBO.

**Hypotheses.** None evaluated here; hypotheses `H1–H_rob` are declared
for later phases.

**Verdict.** Complete. Frozen at `protocol_version 2.5` after the
Phase 6 supersede event; not modified by Phase 7 correction cycle.

**Downstream dependency.** All phases inherit the pre-registration
hashes; any phase that fails hash validation is blocked from execution.

---

## Phase 1 — Walk-forward with weight ablation, purge audit, block-bootstrap CI

**Objective.** Verify that the incumbent v9 model has a walk-forward
edge and that its per-fold PnL distribution is not dominated by any
single fold.

**Frozen inputs.** Pre-registration hashes, `stat_utils`, raw NIFTY
minute-bar + VIX + bhavcopy data.

**Outputs.** Per-fold trade PnL under production thresholds, weight-
ablation report, purge audit, block-bootstrap CI on pooled PF.

**Statistical tests.** Purged block-bootstrap CI on pooled PF.

**Hypotheses evaluated.**
- H1: incumbent v9 model has a walk-forward PF > 1 after friction.

**Verdict.** Complete. Result documented and used as the incumbent
against which Phase 2+ candidates are compared.

**Downstream dependency.** Phase 2 uses the Phase 1 walk-forward split
as its outer purged split.

---

## Phase 2 — Nested purged WF-HPO on PF objective across model families

**Objective.** Search for a better single-family model via nested
purged walk-forward hyperparameter optimisation with PF as the
objective. Families searched: CatBoost, LightGBM, XGBoost, MLP.

**Frozen inputs.** Phase 1 walk-forward split; pre-registration purge
and embargo distances (`EMBARGO_DAYS = 3`, `K_inner = 3`).

**Outputs.** Best hyperparameter configurations per family; per-family
per-fold PnL streams.

**Statistical tests.** Inner-fold objective = pooled PF via bootstrap.

**Hypotheses evaluated.**
- H2: any single-family HPO candidate significantly outperforms the
  Phase 1 incumbent under paired block bootstrap.

**Verdict.** Complete. See per-family selection results.

**Downstream dependency.** Winning per-family configs feed into
Phase 3 as the base learners.

---

## Phase 3 — Genuine architecture diversity (D1–D6)

**Objective.** Guarantee that the ensemble base learners are actually
diverse (not merely retuned copies of one family). Six architectural
axes (D1–D6) explored.

**Frozen inputs.** Phase 2 per-family best configs.

**Outputs.** Six architecturally distinct base models with locked
per-fold predictions.

**Statistical tests.** Diversity diagnostics (correlation, Q-statistic,
disagreement).

**Hypotheses evaluated.**
- H3: base-learner disagreement is above a pre-registered threshold on
  every fold.

**Verdict.** Complete.

**Downstream dependency.** Phase 4 calibrates these six base learners.

---

## Phase 4 — Calibration + re-simulation + regime-conditional analysis

**Objective.** Apply isotonic calibration to each base learner's
predicted probabilities, re-simulate trades under production
thresholds, and re-partition metrics by VIX regime.

**Frozen inputs.** Phase 3 base-model predictions.

**Outputs.** Calibrated per-base-model per-fold predictions
(`logs/phase4/`), regime-conditional PF/Net tables.

**Statistical tests.** Isotonic calibration; regime-stratified pooled
metrics.

**Hypotheses evaluated.**
- H4: calibration does not degrade pooled PF; regime-conditional PF is
  positive in each of {bull, bear, sideways}.

**Verdict.** Complete. Isotonic-calibrated variants promoted to
`calibrated_isotonic` for downstream phases.

**Downstream dependency.** Phase 5 consumes `calibrated_isotonic`
predictions.

---

## Phase 5 — Common-metric target audit + k-NN MI + ensemble output

**Objective.** Choose the ensemble aggregation rule with best out-of-
sample PF. Emit per-fold ensemble predictions and per-fold trade PnL
under production thresholds.

**Frozen inputs.** Phase 4 `calibrated_isotonic` predictions per base
model.

**Outputs.** For each ensemble type (mean, median, weighted,
performance-weighted, min-variance, stacking, confidence-weighted,
uncertainty-weighted):
`logs/phase5/<ensemble>/fold_{1..8}/{predictions.csv, probabilities.csv,
trade_pnl.csv, trades.csv, metrics.json, manifest.json, ensemble.pkl,
weights.json}`.

Fold windows (identical for every ensemble):

| Fold | test_start | test_end | n_test_bars |
|---:|---|---|---:|
| 1 | 2024-07-01 | 2024-10-01 | 4800 |
| 2 | 2024-10-01 | 2025-01-01 | 4575 |
| 3 | 2025-01-01 | 2025-04-01 | 4650 |
| 4 | 2025-04-01 | 2025-07-01 | 4575 |
| 5 | 2025-07-01 | 2025-10-01 | 4800 |
| 6 | 2025-10-01 | 2026-01-01 | 4587 |
| 7 | 2026-01-01 | 2026-04-01 | 4350 |
| 8 | 2026-04-01 | 2026-05-01 | 1415 |

**Statistical tests.** Common-metric target audit; k-NN mutual
information between base-model outputs and label; nested-CV PF
selection.

**Hypotheses evaluated.**
- H5: at least one ensemble aggregation rule has pooled PF > incumbent
  Phase 1 under paired block bootstrap.

**Verdict.** Complete. The **`mean`** ensemble was selected as the
primary target; **`stacking`** kept as a diversity check.

**Downstream dependency.** Phases 6 and 7 consume the
`calibrated_isotonic` Phase-5 outputs as their only frozen prediction
source.

---

## Phase 6 — FWE-controlled threshold optimizer (rewrite)

**Objective.** Deterministically search the 4-dimensional threshold
grid `(call_thr, put_thr, skip_ceil, min_edge)` for candidates that
outperform the production baseline `(0.32, 0.25, 0.65, 0.05)` on the
`mean` and `stacking` ensembles, with family-wise error control
(Hansen SPA / White RC / Holm).

**Frozen inputs.** `logs/phase5/{mean,stacking}/fold_{1..8}/predictions.csv`.

**Grid size.** 360 candidates per target
(`call_thr` ∈ {0.20, 0.25, 0.30, 0.32, 0.35, 0.40},
 `put_thr` ∈ {0.15, 0.20, 0.25, 0.30, 0.35},
 `skip_ceil` ∈ {0.60, 0.65, 0.70, 0.75},
 `min_edge` ∈ {0.03, 0.05, 0.08}).

**Outputs (frozen).**
- `logs/phase6/{mean,stacking}/candidate_results.csv` — 360 rows per target.
- `logs/phase6/{mean,stacking}/charts/*` — 4 chart PNGs + `chart_data.json`.
- `logs/phase6/{mean,stacking}/thr_<hash8>/{manifest.json, result.json,
   trade_pnl_fold_{1..8}.csv}` — per-candidate cache for every
   evaluated candidate (baseline + top-K + winner all present).
- `logs/phase6/summary.json` — top-level machine-readable summary.

**Statistical tests.**
- Paired block-bootstrap 90% CI on ΔPF winner vs. baseline (per target).
- Diebold–Mariano winner vs. baseline (per target).
- Top-10 comparison: Hansen SPA, White RC, Holm–Bonferroni (α=0.10/7).

**Hypotheses evaluated.**
- H_thr (per target): `LB90(ΔPF) > +0.05 AND DM_p < 0.0143` (i.e., FWE-
  adjusted decisive improvement).

**Verdict.** UNDECIDED for both targets.

- `mean` winner `(0.32, 0.15, 0.70, 0.03)`: pooled PF 1.938, Net
  +Rs. 27,483, n=61. LB90(ΔPF) = −1.565 (fails), DM p = 0.029 (fails
  0.0143 threshold), SPA `p_lower` = 0.002.
- `stacking` winner `(0.40, 0.15, 0.75, 0.05)`: pooled PF 1.119, Net
  +Rs. 5,889, n=93. LB90(ΔPF) = +0.030 (fails 0.05 threshold),
  DM p = 0.456.

**Downstream dependency.** Phase 7 consumes the winner threshold
candidates and the frozen `logs/phase6/summary.json` (hash locked in
Phase 7 manifests).

---

## Phase 7 — Robustness & stress testing (with post-audit corrections)

**Objective.** Test whether the Phase-6-approved threshold candidates
survive perturbations along seven axes (R1–R7 in the pre-registered
plan). Verdict controlled by the user-approved H_rob rule.

**Frozen inputs.**
- `logs/phase5/{mean,stacking}/fold_{1..8}/predictions.csv`
- `logs/phase6/summary.json`
- Phase 6 per-candidate per-fold trade PnL cache (used only by the
  post-audit R7 correction cycle for DM recomputation from disk).

**Outputs (post-correction).**
- `logs/phase7/summary.json` — H_rob verdict + criteria + R7 stats.
- `logs/phase7/robustness.json` — R1 (walk-forward variants) + R6
  (stability) + regime tags + tornado.
- `logs/phase7/stress_report.json` — R3 (slippage), R4 (tcost),
  execution-delay curves, break-even multipliers.
- `logs/phase7/jackknife.json` — R2 leave-one-fold-out.
- `logs/phase7/bootstrap.json` — R5 block/paired bootstrap CIs.
- `logs/phase7/Phase7_Report.md` — publication markdown (corrected).
- `logs/phase7/{mean,stacking}/{manifest.json, charts/*}`.
- `logs/phase7/r7_correction.json` + `PHASE7_CORRECTIONS.md` — post-
  audit trail.

**Statistical tests (corrected).**
- Block-bootstrap 90% CI on pooled PF and pooled Net (per target).
- Paired-fold bootstrap 90% CI on ΔPF winner vs. baseline.
- Diebold–Mariano winner vs. baseline (per target) — **corrected to
  test H1: E[winner_perf − baseline_perf] > 0**.
- Hansen SPA over execution-delay variants vs. baseline (per target;
  `p_lower` values persisted from original run transcript).
- Notes: WRC over delay variants deferred; per-family DM/SPA/WRC on
  slippage or tcost stress not run — not in Phase 7 scope.

**Hypotheses evaluated.**
- H_rob (per target): ACCEPT iff ≥ 80 % of scenarios outperform
  baseline PF AND `LB90(ΔPF) > 0` AND no catastrophic scenario
  (PF < 0.8 or Net loss > 2× baseline net) AND `CV(per-fold PF) ≤ 0.5`;
  REJECT if ≥ 3 criteria fail; UNDECIDED otherwise.

**Verdict.** REJECT for both `mean` and `stacking`.

| Target | Outperform % | LB90(ΔPF) | Catastrophic | CV | Verdict |
|---|---:|---:|---:|---:|:---:|
| `mean`     | 71.1 % | +0.136 | 3 | 1.213 | REJECT |
| `stacking` | 89.5 % | −55.270 (artifact) | 6 | 2.048 | REJECT |

**Downstream dependency.** No production deployment. Phase 8 (future,
not started) will use Phase 7 results as its baseline for information-
ceiling analysis.

---

## Cross-phase provenance chain

```
Phase 0 (pre-reg + stat_utils)
  → Phase 1 (walk-forward incumbent)
      → Phase 2 (nested purged WF-HPO)
          → Phase 3 (architecture diversity)
              → Phase 4 (calibration)
                  → Phase 5 (ensemble selection; frozen predictions/PnL)
                      → Phase 6 (threshold optimization; frozen cache)
                          → Phase 7 (robustness)
```

Every arrow is enforced by:
1. Manifest hash validation (`code_hash`, `input_hash`, `data_hash`).
2. Fail-fast on missing inputs or hash mismatch.
3. Pre-registration protocol_version `2.5`, unchanged from Phase 6
   through Phase 7 correction cycle.

## What a new researcher needs to know

- The pre-registered decisive metric for Phase 6 was `H_thr` (per-target
  FWE-controlled improvement over baseline). It came out **UNDECIDED**
  for both targets. This is not a null result; it means the search was
  inconclusive at the pre-registered decisive level.
- The pre-registered decisive metric for Phase 7 is `H_rob` (per-target
  robustness composite). It came out **REJECT** for both targets under
  the user-approved rule adopted before Phase 7 execution.
- Because H_rob rejects both candidates, the production incumbent
  (Phase 1 v9 model at production thresholds) remains the reference. No
  automated deployment is authorised by this experimental record.
- Every scientific conclusion is reproducible from disk with seed 42
  and the frozen `logs/phase5`, `logs/phase6` outputs, except (a) the
  WRC pvalue over delay variants, which is explicitly deferred, and
  (b) the SPA `p_lower` over delay variants, which is transcribed from
  the original run's markdown (source cited in
  `logs/phase7/correction_manifest.json`).
