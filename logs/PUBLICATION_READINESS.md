# OpenClaw – Publication Readiness Assessment

**Scope.** Phases 0 through 7 are being frozen for publication. Phase 8
is not started. This document rates each phase and gives the overall
publication verdict for the frozen record.

**Rating scale.**
- **COMPLETE** — publishable as-is; every reported metric traces to
  disk; no correction needed.
- **COMPLETE WITH MINOR CAVEATS** — publishable with explicit caveats
  documented in the phase's own report; the caveats do not change any
  scientific conclusion.
- **REQUIRES REVISION** — a scientific claim is unsupported or
  incorrect; publication must wait.

## Phase-by-phase ratings

### Phase 0 — Pre-registration + hashing infrastructure — **COMPLETE**

**Justification.** The pre-registration is frozen at `protocol_version
2.5` and is referenced by every downstream manifest via `code_hash`
and `data_hash`. No drift detected in Phase 6 (`--verify-prereg
OK. Pre-registration frozen at 2026-07-12T05:12:34+00:00. No drift.`)
or during the Phase 7 correction cycle (0 file mtimes past the freeze
timestamp). No further work needed.

### Phase 1 — Walk-forward with weight ablation — **COMPLETE**

**Justification.** Purged walk-forward with block-bootstrap CI on
pooled PF is a standard construction; the per-fold PnL, weight
ablation and purge-audit outputs are all on disk. No subsequent phase
has invalidated any Phase 1 conclusion.

### Phase 2 — Nested purged WF-HPO across families — **COMPLETE**

**Justification.** Nested purged CV with a PF objective; per-family
best configs cached with manifests. Used only as an upstream input to
Phase 3; downstream phases don't re-derive Phase 2 numbers.

### Phase 3 — Genuine architecture diversity D1–D6 — **COMPLETE**

**Justification.** Six architecturally distinct base learners with
locked per-fold predictions; diversity diagnostics recorded.

### Phase 4 — Calibration + re-simulation + regime-conditional — **COMPLETE**

**Justification.** Isotonic calibration is standard, monotone, and
locked. Regime-conditional metrics are descriptive only (no decisive
claim made). Calibrated predictions frozen and consumed by Phase 5.

### Phase 5 — Ensemble selection — **COMPLETE**

**Justification.** Both `mean` and `stacking` ensemble outputs
(`predictions.csv`, `trade_pnl.csv`, `manifest.json` per fold) are
frozen and hash-locked in every downstream manifest. Phase 6 and
Phase 7 both verified these hashes at execution time and post hoc.

### Phase 6 — FWE-controlled threshold optimizer — **COMPLETE**

**Justification.** Full 360-candidate deterministic grid, 8-fold
per-candidate replay, per-candidate trade PnL cache. UNDECIDED verdict
for both targets is the pre-registered outcome under `H_thr`; the
Phase-6 report explicitly documents that UNDECIDED is a valid
outcome, not a null result. Machine-readable summary, per-target
`candidate_results.csv`, chart PNGs + `chart_data.json`, all
consistent per audit.

### Phase 7 — Robustness & stress testing — **COMPLETE WITH MINOR CAVEATS**

**Justification.** After the post-audit R7 correction cycle:
- Every H_rob input (paired-bootstrap ΔPF CI, outperform fraction,
  catastrophic-scenario list, per-fold PF CV) is reproducible from
  disk with `seed=42`.
- Diebold–Mariano test now uses the mathematically correct hypothesis
  direction, values persisted into `summary.json`, `stress_report.json`,
  `robustness.json`.
- Break-even multipliers verified to sub-rupee precision by the audit.
- Bootstrap CIs verified byte-identical on re-derivation.
- `Phase7_Report.md` rewritten in place with corrected DM values,
  a real rolling-window PF table, and an explicit note on stacking's
  paired-bootstrap CI numerical artifact.

Remaining caveats (all documented in the report and in
`correction_manifest.json`):
1. **WRC pvalue over execution-delay variants is not on disk.** SPA
   `p_lower` values are transcribed from the original run's markdown
   with provenance. Recomputing WRC (and re-verifying SPA) requires
   persisting delay-variant per-fold PnLs, which was deferred under
   the user's "no replay" instruction.
2. **N = 8 folds limits power** across the whole R5/R7 statistical
   surface. Paired-bootstrap CIs are wide by construction.
3. **Rolling-window PFs above ~4 are denominator-driven.** The
   corrected report calls this out with the actual highest values
   (`mean` folds 6..8 → PF = 15.343 on n = 19; `stacking` folds 1..3
   → PF = 3.098 on n = 32) so no reader can over-interpret.
4. **Stacking's paired-bootstrap ΔPF lower bound of ≈ −55 is a
   numerical artifact** — near-zero baseline gross losses on at least
   one fold. The H_rob rule already treats it as a failing criterion;
   the report explicitly says the number carries no economic meaning.
5. **Fold-shift is implemented as edge-fold drop**, not as
   re-splitting the training window; documented up-front in
   `Phase7_Report.md`.

None of these caveats changes the H_rob REJECT verdict, the production
recommendation, or any Phase 0–6 conclusion.

## Overall assessment

### Is the project scientifically reproducible?

**Yes**, with two documented exceptions. Every result on disk under
`logs/phase{0..7}/` can be reproduced from the frozen inputs plus the
recorded seed. The two exceptions (WRC pvalue over delay variants, and
one small SPA transcription) are named, documented, and do not affect
any published conclusion.

### Is provenance complete?

**Yes.** Every phase has manifest hashes for code, input, and data.
Every downstream phase verifies its upstream hashes on startup and
raises `CacheMismatchError` on drift. Post-audit provenance verified:
0 files under `logs/phase{0..6}/` were touched during the Phase 7
correction cycle.

### Are all reported statistics traceable?

**Yes**, with one class of exception. Every statistic in
`summary.json`, `robustness.json`, `stress_report.json`,
`bootstrap.json`, `jackknife.json`, and the corrected §3–§4 of
`Phase7_Report.md` traces to either (a) a hash-locked input, (b) a
seed-reproducible bootstrap, or (c) a cached per-fold trade PnL. The
exception is the SPA `p_lower` value over execution-delay variants,
which is transcribed from the original run's markdown with the source
citation in `correction_manifest.json.transcribed_values_spa_p_lower`.

### Do any remaining caveats affect the published conclusions?

**No.** The two candidates (`mean` and `stacking` winners) are both
REJECTED by H_rob regardless of the WRC gap and regardless of any
plausible correction to the transcribed SPA value. The production
recommendation ("DO NOT deploy without further evidence") is
insensitive to the remaining caveats.

## Publication verdict

The frozen record (Phase 0 through Phase 7) is **publication-ready**
subject to the caveats named above being reproduced verbatim in any
paper, technical report, or engineering handoff that cites these
results. Phase 7 is COMPLETE WITH MINOR CAVEATS; every earlier phase
is COMPLETE. Phase 8 is not started and its scope is out of the
current freeze.
