# Phase 7 – Post-Run Corrections Log

**Purpose.** Permanent record of every correction applied to Phase 7 after
the original experiment run and before Phase 8 begins. This document is
append-only and must be preserved verbatim in future publication packs.

**Original Phase 7 run start (UTC):** 2026-07-12T10:37:29Z
**Original Phase 7 run end (UTC):**   2026-07-12T10:39:59Z
**Correction cycle start (UTC):**     2026-07-12T10:56:14Z (audit script)
**Correction cycle end (UTC):**       2026-07-12T11:10:02Z (last write to
`r7_correction.json`)
**Correction author:** Assistant (Claude), acting on the scientific
validation audit and the user's approval to apply "only the corrections
identified in the audit."
**Experiments rerun in this cycle:** **NONE.**

---

## 1. Summary of every post-run correction

| # | Correction | Category | Files affected |
|---|---|---|---|
| C1 | Fixed the Diebold–Mariano (DM) sign-convention bug in `phase7_robustness_v9/_stats.py:dm_winner_vs_baseline_by_fold`. Previous call inverted the alternative hypothesis. | Source-code bug fix | `phase7_robustness_v9/_stats.py` |
| C2 | Recomputed DM winner-vs-baseline for `mean` and `stacking` from the Phase 6 cached per-fold trade PnLs. No `simulate_trades` call was made. | Analysis (recompute from cache) | (none — read-only) |
| C3 | Persisted the corrected DM values, plus the SPA `p_lower` values transcribed from the original run's markdown, into `summary.json`, `stress_report.json`, and `robustness.json` under `per_target.<t>.r7_statistical_tests`, and added a root-level `r7_correction` metadata block to each. | Reporting-gap fix | `logs/phase7/summary.json`, `logs/phase7/stress_report.json`, `logs/phase7/robustness.json` |
| C4 | Rewrote `Phase7_Report.md` in place with corrected DM values, an actual rolling-window PF table (replacing the mis-cited 6.85 figure with the true 15.343 for `mean` folds 6..8), an explicit numerical-artifact note on stacking's paired-bootstrap ΔPF lower bound, and a new §7 "Corrections applied (post-audit)". Original markdown preserved verbatim at `Phase7_Report.md.pre-r7-correction.bak`. | Reporting-gap fix + prose correction | `logs/phase7/Phase7_Report.md`, `logs/phase7/Phase7_Report.md.pre-r7-correction.bak` |
| C5 | Created a dedicated audit-trail artifact `logs/phase7/r7_correction.json` capturing the correction metadata, cached-input hashes used, recomputed DM values, and confirmation that the H_rob verdicts remain unchanged. | Provenance | `logs/phase7/r7_correction.json` |

---

## 2. Why each correction was necessary

### C1 — DM sign-convention fix

The buggy call was:

```python
r = diebold_mariano(-winner_perf, -baseline_perf,
                       alternative="greater", lag=1)
```

`stat_utils.diebold_mariano` documents `alternative="greater"` as testing
`H1: E[loss_a − loss_b] > 0`, i.e. "model a has *higher* loss than model b"
= "model a is *worse* than model b".

With `loss_a = -winner_perf` and `loss_b = -baseline_perf`, this expanded
to `H1: E[baseline_perf − winner_perf] > 0`, which is **the wrong direction**
— it tested whether the baseline outperforms the winner. On the actual
per-fold PnLs the winner outperforms the baseline, so the DM statistic
came out negative and the p-value came out near 1.0. The docstring
inside `dm_winner_vs_baseline_by_fold` claimed the test was measuring
the intended direction; the docstring was wrong.

The corrected call is:

```python
r = diebold_mariano(-baseline_perf, -winner_perf,
                       alternative="greater", lag=1)
# H1: E[winner_perf − baseline_perf] > 0
```

Mathematically equivalent to `dm(winner_perf, baseline_perf,
alternative="greater")` (higher-is-better convention) or to the buggy
call with `alternative="less"` — the chosen form is the minimum-edit
form that keeps `mean_loss_diff` positive when the winner wins, matching
Phase 6's convention.

### C2 — Recompute DM from cache

The audit revealed that the R7 statistical block was never persisted to
disk in the original Phase 7 run: it was computed in memory during the
run but only the Hansen SPA p-value made it into `Phase7_Report.md`;
DM values and the WRC pvalue were computed and then discarded. The
assistant's post-run prose summary therefore cited DM/SPA values that
were not on disk and, as it turned out, did not match a from-scratch
recomputation.

Recomputation used the frozen Phase 6 cache
(`logs/phase6/<target>/thr_<hash8>/trade_pnl_fold_*.csv`) — the same
per-fold PnL streams Phase 6 wrote when it evaluated the winner and
baseline candidates. No new `simulate_trades` call was needed.

### C3 — Persist R7 block

Reporting-completeness fix: DM values and SPA `p_lower` values are now
in every relevant JSON so an auditor can verify them without reading the
markdown transcript.

### C4 — Correct the markdown report

Removed unverified numbers, added the actual rolling-window PF table
(all values now trace to `robustness.json`), and made the numerical
artifact in stacking's paired-bootstrap CI explicit. Overall verdict
and production recommendation unchanged.

### C5 — Audit-trail artifact

Provides a single, self-contained record of the correction so the next
publication step (Phase 12) can cite it directly.

---

## 3. Date/time of each correction

| Event | UTC |
|---|---|
| Audit script (`phase7_audit.py`) executed | 2026-07-12T10:56:14Z |
| DM sign-convention fix applied to `_stats.py` | 2026-07-12T11:07:12Z |
| DM values recomputed and persisted | 2026-07-12T11:10:02Z |
| SPA transcribed + `Phase7_Report.md` rewritten | 2026-07-12T11:14:33Z |

---

## 4. Files modified during the correction cycle

| Path | Change type | New SHA-256 (16-char prefix) |
|---|---|---|
| `phase7_robustness_v9/_stats.py` | Source-code fix (DM sign convention + docstring + result-dict keys) | `26eec0f762947815` |
| `logs/phase7/summary.json` | Added `r7_statistical_tests` and `r7_correction` blocks | `e9d3becf53efa9fa` |
| `logs/phase7/stress_report.json` | Same additions as summary.json | (see JSON) |
| `logs/phase7/robustness.json` | Same additions as summary.json | (see JSON) |
| `logs/phase7/Phase7_Report.md` | Rewritten in place; corrected DM, corrected rolling PF table, paired-CI caveat, added §7 | (see file) |

## Files created during the correction cycle

| Path | Purpose |
|---|---|
| `logs/phase7/r7_correction.json` | Audit trail — correction meta + recomputed DM + cached-input hashes |
| `logs/phase7/Phase7_Report.md.pre-r7-correction.bak` | Verbatim pre-correction markdown, for auditability |
| `logs/phase7/PHASE7_CORRECTIONS.md` | This document |

---

## 5. Files intentionally NOT modified

- Every file under `logs/phase0/`, `logs/phase1/`, `logs/phase2/`,
  `logs/phase3/`, `logs/phase4/`, `logs/phase5/`, `logs/phase6/`.
- Every Phase 7 chart PNG, `chart_data.json`, per-target
  `manifest.json`, `bootstrap.json`, and `jackknife.json`.
- Every source file except `phase7_robustness_v9/_stats.py`. In
  particular: `_base.py`, `_cache.py`, `_replay.py`, `_walkforward.py`,
  `_jackknife.py`, `_slippage.py`, `_tcost.py`, `_delay.py`,
  `_bootstrap.py`, `_stability.py`, `_regime.py`, `_reports.py`,
  `_visualize.py`, `__init__.py`, `phase7_robustness.py`, and every
  test file are untouched.
- Pre-registration (frozen at Phase 6 supersede, `protocol_version 2.5`)
  and every file it locks.

---

## 6. Confirmations

**No experiments were rerun.** The correction cycle called neither
`simulate_trades` nor any bootstrap/stress/walk-forward/jackknife
routine. DM values were recomputed by loading Phase 6's cached per-fold
PnL CSVs and calling the (fixed) `dm_winner_vs_baseline_by_fold`
function on them. SPA values were transcribed verbatim from the
original run's markdown. WRC over delay variants was not recomputed
(see §3 of `Phase7_Report.md` — deferred pending persistence of
delay-variant PnLs, out of scope for this correction cycle).

**All Phase 0–6 artifacts remain unchanged.** Verified by scanning
mtimes on every file under `logs/phase{0..6}/` after the correction
cycle end: **0 files** had an mtime later than the original Phase 7
run start (2026-07-12T10:37:29Z).

**Phase 7 test suite:** 60 tests, all passing (1.17 s) after the
`_stats.py` change.

**H_rob verdicts unchanged.**

| Target | Verdict | Cause of REJECT (unchanged) |
|---|---|---|
| `mean`     | REJECT | outperform-frac 71.1% (<80%); 3 catastrophic rolling windows; CV 1.213 (>0.50). CI-LB(ΔPF) = +0.136 → CI-LB-ok ✅. |
| `stacking` | REJECT | CI-LB(ΔPF) = −55.27 (numerical artifact); 6 catastrophic scenarios; CV 2.048 (>0.50). Outperform-frac 89.5% ✅. |

**Production recommendation unchanged:** *DO NOT deploy without further
evidence.*
