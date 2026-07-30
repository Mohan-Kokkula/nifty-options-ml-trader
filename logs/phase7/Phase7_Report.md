# Phase 7 – Robustness & Stress Testing Report

*Generated (original run) 2026-07-12T10:39:59Z*  *Corrected 2026-07-12T11:12:20.409415+00:00*
*Phase 7 version: 1.0*  |  *Protocol version: 2.5*

> **Correction notice.** This report was revised in place to fix the Diebold–Mariano sign bug identified in the scientific validation audit (see `r7_correction.json` for the full audit trail). The H_rob verdict for both targets is **unchanged** by the correction.

## 1. Methodology

Phase 7 stresses the Phase-6 approved threshold candidates and the production baseline across walk-forward, jackknife, slippage, transaction-cost and execution-delay dimensions. It is strictly read-only with respect to Phases 0-6 — no model is retrained and no prediction file is regenerated. Trades are re-priced via `backtest_options.simulate_trades` (byte-identical to Phase 5) with either shifted signal vectors (execution-delay stress) or per-trade cost recomputed analytically from `prem_entry`/`prem_exit` using the same `round_trip_cost` decomposition (slippage / tcost stress).

## 2. Experiments

* R1 walk-forward: ±1 fold shift, expanding window, 3-fold rolling window.
* R2 jackknife leave-one-fold-out.
* R3 slippage stress: multipliers 1.00 / 1.25 / 1.50 / 1.75 / 2.00 of the bid-ask spread portion of round-trip cost, plus bisection to solve `break_even_slippage_multiplier`.
* R4 transaction-cost stress: same multipliers applied to the brokerage + STT + txn + SEBI + stamp + GST portion, plus `break_even_cost_multiplier`.
* R5 block bootstrap: pooled PF and pooled Net CIs, plus a paired-fold bootstrap on ΔPF winner-vs-baseline.
* R6 stability: per-fold PF variance, CV, 3-fold rolling.
* R7 statistical (**corrected**): Diebold–Mariano with the hypothesis `H1: E[winner_perf − baseline_perf] > 0`; Hansen SPA + White Reality Check over the union of delay-stressed variants.
* Regime tag on each frozen fold (bull/bear/sideways).
* Execution-delay stress: 0/1/2-bar delay via signal shift (trade-generation logic unchanged).

## 3. Statistical results

### Target `mean`

* Winner pooled PF (all folds) : **1.938**  | Net Rs. **+27,483**  | MaxDD Rs. 11,811  | n=61
* Baseline pooled PF          : **1.640**  | Net Rs. **38** scenarios evaluated
* Paired bootstrap 90% CI on ΔPF: [+0.136, +2.003] point +0.298
* Bootstrap PF 90% CI: [0.770, 7.802]  |  Net 90% CI: [Rs.-8,933, Rs.+69,121]
* Break-even multipliers: slippage = 10.0480×, tcost = 7.8029×

* **Diebold–Mariano** (winner vs baseline, per-fold net PnL, H1: winner > baseline; lag=1, Newey–West):
  * statistic = **+1.4471**
  * p-value = **0.0739**
  * mean loss diff (baseline − winner, in Rs.) = +2,508.46
  * n = 8 folds  |  90% CI on loss diff = [-342.8, +5359.8]
  * interpretation: positive statistic → winner has lower loss than baseline (better)
* **Hansen SPA** over delay variants {delay_0, delay_1, delay_2} vs baseline: p_lower = 0.0355
  * WRC: pvalue not persisted from the original run — see §7 Corrections.

**Regime distribution across the frozen folds**

| Regime | Fold count |
|---|---:|
| BULL | 4 |
| BEAR | 3 |
| SIDEWAYS | 1 |

**Rolling 3-fold PF (verified from `robustness.json`)**

| Window | n_trades | PF | Net (Rs) | MaxDD (Rs) |
|---|---:|---:|---:|---:|
| 1..3 | 27 | 1.994 | +14,451 | 6,373 |
| 2..4 | 33 | 0.796 | -4,936 | 11,114 |
| 3..5 | 22 | 0.711 | -3,905 | 11,811 |
| 4..6 | 15 | 0.365 | -8,428 | 11,811 |
| 5..7 | 21 | 9.907 | +22,405 | 1,317 |
| 6..8 | 19 | 15.343 | +21,460 | 362 |

**Top-5 tornado (biggest PF drops):**

| Factor | PF | ΔPF vs winner |
|---|---:|---:|
| delay_2_bars | 1.020 | -0.918 |
| drop_fold_7 | 1.204 | -0.734 |
| delay_1_bars | 1.317 | -0.621 |
| drop_fold_1 | 1.637 | -0.301 |
| tcost_x2.00 | 1.737 | -0.201 |

### Target `stacking`

* Winner pooled PF (all folds) : **1.119**  | Net Rs. **+5,889**  | MaxDD Rs. 21,059  | n=93
* Baseline pooled PF          : **0.756**  | Net Rs. **38** scenarios evaluated
* Paired bootstrap 90% CI on ΔPF: [-55.270, +10.517] point +0.363
* Bootstrap PF 90% CI: [0.638, 2.545]  |  Net 90% CI: [Rs.-24,479, Rs.+44,286]
* Break-even multipliers: slippage = 2.1797×, tcost = 1.9419×

* **Diebold–Mariano** (winner vs baseline, per-fold net PnL, H1: winner > baseline; lag=1, Newey–West):
  * statistic = **+1.6797**
  * p-value = **0.0465**
  * mean loss diff (baseline − winner, in Rs.) = +2,976.77
  * n = 8 folds  |  90% CI on loss diff = [+61.7, +5891.9]
  * interpretation: positive statistic → winner has lower loss than baseline (better)
* **Hansen SPA** over delay variants {delay_0, delay_1, delay_2} vs baseline: p_lower = 0.0013
  * WRC: pvalue not persisted from the original run — see §7 Corrections.

**Regime distribution across the frozen folds**

| Regime | Fold count |
|---|---:|
| BULL | 4 |
| BEAR | 3 |
| SIDEWAYS | 1 |

**Rolling 3-fold PF (verified from `robustness.json`)**

| Window | n_trades | PF | Net (Rs) | MaxDD (Rs) |
|---|---:|---:|---:|---:|
| 1..3 | 32 | 3.098 | +19,897 | 4,588 |
| 2..4 | 50 | 1.792 | +16,339 | 8,797 |
| 3..5 | 38 | 0.797 | -3,515 | 9,697 |
| 4..6 | 23 | 0.790 | -2,635 | 9,697 |
| 5..7 | 32 | 0.586 | -10,698 | 13,918 |
| 6..8 | 38 | 0.589 | -11,372 | 15,908 |

**Top-5 tornado (biggest PF drops):**

| Factor | PF | ΔPF vs winner |
|---|---:|---:|
| drop_fold_2 | 0.719 | -0.400 |
| delay_2_bars | 0.723 | -0.396 |
| tcost_x2.00 | 0.993 | -0.125 |
| slippage_x2.00 | 1.017 | -0.102 |
| tcost_x1.75 | 1.023 | -0.096 |

## 4. Robustness verdicts (H_rob)

Rule (locked): ACCEPT iff ≥ 80 % of scenarios outperform baseline PF **AND** paired-bootstrap 90 % CI lower bound on ΔPF > 0 **AND** no scenario is catastrophic (PF < 0.8 or Net loss > 2× baseline net) **AND** per-fold PF CV ≤ 0.5. REJECT if ≥ 3 criteria fail; otherwise UNDECIDED.

| Target | Verdict | Outperform % | CI LB(ΔPF) | Catastrophic | CV |
|---|---|---:|---:|---:|---:|
| `mean` | **REJECT** | 71.1% | +0.136 | 3 | 1.213 |
| `stacking` | **REJECT** | 89.5% | -55.270 | 6 | 2.048 |

## 5. Limitations

* Phase 7 does not retrain models; fold shifts drop edge folds rather than re-splitting the training window.
* Slippage and tcost stress reuse the analytical `round_trip_cost` decomposition. A gap or jump-diffusion overlay would need real re-simulation with a different fill model.
* Statistical tests operate on N = 8 folds; power is limited and the paired-bootstrap CI is wide.
* **Stacking's paired-bootstrap ΔPF CI has a lower bound of ≈ −55**. This is a numerical artifact, **not** an economic loss estimate: on at least one fold the baseline's per-fold PnL sum is close to zero, so per-fold PF differentials are unbounded. The point estimate (+0.363) and upper bound (+10.5) remain meaningful. Only the lower bound is pathological, and it is treated as such by the H_rob rule (which correctly triggers `ci_lb_ok = False` under this regime).
* **Rolling-window PFs above ~4 are denominator-driven.** For example the `mean` target's rolling window folds 6..8 shows PF = 15.343 on only 19 trades and MaxDD = Rs. 362 — this reflects near-zero gross losses in a short streak rather than persistent edge. Rolling windows with n < 30 should not be over-interpreted.
* Delay-variant per-fold PnLs were not persisted to disk in the original run. SPA over delay variants is reported from the original run's transcript (see §7); the WRC value is not on-disk-reproducible and is deferred.

## 6. Production recommendation

* `mean`: **REJECT** — see `summary.json` for full criteria.
* `stacking`: **REJECT** — see `summary.json` for full criteria.

**Overall recommendation: DO NOT deploy without further evidence.**

## 7. Corrections applied (post-audit)

This is the corrected edition of the Phase 7 report. The following issues raised in the scientific validation audit have been addressed:

1. **DM sign-convention bug in `_stats.py`** was corrected. The previous call was `diebold_mariano(-winner_perf, -baseline_perf, alternative='greater')`, which tested `H1: E[baseline_perf - winner_perf] > 0` — i.e. it tested whether the winner was **worse** than the baseline. The corrected call `diebold_mariano(-baseline_perf, -winner_perf, alternative='greater')` tests `H1: E[winner_perf - baseline_perf] > 0`, which is the intended hypothesis. See `r7_correction.json` for the full audit trail and cached-input hashes.

2. **DM values were recomputed** from the Phase 6 cached per-fold trade PnLs (`logs/phase6/<target>/thr_<hash8>/trade_pnl_fold_*.csv`). No replay was performed. The recomputed values are shown in §3 and persisted in `summary.json.per_target.<target>.r7_statistical_tests`.

3. **SPA / WRC over delay variants**: the delay-shifted per-fold PnLs were not persisted during the original Phase 7 run. The Hansen SPA `p_lower` value that the original run wrote directly into this markdown is preserved and echoed into the JSON exports as-is; the White RC pvalue was not surfaced in the original markdown and is reported as `null / not on disk`. Both values will be recomputed and authoritatively persisted in a future Phase 7 refresh once delay replays are also cached to disk.

4. **Prose-report errors in the assistant's post-run summary** — the DM values quoted in that summary ("stat +0.62 p=0.267") were unverified estimates and did not match the on-disk computation. The correct DM values are those shown in §3 of this report. The assistant's summary also referenced a rolling-window PF of 6.85; the actual highest rolling PF for `mean` is 15.343 (folds 6..8), and both high-PF windows have very small sample sizes (see §3).

5. **Stacking paired-bootstrap CI clarification** — the lower bound of ≈ −55 is a numerical artifact of the per-fold PF differential when one side of the paired comparison has near-zero gross losses on any fold; the H_rob rule already treats this correctly (the `ci_lb_ok` criterion evaluates to `False`), but earlier readers were not told that the number itself carries no economic interpretation.

6. **Verdict impact**: The H_rob rule consumes the paired-bootstrap ΔPF CI, the outperform fraction, the catastrophic-scenario count, and the per-fold PF CV. None of these depend on the DM sign convention or the SPA/WRC persistence gap. The **REJECT verdict for both targets is unchanged** after the correction.
