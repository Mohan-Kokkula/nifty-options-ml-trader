# OpenClaw – Reproducibility Checklist

Every entry below is the exact configuration used to produce the
frozen artifacts under `logs/phase{0..7}/`. Follow these steps to
reproduce any phase from scratch.

## Runtime environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| Shell | Git-Bash 5.x on top of Windows (also works from PowerShell 5.1) |
| Working directory | `E:/claude-fix-auto/dynamic-stop-loss/openclaw-v9-kotak/` |
| Python interpreter | **3.12.10** (CPython x86_64) |
| Random seed | **42** everywhere (`Phase7Config.seed`, `--seed 42` on every orchestrator) |

## Package versions (frozen at Phase-7 correction cycle)

| Package | Version |
|---|---|
| numpy | 1.26.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.16.3 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| matplotlib | 3.11.0 |

Any version drift MUST invalidate the code_hash in every downstream
manifest (verified by Phase 7's `verify_cache` gate).

## Deterministic settings

- Master seed: **42**. All bootstrap resamples, paired-bootstrap
  resamples, delay stress, and grid iteration use this seed either
  directly or as the seed for a child `np.random.default_rng`.
- No thread-nondeterminism: XGBoost / LightGBM / CatBoost were trained
  in Phase 2 with fixed `n_jobs` and single-thread parity flags. Phase
  3+ do not retrain — they consume the frozen model outputs, so
  thread-nondeterminism cannot re-enter downstream.
- File I/O uses UTF-8 explicitly (`sys.stdout.reconfigure(encoding=
  "utf-8")` at every entry-point script) to avoid Windows CP1252
  encoding drift.

## Hash types used

| Purpose | Algorithm | Where recorded |
|---|---|---|
| Code file hash | SHA-256 (hex, 64 chars) | `<manifest>.code_hash.{path}` |
| Input file hash | SHA-256 | `<manifest>.input_hash.{path}` |
| Data file hash | SHA-256 | `<manifest>.data_hash.{path}` |
| Candidate identifier | first 8 hex chars of SHA-256 of a JSON dict | directory name `thr_<hash8>/` |

`verify_cache()` (Phase 6 `threshold_opt._manifest`, Phase 7
`phase7_robustness_v9._cache`) rejects any cache whose recomputed
hashes disagree with the recorded values. `CacheMismatchError` is
raised — no silent invalidation.

## Manifest locations (highest-level per phase)

| Phase | Manifest location |
|---|---|
| 0 | Pre-registration (kept alongside project root, referenced by every subsequent manifest) |
| 1 | `logs/phase1/**/manifest.json` |
| 2 | `logs/phase2/**/manifest.json` |
| 3 | `logs/phase3/**/manifest.json` |
| 4 | `logs/phase4/**/manifest.json` |
| 5 | `logs/phase5/<ensemble>/fold_<f>/manifest.json` |
| 6 | `logs/phase6/<target>/thr_<hash8>/manifest.json` + `logs/phase6/summary.json` |
| 7 | `logs/phase7/<target>/manifest.json` + `logs/phase7/summary.json` (+ post-correction: `logs/phase7/r7_correction.json`, `logs/phase7/correction_manifest.json`) |

## Cache locations (persistent)

- Phase 5: `logs/phase5/<ensemble>/fold_<f>/{predictions.csv,
  probabilities.csv, trade_pnl.csv, trades.csv, ensemble.pkl,
  weights.json, metrics.json, manifest.json}`.
- Phase 6: `logs/phase6/<target>/thr_<hash8>/{result.json,
  manifest.json, trade_pnl_fold_{1..8}.csv}` for every 360-candidate
  entry, plus per-target `candidate_results.csv`, `charts/`, and a
  root-level `summary.json`.
- Phase 7: `logs/phase7/<target>/{manifest.json, charts/{*.png,
  chart_data.json}}`, plus the five root-level JSONs and the
  markdown report. **Delay-variant per-fold PnLs are NOT persisted**
  (documented limitation).

## Required inputs

- **NIFTY spot minute-bar CSVs** (5m/15m/30m/60m/day) under `data/` —
  frozen since Phase 1.
- **India VIX** daily series under `data/india_vix.csv`.
- **NSE bhavcopy** for options premia and IV under
  `data/nse_bhavcopy/nifty_options_merged.csv`.
- **Historical expiries** (embedded in `backtest_options.py`).
- All data-file SHA-256 hashes are locked in every Phase 1–7 manifest.

## Expected outputs

| Phase | Primary evidence artifact |
|---|---|
| 1 | Walk-forward per-fold trade PnL + block-bootstrap CI |
| 2 | Per-family best-config manifests + inner-fold PF |
| 3 | Six architecturally distinct base-model prediction sets + diversity report |
| 4 | Calibrated per-base-model predictions + regime-conditional metrics |
| 5 | `logs/phase5/<ensemble>/fold_<f>/predictions.csv` + `trade_pnl.csv` (both `mean` and `stacking` are load-bearing for downstream) |
| 6 | `logs/phase6/summary.json` + `candidate_results.csv` + per-candidate `trade_pnl_fold_*.csv` cache |
| 7 | `logs/phase7/summary.json`, `robustness.json`, `stress_report.json`, `jackknife.json`, `bootstrap.json`, `Phase7_Report.md`, `r7_correction.json`, `correction_manifest.json` |

## Steps required to reproduce each phase

Every step assumes the working directory is
`E:/claude-fix-auto/dynamic-stop-loss/openclaw-v9-kotak/` and the
Python 3.12.10 environment described above is active. `python` calls
below assume that environment.

### Phase 0 — Pre-registration

```
# Frozen at Phase 6 supersede. Recreation is not part of the normal
# reproduction path; if the pre-registration file is lost, its SHA-256
# is recorded in every Phase 1+ manifest under `preregistration_sha`.
```

### Phase 1 — Walk-forward with weight ablation

```
python phase1_walkforward.py --seed 42
# Outputs → logs/phase1/
```

### Phase 2 — Nested purged WF-HPO

```
python phase2_wf_hpo.py --seed 42 --families cat,lgb,mlp,xgb
# Outputs → logs/phase2/
```

### Phase 3 — Architecture diversity

```
python phase3_diversity.py --seed 42
# Outputs → logs/phase3/
```

### Phase 4 — Calibration + regime analysis

```
python phase4_calibration.py --seed 42 --calibrator isotonic
# Outputs → logs/phase4/
```

### Phase 5 — Ensemble selection

```
python phase5_ensemble.py --seed 42 --input calibrated_isotonic \
       --ensembles mean,stacking
# Outputs → logs/phase5/<ensemble>/fold_<f>/*
```

### Phase 6 — Threshold optimization

```
python phase6_threshold_optimizer.py \
       --targets mean,stacking \
       --input calibrated_isotonic \
       --min-trades 50 \
       --seed 42 \
       --verify-prereg
# Outputs → logs/phase6/
# Wall time: ≈ 24 min on Windows-11 laptop (mean 14.8 min + stacking 9.0 min + build_frame 15 s)
```

### Phase 7 — Robustness & stress testing

```
python phase7_robustness.py \
       --targets mean,stacking \
       --seed 42
# Outputs → logs/phase7/
# Wall time: ≈ 3 min after build_frame (15 s)
```

### Phase 7 test suite

```
python -m pytest phase7_robustness_v9/tests -q --no-header
# Expect: 60 passed
```

### Phase 7 post-audit correction cycle (already applied)

Do **not** re-execute. Historical record only:

```
python "path/to/scratchpad/phase7_r7_recompute.py"
python "path/to/scratchpad/phase7_finalise_report.py"
# Read-only re-derivation of R7 stats from Phase-6 cache;
# rewrites logs/phase7/{summary,stress_report,robustness}.json and
# logs/phase7/Phase7_Report.md; leaves Phase 0-6 untouched.
```

## Verification steps for the reproducer

After running any phase, run its `verify_cache` gate:

```python
from phase7_robustness_v9 import verify_cache
from pathlib import Path
assert verify_cache(Path("logs/phase7/mean"), Path("."), "mean")
```

For Phase 6 use the equivalent `threshold_opt._manifest.verify_cache`.

## Known non-reproducible items

None of the on-disk statistical values published in the Phase 7 report
are computed from randomised state without a fixed seed. The only
outputs whose exact numeric value is NOT reproducible from disk in the
current freeze are:

- Hansen SPA `p_lower` over execution-delay variants — the number in
  `summary.json` was transcribed verbatim from the original run's
  markdown; delay-variant per-fold PnLs would need to be persisted
  before a from-scratch recomputation would yield an identical value.
  Any future refresh must persist the delay PnLs first.
- White Reality Check pvalue over execution-delay variants — not on
  disk anywhere; explicitly reported as `null` with a persistence note.
