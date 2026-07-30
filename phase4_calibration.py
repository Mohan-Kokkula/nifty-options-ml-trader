"""
phase4_calibration.py — Orchestrator + aggregator for Phase 4.

Iterates (brain × outer_fold), dispatches to phase4_run_single_fold in
an isolated subprocess (same pattern as Phase 2/3). Aggregates across
folds using stat_utils:

  * per-(brain, config) block-bootstrap 90 % CI on PF
  * paired block-bootstrap 90 % CI on ΔPF between each calibrated
    config and its uncalibrated baseline (same brain)
  * Diebold-Mariano on bar-aligned trade P&L differential
  * H_cal verdict: pre-registered ``LB90(ΔPF_isotonic) > +0.05``

Writes ``logs/phase4/summary.json`` + ``logs/phase4/comparison.txt``.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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

DEFAULT_BRAINS = ["xgb", "lgb", "cat", "mlp"]
CONFIGS = ("uncal", "noop", "platt", "isotonic")

_MODULES_TRACKED = (
    "backtest_threshold_sweep.py",
    "backtest_options.py",
    "phase4_run_single_fold.py",
    "brains/__init__.py",
    "brains/_base.py",
    "brains/_hpo.py",
    "brains/xgb_adapter.py",
    "brains/lgb_adapter.py",
    "brains/cat_adapter.py",
    "brains/mlp_adapter.py",
    "calibrators/__init__.py",
    "calibrators/_base.py",
    "calibrators/_metrics.py",
    "calibrators/noop.py",
    "calibrators/platt.py",
    "calibrators/isotonic.py",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _current_source() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in _MODULES_TRACKED
            if (ROOT / f).exists()}


def _cache_ok(fold_root: Path, expect: dict, k_inner: int
              ) -> tuple[bool, str]:
    required = ["manifest.json", "uncal_metrics.json",
                "calibration_diagnostics.json", "reliability_data.json"]
    for r in required:
        if not (fold_root / r).exists():
            return False, f"missing {r}"
    for cfg in ("noop", "platt", "isotonic"):
        for suff in ("_metrics.json", "_trades.csv", "_trade_pnl.csv",
                     "_calibrator.pkl"):
            if not (fold_root / f"{cfg}{suff}").exists():
                return False, f"missing {cfg}{suff}"
        if not ((fold_root / f"{cfg}_predictions.parquet").exists()
                or (fold_root / f"{cfg}_predictions.csv").exists()):
            return False, f"missing {cfg}_predictions"
    try:
        m = json.loads((fold_root / "manifest.json").read_text())
    except Exception as e:
        return False, f"unreadable manifest ({e})"
    if int(m.get("k_inner", 0)) != k_inner:
        return False, f"k_inner differs ({m.get('k_inner')} vs {k_inner})"
    src = m.get("source_sha256", {})
    for f, h in expect.items():
        if src.get(f) != h:
            return False, f"source SHA changed for {f}"
    return True, "ok"


def _prevent_sleep_windows() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001))
    except Exception:
        return False


def run_one(brain_name: str, i: int, a: date, b: date, seed: int,
            k_inner: int, force: bool, verify_prereg: bool
            ) -> tuple[bool, str]:
    fold_root = ROOT / f"logs/phase4/{brain_name}/fold_{i}"
    expect = _current_source()
    if not force:
        ok, why = _cache_ok(fold_root, expect, k_inner)
        if ok:
            print(f"[{brain_name}/fold {i}] cache HIT; skipping ({why})")
            return False, "cached"
        print(f"[{brain_name}/fold {i}] cache MISS ({why}); recomputing")
    cmd = [sys.executable, "-u", "phase4_run_single_fold.py",
           brain_name, str(i), a.isoformat(), b.isoformat(),
           "--k-inner", str(k_inner),
           "--seed", str(seed)]
    if verify_prereg:
        cmd.append("--verify-prereg")
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        return False, f"FAILED exit={e.returncode}"
    return True, "ran"


# ---------------------------------------------------------------------------
def aggregate(brains: list[str]) -> None:
    import numpy as np
    import pandas as pd
    from stat_utils import (block_bootstrap_ci, diebold_mariano,
                              paired_block_bootstrap_ci,
                              profit_factor, sharpe as sharpe_fn,
                              sortino as sortino_fn, win_rate)

    print("\n" + "=" * 78)
    print("  Phase 4 — Calibration aggregate")
    print("=" * 78)

    streams: dict[str, dict[str, dict[int, np.ndarray]]] = {
        b: {c: {} for c in CONFIGS} for b in brains}
    calib_metrics: dict[str, dict[str, dict[int, dict]]] = {
        b: {c: {} for c in CONFIGS} for b in brains}

    for b in brains:
        for i, (a, bb) in enumerate(FOLDS, 1):
            fr = ROOT / f"logs/phase4/{b}/fold_{i}"
            if not (fr / "manifest.json").exists():
                continue
            ph3_pnl = ROOT / f"logs/phase3/{b}/fold_{i}/trade_pnl.csv"
            if ph3_pnl.exists():
                try:
                    df = pd.read_csv(ph3_pnl)
                    if "trade_pnl" in df.columns and not df.empty:
                        streams[b]["uncal"][i] = df["trade_pnl"].values.astype(float)
                except Exception:
                    pass
            try:
                calib_metrics[b]["uncal"][i] = json.loads(
                    (fr / "uncal_metrics.json").read_text()).get("calibration", {})
            except Exception:
                pass
            for cfg in ("noop", "platt", "isotonic"):
                tp = fr / f"{cfg}_trade_pnl.csv"
                if tp.exists():
                    try:
                        df = pd.read_csv(tp)
                        if "trade_pnl" in df.columns and not df.empty:
                            streams[b][cfg][i] = df["trade_pnl"].values.astype(float)
                    except Exception:
                        pass
                mp = fr / f"{cfg}_metrics.json"
                if mp.exists():
                    try:
                        m = json.loads(mp.read_text())
                        calib_metrics[b][cfg][i] = m.get("calibration", {})
                    except Exception:
                        pass

    print(f"  {'brain':>6} {'cfg':>10} {'folds':>6} {'pooled_n':>9} {'PF':>7}"
          f" {'90% CI':>18} {'Net Rs.':>15} {'WR%':>5} {'Sharpe':>8}"
          f" {'Sortino':>8} {'meanECE':>9} {'meanBrier':>10}")
    print("  " + "-" * 120)
    per_brain: dict[str, dict[str, dict]] = {}
    for b in brains:
        per_brain[b] = {}
        for cfg in CONFIGS:
            s = streams[b][cfg]
            if not s:
                per_brain[b][cfg] = {"status": "MISSING"}
                continue
            pooled = np.concatenate(list(s.values()))
            pf = float(profit_factor(pooled))
            wr = float(win_rate(pooled))
            net = float(pooled.sum())
            try:
                sh = float(sharpe_fn(pooled))
                so = float(sortino_fn(pooled))
            except Exception:
                sh, so = float("nan"), float("nan")
            ci = block_bootstrap_ci(s, profit_factor,
                                     n_resamples=10_000, ci_level=0.90,
                                     seed=42, stat_name=f"{b}_{cfg}_pf")
            eces = [calib_metrics[b][cfg][k].get("top1_ece")
                     for k in calib_metrics[b][cfg]]
            eces = [x for x in eces if isinstance(x, (int, float))]
            briers = [calib_metrics[b][cfg][k].get("brier")
                       for k in calib_metrics[b][cfg]]
            briers = [x for x in briers if isinstance(x, (int, float))]
            mean_ece = float(np.mean(eces)) if eces else float("nan")
            mean_brier = float(np.mean(briers)) if briers else float("nan")
            per_brain[b][cfg] = {
                "n_folds": len(s),
                "n_pooled": int(len(pooled)),
                "pf": pf, "pf_ci90": [ci.lower, ci.upper],
                "net": net, "wr": wr, "sharpe": sh, "sortino": so,
                "mean_top1_ece": mean_ece,
                "mean_brier": mean_brier,
            }
            print(f"  {b:>6} {cfg:>10} {len(s):>6} {len(pooled):>9} {pf:>7.3f} "
                  f"[{ci.lower:>+6.3f}, {ci.upper:>+6.3f}] {net:>+15,.0f} "
                  f"{wr*100:>4.1f}% {sh:>+8.3f} {so:>+8.3f} "
                  f"{mean_ece:>9.4f} {mean_brier:>10.4f}")

    delta_summary: dict[str, dict[str, dict]] = {}
    print("\n  Calibration effect on trading (calibrated - uncal):")
    print(f"  {'brain':>6} {'cfg':>10} {'delta_PF':>10} "
          f"{'90% paired CI':>22} {'DM_p (grtr)':>12} {'delta_Net Rs.':>16}")
    print("  " + "-" * 82)
    for b in brains:
        delta_summary[b] = {}
        s_uncal = streams[b]["uncal"]
        if not s_uncal:
            continue
        for cfg in ("noop", "platt", "isotonic"):
            s_cal = streams[b][cfg]
            common_folds = sorted(set(s_uncal) & set(s_cal))
            if not common_folds:
                delta_summary[b][cfg] = {"status": "MISSING"}
                continue
            a_dict = {k: s_cal[k] for k in common_folds}
            b_dict = {k: s_uncal[k] for k in common_folds}

            def _dpf(a_arr: np.ndarray, b_arr: np.ndarray) -> float:
                pfa = profit_factor(a_arr) if len(a_arr) else float("nan")
                pfb = profit_factor(b_arr) if len(b_arr) else float("nan")
                if not (np.isfinite(pfa) and np.isfinite(pfb)):
                    return float("nan")
                return pfa - pfb

            ci = paired_block_bootstrap_ci(
                a_dict, b_dict, _dpf,
                n_resamples=10_000, ci_level=0.90, seed=42,
                stat_name=f"{b}_{cfg}_dPF")

            a_pieces, b_pieces = [], []
            for k in common_folds:
                fa = ROOT / f"logs/phase4/{b}/fold_{k}/{cfg}_trades.csv"
                fbb = ROOT / f"logs/phase3/{b}/fold_{k}/trades.csv"
                if not (fa.exists() and fbb.exists()):
                    continue
                try:
                    adf = pd.read_csv(fa)
                    bdf = pd.read_csv(fbb)
                except Exception:
                    continue
                if "time" not in adf.columns or "time" not in bdf.columns:
                    continue
                a_by = adf.set_index("time")["net_option"]
                b_by = bdf.set_index("time")["net_option"]
                common = a_by.index.intersection(b_by.index)
                if len(common):
                    a_pieces.append(a_by.loc[common].values.astype(float))
                    b_pieces.append(b_by.loc[common].values.astype(float))
            dm_p, dm_stat, n_bars = None, None, 0
            if a_pieces:
                aa = np.concatenate(a_pieces)
                bb = np.concatenate(b_pieces)
                try:
                    dm = diebold_mariano(aa, bb, alternative="greater",
                                          lag=5, ci_level=0.90)
                    dm_p = dm.pvalue
                    dm_stat = dm.statistic
                    n_bars = dm.n
                except Exception:
                    pass
            net_delta = (per_brain[b][cfg]["net"]
                         - per_brain[b]["uncal"]["net"])
            delta_summary[b][cfg] = {
                "delta_pf_point": ci.point_estimate,
                "delta_pf_ci_lower_90": ci.lower,
                "delta_pf_ci_upper_90": ci.upper,
                "dm_statistic": dm_stat,
                "dm_pvalue": dm_p,
                "dm_n_bars": n_bars,
                "delta_net": net_delta,
            }
            dm_p_s = f"{dm_p:.4f}" if isinstance(dm_p, float) else "  n/a  "
            print(f"  {b:>6} {cfg:>10} {ci.point_estimate:>+10.3f} "
                  f"[{ci.lower:>+7.3f}, {ci.upper:>+7.3f}] {dm_p_s:>12} "
                  f"{net_delta:>+16,.0f}")

    print("\n  H_cal verdict (pre-registered acceptance criteria):")
    print(f"    rule: LB90(dPF_isotonic) > +0.05 for AT LEAST ONE brain")
    passing_brains = []
    for b in brains:
        row = delta_summary.get(b, {}).get("isotonic", {})
        lb = row.get("delta_pf_ci_lower_90")
        if isinstance(lb, (int, float)) and lb > 0.05:
            passing_brains.append(b)
    if passing_brains:
        decision = f"REJECT H0_cal (brains passing: {passing_brains})"
    else:
        decision = "FAIL TO REJECT H0_cal"
    print(f"    passing brains       : {passing_brains}")
    print(f"    decision             : {decision}")

    summary = {
        "brains": brains,
        "configs": list(CONFIGS),
        "per_brain": per_brain,
        "delta_vs_uncal": delta_summary,
        "h_cal_verdict": {
            "rule": "LB90(dPF_isotonic) > +0.05 for at least one brain",
            "passing_brains": passing_brains,
            "decision": decision,
        },
    }
    (ROOT / "logs/phase4").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs/phase4/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n  wrote logs/phase4/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default=",".join(DEFAULT_BRAINS))
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-prereg", action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    brains = [b.strip() for b in args.brains.split(",") if b.strip()]

    if args.aggregate_only:
        aggregate(brains)
        return

    if args.folds == "all":
        selected = list(range(1, len(FOLDS) + 1))
    else:
        selected = [int(x) for x in args.folds.split(",") if x.strip()]

    (ROOT / "logs/phase4").mkdir(parents=True, exist_ok=True)

    sleep_prevented = _prevent_sleep_windows()
    if platform.system() == "Windows":
        print(f"[pre-flight] Windows sleep prevention: {sleep_prevented}")

    statuses = []
    for b in brains:
        for i in selected:
            a, bb = FOLDS[i - 1]
            ran, st = run_one(b, i, a, bb, args.seed, args.k_inner,
                               args.force, args.verify_prereg)
            statuses.append((b, i, st))

    print("\n  Run summary:")
    for b, i, st in statuses:
        print(f"    {b} fold {i}: {st}")

    if not args.skip_aggregate:
        aggregate(brains)


if __name__ == "__main__":
    main()
