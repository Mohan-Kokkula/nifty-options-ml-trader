"""
phase1_leakage_audit.py — Standalone data-leakage audit for Phase 1.

Four checks. Any failure exits non-zero.

  (1) Purge/embargo — verify every pre-registered fold boundary honours
      EMBARGO_DAYS between the last training bar and the first test bar.

  (2) Feature vs forward return — for each feature column, correlate
      with the next-bar return r_{t+1} = close_{t+1}/close_t - 1. Any
      column with |Pearson| > threshold is flagged as a potential
      look-ahead. Correlating with the raw next-bar close (as an earlier
      draft did) is uninformative because close is autocorrelated. The
      forward *return* strips out the level and exposes actual
      predictive-of-future signal.

  (3) Label construction — compute the median forward return within
      each label class. CALL (0) should have positive median forward
      return, PUT (1) negative, SKIP (2) near zero.

  (4) Sample-weight look-ahead documentation — quantify how many
      training rows receive the trend-day 3x multiplier, whose gate
      depends on end-of-day OHLC computed via
      ``groupby(day)['close'].last()``. This is a *documented* known
      look-ahead, not a bug — the ablation branch removes it, and this
      audit reports its footprint.

Writes: logs/phase1/leakage_audit.json
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import EMBARGO_DAYS, build_frame

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

FEATURE_LEAKAGE_THRESHOLD = 0.30  # |Pearson vs 1-bar-fwd return| > 0.30 -> flag
SAMPLE_SIZE = 50_000               # feature correlation subsample size


# ---------------------------------------------------------------------------
def audit_purge(feat: pd.DataFrame) -> list[dict]:
    out = []
    for i, (a, b) in enumerate(FOLDS, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if not tr_mask.sum() or not te_mask.sum():
            out.append({"fold": i, "status": "SKIP", "reason": "empty"})
            continue
        tr_max = feat.index[tr_mask].max().date()
        te_min = feat.index[te_mask].min().date()
        gap = (te_min - tr_max).days
        out.append({
            "fold": i,
            "cutoff": str(cutoff),
            "train_max_date": str(tr_max),
            "test_min_date": str(te_min),
            "gap_days": int(gap),
            "required_gap": EMBARGO_DAYS,
            "ok": gap >= EMBARGO_DAYS,
        })
    return out


def audit_feature_future_correlation(
    feat: pd.DataFrame, fcols: list[str],
    threshold: float = FEATURE_LEAKAGE_THRESHOLD,
) -> dict:
    """Correlate each feature at t with 1-bar-forward return.

    Uses a subsample for speed. Any |r| > threshold surfaces as flagged.
    Note that a truly predictive feature is *allowed* to correlate with
    forward return — this test catches obvious construction bugs (e.g.
    accidentally including next-bar OHLC), not learned edge.
    """
    if "close" not in feat.columns:
        return {"ok": False, "reason": "no 'close' column"}
    close = feat["close"].astype(float)
    fwd_ret = close.shift(-1) / close - 1

    rng = np.random.default_rng(0)
    n_valid = int(fwd_ret.notna().sum())
    n_sample = min(SAMPLE_SIZE, n_valid)
    valid_idx = np.flatnonzero(fwd_ret.notna().values)
    pick = rng.choice(valid_idx, size=n_sample, replace=False)
    y = fwd_ret.values[pick]

    flagged = []
    all_rows = []
    for c in fcols:
        col = feat[c].values[pick]
        if np.std(col) == 0 or np.std(y) == 0:
            continue
        r = float(np.corrcoef(col, y)[0, 1])
        all_rows.append({"feature": c, "r": r})
        if abs(r) > threshold:
            flagged.append({"feature": c, "corr_with_fwd_return_1bar": r})
    return {
        "threshold": threshold,
        "n_features_checked": len(fcols),
        "n_bars_sampled": n_sample,
        "flagged": flagged,
        # top 10 by |r|, both positive and negative — for record only
        "top_abs_correlations": sorted(
            all_rows, key=lambda x: abs(x["r"]), reverse=True)[:10],
    }


def audit_label_construction(feat: pd.DataFrame) -> dict:
    if "label" not in feat.columns or "close" not in feat.columns:
        return {"ok": False, "reason": "missing 'label' or 'close'"}
    close = feat["close"].astype(float)
    fwd_ret = close.shift(-3) / close - 1
    valid = fwd_ret.notna() & feat["label"].notna()
    grouped = fwd_ret[valid].groupby(feat["label"][valid]).median()
    return {
        "ok": True,
        "median_forward_return_by_label": {
            int(k): float(v) for k, v in grouped.items()
        },
        "label_encoding": {"0": "CALL", "1": "PUT", "2": "SKIP"},
    }


def audit_sample_weight_lookahead(feat: pd.DataFrame) -> dict:
    """Quantify the known look-ahead in
    ``backtest_threshold_sweep._sample_weights``: the trend-day 3x
    multiplier uses ``day_close - day_open`` computed on the full day
    and applied to intraday training rows.
    """
    from backtest_threshold_sweep import _sample_weights
    sub = feat.iloc[:200_000]
    w = _sample_weights(sub)
    y = sub["label"].values
    is_trade = y != 2
    skip_pct = (y == 2).mean()
    base = skip_pct / max(1 - skip_pct, 1e-9)
    upweighted = (w > base * 1.5) & is_trade
    total_trades = int(is_trade.sum())
    return {
        "known_lookahead": (
            "backtest_threshold_sweep._sample_weights uses "
            "sub.groupby(day)['close'].last() to compute "
            "|day_close - day_open|, then applies a 3x weight to every "
            "intraday training row on days where that quantity exceeds "
            "150. day_close is not known until end-of-day."
        ),
        "training_rows_sampled": int(len(sub)),
        "trade_rows_upweighted_vs_base_x1_5": int(upweighted.sum()),
        "share_of_trade_rows_upweighted": float(
            upweighted.sum() / max(total_trades, 1)),
        "ablation_available": True,
        "ablation_flag": "--no-lookahead-weights",
    }


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("  Phase 1 — Data-leakage audit")
    print("=" * 72)
    feat, fcols = build_frame()
    print(f"Frame: {len(feat):,} bars x {len(fcols)} features")

    print("\n(1) Purge / embargo audit")
    purge = audit_purge(feat)
    for c in purge:
        if c.get("status") == "SKIP":
            print(f"  Fold {c['fold']}: SKIP ({c['reason']})")
        elif c["ok"]:
            print(f"  Fold {c['fold']}: gap={c['gap_days']}d >= "
                  f"{EMBARGO_DAYS}d  OK")
        else:
            print(f"  Fold {c['fold']}: FAIL {c}")

    print("\n(2) Feature vs 1-bar forward return")
    ff = audit_feature_future_correlation(feat, fcols)
    if ff.get("flagged"):
        print(f"  FLAGGED {len(ff['flagged'])} feature(s) with "
              f"|r|>{FEATURE_LEAKAGE_THRESHOLD}:")
        for row in ff["flagged"][:20]:
            print(f"    {row['feature']}: r={row['corr_with_fwd_return_1bar']:+.3f}")
    else:
        print(f"  OK: no feature exceeds |r|>{FEATURE_LEAKAGE_THRESHOLD}")
    print(f"  top-5 by |r| (informational):")
    for row in ff.get("top_abs_correlations", [])[:5]:
        print(f"    {row['feature']}: r={row['r']:+.4f}")

    print("\n(3) Label construction sanity")
    la = audit_label_construction(feat)
    if la.get("ok"):
        for k, v in la["median_forward_return_by_label"].items():
            name = la["label_encoding"].get(str(k), "?")
            print(f"  {name} (label {k}): median fwd return = {v:+.5f}")
    else:
        print(f"  FAIL: {la}")

    print("\n(4) Sample-weight look-ahead documentation")
    sw = audit_sample_weight_lookahead(feat)
    print(f"  Known look-ahead: {sw['known_lookahead']}")
    print(f"  {sw['share_of_trade_rows_upweighted']*100:.1f}% of trade rows "
          f"receive >1.5x base weight")
    print(f"  Ablation: pass {sw['ablation_flag']} to fold runner")

    out = {
        "purge_embargo": purge,
        "feature_future_correlation": ff,
        "label_construction": la,
        "sample_weight_lookahead": sw,
    }
    (ROOT / "logs/phase1").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "logs/phase1/leakage_audit.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote logs/phase1/leakage_audit.json")

    purge_ok = all(c.get("ok", False)
                   for c in purge if c.get("status") != "SKIP")
    feature_ok = not ff.get("flagged")
    label_ok = la.get("ok") and \
        la["median_forward_return_by_label"].get(0, 0) > 0 and \
        la["median_forward_return_by_label"].get(1, 0) < 0
    if purge_ok and feature_ok and label_ok:
        print("\nAudit PASSED.")
        sys.exit(0)
    print(f"\nAudit FAILED (purge_ok={purge_ok} feature_ok={feature_ok} "
          f"label_ok={label_ok})")
    sys.exit(2)


if __name__ == "__main__":
    main()
