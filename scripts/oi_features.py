"""
oi_features.py — PART D: intraday OI feature engine (features ONLY, no models).

Operates on the oi_archiver output (1-min ATM series). All features are
fixed-strike-aware: rows are grouped by atm_strike, and OI changes are only
computed within runs of the SAME strike (an ATM re-reference is a different
instrument; deltas across the splice are masked NaN).

Features (per pre-registration in the OI panel review):
  atm_oi_change_{5,15,30}m : net (put_oi - call_oi) change over the window,
                             normalized by the day's first valid OI level
  call_unwinding           : price up & call_oi falling     (bullish-coded +1)
  put_writing              : price up & put_oi rising       (bullish-coded +1)
  short_covering           : price up & total OI falling
  long_buildup             : price up & total OI rising
  pcr_velocity             : d(pcr)/dt over 15m (robust Theil-Sen-lite: median
                             of pairwise slopes on 5 points)
  oi_acceleration          : 2nd difference of 15m ATM OI change (flagged by
                             the panel as noise-prone; emitted for diagnostics)

Diagnostics report: reports/oi_feature_diagnostics.json — NaN rates, ranges,
stability. Runs on the real archive when present; otherwise on clearly-marked
synthetic data to validate the formulas end-to-end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT / "data" / "openalgo_oi"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: one day of 1-min rows with the archiver schema."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    out = pd.DataFrame(index=df.index)

    # fixed-strike runs: deltas valid only within the same ATM strike
    same = df["atm_strike"].eq(df["atm_strike"].shift())
    base_oi = max(float(df[["call_oi", "put_oi"]].iloc[0].sum()), 1.0)

    for w in (5, 15, 30):
        d_call = df["call_oi"].diff(w)
        d_put = df["put_oi"].diff(w)
        run_ok = same.rolling(w).sum().eq(w)         # whole window same strike
        net = (d_put - d_call).where(run_ok) / base_oi
        out[f"atm_oi_change_{w}m"] = net

    px_up = df["spot"].diff(5) > 0
    d_call5 = df["call_oi"].diff(5).where(same.rolling(5).sum().eq(5))
    d_put5 = df["put_oi"].diff(5).where(same.rolling(5).sum().eq(5))
    d_tot5 = d_call5 + d_put5
    out["call_unwinding"] = ((px_up) & (d_call5 < 0)).astype(float).where(d_call5.notna())
    out["put_writing"] = ((px_up) & (d_put5 > 0)).astype(float).where(d_put5.notna())
    out["short_covering"] = ((px_up) & (d_tot5 < 0)).astype(float).where(d_tot5.notna())
    out["long_buildup"] = ((px_up) & (d_tot5 > 0)).astype(float).where(d_tot5.notna())

    # pcr_velocity: median pairwise slope over last 5 points spanning 15m
    pcr = df["pcr"].values
    n = len(df)
    vel = np.full(n, np.nan)
    for i in range(15, n):
        pts = pcr[i - 15:i + 1:4][-5:]
        if len(pts) >= 3 and not np.isnan(pts).any():
            slopes = [(pts[b] - pts[a]) / (b - a)
                      for a in range(len(pts)) for b in range(a + 1, len(pts))]
            vel[i] = float(np.median(slopes))
    out["pcr_velocity"] = vel

    out["oi_acceleration"] = out["atm_oi_change_15m"].diff(5)

    # ── Priority-4 additions (per roadmap), fixed-strike-masked ──────────
    run5 = same.rolling(5).sum().eq(5)
    # ATM call/put OI % change over 5m (NaN across an ATM re-reference)
    out["atm_ce_oi_chg_pct"] = (df["call_oi"].pct_change(5)).where(run5)
    out["atm_pe_oi_chg_pct"] = (df["put_oi"].pct_change(5)).where(run5)
    # signed buildup: +1 price & total-OI move same way, -1 opposite (regime)
    d_spot5 = df["spot"].diff(5)
    d_tot_oi5 = (df["call_oi"] + df["put_oi"]).diff(5).where(run5)
    out["oi_buildup_signed"] = (np.sign(d_spot5) * np.sign(d_tot_oi5)).where(
        d_tot_oi5.notna())
    # net OI flow: (Δput - Δcall) normalized by gross flow, in [-1, 1]
    denom = (d_put5.abs() + d_call5.abs()).replace(0, np.nan)
    out["net_oi_flow"] = ((d_put5 - d_call5) / denom).where(run5)
    return out


def _synthetic_day(seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-12 09:15", "2026-06-12 15:30", freq="1min")
    n = len(ts)
    spot = 23300 + np.cumsum(rng.normal(0, 4, n))
    atm = (spot / 50).round() * 50
    return pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "spot": spot.round(2), "atm_strike": atm.astype(int),
        "call_oi": (5e6 + np.cumsum(rng.normal(2000, 8000, n))).astype(int),
        "put_oi": (6e6 + np.cumsum(rng.normal(3000, 9000, n))).astype(int),
        "call_iv": 14 + rng.normal(0, 0.2, n),
        "put_iv": 15 + rng.normal(0, 0.2, n),
        "pcr": 1.05 + np.cumsum(rng.normal(0, 0.002, n)),
    })


def main():
    files = sorted(ARCHIVE.glob("oi_2*.csv")) + sorted(ARCHIVE.glob("oi_2*.parquet"))
    files = [f for f in files if "gaps" not in f.name]
    if files:
        src, synthetic = files[-1].name, False
        df = (pd.read_parquet(files[-1]) if files[-1].suffix == ".parquet"
              else pd.read_csv(files[-1]))
    else:
        src, synthetic = "SYNTHETIC (no archive data yet)", True
        df = _synthetic_day()
    feats = compute_features(df)

    diag = {"source": src, "synthetic": synthetic, "rows": len(feats),
            "features": {}}
    for c in feats.columns:
        s = feats[c]
        diag["features"][c] = {
            "nan_rate": round(float(s.isna().mean()), 4),
            "mean": round(float(s.mean()), 6) if s.notna().any() else None,
            "std": round(float(s.std()), 6) if s.notna().any() else None,
            "min": round(float(s.min()), 6) if s.notna().any() else None,
            "max": round(float(s.max()), 6) if s.notna().any() else None,
        }
    diag["notes"] = [
        "fixed-strike: OI deltas masked across ATM re-references",
        "oi_acceleration emitted for diagnostics only (panel: noise-prone)",
        "no model training performed (per Part D requirements)",
    ]
    out = REPORTS / "oi_feature_diagnostics.json"
    json.dump(diag, open(out, "w"), indent=1)
    print(f"Diagnostics ({src}): {len(feats)} rows, "
          f"{len(feats.columns)} features -> {out}")
    for c, d in diag["features"].items():
        print(f"  {c:22s} nan={d['nan_rate']:.1%} mean={d['mean']}")


if __name__ == "__main__":
    main()
