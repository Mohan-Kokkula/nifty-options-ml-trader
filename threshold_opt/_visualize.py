"""Chart data + optional PNG rendering.

The authoritative output is ``chart_data.json``; PNG rendering is
optional and silently skipped when matplotlib is not available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._evaluate import CandidateResult


def _to_records(results: list[CandidateResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        c = r.candidate
        rows.append({
            "call_thr": float(c.call_thr),
            "put_thr": float(c.put_thr),
            "skip_ceil": float(c.skip_ceil),
            "min_edge": float(c.min_edge),
            "pf": r.pooled.get("pf"),
            "net": r.pooled.get("net"),
            "n_trades": r.pooled.get("n", 0),
            "passes_min_trades": bool(r.passes_min_trades),
        })
    return pd.DataFrame(rows)


def build_chart_data(
    results: list[CandidateResult],
    hold_skip: float = 0.65,
    hold_edge: float = 0.05,
) -> dict:
    """Return chart data holding ``skip_ceil`` and ``min_edge`` fixed.

    All 4 chart panels share the same slice so the marginal effects of
    ``call_thr`` and ``put_thr`` can be compared like-for-like.
    """
    df = _to_records(results)
    mask = ((df["skip_ceil"] == hold_skip)
            & (df["min_edge"] == hold_edge))
    sub = df[mask]

    def _agg_by(col: str, value: str) -> dict:
        if sub.empty:
            return {}
        out: dict = {}
        for k, grp in sub.groupby(col):
            vals = [v for v in grp[value].tolist()
                     if isinstance(v, (int, float)) and np.isfinite(v)]
            out[float(k)] = {
                "count": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "min":  float(np.min(vals))  if vals else None,
                "max":  float(np.max(vals))  if vals else None,
            }
        return out

    def _pivot(values: str) -> dict:
        if sub.empty:
            return {"index": [], "columns": [], "data": []}
        piv = sub.pivot_table(index="call_thr", columns="put_thr",
                                 values=values, aggfunc="mean")
        return {
            "index": [float(x) for x in piv.index],
            "columns": [float(x) for x in piv.columns],
            "data": [[float(v) if isinstance(v, (int, float))
                                 and np.isfinite(v) else None
                       for v in row]
                      for row in piv.values],
        }

    return {
        "hold": {"skip_ceil": float(hold_skip),
                  "min_edge": float(hold_edge),
                  "n_candidates_in_slice": int(len(sub))},
        "call_vs_pf": _agg_by("call_thr", "pf"),
        "put_vs_pf": _agg_by("put_thr", "pf"),
        "call_x_put_pf": _pivot("pf"),
        "call_x_put_trade_count": _pivot("n_trades"),
    }


def write_chart_pngs(
    chart_data: dict,
    out_dir: Path,
    title_prefix: str = "",
) -> list[Path]:
    """Render the four charts to PNG. Silent skip if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Panel 1: CALL vs PF
    call_vs = chart_data.get("call_vs_pf", {})
    if call_vs:
        xs = sorted(call_vs.keys())
        ys = [call_vs[x]["mean"] if call_vs[x]["mean"] is not None else np.nan
              for x in xs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o")
        ax.axvline(0.32, linestyle="--", color="grey",
                    label="production call_thr=0.32")
        ax.set_xlabel("call_thr"); ax.set_ylabel("mean pooled PF")
        ax.set_title(f"{title_prefix}CALL threshold vs mean pooled PF")
        ax.grid(True, alpha=0.3); ax.legend()
        p = out_dir / "call_vs_pf.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig); written.append(p)

    # Panel 2: PUT vs PF
    put_vs = chart_data.get("put_vs_pf", {})
    if put_vs:
        xs = sorted(put_vs.keys())
        ys = [put_vs[x]["mean"] if put_vs[x]["mean"] is not None else np.nan
              for x in xs]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, marker="o")
        ax.axvline(0.25, linestyle="--", color="grey",
                    label="production put_thr=0.25")
        ax.set_xlabel("put_thr"); ax.set_ylabel("mean pooled PF")
        ax.set_title(f"{title_prefix}PUT threshold vs mean pooled PF")
        ax.grid(True, alpha=0.3); ax.legend()
        p = out_dir / "put_vs_pf.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig); written.append(p)

    # Panel 3: CALL x PUT PF heatmap
    piv = chart_data.get("call_x_put_pf", {})
    if piv.get("data"):
        fig, ax = plt.subplots(figsize=(8, 6))
        arr = np.array(piv["data"], dtype=float)
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(piv["columns"])))
        ax.set_xticklabels([f"{x:.2f}" for x in piv["columns"]])
        ax.set_yticks(range(len(piv["index"])))
        ax.set_yticklabels([f"{x:.2f}" for x in piv["index"]])
        ax.set_xlabel("put_thr"); ax.set_ylabel("call_thr")
        ax.set_title(f"{title_prefix}CALL x PUT pooled PF heatmap")
        fig.colorbar(im, ax=ax, label="pooled PF")
        p = out_dir / "call_x_put_pf_heatmap.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig); written.append(p)

    # Panel 4: CALL x PUT trade count heatmap
    piv = chart_data.get("call_x_put_trade_count", {})
    if piv.get("data"):
        fig, ax = plt.subplots(figsize=(8, 6))
        arr = np.array(piv["data"], dtype=float)
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
        ax.set_xticks(range(len(piv["columns"])))
        ax.set_xticklabels([f"{x:.2f}" for x in piv["columns"]])
        ax.set_yticks(range(len(piv["index"])))
        ax.set_yticklabels([f"{x:.2f}" for x in piv["index"]])
        ax.set_xlabel("put_thr"); ax.set_ylabel("call_thr")
        ax.set_title(f"{title_prefix}CALL x PUT trade count heatmap")
        fig.colorbar(im, ax=ax, label="pooled trade count")
        p = out_dir / "call_x_put_trade_count_heatmap.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig); written.append(p)

    return written


def save_chart_data(chart_data: dict, out_dir: Path) -> Path:
    """Write ``chart_data.json`` — the authoritative source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "chart_data.json"
    with open(p, "w") as fh:
        json.dump(chart_data, fh, indent=2, default=str)
    return p
