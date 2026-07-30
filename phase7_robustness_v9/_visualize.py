"""Chart-data JSON + optional matplotlib PNG rendering for Phase 7."""
from __future__ import annotations

import json
from pathlib import Path


def build_chart_data(target_result: dict) -> dict:
    """Return a JSON-safe payload for degradation, drawdown, rolling PF, and
    slippage sensitivity curves."""
    slip = target_result["slippage"]["curve"]
    tcost = target_result["tcost"]["curve"]
    delay = target_result["exec_delay"]["curve"]

    slippage_series = {
        "multipliers": [float(k) for k in sorted(slip.keys(), key=float)],
        "pf":  [slip[k]["pooled_pf"]  for k in sorted(slip.keys(), key=float)],
        "net": [slip[k]["pooled_net"] for k in sorted(slip.keys(), key=float)],
        "dd":  [slip[k]["pooled_max_dd"] for k in sorted(slip.keys(), key=float)],
    }
    tcost_series = {
        "multipliers": [float(k) for k in sorted(tcost.keys(), key=float)],
        "pf":  [tcost[k]["pooled_pf"]  for k in sorted(tcost.keys(), key=float)],
        "net": [tcost[k]["pooled_net"] for k in sorted(tcost.keys(), key=float)],
        "dd":  [tcost[k]["pooled_max_dd"] for k in sorted(tcost.keys(), key=float)],
    }
    delay_series = {
        "delays": [int(k) for k in sorted(delay.keys(), key=int)],
        "pf":  [delay[k]["pooled_pf"]  for k in sorted(delay.keys(), key=int)],
        "net": [delay[k]["pooled_net"] for k in sorted(delay.keys(), key=int)],
    }
    rolling = target_result["stability"]["rolling_metrics"]
    rolling_series = {
        "label": [f"{r['start_fold']}-{r['end_fold']}" for r in rolling],
        "pf":  [r["pf"]  for r in rolling],
        "net": [r["net"] for r in rolling],
        "dd":  [r["max_dd"] for r in rolling],
    }
    tornado = target_result["tornado"][:10]
    tornado_series = {
        "factor":  [r["factor"] for r in tornado],
        "delta":   [r["delta_pf_vs_baseline"] for r in tornado],
        "pf":      [r["pooled_pf"] for r in tornado],
    }
    return {
        "slippage_sensitivity": slippage_series,
        "tcost_sensitivity":    tcost_series,
        "exec_delay_sensitivity": delay_series,
        "rolling_metrics": rolling_series,
        "tornado_top10":  tornado_series,
    }


def save_chart_data(target_result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build_chart_data(target_result)
    p = out_dir / "chart_data.json"
    p.write_text(json.dumps(d, indent=2))
    return p


def write_chart_pngs(target_result: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build_chart_data(target_result)
    written: list[Path] = []

    # slippage degradation
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d["slippage_sensitivity"]["multipliers"],
              d["slippage_sensitivity"]["pf"], marker="o")
    ax.set_xlabel("Slippage multiplier")
    ax.set_ylabel("Pooled PF")
    ax.set_title("Slippage sensitivity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "slippage_sensitivity.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # tcost
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d["tcost_sensitivity"]["multipliers"],
              d["tcost_sensitivity"]["pf"], marker="o", color="tab:orange")
    ax.set_xlabel("Transaction-cost multiplier")
    ax.set_ylabel("Pooled PF")
    ax.set_title("Transaction-cost sensitivity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "tcost_sensitivity.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # delay
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(b) for b in d["exec_delay_sensitivity"]["delays"]],
             d["exec_delay_sensitivity"]["pf"], color="tab:green")
    ax.set_xlabel("Execution delay (bars)")
    ax.set_ylabel("Pooled PF")
    ax.set_title("Execution-delay stress")
    fig.tight_layout()
    p = out_dir / "delay_stress.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # rolling PF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d["rolling_metrics"]["label"], d["rolling_metrics"]["pf"],
              marker="s", color="tab:purple")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Rolling 3-fold window")
    ax.set_ylabel("Pooled PF")
    ax.set_title("Rolling PF")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "rolling_pf.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    # tornado
    fig, ax = plt.subplots(figsize=(7, 5))
    y = list(range(len(d["tornado_top10"]["factor"])))
    ax.barh(y, d["tornado_top10"]["delta"], color="tab:red")
    ax.set_yticks(y)
    ax.set_yticklabels(d["tornado_top10"]["factor"])
    ax.invert_yaxis()
    ax.set_xlabel("ΔPF vs. winner (all folds)")
    ax.set_title("Tornado: worst PF impact by stress factor")
    fig.tight_layout()
    p = out_dir / "tornado.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)

    return written
