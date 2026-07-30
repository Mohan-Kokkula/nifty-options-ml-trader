"""
shadow_daily_report.py — daily agreement/metrics report for the production
shadow framework (core/shadow_framework.py).

Reads data/shadow/predictions/shadow_<date>.jsonl (written every cycle by
core/shadow_framework.observe) and writes
data/shadow/reports/report_<date>.json containing:

  - n_cycles
  - agreement_rate                 (direction match, primary vs shadow)
  - direction_distribution         (CALL/PUT/SKIP counts, per model)
  - trade_count                    (primary, shadow, diff = shadow - primary)
  - trade_decision_agreement_rate
  - confidence                     (per-model mean, mean diff, mean |diff|)

Usage:
  python scripts/shadow_daily_report.py                # today
  python scripts/shadow_daily_report.py --date 2026-06-12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import shadow_framework as _sf  # noqa: E402


def generate_report(date: str | None = None) -> dict:
    date = date or datetime.now().strftime("%Y-%m-%d")
    pred_dir, reports_dir = _sf.PRED_DIR, _sf.REPORTS_DIR
    path = pred_dir / f"shadow_{date}.jsonl"

    recs = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue

    try:
        source = str(path.relative_to(ROOT))
    except ValueError:
        source = str(path)

    n = len(recs)
    report = {
        "date": date,
        "n_cycles": n,
        "source": source,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if n == 0:
        report["note"] = "no shadow observations for this date"
    else:
        agree = sum(1 for r in recs if r["compare"]["direction_match"])
        trade_match = sum(1 for r in recs if r["compare"]["trade_decision_match"])
        p_dirs = Counter(r["primary"]["direction"] for r in recs)
        s_dirs = Counter(r["shadow"]["direction"] for r in recs)
        p_trades = sum(1 for r in recs if r["primary"]["trade_decision"])
        s_trades = sum(1 for r in recs if r["shadow"]["trade_decision"])
        p_conf = [r["primary"]["confidence"] for r in recs]
        s_conf = [r["shadow"]["confidence"] for r in recs]
        diffs = [r["compare"]["confidence_diff"] for r in recs]

        report.update({
            "agreement_rate": round(agree / n, 4),
            "direction_distribution": {
                "primary": dict(p_dirs),
                "shadow": dict(s_dirs),
            },
            "trade_count": {
                "primary": p_trades,
                "shadow": s_trades,
                "diff": s_trades - p_trades,
            },
            "trade_decision_agreement_rate": round(trade_match / n, 4),
            "confidence": {
                "primary_mean": round(sum(p_conf) / n, 2),
                "shadow_mean": round(sum(s_conf) / n, 2),
                "mean_diff": round(sum(diffs) / n, 2),
                "mean_abs_diff": round(sum(abs(d) for d in diffs) / n, 2),
            },
            "shadow_model_dir": recs[-1]["shadow"].get("model_dir"),
        })

    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"report_{date}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    report = generate_report(args.date)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
