import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
r = json.load(open(ROOT / "data/validation_results/report.json"))


def fm(m):
    if not m or m.get("trades", 0) == 0:
        return "no trades"
    return (f"n={m['trades']:5d} wr={m['wr']:.1%} pf={m['pf']:.2f} "
            f"shp={m.get('sharpe', 0):5.2f} dd={m['maxdd']:>9,.0f} "
            f"net={m['net']:>11,.0f} avgR={m['avg_r']:+.3f} "
            f"exp={m['expectancy']:+7.0f}")


print("=== WALK-FORWARD (V9 engine; train<=2024-08, val<=2025-07, TEST after) ===")
for v in ("OLD", "NEW"):
    print(f"\n  {v}:")
    for k, m in r[v]["wf"].items():
        print(f"    {k:24s} {fm(m)}")

print("\n=== YEARLY (OLD | NEW) ===")
ys = sorted(set(r["OLD"]["yearly"]) | set(r["NEW"]["yearly"]), key=int)
for y in ys:
    o = r["OLD"]["yearly"].get(y, {})
    n_ = r["NEW"]["yearly"].get(y, {})
    print(f"  {y}: OLD n={o.get('trades',0):4d} wr={o.get('wr',0):.0%} "
          f"pf={o.get('pf',0):4.2f} net={o.get('net',0):>9,.0f} "
          f"dd={o.get('maxdd',0):>8,.0f} | NEW n={n_.get('trades',0):4d} "
          f"wr={n_.get('wr',0):.0%} pf={n_.get('pf',0):4.2f} "
          f"net={n_.get('net',0):>9,.0f} dd={n_.get('maxdd',0):>8,.0f}")

for v in ("OLD", "NEW"):
    print(f"\n=== REGIMES ({v}) ===")
    for k, m in r[v].get("regimes", {}).items():
        print(f"  {k:18s} {fm(m)}")

print("\n=== MONTE CARLO ===")
for v in ("OLD", "NEW"):
    mc = r[v].get("monte_carlo", {})
    print(f"  {v}: dd_p50={mc.get('dd_p50',0):>9,.0f} "
          f"dd_p95={mc.get('dd_p95',0):>9,.0f} dd_p99={mc.get('dd_p99',0):>9,.0f} "
          f"P(ruin50%)={mc.get('prob_ruin_50pct_capital',0):.4f} "
          f"annPnL={mc.get('ann_pnl',0):>9,.0f}")

print("\n=== OVERALL (full 11.4y, model-in-sample-dominated) ===")
print("  PRE :", fm(r["pre"]))
print("  OLD :", fm(r["OLD"]["overall"]))
print("  NEW :", fm(r["NEW"]["overall"]))

print("\n=== SOLO fixes (OLD + one fix) ===")
for k, m in r["solo"].items():
    print(f"  {k:9s} {fm(m)}")
print("\n=== ABLATIONS (NEW minus one fix) ===")
for k, m in r["abl"].items():
    print(f"  {k:9s} {fm(m)}")

print("\n=== V10 SECONDARY WINDOW ===")
vw = r.get("v10_window", {})
print("  range:", vw.get("range"))
for v in ("OLD", "NEW"):
    if v in vw:
        print(f"  V10 {v}:", fm(vw[v]))
