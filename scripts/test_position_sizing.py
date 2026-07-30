#!/usr/bin/env python3
"""
Unit tests for dynamic position sizing (core/claude_pilot.py _compute_position_size).

Standalone script (repo convention — see scripts/test_*.py siblings), not
pytest-collected (pytest.ini testpaths = stat_utils/tests only). Run with:
    python scripts/test_position_sizing.py

Calls _compute_position_size on a lightweight stub instead of a real
ClaudePilot — constructing a real ClaudePilot has side effects (writes to
the production trade journal, spins up an IV engine) that don't belong in
a unit test of a pure sizing calculation.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claude_pilot import ClaudePilot, PilotConfig


def make_stub(config: PilotConfig, max_loss_per_trade: float = 2000.0):
    """A minimal object exposing only what _compute_position_size reads."""
    risk = SimpleNamespace(max_loss_per_trade=max_loss_per_trade)
    trader = SimpleNamespace(risk=risk)
    return SimpleNamespace(config=config, trader=trader, _size_halved_remaining=0)


def compute(stub, sl_pts, ml_confidence, delta=0.5):
    return ClaudePilot._compute_position_size(stub, sl_pts, ml_confidence, delta)


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True
    cfg = PilotConfig(
        account_equity=100_000.0,
        max_risk_pct=0.02,      # risk ₹2,000/trade at full Kelly
        kelly_fraction=0.25,
        min_lots=1,
        max_lots=5,
        lot_size=65,
    )

    # 1. Baseline: moderate SL, mid confidence — should return >= min_lots
    stub = make_stub(cfg, max_loss_per_trade=2000.0)
    lots = compute(stub, sl_pts=40.0, ml_confidence=0.65, delta=0.5)
    all_ok &= check(f"baseline sizing returns >=1 lot (got {lots})", lots >= 1)

    # 2. Never exceeds max_lots regardless of confidence
    stub = make_stub(cfg, max_loss_per_trade=1_000_000.0)  # disable the cap for this case
    lots = compute(stub, sl_pts=5.0, ml_confidence=1.0, delta=0.9)
    all_ok &= check(f"lots clamped to max_lots=5 (got {lots})", lots <= cfg.max_lots)

    # 3. Never below min_lots (unless the trade-skip sentinel 0 fires)
    stub = make_stub(cfg, max_loss_per_trade=2000.0)
    lots = compute(stub, sl_pts=40.0, ml_confidence=0.50, delta=0.5)
    all_ok &= check(f"lots >= min_lots or 0-skip sentinel (got {lots})", lots == 0 or lots >= cfg.min_lots)

    # 4. MAX_LOSS_PER_TRADE (via RiskManager) hard-caps lots
    # loss_per_lot = sl_pts * delta * lot_size = 100 * 0.5 * 65 = 3250/lot
    # max_loss_per_trade = 2000 -> max 0 lots afford-able -> must return 0 (skip)
    stub = make_stub(cfg, max_loss_per_trade=2000.0)
    lots = compute(stub, sl_pts=100.0, ml_confidence=0.90, delta=0.5)
    all_ok &= check(f"trade skipped (0) when 1 lot exceeds MAX_LOSS_PER_TRADE (got {lots})", lots == 0)

    # 5. RiskManager integration: a HIGHER max_loss_per_trade allows more lots
    # for the same setup that was capped to 0 above.
    stub_tight = make_stub(cfg, max_loss_per_trade=2000.0)
    stub_loose = make_stub(cfg, max_loss_per_trade=20000.0)
    lots_tight = compute(stub_tight, sl_pts=100.0, ml_confidence=0.90, delta=0.5)
    lots_loose = compute(stub_loose, sl_pts=100.0, ml_confidence=0.90, delta=0.5)
    all_ok &= check(
        f"raising RiskManager.max_loss_per_trade increases allowed lots "
        f"(tight={lots_tight}, loose={lots_loose})",
        lots_loose > lots_tight,
    )

    # 6. Zero/negative SL distance defaults to min_lots rather than crashing
    stub = make_stub(cfg, max_loss_per_trade=2000.0)
    lots = compute(stub, sl_pts=0.0, ml_confidence=0.65, delta=0.5)
    all_ok &= check(f"sl_pts<=0 defaults to min_lots (got {lots})", lots == cfg.min_lots)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
