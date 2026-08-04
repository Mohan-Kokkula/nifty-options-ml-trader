#!/usr/bin/env python3
"""
Unit tests for futures position sizing (core/claude_pilot.py
_compute_futures_position_size). Only reachable when
PilotConfig.execution_mode == "futures" (default "options" — unused in
today's live behavior).

Standalone script (repo convention, mirrors scripts/test_position_sizing.py),
not pytest-collected. Run with:
    python scripts/test_futures_position_sizing.py

Uses a lightweight stub instead of a real ClaudePilot, same rationale as
test_position_sizing.py: constructing a real ClaudePilot has side effects
that don't belong in a unit test of a pure sizing calculation.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.claude_pilot import ClaudePilot, PilotConfig


def make_stub(config: PilotConfig, max_loss_per_trade: float = 2000.0):
    risk = SimpleNamespace(max_loss_per_trade=max_loss_per_trade)
    trader = SimpleNamespace(risk=risk)  # no get_funds_available -> exercises fail-open path
    return SimpleNamespace(config=config, trader=trader, _size_halved_remaining=0)


def compute(stub, sl_pts, ml_confidence, spot):
    return ClaudePilot._compute_futures_position_size(stub, sl_pts, ml_confidence, spot)


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True
    NIFTY_SPOT = 24600.0

    # Large equity so margin isn't the binding constraint for these cases.
    cfg = PilotConfig(
        account_equity=5_000_000.0,
        max_risk_pct=0.02,
        kelly_fraction=0.25,
        min_lots=1,
        max_lots=5,
        lot_size=65,
        futures_margin_pct_estimate=0.15,
    )

    # 1. Baseline: moderate SL, mid confidence, ample equity -> >= min_lots
    stub = make_stub(cfg, max_loss_per_trade=20_000.0)
    lots = compute(stub, sl_pts=40.0, ml_confidence=0.65, spot=NIFTY_SPOT)
    all_ok &= check(f"baseline sizing returns >=1 lot (got {lots})", lots >= 1)

    # 2. Never exceeds max_lots regardless of confidence
    stub = make_stub(cfg, max_loss_per_trade=10_000_000.0)
    lots = compute(stub, sl_pts=5.0, ml_confidence=1.0, spot=NIFTY_SPOT)
    all_ok &= check(f"lots clamped to max_lots=5 (got {lots})", lots <= cfg.max_lots)

    # 3. delta=1.0 fixed: loss_per_lot = sl_pts * lot_size (no delta discount,
    #    unlike the options sizer) -- verify a tight MAX_LOSS_PER_TRADE bites
    #    harder here than it would for an equivalent options trade.
    #    loss_per_lot = 100 * 65 = 6500/lot; max_loss=2000 -> 0 lots afford-able.
    stub = make_stub(cfg, max_loss_per_trade=2000.0)
    lots = compute(stub, sl_pts=100.0, ml_confidence=0.90, spot=NIFTY_SPOT)
    all_ok &= check(
        f"trade skipped (0) when 1 lot's point-loss exceeds MAX_LOSS_PER_TRADE (got {lots})",
        lots == 0,
    )

    # 4. NEW vs options: margin constraint blocks sizing on SMALL equity even
    #    when the point-based risk math alone would allow lots. This is the
    #    real difference futures sizing needed that options never did --
    #    1 lot notional = 24600*65 = ~1,599,000; at 15% estimate that's
    #    ~239,850 margin/lot -- an account with only 100,000 equity can't
    #    afford even 1 lot on margin grounds, regardless of risk-based sizing.
    small_cfg = PilotConfig(
        account_equity=100_000.0, max_risk_pct=0.02, kelly_fraction=0.25,
        min_lots=1, max_lots=5, lot_size=65, futures_margin_pct_estimate=0.15,
    )
    stub = make_stub(small_cfg, max_loss_per_trade=1_000_000.0)  # risk cap not the binding one here
    lots = compute(stub, sl_pts=30.0, ml_confidence=0.65, spot=NIFTY_SPOT)
    all_ok &= check(
        f"margin constraint blocks trade (0) on small equity even with loose risk cap (got {lots})",
        lots == 0,
    )

    # 5. Margin constraint REDUCES (not necessarily zeroes) lots for
    #    mid-sized equity that can afford some but not all Kelly-sized lots.
    mid_cfg = PilotConfig(
        account_equity=800_000.0, max_risk_pct=0.05, kelly_fraction=1.0,
        min_lots=1, max_lots=10, lot_size=65, futures_margin_pct_estimate=0.15,
    )
    stub = make_stub(mid_cfg, max_loss_per_trade=1_000_000.0)
    lots = compute(stub, sl_pts=30.0, ml_confidence=0.95, spot=NIFTY_SPOT)
    # margin/lot ~= 239,850 -> 800,000/239,850 ~= 3 lots max by margin
    all_ok &= check(f"margin caps lots to an affordable count (got {lots}, expect <=3)", 0 < lots <= 3)

    # 6. Zero/negative SL distance defaults to min_lots rather than crashing
    stub = make_stub(cfg, max_loss_per_trade=20_000.0)
    lots = compute(stub, sl_pts=0.0, ml_confidence=0.65, spot=NIFTY_SPOT)
    all_ok &= check(f"sl_pts<=0 defaults to min_lots (got {lots})", lots == cfg.min_lots)

    # 7. Missing get_funds_available on the stub's trader must NOT crash --
    #    the live-funds check is best-effort/logging-only (see make_stub:
    #    trader has no get_funds_available attribute at all).
    stub = make_stub(cfg, max_loss_per_trade=20_000.0)
    try:
        lots = compute(stub, sl_pts=40.0, ml_confidence=0.65, spot=NIFTY_SPOT)
        all_ok &= check(f"missing get_funds_available() doesn't crash sizing (got {lots})", lots >= 1)
    except Exception as e:
        all_ok &= check(f"missing get_funds_available() doesn't crash sizing (raised {e!r})", False)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
