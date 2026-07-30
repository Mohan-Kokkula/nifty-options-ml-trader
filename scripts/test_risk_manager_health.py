#!/usr/bin/env python3
"""
Unit tests for RiskManager's automatic strategy-health pause/resume
(security/__init__.py) and its persistence across restarts.

Standalone script (repo convention), run with:
    python scripts/test_risk_manager_health.py

Uses a temp state file (never data/risk_state.json) and monkeypatches
time.time() where needed to test cooldown expiry without sleeping.
"""

import sys
import time as _time_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from security import RiskManager


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True
    tmp_state = Path(__file__).parent.parent / "data" / "_test_risk_health_state.json"
    if tmp_state.exists():
        tmp_state.unlink()

    # ── 1. Consecutive-loss pause ────────────────────────────────────
    rm = RiskManager(
        max_daily_loss=5000.0,
        max_loss_per_trade=2000.0,
        max_open_positions=3,
        state_file=str(tmp_state),
        drawdown_pause_pct=0.70,
        consecutive_loss_pause_threshold=3,
        pause_cooldown_min=30,
    )
    rm.record_trade_open(); rm.record_trade_close(pnl=-500)
    rm.record_trade_open(); rm.record_trade_close(pnl=-500)
    allowed, _ = rm.can_open_trade()
    all_ok &= check(f"2 consecutive losses (threshold=3): still allowed (got {allowed})", allowed is True)

    rm.record_trade_open(); rm.record_trade_close(pnl=-500)  # 3rd consecutive loss -> pause
    allowed, reason = rm.can_open_trade()
    all_ok &= check(f"3rd consecutive loss triggers pause (allowed={allowed}, reason={reason!r})", allowed is False)
    all_ok &= check("pause_reason mentions consecutive losses", "consecutive losses" in rm.pause_reason)

    # ── 2. Cooldown auto-resume (simulate time passing) ──────────────
    rm.paused_until = _time_module.time() - 1  # pretend cooldown already elapsed
    allowed, _ = rm.can_open_trade()
    all_ok &= check(f"after cooldown elapses, trading resumes automatically (got {allowed})", allowed is True)

    tmp_state.unlink()

    # ── 3. Drawdown-pct pause (separate instance, fresh state) ───────
    rm2 = RiskManager(
        max_daily_loss=5000.0,
        max_loss_per_trade=2000.0,
        max_open_positions=3,
        state_file=str(tmp_state),
        drawdown_pause_pct=0.70,   # pause at 70% of 5000 = -3500
        consecutive_loss_pause_threshold=99,  # disable streak trigger for this test
        pause_cooldown_min=60,
    )
    rm2.record_trade_open(); rm2.record_trade_close(pnl=-2000)
    allowed, _ = rm2.can_open_trade()
    all_ok &= check(f"daily_pnl=-2000 (<70% of 5000=3500): still allowed (got {allowed})", allowed is True)

    rm2.record_trade_open(); rm2.record_trade_close(pnl=-2000)  # daily_pnl now -4000 >= -3500 threshold
    allowed, reason = rm2.can_open_trade()
    all_ok &= check(f"daily_pnl=-4000 (>=70% of 5000): pause triggers (allowed={allowed})", allowed is False)
    all_ok &= check("pause_reason mentions drawdown", "drawdown" in rm2.pause_reason)
    tmp_state.unlink()

    # ── 4. Persistence round-trip (pause state survives a restart) ───
    rm3 = RiskManager(
        max_daily_loss=5000.0, max_loss_per_trade=2000.0, max_open_positions=3,
        state_file=str(tmp_state), consecutive_loss_pause_threshold=1, pause_cooldown_min=60,
    )
    rm3.record_trade_open(); rm3.record_trade_close(pnl=-100)  # 1 loss triggers pause (threshold=1)
    all_ok &= check("rm3 is paused after 1 loss (threshold=1)", rm3.paused_until > 0)

    rm4 = RiskManager(
        max_daily_loss=5000.0, max_loss_per_trade=2000.0, max_open_positions=3,
        state_file=str(tmp_state), consecutive_loss_pause_threshold=1, pause_cooldown_min=60,
    )
    all_ok &= check(
        f"new instance restores pause state from disk (paused_until={rm4.paused_until})",
        rm4.paused_until == rm3.paused_until and rm4.pause_reason == rm3.pause_reason,
    )
    allowed, _ = rm4.can_open_trade()
    all_ok &= check(f"restored instance also blocks new trades (got {allowed})", allowed is False)
    tmp_state.unlink()

    # ── 5. Winning trade resets the consecutive-loss streak ──────────
    rm5 = RiskManager(
        max_daily_loss=5000.0, max_loss_per_trade=2000.0, max_open_positions=3,
        state_file=str(tmp_state), consecutive_loss_pause_threshold=3,
    )
    rm5.record_trade_open(); rm5.record_trade_close(pnl=-500)
    rm5.record_trade_open(); rm5.record_trade_close(pnl=-500)
    rm5.record_trade_open(); rm5.record_trade_close(pnl=+1000)  # win resets streak
    all_ok &= check(f"win resets consecutive_losses to 0 (got {rm5.consecutive_losses})", rm5.consecutive_losses == 0)
    tmp_state.unlink()

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
