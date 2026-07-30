"""
check_lot_size.py — Verify NIFTY current lot size and update settings.env.

NIFTY lot size has changed multiple times (50 → 25 → 75 since 2024).
Trading partial lots can cause:
  - Order rejection
  - Full-lot brokerage on partial qty
  - P&L calculation discrepancies

This script:
  1. Logs into Kotak Neo
  2. Fetches NIFTY option chain
  3. Reads `iLotSize` field from any contract
  4. Compares to DEFAULT_QTY in settings.env
  5. Prompts to update if mismatch

USAGE:
  python scripts/check_lot_size.py            # check + prompt
  python scripts/check_lot_size.py --auto-fix # update without asking
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "config" / "settings.env")


def _next_tuesday() -> str:
    today = datetime.now().date()
    days = (1 - today.weekday()) % 7
    if days == 0:
        days = 7
    return (today + timedelta(days=days)).strftime("%d%b%y").upper()


def fetch_nifty_lot_size() -> int:
    from neo_api_client import NeoAPI
    import pyotp

    print("→ Logging into Kotak Neo...")
    neo = NeoAPI(
        consumer_key=os.environ["KOTAK_CONSUMER_KEY"],
        environment="prod",
        neo_fin_key="neotradeapi",
    )
    neo.totp_login(
        mobile_number=os.environ["KOTAK_MOBILE"],
        ucc=os.environ["KOTAK_UCC"],
        totp=pyotp.TOTP(os.environ["KOTAK_TOTP_SECRET"]).now(),
    )
    neo.totp_validate(mpin=os.environ["KOTAK_MPIN"])
    if not neo.configuration.serverId:
        neo.configuration.serverId = "server_1"

    # Try multiple expiries to find one with data
    candidates = []
    today = datetime.now().date()
    days_to_tue = (1 - today.weekday()) % 7
    if days_to_tue == 0:
        days_to_tue = 7
    for i in range(4):
        d = today + timedelta(days=days_to_tue + 7 * i)
        candidates.append(d.strftime("%d%b%y").upper())

    print(f"→ Fetching NIFTY option chain...")
    results = None
    for exp in candidates:
        r = neo.search_scrip(
            exchange_segment="nse_fo", symbol="nifty", expiry=exp,
        )
        if isinstance(r, list) and len(r) > 0:
            results = r
            print(f"  Found {len(r)} contracts (expiry={exp})")
            break

    if not results:
        raise RuntimeError("No data from search_scrip — Kotak issue")

    # Find any NIFTY OPTIDX contract (not FINNIFTY/MIDCPNIFTY)
    for c in results:
        sym = str(c.get("pSymbolName", "")).upper()
        inst = str(c.get("pInstType", ""))
        if (inst == "OPTIDX" and sym == "NIFTY"
                and "FINNIFTY" not in sym and "MIDCPNIFTY" not in sym):
            lot = c.get("iLotSize") or c.get("lotSize") or c.get("lot_size")
            if lot:
                return int(lot)

    raise RuntimeError("Could not find iLotSize in any contract")


def read_current_default_qty() -> tuple[int | None, Path]:
    env_path = ROOT / "config" / "settings.env"
    if not env_path.exists():
        return None, env_path
    text = env_path.read_text()
    m = re.search(r"^\s*DEFAULT_QTY\s*=\s*(\d+)", text, re.MULTILINE)
    return (int(m.group(1)) if m else None), env_path


def update_default_qty(env_path: Path, new_qty: int) -> bool:
    text = env_path.read_text()
    if re.search(r"^\s*DEFAULT_QTY\s*=", text, re.MULTILINE):
        new_text = re.sub(
            r"^\s*DEFAULT_QTY\s*=\s*\d+",
            f"DEFAULT_QTY={new_qty}",
            text, count=1, flags=re.MULTILINE,
        )
    else:
        new_text = text.rstrip() + f"\n# Added by check_lot_size.py\nDEFAULT_QTY={new_qty}\n"
    env_path.write_text(new_text)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-fix", action="store_true",
                        help="Update settings.env without confirmation")
    args = parser.parse_args()

    print("═" * 60)
    print(" NIFTY Lot Size Verification")
    print("═" * 60)

    try:
        actual_lot = fetch_nifty_lot_size()
    except Exception as e:
        print(f"❌ Failed to fetch lot size: {e}")
        sys.exit(1)

    current_qty, env_path = read_current_default_qty()
    print(f"\n→ Kotak says NIFTY lot size: {actual_lot}")
    print(f"→ Your config DEFAULT_QTY:   {current_qty}")

    if current_qty is None:
        print(f"\n⚠️  DEFAULT_QTY not set in {env_path}")
        if args.auto_fix or input(f"Add DEFAULT_QTY={actual_lot}? [y/N] ").lower() == "y":
            update_default_qty(env_path, actual_lot)
            print(f"✅ Added DEFAULT_QTY={actual_lot} to {env_path}")
        return

    if current_qty == actual_lot:
        print(f"\n✅ Match — no action needed")
        return

    print(f"\n⚠️  MISMATCH: trading {current_qty} but lot is {actual_lot}")
    print(f"  Risks of trading {current_qty} when lot={actual_lot}:")
    if current_qty < actual_lot:
        print(f"    - Order may be rejected as 'invalid quantity'")
        print(f"    - Even if accepted, charged full-lot brokerage on {current_qty}")
    else:
        print(f"    - Effective position is {current_qty // actual_lot} lots + "
              f"{current_qty % actual_lot} extra (also rejected)")

    if args.auto_fix or input(f"\nUpdate DEFAULT_QTY={current_qty} → {actual_lot}? [y/N] ").lower() == "y":
        update_default_qty(env_path, actual_lot)
        print(f"✅ Updated {env_path}")
        print(f"   Restart bot for change to take effect:")
        print(f"   sudo systemctl restart openclaw-v9-kotak")


if __name__ == "__main__":
    main()
