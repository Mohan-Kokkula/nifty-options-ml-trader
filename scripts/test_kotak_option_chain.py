"""
test_kotak_option_chain.py — verify Kotak Neo option chain works end-to-end.

Run this on Hostinger (or anywhere with credentials) AFTER market hours.
It will:
  1. Connect to Kotak Neo using your config/settings.env
  2. Pull the raw scrip list (no parsing)
  3. Dump the first 3 scrips so we can SEE the field names
  4. Call get_option_chain() with the patched code
  5. Print the result

Usage:
    python scripts/test_kotak_option_chain.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load settings.env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / "settings.env")
except ImportError:
    pass


def main():
    print("=" * 70)
    print(" Kotak Neo Option Chain Diagnostic")
    print("=" * 70)

    # ── 1. Boot the client ──────────────────────────────────────────
    try:
        from security import RateLimiter
        from core.kotak_neo_client import KotakNeoClient
    except ImportError as e:
        print(f"\n[FAIL] Could not import core modules: {e}")
        return 1

    creds = {
        "consumer_key":  os.getenv("KOTAK_CONSUMER_KEY", ""),
        "neo_fin_key":   os.getenv("KOTAK_NEO_FIN_KEY", ""),
        "mobile":        os.getenv("KOTAK_MOBILE", ""),
        "password":      os.getenv("KOTAK_PASSWORD", ""),
        "totp_secret":   os.getenv("KOTAK_TOTP_SECRET", ""),
        "ucc":           os.getenv("KOTAK_UCC", ""),
        "mpin":          os.getenv("KOTAK_MPIN", ""),
    }
    missing = [k for k, v in creds.items() if not v and k != "neo_fin_key"]
    if missing:
        print(f"\n[FAIL] Missing creds in settings.env: {missing}")
        return 1
    print(f"\n[OK] Loaded credentials for UCC={creds['ucc']}")

    print("\n[1/4] Logging into Kotak Neo (TOTP)...")
    try:
        client = KotakNeoClient(
            consumer_key=creds["consumer_key"],
            rate_limiter=RateLimiter(),
            neo_fin_key=creds["neo_fin_key"],
            mobile=creds["mobile"],
            password=creds["password"],
            totp_secret=creds["totp_secret"],
            ucc=creds["ucc"],
            mpin=creds["mpin"],
        )
        print("[OK] Logged in")
    except Exception as e:
        print(f"[FAIL] Login failed: {e}")
        return 1

    # ── 2. Find nearest expiry ──────────────────────────────────────
    print("\n[2/4] Finding nearest expiry...")
    import os
    from datetime import date, timedelta

    # Try from settings.env first
    nifty_expiry_env = os.getenv("NIFTY_EXPIRY", "").strip().upper()
    print(f"      NIFTY_EXPIRY in env = '{nifty_expiry_env}'")

    expiry_neo = client._get_nearest_expiry()

    # Fallback: if _get_nearest_expiry() still returns empty, compute
    # the next Thursday manually as a safety net for the diagnostic.
    if not expiry_neo:
        print("      _get_nearest_expiry() returned empty — using computed Thursday fallback")
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
        if days_ahead == 0:
            days_ahead = 7
        next_thu = today + timedelta(days=days_ahead)
        expiry_neo = next_thu.strftime("%d%b%Y").upper()  # e.g. 29MAY2026
        print(f"      Computed fallback expiry: {expiry_neo}")

    print(f"[OK] Nearest expiry (Neo format): {expiry_neo}")

    # ── 3. RAW scrip dump (the truth from Kotak API) ────────────────
    print("\n[3/4] Calling search_scrip — RAW result (no filtering)...")
    raw = client._call_with_retry(
        client._neo.search_scrip,
        exchange_segment="nse_fo",
        symbol="nifty",
        expiry=expiry_neo,
    )

    if not isinstance(raw, list) or not raw:
        print(f"[FAIL] search_scrip returned: {type(raw).__name__} {raw}")
        return 1

    print(f"[OK] Got {len(raw)} raw scrips for {expiry_neo}")
    print("\n--- FIRST SCRIP (full keys) ---")
    print(json.dumps(raw[0], indent=2, default=str))
    print("\n--- LAST SCRIP (full keys) ---")
    print(json.dumps(raw[-1], indent=2, default=str))

    # Show all unique strike-price-like keys
    strike_keys = set()
    for r in raw[:20]:
        for k in r.keys():
            if "strike" in k.lower() or "strk" in k.lower():
                strike_keys.add(k)
    print(f"\nKeys that LOOK like strike fields: {strike_keys or '(none found!)'}")

    # Show all unique option-type-like keys
    opt_keys = set()
    for r in raw[:20]:
        for k in r.keys():
            if "option" in k.lower() or "opt" in k.lower() or "type" in k.lower():
                opt_keys.add(k)
    print(f"Keys that LOOK like option-type fields: {opt_keys or '(none found!)'}")

    # ── 4. Run the patched get_option_chain ─────────────────────────
    print("\n[4/4] Calling patched get_option_chain()...")
    try:
        chain = client.get_option_chain(
            underlying="NIFTY",
            exchange="NFO",
            expiry_date=None,        # let it use nearest
            strike_count=10,
        )
    except Exception as e:
        print(f"[FAIL] get_option_chain raised: {e}")
        return 1

    n_strikes = len(chain.get("chain", []))
    spot = chain.get("spot", 0)
    pcr = chain.get("pcr", 0)
    print(f"[{'OK' if n_strikes > 0 else 'FAIL'}] "
          f"strikes={n_strikes} spot={spot} PCR={pcr}")

    if n_strikes > 0:
        first = chain["chain"][0]
        print("\n--- First chain entry ---")
        print(json.dumps(first, indent=2, default=str))
        print("\n[SUCCESS] Kotak option chain is working!")
        return 0
    else:
        print("\n[FAIL] Chain is empty — paste the FIRST SCRIP dump above to Claude.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
