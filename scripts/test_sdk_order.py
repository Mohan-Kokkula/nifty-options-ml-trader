"""
test_sdk_order.py — Test order placement using the SDK's native place_order
(NOT raw curl). This isolates whether 100008 is from our headers or from
Kotak portal-side auth.

If SDK place_order works → our raw curl headers are wrong
If SDK ALSO returns 100008 → IP/permission issue on Kotak portal
"""
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "config" / "settings.env")

from neo_api_client import NeoAPI
import pyotp


def main():
    print("=" * 70)
    print("Step 1: Initialize SDK")
    print("=" * 70)

    neo = NeoAPI(
        consumer_key=os.environ["KOTAK_CONSUMER_KEY"],
        environment="prod",
        neo_fin_key="neotradeapi",
    )
    print(f"  consumer_key: {os.environ['KOTAK_CONSUMER_KEY'][:12]}...")
    print(f"  base_url:     {neo.configuration.base_url}")

    print("\n" + "=" * 70)
    print("Step 2: TOTP login")
    print("=" * 70)
    totp = pyotp.TOTP(os.environ["KOTAK_TOTP_SECRET"]).now()
    r1 = neo.totp_login(
        mobile_number=os.environ["KOTAK_MOBILE"],
        ucc=os.environ["KOTAK_UCC"],
        totp=totp,
    )
    if "data" not in r1:
        print(f"  ❌ totp_login failed: {json.dumps(r1, indent=2)}")
        sys.exit(1)
    print(f"  ✅ totp_login OK")

    print("\n" + "=" * 70)
    print("Step 3: TOTP validate (MPIN)")
    print("=" * 70)
    r2 = neo.totp_validate(mpin=os.environ["KOTAK_MPIN"])
    if "data" not in r2:
        print(f"  ❌ totp_validate failed: {json.dumps(r2, indent=2)}")
        sys.exit(1)
    print(f"  ✅ totp_validate OK")
    print(f"     edit_sid:   {neo.configuration.edit_sid}")
    print(f"     edit_token: {(neo.configuration.edit_token or '')[:30]}...")
    print(f"     serverId:   {neo.configuration.serverId!r}")
    print(f"     base_url:   {neo.configuration.base_url}")

    # Force serverId if empty (we know server_1 works for limits)
    if not neo.configuration.serverId:
        neo.configuration.serverId = "server_1"
        print(f"     ⚠️ serverId was empty — forced to server_1")

    print("\n" + "=" * 70)
    print("Step 4: Test that read APIs work (limits + positions)")
    print("=" * 70)
    print(f"  Calling limits()...")
    lim = neo.limits()
    print(f"  Result: {str(lim)[:200]}")
    if "Category" in str(lim) or "EntityId" in str(lim) or "data" in str(lim).lower():
        print(f"  ✅ limits() works — read APIs are authorized")
    else:
        print(f"  ❌ limits() failed — fundamental auth issue")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Step 5: Resolve trading symbol")
    print("=" * 70)

    # Try multiple expiry candidates — Tuesday is current weekly NIFTY expiry
    from datetime import datetime as _dt, timedelta as _td
    today = _dt.now().date()
    candidates = []
    # Next 4 Tuesdays (weeklies)
    days_to_tue = (1 - today.weekday()) % 7
    if days_to_tue == 0:
        days_to_tue = 7    # already Tue → next Tue
    for i in range(4):
        d = today + _td(days=days_to_tue + 7*i)
        candidates.append(d.strftime("%d%b%y").upper())
    # Also no-expiry (returns all contracts)
    candidates.append("")

    print(f"  Trying expiry candidates: {candidates}")
    results = None
    used_expiry = None
    for exp in candidates:
        try:
            r = neo.search_scrip(
                exchange_segment="nse_fo",
                symbol="nifty",
                expiry=exp,
            )
        except Exception as e:
            print(f"    {exp!r}: exception {e}")
            continue
        if isinstance(r, list) and len(r) > 0:
            print(f"    {exp!r} → {len(r)} contracts ✓")
            results = r
            used_expiry = exp
            break
        else:
            print(f"    {exp!r} → {r}")

    if not results:
        print(f"  ❌ No expiry returned data — Kotak might be down or auth broken")
        sys.exit(1)

    # Find any NIFTY CE near 24050
    target_strike = 24050
    matches = []
    for r in results:
        sym = str(r.get("pSymbolName", "")).upper()
        if "FINNIFTY" in sym or "NIFTYBEES" in sym or "MIDCPNIFTY" in sym:
            continue
        if "NIFTY" not in sym:
            continue
        trd = str(r.get("pTrdSymbol", ""))
        if str(target_strike) in trd and "CE" in trd.upper():
            matches.append(r)

    if not matches:
        # Show available NIFTY CE strikes near 24050
        print(f"  ❌ No NIFTY {target_strike}CE found in expiry {used_expiry}")
        print(f"  Available NIFTY CE contracts (sample):")
        nifty_ce = [
            r for r in results
            if "NIFTY" in str(r.get("pSymbolName","")).upper()
            and "FIN" not in str(r.get("pSymbolName","")).upper()
            and "BEES" not in str(r.get("pSymbolName","")).upper()
            and r.get("pOptionType") == "CE"
        ]
        # Sort by distance to target strike
        nifty_ce.sort(key=lambda r: abs(int(float(r.get("dStrikePrice", 0) or 99999)) - target_strike))
        for r in nifty_ce[:10]:
            print(f"    {r.get('pTrdSymbol')}  expiry={r.get('pExpiryDate')}  strike={r.get('dStrikePrice')}")
        if nifty_ce:
            # Use nearest available
            chosen = nifty_ce[0]
            print(f"  ⚡ Using nearest available: {chosen.get('pTrdSymbol')}")
            trading_symbol = chosen.get("pTrdSymbol")
        else:
            sys.exit(1)
    else:
        chosen = matches[0]
        trading_symbol = chosen.get("pTrdSymbol")
        print(f"  ✅ Resolved: {trading_symbol}")

    print("\n" + "=" * 70)
    print("Step 6: Place order via SDK's native place_order()")
    print("=" * 70)
    print(f"  Note: this uses neo.place_order(...) — SDK's exact code path")
    print(f"  No custom headers, no raw REST. Just the SDK.")
    print(f"")
    print(f"  Calling neo.place_order(")
    print(f"    exchange_segment='nse_fo',")
    print(f"    product='MIS',")
    print(f"    price='150',")
    print(f"    order_type='L',")
    print(f"    quantity='75',")
    print(f"    validity='DAY',")
    print(f"    trading_symbol='{trading_symbol}',")
    print(f"    transaction_type='B',")
    print(f"  )")

    t0 = time.time()
    try:
        resp = neo.place_order(
            exchange_segment="nse_fo",
            product="MIS",
            price="150",                 # well below market — won't fill
            order_type="L",              # LIMIT
            quantity="75",
            validity="DAY",
            trading_symbol=trading_symbol,
            transaction_type="B",
        )
    except Exception as e:
        print(f"\n  ❌ SDK raised: {e}")
        sys.exit(1)
    dt = (time.time() - t0) * 1000
    print(f"\n  ⏱  Response in {dt:.0f}ms")
    print(f"  Body:")
    print(json.dumps(resp, indent=4, default=str))

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if isinstance(resp, dict):
        if resp.get("stat") == "Ok" or resp.get("nOrdNo"):
            print("  ✅✅✅ ORDER PLACED SUCCESSFULLY via SDK")
            print(f"     Order ID: {resp.get('nOrdNo')}")
            print(f"     This means: SDK works, but our raw curl was missing something.")
            print(f"     We can simplify test_curl_pe.py to use neo.place_order() instead.")
        elif resp.get("stCode") == 100008:
            print("  ❌ STILL 100008 unauthorized — even via SDK")
            print(f"     This means: NOT a code issue. Portal-side problem.")
            print(f"     Action items:")
            print(f"       1. Confirm IP registered MATCHES this VPS's public IP:")
            print(f"          curl -s https://api.ipify.org")
            print(f"       2. Logout fully + relogin via this script (fresh session post-IP-reg)")
            print(f"       3. Confirm F&O trading scope is enabled on the registered API app")
            print(f"       4. If issue persists → contact Kotak Neo support with this log")
        elif resp.get("stCode") == 100009:
            print("  ⚠️ stCode 100009 — usually means margin/funds insufficient")
            print(f"     Check: trader.client.limits() — do you have funds for ₹150 × 75?")
        else:
            print(f"  ❌ Unexpected error: {resp}")
            print(f"     stCode: {resp.get('stCode')}")
            print(f"     errMsg: {resp.get('errMsg')}")


if __name__ == "__main__":
    main()
