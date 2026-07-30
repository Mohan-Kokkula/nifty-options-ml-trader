"""
test_curl_pe.py — Standalone Kotak Neo PE order test.

Performs the EXACT REST flow from the official Kotak Neo API spec:
  1. totp_login  → view token
  2. totp_validate → edit token + sid + serverId + base_url
  3. search_scrip → resolve trading symbol for NIFTY PE
  4. POST /quick/order/rule/ms/place with jData URL-encoded form

USAGE:
  # Dry-run (prints what would be sent, no order placed):
  python scripts/test_curl_pe.py --strike 23900 --dry-run

  # Place real PE order:
  python scripts/test_curl_pe.py --strike 23900 --qty 75

  # Custom expiry (default = next weekly Tuesday):
  python scripts/test_curl_pe.py --strike 23900 --expiry 29APR26

  # Limit order with specific price:
  python scripts/test_curl_pe.py --strike 23900 --price 145.50 --type L

ENV (from config/settings.env):
  KOTAK_CONSUMER_KEY
  KOTAK_NEO_FIN_KEY            (default: neotradeapi)
  KOTAK_MOBILE
  KOTAK_UCC
  KOTAK_MPIN
  KOTAK_TOTP_SECRET            (base32 from authenticator app)
  KOTAK_ENVIRONMENT            (prod / uat — default: prod)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Load env from settings.env if present
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ENV_FILE = ROOT / "config" / "settings.env"
if ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except ImportError:
        # Minimal manual load
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


import requests


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _gen_totp(secret: str) -> str:
    import pyotp
    return pyotp.TOTP(secret).now()


def _next_tuesday(today: datetime | None = None) -> str:
    """Next NIFTY weekly expiry (Tuesday) in DDMMMYY format like '29APR26'."""
    today = today or datetime.now()
    days_ahead = (1 - today.weekday()) % 7   # Tue=1
    if days_ahead == 0 and today.hour >= 15 and today.minute >= 30:
        days_ahead = 7   # already past today's expiry close → next week
    target = today + timedelta(days=days_ahead)
    return target.strftime("%d%b%y").upper()


def _print_section(title: str, char: str = "─"):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(char * 70)


# ──────────────────────────────────────────────────────────────────────
# Step 1 + 2: Login (TOTP + MPIN) using SDK
# ──────────────────────────────────────────────────────────────────────

def login_via_sdk():
    from neo_api_client import NeoAPI

    consumer_key = os.environ["KOTAK_CONSUMER_KEY"]
    environment  = os.getenv("KOTAK_ENVIRONMENT", "prod")
    # neo_fin_key MUST be "neotradeapi" per Kotak v2 spec — env may be empty
    neo_fin_key  = os.getenv("KOTAK_NEO_FIN_KEY", "").strip() or "neotradeapi"
    mobile       = os.environ["KOTAK_MOBILE"]
    ucc          = os.environ["KOTAK_UCC"]
    mpin         = os.environ["KOTAK_MPIN"]
    totp_secret  = os.environ["KOTAK_TOTP_SECRET"]

    _print_section("STEP 1: Initialize SDK + TOTP login")
    print(f"  consumer_key:  {consumer_key[:12]}...")
    print(f"  environment:   {environment}")
    print(f"  neo_fin_key:   {neo_fin_key}")
    print(f"  mobile:        {mobile}")
    print(f"  ucc:           {ucc}")

    neo = NeoAPI(
        consumer_key=consumer_key,
        environment=environment,
        neo_fin_key=neo_fin_key,
    )

    # Step A — totp_login (mobile + ucc + 6-digit TOTP → view token)
    totp = _gen_totp(totp_secret)
    print(f"\n  Generating TOTP: {totp}")
    print(f"  Calling totp_login(mobile={mobile}, ucc={ucc}, totp=***)...")
    r1 = neo.totp_login(mobile_number=mobile, ucc=ucc, totp=totp)
    if "data" not in r1:
        print(f"\n  ❌ totp_login failed: {json.dumps(r1, indent=2)[:500]}")
        sys.exit(1)
    print(f"  ✅ totp_login OK (view token len={len(neo.configuration.view_token or '')})")

    # Step B — totp_validate (mpin → edit token)
    _print_section("STEP 2: TOTP validate (MPIN)")
    print(f"  Calling totp_validate(mpin=***)...")
    r2 = neo.totp_validate(mpin=mpin)
    if "data" not in r2:
        print(f"\n  ❌ totp_validate failed: {json.dumps(r2, indent=2)[:500]}")
        sys.exit(1)

    print(f"  ✅ totp_validate OK")
    print(f"     edit_token: {(neo.configuration.edit_token or '')[:30]}...")
    print(f"     edit_sid:   {neo.configuration.edit_sid}")
    print(f"     serverId:   {neo.configuration.serverId!r}")
    print(f"     base_url:   {neo.configuration.base_url}")

    # If serverId is empty, probe by trying each candidate.
    # SUCCESS check: limits() returns a dict with 'Category' or 'EntityId' field
    # (NOT wrapped in {'data': ...}) — confirmed from real responses.
    if not neo.configuration.serverId:
        print(f"\n  ⚠️  serverId is empty — probing server_1..5 ...")
        candidates = [f"server_{i}" for i in range(1, 6)]
        worked = None
        for sid in candidates:
            neo.configuration.serverId = sid
            try:
                lim = neo.limits()
                # Real success indicator: the dict contains account fields
                if isinstance(lim, dict) and (
                    "Category" in lim or "EntityId" in lim
                    or "BoardLotLimit" in lim or "data" in lim
                ):
                    print(f"     ✅ {sid} works (limits → {list(lim.keys())[:5]})")
                    worked = sid
                    break
                else:
                    print(f"     ❌ {sid}: unexpected response {str(lim)[:80]}")
            except Exception as e:
                print(f"     ❌ {sid}: exception — {str(e)[:80]}")
        if worked is None:
            print(f"     ⚠️  Defaulting serverId to 'server_1'")
            neo.configuration.serverId = "server_1"

    # Force neo_fin_key on configuration (env may have been empty)
    try:
        neo.configuration.neo_fin_key = neo_fin_key
    except Exception:
        pass

    return neo


# ──────────────────────────────────────────────────────────────────────
# Step 3: Resolve trading symbol via search_scrip
# ──────────────────────────────────────────────────────────────────────

def resolve_trading_symbol(neo, expiry: str, strike: int, option_type: str) -> str:
    _print_section(f"STEP 3: Resolve trading symbol (NIFTY {expiry} {strike} {option_type})")
    print(f"  Calling search_scrip(exchange_segment='nse_fo', symbol='nifty', expiry={expiry!r})...")

    results = neo.search_scrip(
        exchange_segment="nse_fo",
        symbol="nifty",
        expiry=expiry,
    )

    if not isinstance(results, list) or not results:
        print(f"  ❌ search_scrip returned: {results}")
        sys.exit(1)

    print(f"  Got {len(results)} contracts. Filtering...")

    # ── DIAGNOSTIC: dump first 3 results to see actual structure ─────
    print(f"\n  🔍 Sample raw contracts (first 3):")
    for i, r in enumerate(results[:3]):
        print(f"    [{i}] keys={list(r.keys())[:15]}")
        for k in ("pSymbolName", "pTrdSymbol", "pSymbol", "pOptionType",
                  "dStrikePrice", "pExpiryDate", "pInstType", "pSegment"):
            if k in r:
                print(f"        {k} = {r[k]!r}")

    # ── DIAGNOSTIC: unique pSymbolName values ────────────────────────
    unique_syms = sorted({str(r.get("pSymbolName", "")) for r in results})
    print(f"\n  🔍 Unique pSymbolName values ({len(unique_syms)}):")
    for s in unique_syms[:20]:
        print(f"    {s!r}")
    if len(unique_syms) > 20:
        print(f"    ... +{len(unique_syms) - 20} more")

    # ── DIAGNOSTIC: unique strike scales (in case paise vs rupees) ───
    raw_strikes = sorted({
        float(r.get("dStrikePrice", 0))
        for r in results
        if r.get("dStrikePrice")
    })
    print(f"\n  🔍 Strike value samples (showing first 20 + last 5):")
    for s in raw_strikes[:20]:
        print(f"    {s}")
    if len(raw_strikes) > 25:
        print(f"    ...")
        for s in raw_strikes[-5:]:
            print(f"    {s}")

    # ── DIAGNOSTIC: unique pInstType values ──────────────────────────
    unique_inst = sorted({str(r.get("pInstType", "")) for r in results})
    print(f"\n  🔍 Unique pInstType values: {unique_inst}")

    # Filter — try both rupee and paise scaling for strike
    print(f"\n  Filtering for NIFTY {strike}{option_type}...")
    matches = []
    matches_paise = []
    for r in results:
        sym = str(r.get("pSymbolName", "")).upper()
        if "FINNIFTY" in sym or "NIFTYBEES" in sym or "MIDCPNIFTY" in sym:
            continue
        if "NIFTY" not in sym:
            continue
        try:
            r_strike = float(r.get("dStrikePrice", 0))
        except Exception:
            continue
        r_opt = str(r.get("pOptionType", "")).upper()
        if r_opt != option_type.upper():
            continue
        if r_strike == strike:
            matches.append(r)
        elif r_strike == strike * 100:   # paise scale
            matches_paise.append(r)

    if matches_paise and not matches:
        print(f"  ⚠️  Strikes are in PAISE scale — using those matches")
        matches = matches_paise

    if not matches:
        # Final attempt: tolerant match on substring
        print(f"\n  Trying tolerant substring match on pTrdSymbol containing '{strike}' and '{option_type}'...")
        for r in results:
            trd = str(r.get("pTrdSymbol", ""))
            if (str(strike) in trd and option_type.upper() in trd.upper() and
                "NIFTY" in trd.upper() and "FINNIFTY" not in trd.upper()):
                matches.append(r)
                print(f"    Match: {trd}")
                if len(matches) >= 3:
                    break

    if not matches:
        print(f"\n  ❌ No NIFTY {strike}{option_type} found for expiry {expiry}.")
        # Show ALL NIFTY contracts in result (regardless of strike)
        nifty_only = [
            r for r in results[:500]
            if "NIFTY" in str(r.get("pSymbolName", "")).upper()
            and "FIN" not in str(r.get("pSymbolName", "")).upper()
            and "BEES" not in str(r.get("pSymbolName", "")).upper()
        ]
        print(f"\n  NIFTY-only contracts: {len(nifty_only)} of {len(results)} total")
        if nifty_only:
            print(f"  Sample NIFTY trading symbols (first 10):")
            for r in nifty_only[:10]:
                print(f"    pTrdSymbol={r.get('pTrdSymbol')!r}  "
                      f"strike={r.get('dStrikePrice')}  "
                      f"opt={r.get('pOptionType')}  "
                      f"expiry={r.get('pExpiryDate')}")
        sys.exit(1)

    chosen = matches[0]
    trading_symbol = chosen.get("pTrdSymbol") or chosen.get("trading_symbol")
    token = chosen.get("pSymbol")
    print(f"\n  ✅ Resolved: {trading_symbol}  (token={token})")
    return trading_symbol


# ──────────────────────────────────────────────────────────────────────
# Step 4: Place order via raw REST POST (per official docs)
# ──────────────────────────────────────────────────────────────────────

def place_order_raw(neo, trading_symbol: str, qty: int, price: float,
                    order_type: str, transaction: str, product: str,
                    retention: str, dry_run: bool) -> dict:
    """
    POST <base_url>/quick/order/rule/ms/place
    Body: jData = URL-encoded JSON (see Kotak API spec).
    """
    _print_section("STEP 4: Place order (raw REST)")

    # Build jData per official spec
    jdata_obj = {
        "am": "NO",                           # AMO order? NO
        "dq": "0",                            # disclosed quantity
        "es": "nse_fo",                       # exchange segment (NSE F&O)
        "mp": "0",                            # market protection
        "pc": product,                        # product (MIS / NRML / CNC)
        "pf": "N",                            # portfolio flag
        "pr": f"{price:.2f}" if price > 0 else "0",   # price
        "pt": order_type,                     # MKT / L / SL / SL-M
        "qt": str(qty),                       # quantity
        "rt": retention,                      # DAY / IOC
        "tp": "0",                            # trigger price (for SL/SL-M)
        "ts": trading_symbol,                 # trading symbol
        "tt": transaction,                    # B (buy) / S (sell)
    }
    jdata_str = json.dumps(jdata_obj, separators=(",", ":"))

    # URL — base_url comes from tradeApiValidate response
    base_url = neo.configuration.base_url
    if not base_url:
        # Fallback to public gateway
        base_url = "https://gw-napi.kotaksecurities.com"
        print(f"  ⚠️  base_url empty — using fallback: {base_url}")
    path = "/quick/order/rule/ms/place"
    server_id = neo.configuration.serverId
    url = f"{base_url}{path}?sId={server_id}"

    # neo-fin-key MUST be the literal string "neotradeapi" per Kotak v2 spec.
    # Pull from configuration (set during login), fall back to literal.
    nfk = (getattr(neo.configuration, "neo_fin_key", "")
           or os.getenv("KOTAK_NEO_FIN_KEY", "").strip()
           or "neotradeapi")
    headers = {
        "accept":       "application/json",
        "Sid":          neo.configuration.edit_sid,
        "Auth":         neo.configuration.edit_token,
        "neo-fin-key":  nfk,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    form_body = {"jData": jdata_str}

    # Print everything for diagnostics
    print(f"  URL:    {url}")
    print(f"  HEADERS:")
    for k, v in headers.items():
        if k.lower() in ("auth", "sid"):
            v = (v[:20] + "...") if v else "(empty)"
        print(f"    {k}: {v}")
    print(f"  jData (decoded):")
    print(f"    {json.dumps(jdata_obj, indent=4).replace(chr(10), chr(10) + '    ')}")

    if dry_run:
        print(f"\n  🟡 DRY-RUN: not sending. Use --no-dry-run / remove --dry-run to fire.")
        return {"dry_run": True, "url": url, "jdata": jdata_obj}

    print(f"\n  📤 POSTing to Kotak...")
    t0 = time.time()
    r = requests.post(url, headers=headers, data=form_body, timeout=15)
    dt_ms = (time.time() - t0) * 1000

    print(f"\n  ⏱  Response in {dt_ms:.0f}ms")
    print(f"  HTTP {r.status_code}")
    print(f"  Body:")
    try:
        body = r.json()
        print(f"    {json.dumps(body, indent=4).replace(chr(10), chr(10) + '    ')}")
    except Exception:
        body = {"raw": r.text}
        print(f"    {r.text[:1000]}")

    return {
        "http": r.status_code,
        "elapsed_ms": dt_ms,
        "body": body,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Kotak Neo PE order test (raw REST per official spec)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strike", type=int, required=True, help="Option strike price (e.g. 23900)")
    parser.add_argument("--option", choices=["PE", "CE"], default="PE", help="Option type")
    parser.add_argument("--expiry", default=None,
                        help="Expiry DDMMMYY (e.g. 29APR26). Default: next Tuesday.")
    parser.add_argument("--qty", type=int, default=75, help="Lot quantity (NIFTY lot=75)")
    parser.add_argument("--price", type=float, default=0.0,
                        help="Limit price (0 = market). Use only with --type L")
    parser.add_argument("--type", choices=["MKT", "L", "SL", "SL-M"], default="MKT",
                        help="Order type")
    parser.add_argument("--side", choices=["B", "S"], default="B",
                        help="Buy (B) or Sell (S)")
    parser.add_argument("--product", choices=["MIS", "NRML", "CNC"], default="MIS",
                        help="Product code (MIS=intraday)")
    parser.add_argument("--retention", choices=["DAY", "IOC"], default="DAY",
                        help="Retention (DAY/IOC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print request without sending")
    args = parser.parse_args()

    expiry = args.expiry or _next_tuesday()

    # Sanity check env
    required = ["KOTAK_CONSUMER_KEY", "KOTAK_MOBILE", "KOTAK_UCC", "KOTAK_MPIN", "KOTAK_TOTP_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"❌ Missing env vars: {missing}")
        print(f"   Set them in {ENV_FILE} or export to shell.")
        sys.exit(1)

    print(f"╔════════════════════════════════════════════════════════════╗")
    print(f"║      Kotak Neo PE Order Test — Standalone                  ║")
    print(f"║      Strike:  {args.strike} {args.option}                          ║")
    print(f"║      Expiry:  {expiry}                                       ║")
    print(f"║      Qty:     {args.qty}                                          ║")
    print(f"║      Price:   {args.price if args.price else 'MARKET'}                                     ║")
    print(f"║      Side:    {'BUY' if args.side == 'B' else 'SELL'}                                       ║")
    print(f"║      Product: {args.product}                                          ║")
    print(f"║      Mode:    {'DRY-RUN' if args.dry_run else 'LIVE'}                                  ║")
    print(f"╚════════════════════════════════════════════════════════════╝")

    # 1+2: Login
    neo = login_via_sdk()

    # 3: Resolve trading symbol
    trading_symbol = resolve_trading_symbol(
        neo, expiry=expiry, strike=args.strike, option_type=args.option
    )

    # 4: Place order
    result = place_order_raw(
        neo,
        trading_symbol=trading_symbol,
        qty=args.qty,
        price=args.price,
        order_type=args.type,
        transaction=args.side,
        product=args.product,
        retention=args.retention,
        dry_run=args.dry_run,
    )

    _print_section("FINAL RESULT", char="═")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
