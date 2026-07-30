#!/usr/bin/env python3
"""
Example usage — demonstrates smart auto-strike selection.
Run the server first: python main.py
Then run this: python scripts/example_trades.py
"""

import json
import requests

BASE = "http://localhost:8080"


def call(method, endpoint, payload=None):
    """Make an API request and print the result."""
    url = f"{BASE}{endpoint}"
    print(f"\n{'='*65}")
    print(f"{method} {url}")
    if payload:
        print(f"Body: {json.dumps(payload, indent=2)}")
    print("-" * 65)

    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(data, indent=2, default=str)}")
        return data
    except requests.ConnectionError:
        print("ERROR: Cannot connect. Is the server running? (python main.py)")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    print("=" * 65)
    print("  OpenClaw Nifty Options Trader — Smart Strike Examples")
    print("=" * 65)

    # 1. Health check
    print("\n📡 [1/10] Health check")
    call("GET", "/health")

    # 2. Check funds
    print("\n💰 [2/10] Available funds")
    call("POST", "/funds", {})

    # -----------------------------------------------------------------
    # STRIKE SELECTION (preview — no orders placed)
    # -----------------------------------------------------------------

    # 3. ATM strike recommendation
    print("\n🎯 [3/10] Get ATM strike for CE")
    call("POST", "/strikes", {
        "option_type": "CE",
        "strike_mode": "ATM",
    })

    # 4. OTM_2 strike recommendation
    print("\n🎯 [4/10] Get OTM_2 strike for PE")
    call("POST", "/strikes", {
        "option_type": "PE",
        "strike_mode": "OTM_2",
    })

    # 5. Premium-based strike recommendation
    print("\n🎯 [5/10] Find CE strike with premium near ₹150")
    call("POST", "/strikes", {
        "option_type": "CE",
        "strike_mode": "PREMIUM",
        "target_premium": 150,
    })

    # 6. Mini option chain
    print("\n📊 [6/10] Option chain (±3 strikes)")
    call("POST", "/strikes", {
        "chain": True,
        "width": 3,
    })

    # -----------------------------------------------------------------
    # TRADES (auto-strike)
    # -----------------------------------------------------------------

    # 7. Buy ATM CE (no strike specified = auto ATM)
    print("\n🟢 [7/10] Buy NIFTY ATM CE (auto-selected)")
    call("POST", "/trade", {
        "action": "BUY",
        "option_type": "CE",
    })

    # 8. Buy OTM_1 PE
    print("\n🟢 [8/10] Buy NIFTY PE 1 strike OTM (auto-selected)")
    call("POST", "/trade", {
        "action": "BUY",
        "option_type": "PE",
        "strike_mode": "OTM_1",
    })

    # 9. Sell CE by premium target
    print("\n🔴 [9/10] Sell NIFTY CE with premium near ₹200 (auto-selected)")
    call("POST", "/trade", {
        "action": "SELL",
        "option_type": "CE",
        "strike_mode": "PREMIUM",
        "target_premium": 200,
    })

    # 10. Manual strike (override auto-selection)
    print("\n🟢 [10/10] Buy NIFTY 24500 CE (manual strike)")
    call("POST", "/trade", {
        "action": "BUY",
        "option_type": "CE",
        "strike": 24500,
    })

    print("\n" + "=" * 65)
    print("  Done! Check your email and WhatsApp for trade alerts.")
    print("  Review audit log: cat /tmp/trade_audit.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
