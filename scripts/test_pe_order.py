"""
Standalone test: place a single PE IOC LIMIT order on Kotak Neo.
Uses the SAME payload shape as core/kotak_neo_client.py place_option_order().

Usage on Hostinger VPS:
    cd /opt/openclaw-v9-kotak
    source .venv/bin/activate          # or: uv venv activate
    python scripts/test_pe_order.py
"""
import os, sys, pyotp
from neo_api_client import NeoAPI
from dotenv import load_dotenv

load_dotenv("config/settings.env")

# -------- EDIT THESE BEFORE RUNNING --------
TRADING_SYMBOL = "NIFTY29APR2625000PE"   # exact pTrdSymbol from search_scrip
QUANTITY       = "65"                     # 1 lot
LIMIT_PRICE    = "1.00"                   # set well BELOW market so IOC cancels safely
                                          # (use real ask only if you actually want a fill)
TRANSACTION    = "B"                    # "BUY" entry, "SELL" exit
# -------------------------------------------

CONSUMER_KEY  = os.getenv("KOTAK_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("KOTAK_CONSUMER_SECRET")  # optional
NEO_FIN_KEY   = os.getenv("KOTAK_NEO_FIN_KEY")
MOBILE        = os.getenv("KOTAK_MOBILE")             # e.g. +919876543210
UCC           = os.getenv("KOTAK_UCC")
MPIN          = os.getenv("KOTAK_MPIN")
TOTP_SECRET   = os.getenv("KOTAK_TOTP_SECRET")

print(f"Logging in as UCC={UCC} mobile={MOBILE}")
neo = NeoAPI(consumer_key=CONSUMER_KEY, environment="prod", neo_fin_key=NEO_FIN_KEY)

# Step 1: TOTP login
totp_code = pyotp.TOTP(TOTP_SECRET.strip().replace(" ", "")).now()
r1 = neo.totp_login(mobile_number=MOBILE, ucc=UCC, totp=totp_code)
print(f"totp_login → {r1}")

# Step 2: MPIN validate
r2 = neo.totp_validate(mpin=MPIN)
print(f"totp_validate → {r2}")

# Step 3: place_order — EXACT payload our bot sends
order_req = {
    "exchange_segment": "nse_fo",
    "product":          "MIS",
    "price":            "1.00",
    "order_type":       "L",
    "quantity":         "65",
    "validity":         "IOC",
    "trading_symbol":   "NIFTY29APR2625000PE",
    "transaction_type": "B",
    "trigger_price":    "0",
    "disclosed_quantity": "0",
    "amo":              "NO",
    "market_protection": "0",
    "pf":               "N",
}

print("\n📤 SDK place_order REQUEST:")
for k, v in order_req.items():
    print(f"   {k:20s} = {v!r}")

resp = neo.place_order(**order_req)
print(f"\n📥 SDK RESPONSE: {resp}")
print("\n📤 REQUEST:")
for k, v in order_req.items():
    print(f"   {k:18s} = {v!r}")

resp = neo.place_order(**order_req)

print("\n📥 RESPONSE:")
print(resp)