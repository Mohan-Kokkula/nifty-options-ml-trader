"""
Quick sanity check: does Kotak Neo return a valid Nifty spot?
Run:  python test_spot.py
"""
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from dotenv import load_dotenv
load_dotenv("config/settings.env")

from config import load_config
from core.kotak_neo_client import KotakNeoClient
from security import RateLimiter

cfg = load_config()
print(f"Consumer key (first 12 chars): {cfg.kotak.consumer_key[:12]}...")
print(f"Mobile (last 4)              : ****{cfg.kotak.mobile[-4:] if cfg.kotak.mobile else 'N/A'}")
print(f"UCC set                      : {bool(cfg.kotak.ucc)}  ({cfg.kotak.ucc[:3]}***)" if cfg.kotak.ucc else "UCC set                      : False")
print(f"MPIN set                     : {bool(cfg.kotak.mpin or cfg.kotak.password)}")
print(f"TOTP secret set              : {bool(cfg.kotak.totp_secret)}")

rate_limiter = RateLimiter(cfg.security.rate_limit_per_sec)
client = KotakNeoClient(
    consumer_key=cfg.kotak.consumer_key,
    environment=cfg.kotak.environment,
    neo_fin_key=cfg.kotak.neo_fin_key or None,
    rate_limiter=rate_limiter,
    mobile=cfg.kotak.mobile,
    password=cfg.kotak.password,
    totp_secret=cfg.kotak.totp_secret,
    ucc=cfg.kotak.ucc,
    mpin=cfg.kotak.mpin,
)

print("\n--- Token resolution ---")
try:
    token, seg = client._resolve_nifty_token()
    print(f"  token = {token!r}")
    print(f"  seg   = {seg!r}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

print("\n--- Spot price (1st call — should also resolve token internally) ---")
try:
    spot = client._get_spot_price()
    print(f"  spot = {spot}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- Spot price (2nd call — should reuse cached token) ---")
try:
    spot = client._get_spot_price()
    print(f"  spot = {spot}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- get_quote('NIFTY','NSE_INDEX') ---")
print(f"  {client.get_quote('NIFTY', 'NSE_INDEX')}")
