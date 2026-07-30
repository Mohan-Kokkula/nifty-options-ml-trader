"""Verify get_option_chain() now returns real OI after the fix."""
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
from dotenv import load_dotenv; load_dotenv("config/settings.env")
sys.path.insert(0, ".")
from config import load_config
from core.kotak_neo_client import KotakNeoClient
from security import RateLimiter

cfg = load_config()
rl = RateLimiter(cfg.security.rate_limit_per_sec)
client = KotakNeoClient(
    consumer_key=cfg.kotak.consumer_key, environment=cfg.kotak.environment,
    neo_fin_key=None, rate_limiter=rl,
    mobile=cfg.kotak.mobile, password=cfg.kotak.password,
    totp_secret=cfg.kotak.totp_secret, ucc=cfg.kotak.ucc, mpin=cfg.kotak.mpin,
)

print("=== Testing get_option_chain() with OI fix ===")
chain = client.get_option_chain(
    underlying="NIFTY", exchange="NFO", expiry_date="26MAY26", strike_count=5
)

strikes = chain.get("chain", [])
total_ce = chain.get("total_ce_oi", 0)
total_pe = chain.get("total_pe_oi", 0)
pcr      = chain.get("pcr", 0)
spot     = chain.get("spot", 0)

print(f"\nspot       : {spot:,.2f}")
print(f"strikes    : {len(strikes)}")
print(f"total_ce_oi: {total_ce:,.0f}")
print(f"total_pe_oi: {total_pe:,.0f}")
print(f"PCR        : {pcr:.3f}")
print(f"\nATM chain sample:")
for e in strikes[:6]:
    ce_oi  = e.get("ce_oi", 0)
    pe_oi  = e.get("pe_oi", 0)
    ce_ltp = e.get("ce_ltp", 0)
    pe_ltp = e.get("pe_ltp", 0)
    print(f"  {e['strike']:>6} | CE ltp={ce_ltp:>8.2f}  OI={ce_oi:>12,.0f}"
          f"  |  PE ltp={pe_ltp:>8.2f}  OI={pe_oi:>12,.0f}")

if total_ce > 0 and total_pe > 0:
    print("\n✅ OI data is working!")
else:
    print("\n❌ OI still 0 — check logs above")
