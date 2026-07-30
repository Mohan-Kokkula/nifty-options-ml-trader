"""Probe raw search_scrip + quotes responses after successful TOTP login."""
import json, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from dotenv import load_dotenv
load_dotenv("config/settings.env")

from config import load_config
from core.kotak_neo_client import KotakNeoClient
from security import RateLimiter

cfg = load_config()
rl = RateLimiter(cfg.security.rate_limit_per_sec)
client = KotakNeoClient(
    consumer_key=cfg.kotak.consumer_key,
    environment=cfg.kotak.environment,
    neo_fin_key=cfg.kotak.neo_fin_key or None,
    rate_limiter=rl,
    mobile=cfg.kotak.mobile,
    password=cfg.kotak.password,
    totp_secret=cfg.kotak.totp_secret,
    ucc=cfg.kotak.ucc,
    mpin=cfg.kotak.mpin,
)
neo = client._neo

def dump(label, obj):
    print(f"\n=== {label} ===")
    try:
        print(json.dumps(obj, indent=2, default=str)[:2500])
    except Exception:
        print(repr(obj)[:2500])

# 1) search across segments/terms
for seg in ["nse_cm", "nse_idx", "nse_fo"]:
    for term in ["nifty 50", "NIFTY 50", "nifty", "NIFTY"]:
        try:
            r = neo.search_scrip(exchange_segment=seg, symbol=term)
            n = len(r) if isinstance(r, list) else "dict"
            print(f"  search {seg:8s} {term!r:14s} → {n}")
            if isinstance(r, list) and r:
                # Print first 3 matches
                for item in r[:3]:
                    print(f"      {item}")
        except Exception as e:
            print(f"  search {seg}/{term} EXC: {e}")

# 2) quotes using a few candidate tokens/segments
for tok, seg in [("26000","nse_cm"), ("26000","nse_idx"), ("Nifty 50","nse_cm")]:
    try:
        r = neo.quotes(
            instrument_tokens=[{"instrument_token": tok, "exchange_segment": seg}],
            quote_type="ltp",
        )
        dump(f"quotes token={tok} seg={seg}", r)
    except Exception as e:
        print(f"quotes {tok}/{seg} EXC: {e}")

# 3) also try without quote_type
try:
    r = neo.quotes(instrument_tokens=[{"instrument_token":"26000","exchange_segment":"nse_cm"}])
    dump("quotes default (no quote_type)", r)
except Exception as e:
    print(f"quotes default EXC: {e}")
