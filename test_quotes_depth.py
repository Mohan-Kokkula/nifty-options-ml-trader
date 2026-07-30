"""Quick test: find which quote_type returns OI for NIFTY options."""
import json, logging
logging.basicConfig(level=logging.WARNING)
from dotenv import load_dotenv; load_dotenv("config/settings.env")
import sys; sys.path.insert(0, ".")
from config import load_config
from core.kotak_neo_client import KotakNeoClient
from security import RateLimiter

cfg = load_config()
rl  = RateLimiter(cfg.security.rate_limit_per_sec)
client = KotakNeoClient(
    consumer_key=cfg.kotak.consumer_key, environment=cfg.kotak.environment,
    neo_fin_key=None, rate_limiter=rl,
    mobile=cfg.kotak.mobile, password=cfg.kotak.password,
    totp_secret=cfg.kotak.totp_secret, ucc=cfg.kotak.ucc, mpin=cfg.kotak.mpin,
)
neo = client._neo

# Get 2 ATM NIFTY options
scrips = neo.search_scrip(exchange_segment="nse_fo", symbol="nifty", expiry="26May2026")
opts = [s for s in (scrips if isinstance(scrips, list) else [])
        if s.get("pSymbolName") == "NIFTY" and s.get("pInstType") == "OPTIDX"]

def strike_val(s):
    v = s.get("dStrikePrice;", -1)
    if v in (None, "", -1): return 0
    f = float(v)
    return f / 100 if f > 1_000_000 else f

# Nearest ATM: one CE and one PE
atm_opts = sorted(opts, key=lambda s: abs(strike_val(s) - 23800))[:2]
tokens = [{"instrument_token": str(s.get("pSymbol", "")), "exchange_segment": "nse_fo"}
          for s in atm_opts]
print(f"Testing with tokens: {[t['instrument_token'] for t in tokens]}")
print(f"  symbols: {[s.get('pTrdSymbol') for s in atm_opts]}")

# Test all quote_type variants
for qt in [None, "ltp", "depth", "52w"]:
    try:
        if qt is None:
            r = neo.quotes(instrument_tokens=tokens)
        else:
            r = neo.quotes(instrument_tokens=tokens, quote_type=qt)

        typ = type(r).__name__
        if isinstance(r, list) and r:
            # Show all keys of first record
            keys = list(r[0].keys())
            print(f"\nquote_type={qt!r} -> list[{len(r)}] | keys={keys}")
            # Show values for first record
            for k in keys:
                print(f"  {k}: {r[0][k]}")
        elif isinstance(r, dict):
            print(f"\nquote_type={qt!r} -> dict: {json.dumps(r)[:300]}")
        else:
            print(f"\nquote_type={qt!r} -> {typ}: {str(r)[:200]}")
    except Exception as e:
        print(f"\nquote_type={qt!r} -> EXCEPTION: {e}")
