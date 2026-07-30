"""
Raw probe of Kotak Neo SDK calls.
Run:  python test_kotak_raw.py
"""
import logging, json, traceback
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

from dotenv import load_dotenv
load_dotenv("config/settings.env")

from config import load_config
from core.kotak_neo_client import KotakNeoClient
from security import RateLimiter

cfg = load_config()
client = KotakNeoClient(cfg, rate_limiter=RateLimiter(cfg.security.rate_limit_per_sec))
neo = client._neo

def show(label, fn):
    print(f"\n========== {label} ==========")
    try:
        result = fn()
        print(f"TYPE: {type(result).__name__}")
        try:
            print(json.dumps(result, indent=2, default=str)[:2000])
        except Exception:
            print(repr(result)[:2000])
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

# 1. Try every variant of search_scrip for Nifty
for seg in ["nse_cm", "nse_idx", "NSE_INDEX", "nse_fo"]:
    for term in ["nifty 50", "nifty50", "NIFTY 50", "NIFTY", "nifty"]:
        show(
            f"search_scrip(exchange_segment={seg!r}, symbol={term!r})",
            lambda s=seg, t=term: neo.search_scrip(exchange_segment=s, symbol=t),
        )

# 2. Direct quote with the assumed token 26000 on every plausible segment
for seg in ["nse_cm", "nse_idx", "NSE_INDEX"]:
    show(
        f"quotes(token=26000, segment={seg!r})",
        lambda s=seg: neo.quotes(
            instrument_tokens=[{"instrument_token": "26000", "exchange_segment": s}],
            quote_type="ltp",
        ),
    )

# 3. Limits — confirms session is alive (this is what the health check uses)
show("limits(segment='FO')", lambda: neo.limits(segment="FO"))
show("limits(segment='CM')", lambda: neo.limits(segment="CM"))
