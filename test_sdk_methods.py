"""List public methods on the installed neo-api-client NeoAPI class."""
import neo_api_client
print(f"neo-api-client version: {getattr(neo_api_client, '__version__', 'unknown')}")
print(f"Module path: {neo_api_client.__file__}")
print()

from neo_api_client import NeoAPI
api = NeoAPI(consumer_key="dummy", environment="prod")

methods = sorted(m for m in dir(api) if not m.startswith("_"))
print("Public methods on NeoAPI:")
for m in methods:
    print(f"  {m}")
