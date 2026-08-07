"""Tests for the futures archiver's quote-payload extraction.

Reproduces the production failure of 2026-08-07: symbol resolution
succeeded (NIFTY26AUGFUT) but every tick logged "no snapshot ... (no
quote/depth returned)" and data/futures_archive stayed empty. Cause:
snapshot() read only data["ltp"] on a dict, while Kotak payloads also
arrive list-wrapped and use alternate key spellings -- the exact
tolerance normalize_depth() already had, never applied to the quote path.

Run standalone:  python scripts/test_quote_extract.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.futures_archiver import _extract_ltp_volume, _unwrap  # noqa: E402

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


print("\n-- payload shapes --")
check("dict under 'data'",
      _extract_ltp_volume({"data": {"ltp": 24610.5, "volume": 55410}})
      == (24610.5, 55410.0))
check("LIST under 'data' (the shape normalize_depth handles)",
      _extract_ltp_volume({"data": [{"ltp": 24610.5, "volume": 55410}]})
      == (24610.5, 55410.0))
check("bare dict, no 'data' wrapper",
      _extract_ltp_volume({"ltp": 24610.5, "volume": 55410}) == (24610.5, 55410.0))
check("bare list", _extract_ltp_volume([{"ltp": 100.0, "v": 5}]) == (100.0, 5.0))

print("\n-- alternate key spellings --")
for key in ("ltp", "last_price", "lp", "lastPrice", "ltP", "last_traded_price"):
    check(f"price key '{key}'",
          _extract_ltp_volume({"data": {key: 24610.5}})[0] == 24610.5)
for key in ("volume", "v", "vol", "tradedQty", "volume_traded", "vtt", "vTrdQty"):
    check(f"volume key '{key}'",
          _extract_ltp_volume({"data": {"ltp": 1.0, key: 999}})[1] == 999.0)

print("\n-- degenerate inputs must not raise --")
for bad in (None, {}, [], {"data": None}, {"data": []}, {"data": {}},
            "not-a-dict", {"data": {"ltp": None}}, {"data": {"ltp": "abc"}},
            {"data": {"ltp": 0}}):
    try:
        ltp, vol = _extract_ltp_volume(bad)
        ok = (ltp == 0.0 and vol == 0.0)
    except Exception as e:
        ok = False
        print(f"        raised {type(e).__name__}: {e}")
    check(f"{str(bad)[:34]:<34} -> (0.0, 0.0)", ok)

print("\n-- volume must survive a missing price and vice versa --")
check("price present, volume absent",
      _extract_ltp_volume({"data": {"ltp": 24610.5}}) == (24610.5, 0.0))
check("volume present, price absent -> ltp 0 (row correctly rejected)",
      _extract_ltp_volume({"data": {"volume": 55410}}) == (0.0, 55410.0))

print("\n-- _unwrap --")
check("unwraps nested list-in-data", _unwrap({"data": [{"a": 1}]}) == {"a": 1})
check("empty list -> {}", _unwrap({"data": []}) == {})
check("non-dict -> {}", _unwrap(42) == {})

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
