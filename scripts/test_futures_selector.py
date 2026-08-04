"""
Standalone test for core/futures_selector.py — run directly:
    python scripts/test_futures_selector.py

Not pytest-discovered (matches this repo's convention — see pytest.ini
testpaths). Uses a mock broker client so it runs with no live session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock
from datetime import date
from core.futures_selector import FuturesSelector


def _mock_client(search_scrip_return):
    client = MagicMock()
    client._neo.search_scrip = MagicMock(return_value=search_scrip_return)
    return client


def _wanted_symbol():
    from core.vwap_engine import _current_futures_symbol
    return _current_futures_symbol().upper()


def test_exact_match():
    wanted = _wanted_symbol()
    client = _mock_client([
        {"pTrdSymbol": wanted, "pSymbol": "58072"},
        {"pTrdSymbol": "NIFTY24500CE", "pOptionType": "CE"},
    ])
    sel = FuturesSelector(client)
    sym = sel.resolve_trading_symbol()
    assert sym == wanted, f"expected {wanted}, got {sym}"
    print(f"PASS: exact match resolves to {sym}")


def test_fallback_when_exact_missing():
    client = _mock_client([
        {"pTrdSymbol": "NIFTY99XXXFUT", "pOptionType": ""},
        {"pTrdSymbol": "NIFTY24500CE", "pOptionType": "CE"},
    ])
    sel = FuturesSelector(client)
    sym = sel.resolve_trading_symbol()
    assert sym == "NIFTY99XXXFUT", f"expected fallback, got {sym}"
    print(f"PASS: fallback resolves to {sym} when exact symbol absent")


def test_empty_results_returns_none():
    client = _mock_client([])
    sel = FuturesSelector(client)
    assert sel.resolve_trading_symbol() is None
    print("PASS: empty search_scrip result -> None (fail closed)")


def test_non_list_response_returns_none():
    # Kotak returns a dict like {"Error Message": "..."} pre-auth
    client = _mock_client({"Error Message": "Complete 2fa"})
    sel = FuturesSelector(client)
    assert sel.resolve_trading_symbol() is None
    print("PASS: non-list (pre-auth) response -> None (fail closed)")


def test_no_options_contaminate_fallback():
    # Options contracts must never be mistaken for the futures fallback
    client = _mock_client([
        {"pTrdSymbol": "NIFTYFUT", "pOptionType": "CE"},  # malformed/option, must be skipped
    ])
    sel = FuturesSelector(client)
    assert sel.resolve_trading_symbol() is None
    print("PASS: option-tagged row never matches futures fallback")


def test_caching_avoids_repeat_lookup():
    wanted = _wanted_symbol()
    client = _mock_client([{"pTrdSymbol": wanted, "pSymbol": "58072"}])
    sel = FuturesSelector(client)
    sym1 = sel.resolve_trading_symbol()
    sym2 = sel.resolve_trading_symbol()
    assert sym1 == sym2 == wanted
    assert client._neo.search_scrip.call_count == 1, (
        f"expected 1 broker call (cached), got {client._neo.search_scrip.call_count}"
    )
    print("PASS: second call served from cache, no repeat broker lookup")


def test_invalidate_cache_forces_relookup():
    wanted = _wanted_symbol()
    client = _mock_client([{"pTrdSymbol": wanted, "pSymbol": "58072"}])
    sel = FuturesSelector(client)
    sel.resolve_trading_symbol()
    sel.invalidate_cache()
    sel.resolve_trading_symbol()
    assert client._neo.search_scrip.call_count == 2
    print("PASS: invalidate_cache() forces a fresh broker lookup")


def test_no_broker_client_returns_none():
    class NoNeo:
        pass
    sel = FuturesSelector(NoNeo())
    assert sel.resolve_trading_symbol() is None
    print("PASS: missing _neo attribute -> None (fail closed), no crash")


if __name__ == "__main__":
    test_exact_match()
    test_fallback_when_exact_missing()
    test_empty_results_returns_none()
    test_non_list_response_returns_none()
    test_no_options_contaminate_fallback()
    test_caching_avoids_repeat_lookup()
    test_invalidate_cache_forces_relookup()
    test_no_broker_client_returns_none()
    print("\nALL TESTS PASSED")
