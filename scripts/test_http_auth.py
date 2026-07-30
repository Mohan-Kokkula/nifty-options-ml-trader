#!/usr/bin/env python3
"""
Regression test for C1: the HTTP API must require authentication when
API_AUTH_TOKEN is configured, and must remain backward compatible
(no auth enforced) when it isn't set. Also covers the POST /login
session-token flow (MPIN-based, added on top of C1).

Standalone script (repo convention), run with:
    python scripts/test_http_auth.py

Covers main.py::TradeHandler._check_auth() and _handle_login().

Bug being regression-tested: previously, do_GET/do_POST had no
authentication check at all -- any caller able to reach the port could
place trades, close positions, or cancel all orders.
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as main_module


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def _make_handler(auth_header=None):
    handler = object.__new__(main_module.TradeHandler)
    handler.headers = {}
    if auth_header is not None:
        handler.headers["Authorization"] = auth_header
    handler._respond = MagicMock()
    return handler


def _reset_login_state():
    main_module._login_activated = False
    main_module._active_token = {"value": None, "expires_at": 0.0}
    main_module._login_failed_attempts = 0
    main_module._login_locked_until = 0.0


def main():
    all_ok = True
    _reset_login_state()
    main_module._logger = MagicMock()  # set at runtime by main(); unset in a bare import

    # ── No token configured: backward compatible, auth not enforced ────
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("API_AUTH_TOKEN", None)
        h = _make_handler()
        all_ok &= check(
            "no API_AUTH_TOKEN configured -> request allowed (backward compatible)",
            h._check_auth() is True,
        )
        all_ok &= check("no 401 sent when auth not enforced", not h._respond.called)

    # ── Token configured, no header: rejected ───────────────────────────
    with patch.dict(os.environ, {"API_AUTH_TOKEN": "secret123"}):
        h2 = _make_handler()
        all_ok &= check("missing Authorization header -> rejected", h2._check_auth() is False)
        all_ok &= check(
            "401 sent for missing header",
            h2._respond.call_args[0] == (401, {"error": "Unauthorized"}),
        )

    # ── Token configured, wrong header: rejected ────────────────────────
    with patch.dict(os.environ, {"API_AUTH_TOKEN": "secret123"}):
        h3 = _make_handler(auth_header="Bearer wrong-token")
        all_ok &= check("mismatched token -> rejected", h3._check_auth() is False)

    # ── Token configured, correct header: allowed ───────────────────────
    with patch.dict(os.environ, {"API_AUTH_TOKEN": "secret123"}):
        h4 = _make_handler(auth_header="Bearer secret123")
        all_ok &= check("matching token -> allowed", h4._check_auth() is True)
        all_ok &= check("no 401 sent when token matches", not h4._respond.called)

    # ── /login: correct MPIN issues a token and activates enforcement ───
    _reset_login_state()
    with patch.dict(os.environ, {"KOTAK_MPIN": "1234", "API_AUTH_TOKEN": ""}):
        os.environ.pop("API_AUTH_TOKEN", None)
        h5 = _make_handler()
        h5._handle_login({"mpin": "1234"})
        status, payload = h5._respond.call_args[0]
        all_ok &= check("correct MPIN -> HTTP 200", status == 200)
        all_ok &= check("correct MPIN -> token present in response", bool(payload.get("token")))
        all_ok &= check("login activates enforcement", main_module._login_activated is True)

        issued_token = payload["token"]
        h6 = _make_handler(auth_header=f"Bearer {issued_token}")
        all_ok &= check("issued session token is accepted by _check_auth", h6._check_auth() is True)

        h7 = _make_handler()  # no header at all, after a login has occurred
        all_ok &= check(
            "after a login, an unauthenticated request is now rejected "
            "(enforcement turned on, unlike before any /login)",
            h7._check_auth() is False,
        )

    # ── /login: wrong MPIN is rejected, does not issue a token ──────────
    _reset_login_state()
    with patch.dict(os.environ, {"KOTAK_MPIN": "1234"}):
        h8 = _make_handler()
        h8._handle_login({"mpin": "0000"})
        status8, payload8 = h8._respond.call_args[0]
        all_ok &= check("wrong MPIN -> HTTP 401", status8 == 401)
        all_ok &= check("wrong MPIN -> no token issued", main_module._login_activated is False)

    # ── /login: 5 failed attempts locks out further attempts ────────────
    _reset_login_state()
    with patch.dict(os.environ, {"KOTAK_MPIN": "1234"}):
        for _ in range(5):
            h = _make_handler()
            h._handle_login({"mpin": "wrong"})
        h9 = _make_handler()
        h9._handle_login({"mpin": "wrong"})
        status9, payload9 = h9._respond.call_args[0]
        all_ok &= check("6th failed attempt after 5 failures -> HTTP 429 (locked out)", status9 == 429)

    # ── /login: expired session token is no longer accepted ─────────────
    _reset_login_state()
    with patch.dict(os.environ, {"KOTAK_MPIN": "1234"}):
        os.environ.pop("API_AUTH_TOKEN", None)
        main_module._login_activated = True
        main_module._active_token = {"value": "expired-tok", "expires_at": time.time() - 10}
        h10 = _make_handler(auth_header="Bearer expired-tok")
        all_ok &= check("expired session token is rejected", h10._check_auth() is False)

    _reset_login_state()
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
