#!/usr/bin/env python3
"""
Regression test for C4: SDK-level HTTP timeout patch.

Standalone script (repo convention), run with:
    python scripts/test_sdk_timeout_patch.py

Covers core/sdk_timeout_patch.py — the vendored neo_api_client SDK has no
native timeout anywhere (neo_api_client/rest.py's RESTClientObject.request()
calls requests.post()/requests.get() with no timeout= kwarg), so a hung
broker connection previously blocked the calling thread indefinitely.

2026-07-21 operational-safety fix #1: order-placement/cancellation URLs are
NO LONGER excluded from the timeout (previously excluded, tagged "C6").
This file's exclusion-classification checks were rewritten accordingly —
every POST/GET gets the timeout now, with no exceptions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.sdk_timeout_patch as patch_mod


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True

    # ── No exclusions module-level: the old _EXCLUDED_URL_SUBSTRINGS /
    # _is_excluded() were removed entirely by operational-safety fix #1 ──
    all_ok &= check(
        "_is_excluded no longer exists (exclusion mechanism removed)",
        not hasattr(patch_mod, "_is_excluded"),
    )
    all_ok &= check(
        "_EXCLUDED_URL_SUBSTRINGS no longer exists",
        not hasattr(patch_mod, "_EXCLUDED_URL_SUBSTRINGS"),
    )

    # ── Timeout injection: unconditional, no exceptions for any URL ────
    fake_requests = MagicMock()
    fake_requests.post.return_value = "POST_RESPONSE"
    fake_requests.get.return_value = "GET_RESPONSE"
    original_real = patch_mod._real_requests
    patch_mod._real_requests = fake_requests
    try:
        proxy = patch_mod._TimeoutInjectingRequests()

        proxy.post(url="https://x/quick/user/positions", headers={}, data=None)
        _, kwargs = fake_requests.post.call_args
        all_ok &= check(
            "non-order POST gets a timeout injected",
            kwargs.get("timeout") == (patch_mod.CONNECT_TIMEOUT_SEC, patch_mod.READ_TIMEOUT_SEC),
        )

        fake_requests.post.reset_mock()
        proxy.post(url="https://x/quick/order/rule/ms/place", headers={}, data=None)
        _, kwargs = fake_requests.post.call_args
        all_ok &= check(
            "order-placement POST NOW ALSO gets a timeout injected (C6 resolved)",
            kwargs.get("timeout") == (patch_mod.CONNECT_TIMEOUT_SEC, patch_mod.READ_TIMEOUT_SEC),
        )

        fake_requests.post.reset_mock()
        proxy.post(url="https://x/quick/order/cancel", headers={}, data=None)
        _, kwargs = fake_requests.post.call_args
        all_ok &= check(
            "order-cancellation POST NOW ALSO gets a timeout injected (C6 resolved)",
            kwargs.get("timeout") == (patch_mod.CONNECT_TIMEOUT_SEC, patch_mod.READ_TIMEOUT_SEC),
        )

        proxy.get(url="https://x/quick/user/limits", headers={})
        _, kwargs = fake_requests.get.call_args
        all_ok &= check(
            "GET gets a timeout injected",
            kwargs.get("timeout") == (patch_mod.CONNECT_TIMEOUT_SEC, patch_mod.READ_TIMEOUT_SEC),
        )

        fake_requests.post.reset_mock()
        proxy.post(url="https://x/quick/user/positions", headers={}, timeout=(1.0, 1.0))
        _, kwargs = fake_requests.post.call_args
        all_ok &= check(
            "an explicitly-passed timeout is preserved, not overridden",
            kwargs.get("timeout") == (1.0, 1.0),
        )
    finally:
        patch_mod._real_requests = original_real

    # ── patch_sdk_timeouts(): applies, idempotent, fail-loud ────────────
    from neo_api_client import rest as real_rest_module

    saved_requests_attr = real_rest_module.requests

    patch_mod._patched = False
    try:
        patch_mod.patch_sdk_timeouts()
        all_ok &= check(
            "patch_sdk_timeouts replaces neo_api_client.rest.requests",
            isinstance(real_rest_module.requests, patch_mod._TimeoutInjectingRequests),
        )

        patched_instance = real_rest_module.requests
        patch_mod.patch_sdk_timeouts()  # second call must be a no-op
        all_ok &= check(
            "second call is idempotent (no re-patch)",
            real_rest_module.requests is patched_instance,
        )
    finally:
        real_rest_module.requests = saved_requests_attr
        patch_mod._patched = True

    # Fail-loud path: SDK structure no longer matches the patch target.
    patch_mod._patched = False
    delattr(real_rest_module, "requests")
    try:
        raised = False
        try:
            patch_mod.patch_sdk_timeouts()
        except RuntimeError:
            raised = True
        all_ok &= check(
            "missing 'requests' attribute on SDK raises RuntimeError (fail-loud)", raised
        )
    finally:
        real_rest_module.requests = saved_requests_attr
        patch_mod._patched = True

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
