"""
sdk_timeout_patch.py — C4: SDK-level HTTP timeout for the vendored
neo_api_client SDK.

The installed neo_api_client SDK (site-packages, not part of this repo) has
no native timeout support anywhere: RESTClientObject.request()
(neo_api_client/rest.py) calls requests.post()/requests.get() with no
timeout= kwarg, so a hung broker connection blocks the calling thread
indefinitely. All api/*.py SDK modules route through this one shared
chokepoint (self.rest_client.request(...)), except scrip_search.py's
separate direct requests.get() call for the scrip-master CSV, which this
patch does not cover.

This patches ONLY the `requests` name inside neo_api_client.rest's own
module namespace — not the global `requests` module — so no other file in
this codebase (notifications/telegram_notifier.py, etc.) is affected.

2026-07-21 operational-safety fix #1 (resolves "C6"): order-placement and
order-cancellation URLs are NO LONGER excluded from this timeout. They
previously were, out of concern that a timeout could risk a duplicate
order on retry. That concern is now handled correctly at the call site
instead of by leaving these calls unbounded forever:
core.kotak_neo_client._call_with_retry() already never blindly retries an
order call (is_order=True skips retry after one attempt, unchanged), and
now additionally raises the distinctly-typed
core.kotak_neo_client.BrokerCallTimeout on a timeout specifically, so
callers can never mistake "timed out, outcome unknown" for "confirmed
success" or "confirmed failure." Every order call is therefore bounded by
this same timeout with no exceptions.
"""

import logging

import requests as _real_requests

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SEC = 5.0
READ_TIMEOUT_SEC = 10.0

_patched = False


class _TimeoutInjectingRequests:
    """Drop-in replacement for the `requests` module as seen from inside
    neo_api_client.rest — adds a default timeout to every .post()/.get()
    call, with no exclusions. A caller-supplied timeout is never
    overridden. Everything else delegates to the real requests module."""

    def post(self, *args, **kwargs):
        kwargs.setdefault("timeout", (CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC))
        return _real_requests.post(*args, **kwargs)

    def get(self, *args, **kwargs):
        kwargs.setdefault("timeout", (CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC))
        return _real_requests.get(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_real_requests, name)


def patch_sdk_timeouts() -> None:
    """Apply the timeout patch once. Idempotent — safe to call more than
    once. Raises RuntimeError instead of silently no-op'ing if the SDK's
    structure no longer matches what this patch targets (e.g. after an
    SDK version upgrade), so a missing timeout fails loudly at startup
    rather than silently reintroducing unbounded blocking calls."""
    global _patched
    if _patched:
        return

    try:
        from neo_api_client import rest as _rest_module
    except ImportError as e:
        raise RuntimeError(
            f"C4 timeout patch: neo_api_client.rest not importable ({e}) — "
            f"SDK calls would run with NO timeout. Aborting startup."
        ) from e

    if not hasattr(_rest_module, "requests"):
        raise RuntimeError(
            "C4 timeout patch: neo_api_client.rest has no 'requests' "
            "attribute — SDK internals changed, patch target is stale."
        )

    _rest_module.requests = _TimeoutInjectingRequests()

    if not isinstance(_rest_module.requests, _TimeoutInjectingRequests):
        raise RuntimeError("C4 timeout patch: assignment did not take effect.")

    _patched = True
    logger.info(
        f"C4: SDK HTTP timeout patch applied "
        f"(connect={CONNECT_TIMEOUT_SEC}s, read={READ_TIMEOUT_SEC}s; "
        f"no exclusions -- order placement/cancellation now bounded too)"
    )
