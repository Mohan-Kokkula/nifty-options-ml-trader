"""
watchdog_helper.py — run a function with a hard timeout.

Used to prevent any single gate/API call from hanging the pilot loop.
If a call takes longer than timeout_sec, returns default value.

USAGE:
    from core.watchdog_helper import call_with_timeout

    result = call_with_timeout(
        fn=lambda: my_slow_function(arg1, arg2),
        timeout_sec=5.0,
        default=None,
        name="my_slow_function",
    )
"""
from __future__ import annotations
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


def call_with_timeout(
    fn: Callable,
    timeout_sec: float = 5.0,
    default: Any = None,
    name: str = "anonymous",
) -> Any:
    """
    Run `fn()` with a hard timeout. Returns `default` if it doesn't
    complete in `timeout_sec` seconds.

    Worker runs as daemon — if it hangs forever, it gets cleaned up
    when the bot process exits (not before).
    """
    result_box = {"value": default, "done": False, "exc": None}

    def _runner():
        try:
            result_box["value"] = fn()
        except Exception as e:
            result_box["exc"] = e
        finally:
            result_box["done"] = True

    worker = threading.Thread(target=_runner, daemon=True, name=f"wd_{name}")
    worker.start()
    worker.join(timeout=timeout_sec)

    if not result_box["done"]:
        logger.warning(
            f"⚠️ WATCHDOG TIMEOUT: {name}() took >{timeout_sec}s — "
            f"using default value (worker still running in background)"
        )
        return default
    if result_box["exc"]:
        logger.debug(f"WATCHDOG: {name}() raised {result_box['exc']}")
        return default
    return result_box["value"]
