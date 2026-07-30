#!/usr/bin/env python3
"""
Regression test: KotakNeoClient._expiry_oa_to_neo() must accept every
expiry format core.expiry_utils._parse_expiry_str() accepts, not just
the single no-dash-2-digit-year format it originally handled.

Standalone script (repo convention), run with:
    python scripts/test_expiry_oa_to_neo.py

Covers core/kotak_neo_client.py::KotakNeoClient._expiry_oa_to_neo().

Bug being regression-tested: scripts/oi_archiver.py's fallback path
formats the expiry as "21-JUL-26" (dash + 2-digit year, via
d.strftime("%d-%b-%y")), which the original _expiry_oa_to_neo() could not
parse (it only tried "%d%b%y", no dash) -- it silently returned the
malformed string unchanged, and get_option_chain()'s search_scrip() call
then found no matching contract, even with a correct, non-stale
NIFTY_EXPIRY configured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kotak_neo_client import KotakNeoClient


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main():
    all_ok = True
    client = object.__new__(KotakNeoClient)

    all_ok &= check(
        "dash + 2-digit year (the reported bug) normalizes correctly",
        client._expiry_oa_to_neo("21-JUL-26") == "21JUL2026",
    )
    all_ok &= check(
        "no-dash 2-digit year (already worked) is unaffected",
        client._expiry_oa_to_neo("21JUL26") == "21JUL2026",
    )
    all_ok &= check(
        "no-dash 4-digit year (already-correct Neo format) is unaffected",
        client._expiry_oa_to_neo("21JUL2026") == "21JUL2026",
    )
    all_ok &= check(
        "dash + 4-digit year normalizes correctly",
        client._expiry_oa_to_neo("21-JUL-2026") == "21JUL2026",
    )
    all_ok &= check(
        "lowercase input is handled (case-insensitive)",
        client._expiry_oa_to_neo("21-jul-26") == "21JUL2026",
    )
    all_ok &= check(
        "unrecognized format returns unchanged, uppercased (pre-existing fallback preserved)",
        client._expiry_oa_to_neo("garbage") == "GARBAGE",
    )

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
