"""
generate_truedata_symbols.py — Generate symbol list for Truedata Velocity bulk-add.

WHY THE ACTIVATION ERROR HAPPENED
-----------------------------------
Velocity is a REAL-TIME feed platform. It can only activate contracts that are
currently trading on NSE. Expired weekly expiries (e.g. NIFTY260512 = 12-May-2026)
are no longer traded, so Velocity cannot subscribe to their live feed.

CORRECT WORKFLOW
-----------------------------------
┌─────────────────────────────────────────────────────────────────────────────┐
│  Velocity (real-time)    → Only CURRENT + NEXT weekly expiry symbols        │
│  Historical REST API     → All expired contracts for V10 training data      │
└─────────────────────────────────────────────────────────────────────────────┘

USAGE:
    python scripts/generate_truedata_symbols.py
    → writes three files in data/truedata_archive/:
        velocity_current_LIVE.txt   ← paste into Velocity (active expiry only)
        velocity_next_week.txt      ← next week's expiry (add 1-2 days before)
        historical_all_expiries.txt ← all symbols for historical API download
"""
from datetime import date, timedelta
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# ATM range: NIFTY was in 22000-25500 range during Jan-May 2026
SPOT_MIN = 22000
SPOT_MAX = 26000
STRIKE_STEP = 50

# For live Velocity subscription, only use a tight band around current spot
# (reduces symbols needed; Velocity has limits on free trial)
LIVE_SPOT_CENTRE = 24700   # Update to today's NIFTY spot
LIVE_BAND_STRIKES = 20     # ±20 strikes = 40 strikes each side × 2 = 80 CE+PE


# ──────────────────────────────────────────────────────────────────────────────
# NIFTY weekly expiry dates (Tuesdays, check NSE for holiday shifts)
# Format for Truedata: YYMMDD → "260526" = 26 May 2026
# ──────────────────────────────────────────────────────────────────────────────

# All weekly expiries from Jan 2026 to current (for historical API download)
HISTORICAL_EXPIRIES = [
    # ── Jan 2026 ──
    "260106",  # 06 Jan 2026
    "260113",  # 13 Jan 2026
    "260120",  # 20 Jan 2026
    "260127",  # 27 Jan 2026
    # ── Feb 2026 ──
    "260203",  # 03 Feb 2026
    "260210",  # 10 Feb 2026
    "260217",  # 17 Feb 2026
    "260224",  # 24 Feb 2026
    # ── Mar 2026 ──
    "260303",  # 03 Mar 2026
    "260310",  # 10 Mar 2026
    "260317",  # 17 Mar 2026
    "260324",  # 24 Mar 2026
    "260331",  # 31 Mar 2026
    # ── Apr 2026 ──
    "260407",  # 07 Apr 2026
    "260414",  # 14 Apr 2026 (check: Ambedkar Jayanti — may be shifted)
    "260421",  # 21 Apr 2026
    "260428",  # 28 Apr 2026
    # ── May 2026 ──
    "260505",  # 05 May 2026
    "260512",  # 12 May 2026
    "260519",  # 19 May 2026  ← expired TODAY (2026-05-19)
]

# Current ACTIVE expiry (can be subscribed in Velocity)
CURRENT_ACTIVE_EXPIRY = "260526"   # 26 May 2026 — UPDATE EVERY TUESDAY

# Next expiry (add to Velocity 1-2 days before current expires)
NEXT_EXPIRY = "260602"             # 02 Jun 2026 — UPDATE EVERY TUESDAY


# ──────────────────────────────────────────────────────────────────────────────
# Symbol generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_symbols_for_expiry(
    expiry: str,
    spot_min: int,
    spot_max: int,
    step: int = 50,
) -> list[str]:
    """Generate CE+PE symbols for a single expiry, all strikes in range."""
    symbols = []
    for strike in range(spot_min, spot_max + 1, step):
        symbols.append(f"NIFTY{expiry}{strike}CE")
        symbols.append(f"NIFTY{expiry}{strike}PE")
    return symbols


def generate_live_symbols(
    expiry: str,
    centre: int,
    band: int = 20,
    step: int = 50,
) -> list[str]:
    """
    Tight band for Velocity — only ±band strikes around centre.
    Keeps live symbol count low (free trial has limits).
    """
    low  = centre - band * step
    high = centre + band * step
    return generate_symbols_for_expiry(expiry, low, high, step)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("data/truedata_archive")
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = ["NIFTY-I", "NIFTY", "INDIAVIX"]  # index + futures always needed

    # ── 1. Velocity LIVE file (current expiry only, tight band) ──────────────
    live_symbols = generate_live_symbols(
        expiry=CURRENT_ACTIVE_EXPIRY,
        centre=LIVE_SPOT_CENTRE,
        band=LIVE_BAND_STRIKES,
    )
    live_path = out_dir / "velocity_current_LIVE.txt"
    live_path.write_text("\n".join(headers + live_symbols), encoding="utf-8")

    # ── 2. Velocity NEXT week file ────────────────────────────────────────────
    next_symbols = generate_live_symbols(
        expiry=NEXT_EXPIRY,
        centre=LIVE_SPOT_CENTRE,
        band=LIVE_BAND_STRIKES,
    )
    next_path = out_dir / "velocity_next_week.txt"
    next_path.write_text("\n".join(headers + next_symbols), encoding="utf-8")

    # ── 3. Historical API file (all expired expiries, full strike range) ──────
    all_hist_symbols = []
    for exp in HISTORICAL_EXPIRIES:
        all_hist_symbols.extend(
            generate_symbols_for_expiry(exp, SPOT_MIN, SPOT_MAX, STRIKE_STEP)
        )
    hist_path = out_dir / "historical_all_expiries.txt"
    hist_path.write_text("\n".join(all_hist_symbols), encoding="utf-8")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("TRUEDATA SYMBOL FILES GENERATED")
    print("=" * 70)
    print()
    print("[LIVE] FOR VELOCITY (real-time subscription):")
    print(f"   Current expiry ({CURRENT_ACTIVE_EXPIRY}): {live_path}")
    print(f"   -> {len(headers) + len(live_symbols)} symbols (paste this into Velocity now)")
    print()
    print(f"   Next expiry ({NEXT_EXPIRY}): {next_path}")
    print(f"   -> {len(headers) + len(next_symbols)} symbols (add to Velocity next Monday)")
    print()
    print("[HIST] FOR HISTORICAL API (run download_truedata_historical.py):")
    print(f"   All expired expiries: {hist_path}")
    print(f"   -> {len(all_hist_symbols)} option symbols across {len(HISTORICAL_EXPIRIES)} expiries")
    print()
    print("HOW TO USE VELOCITY (fix for the activation error):")
    print("  1. Open Velocity -> symbol list")
    print("  2. Delete all NIFTY option entries (keep NIFTY-I, NIFTY, INDIAVIX)")
    print(f"  3. Open: {live_path.absolute()}")
    print("  4. Select All (Ctrl+A) -> Copy (Ctrl+C)")
    print("  5. In Velocity: 'Add symbol(s)' -> Paste -> OK")
    print("  6. These are all 26-May-2026 expiry = ACTIVE = no error")
    print()
    print("WHY expired expiries fail in Velocity:")
    print("  Velocity = LIVE FEED. Expired contracts have no live market data.")
    print("  Use download_truedata_historical.py for historical option chain data.")
    print()
    print("First 10 live symbols:")
    for s in (headers + live_symbols)[:10]:
        print(f"   {s}")


if __name__ == "__main__":
    main()
