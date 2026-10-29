"""
gex_live_scanner.py — Live GEX Options Signal Scanner (3-min cycle)
===================================================================
Scrapes NSE options chain every 3 minutes during market hours,
runs the S18 GEX strategy, and sends Telegram alerts on signals.

Usage:
    python scripts/gex_live_scanner.py

Requires in settings.env:
    TELEGRAM_ENABLED=true
    TELEGRAM_BOT_TOKEN=<your_bot_token>
    TELEGRAM_CHAT_ID=<your_chat_id>

Setup Telegram bot:
    1. Message @BotFather on Telegram -> /newbot -> copy the token
    2. Message your bot, then open:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       Find "chat":{"id": 123456789} -> that's your CHAT_ID
"""

import os
import sys
import time
import logging
from datetime import datetime, date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from dotenv import load_dotenv
load_dotenv(PROJECT / "config" / "settings.env")

import pandas as pd

from core.gex_engine import compute_gex_from_chain
from core.gex_options_strategy import GEXOptionsStrategy, format_signal
from notifications.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gex_scanner")

SCAN_INTERVAL = 180  # 3 minutes
COOLDOWN_MINUTES = 9
MARKET_OPEN = 915
MARKET_CLOSE = 1530
NIFTY_SYMBOL = "NIFTY"


_cached_spot = 0.0


def scrape_options_chain() -> pd.DataFrame:
    """Fetch NSE options chain via nsepython (bypasses Akamai)."""
    global _cached_spot
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            from nsepython import option_chain
            raw = option_chain("NIFTY")

            if not raw or "data" not in raw:
                log.error("nsepython returned empty data")
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return pd.DataFrame()

            records = raw["data"]
            timestamp = raw.get("timestamp", "")

            # Find nearest expiry from the data
            expiry_dates = set()
            for rec in records:
                ce = rec.get("CE", {})
                pe = rec.get("PE", {})
                exp = ce.get("expiryDate") or pe.get("expiryDate")
                if exp:
                    expiry_dates.add(exp)

            nearest_expiry = None
            if expiry_dates:
                from datetime import datetime as dt
                parsed = []
                for e in expiry_dates:
                    try:
                        parsed.append((dt.strptime(e, "%d-%b-%Y"), e))
                    except ValueError:
                        pass
                if parsed:
                    parsed.sort()
                    nearest_expiry = parsed[0][1]

            # Extract spot from underlying value
            for rec in records:
                ce = rec.get("CE", {})
                pe = rec.get("PE", {})
                uv = ce.get("underlyingValue") or pe.get("underlyingValue")
                if uv and float(uv) > 0:
                    _cached_spot = float(uv)
                    break

            rows = []
            for rec in records:
                ce = rec.get("CE", {})
                pe = rec.get("PE", {})
                if not ce and not pe:
                    continue

                # Filter to nearest expiry only
                rec_exp = ce.get("expiryDate") or pe.get("expiryDate")
                if nearest_expiry and rec_exp != nearest_expiry:
                    continue

                strike_raw = rec.get("strikePrice",
                                     ce.get("strikePrice", pe.get("strikePrice", 0)))
                strike = float(str(strike_raw).strip())

                rows.append({
                    "strike": strike,
                    "call_oi": ce.get("openInterest", 0),
                    "call_chg_oi": ce.get("changeinOpenInterest", 0),
                    "call_vol": ce.get("totalTradedVolume", 0),
                    "call_iv": ce.get("impliedVolatility", 0),
                    "call_ltp": ce.get("lastPrice", 0),
                    "put_oi": pe.get("openInterest", 0),
                    "put_chg_oi": pe.get("changeinOpenInterest", 0),
                    "put_vol": pe.get("totalTradedVolume", 0),
                    "put_iv": pe.get("impliedVolatility", 0),
                    "put_ltp": pe.get("lastPrice", 0),
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("strike").reset_index(drop=True)
                log.info(f"Chain scraped: {len(df)} strikes (expiry={nearest_expiry}, "
                         f"spot={_cached_spot:.1f}, ts={timestamp})")
                today = date.today().strftime("%Y%m%d")
                save_path = PROJECT / "data" / f"nifty_options_chain_{today}.csv"
                df.to_csv(save_path, index=False)
            return df

        except Exception as e:
            log.error(f"Chain scrape attempt {attempt+1} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)

    return pd.DataFrame()


def get_nifty_spot() -> float:
    """Return spot price cached from the last option_chain call."""
    return _cached_spot


def load_chain_from_file() -> pd.DataFrame:
    """Fallback: load today's chain from CSV if scraping fails."""
    today = date.today().strftime("%Y%m%d")
    path = PROJECT / "data" / f"nifty_options_chain_{today}.csv"
    if path.exists():
        return pd.read_csv(path)
    # Try yesterday
    from datetime import timedelta
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    path2 = PROJECT / "data" / f"nifty_options_chain_{yesterday}.csv"
    if path2.exists():
        log.warning(f"Using yesterday's chain: {path2.name}")
        return pd.read_csv(path2)
    return pd.DataFrame()


def compute_dte() -> int:
    """Days to expiry for current Nifty weekly (Monday expiry)."""
    today = date.today()
    weekday = today.weekday()  # Mon=0 ... Sun=6
    if weekday == 0:
        return 0  # Expiry day
    days_to_mon = (7 - weekday) % 7
    return days_to_mon


def format_telegram_signal(signal, spot):
    """Format signal for Telegram HTML message."""
    action = signal["action"]
    if action == "NO_TRADE":
        return None

    lvl = signal["levels"]
    m = signal["metrics"]
    now = datetime.now().strftime("%H:%M")

    emoji = "🟢" if "CE" in action else "🔴" if "PE" in action else "🔵"

    lines = [
        f"{emoji} <b>GEX SIGNAL — {action}</b>",
        "",
        f"<b>Strike:</b> {signal['strike']:.0f}",
        f"<b>Spot:</b> {spot:.2f}",
        f"<b>Confidence:</b> {signal['confidence']}%",
        f"<b>SL:</b> {signal['sl_pct']}% | <b>TP:</b> {signal['tp_pct']}%",
        "",
        f"<b>Regime:</b> {signal['regime'].upper()}",
        f"<b>Flow:</b> {m['flow']} (chgPCR={m['chg_oi_pcr']:.2f})",
        f"<b>GEX Levels:</b>",
        f"  Call Wall: {lvl['call_wall']:.0f}",
        f"  Put Wall: {lvl['put_wall']:.0f}",
        f"  Gamma Flip: {lvl['gamma_flip']:.0f}",
        f"  Max Pain: {lvl['max_pain']:.0f}",
        "",
        f"<b>Reason:</b>",
        f"<i>{signal['reason']}</i>",
        "",
        f"⏰ {now} IST | DTE={m['dte']}",
    ]
    return "\n".join(lines)


def main():
    log.info("=" * 60)
    log.info("  GEX Live Scanner — S18 Options Strategy")
    log.info("=" * 60)

    # Init Telegram
    tg = TelegramNotifier(
        enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )

    if tg.enabled:
        tg.send_message(
            "🚀 <b>GEX Scanner Started</b>\n\n"
            "Scanning every 3 min for S18 options signals.\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')} IST"
        )
        log.info("Telegram connected and startup message sent")
    else:
        log.warning("Telegram NOT enabled — signals will print to console only")
        log.warning("Set TELEGRAM_ENABLED=true in settings.env")

    strat = GEXOptionsStrategy()
    last_signal_time = 0
    last_signal_action = ""
    day_open = 0.0
    day_high = 0.0
    day_low = 99999.0
    chain_df = pd.DataFrame()
    chain_fetch_count = 0

    log.info(f"DTE: {compute_dte()} | Scan interval: {SCAN_INTERVAL}s")
    log.info("Waiting for market hours...")

    while True:
        now = datetime.now()
        hhmm = now.hour * 100 + now.minute

        # Outside market hours
        if hhmm < MARKET_OPEN - 5 or hhmm > MARKET_CLOSE:
            if hhmm > MARKET_CLOSE and chain_fetch_count > 0:
                log.info("Market closed. Scanner stopping.")
                if tg.enabled:
                    tg.send_message(
                        "🔔 <b>GEX Scanner — Market Closed</b>\n"
                        f"Scans today: {chain_fetch_count}"
                    )
                break
            time.sleep(30)
            continue

        try:
            # Refresh chain every cycle
            new_chain = scrape_options_chain()
            if new_chain.empty:
                new_chain = load_chain_from_file()
            if new_chain.empty:
                log.warning("No options chain data available, retrying...")
                time.sleep(SCAN_INTERVAL)
                continue
            chain_df = new_chain
            chain_fetch_count += 1

            # Get spot price
            spot = get_nifty_spot()
            if spot == 0:
                # Try from chain LTP
                atm = chain_df.iloc[(chain_df["strike"] - 22800).abs().argsort()[:1]]
                if not atm.empty:
                    call_ltp = float(atm.iloc[0]["call_ltp"])
                    put_ltp = float(atm.iloc[0]["put_ltp"])
                    spot = float(atm.iloc[0]["strike"]) + call_ltp - put_ltp
                else:
                    log.warning("Cannot determine spot price")
                    time.sleep(SCAN_INTERVAL)
                    continue

            # Track intraday range
            if day_open == 0:
                day_open = spot
            day_high = max(day_high, spot)
            day_low = min(day_low, spot)

            dte = compute_dte()

            # Compute GEX and analyze
            gex = compute_gex_from_chain(chain_df, spot, dte)
            signal = strat.analyze(
                chain_df, spot, gex, dte,
                day_open, day_high, day_low, hhmm
            )

            # Log every scan
            regime_short = signal["regime"][0].upper()
            flip = signal["levels"]["gamma_flip"]
            m = signal["metrics"]
            action = signal["action"]

            log.info(
                f"Spot={spot:.1f} GEX={regime_short} Flip={flip:.0f}({spot-flip:+.0f}) "
                f"Flow={m['flow']} Move={m['intraday_move']:+.0f}({m['move_pct_of_range']:.0%}) "
                f"-> {action}"
            )

            # Send alert if signal fires and cooldown passed
            if action != "NO_TRADE":
                minutes_since = (hhmm - last_signal_time) if last_signal_time else 999
                # Convert HHMM diff to actual minutes
                h_diff = minutes_since // 100
                m_diff = minutes_since % 100
                actual_minutes = h_diff * 60 + m_diff

                is_same = (action == last_signal_action)
                cooldown_ok = actual_minutes >= COOLDOWN_MINUTES or not is_same

                if cooldown_ok:
                    log.info(f"*** SIGNAL: {action} {signal['strike']:.0f} "
                             f"conf={signal['confidence']}% ***")

                    # Console output
                    print("\n" + format_signal(signal) + "\n")

                    # Telegram alert
                    msg = format_telegram_signal(signal, spot)
                    if msg and tg.enabled:
                        tg.send_message(msg)
                        log.info("Telegram alert sent")

                    last_signal_time = hhmm
                    last_signal_action = action
                else:
                    log.info(f"  (repeat {action}, cooldown {actual_minutes}m < {COOLDOWN_MINUTES}m)")

        except Exception as e:
            log.error(f"Scan error: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
