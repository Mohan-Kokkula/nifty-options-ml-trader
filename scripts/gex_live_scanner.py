"""
gex_live_scanner.py — Live GEX Options Signal Scanner (3-min cycle)
===================================================================
Runs in three sessions using NSE GEX levels:

  Pre-market  (06:30–09:14): Gift Nifty spot + previous day's NSE chain
  Market      (09:15–15:30): NSE spot + live NSE chain (refreshed each cycle)
  Post-market (16:35–02:45): Gift Nifty spot + day's final NSE chain

Gift Nifty trades ~21 hrs/day on NSE IX. We use its price as the
"current Nifty level" outside NSE hours, against NSE's option chain
(where the real dealer gamma positioning lives).

Usage:
    python scripts/gex_live_scanner.py

Requires in settings.env:
    TELEGRAM_ENABLED=true
    TELEGRAM_BOT_TOKEN=<your_bot_token>
    TELEGRAM_CHAT_ID=<your_chat_id>
"""

import os
import sys
import time
import json
import re
import logging
from datetime import datetime, date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from dotenv import load_dotenv
load_dotenv(PROJECT / "config" / "settings.env")

import pandas as pd
import requests

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
NIFTY_SYMBOL = "NIFTY"

# Session boundaries (HHMM)
GIFT_SESSION1_START = 630   # Gift Nifty session 1
NSE_OPEN = 915
NSE_CLOSE = 1530
GIFT_SESSION2_START = 1635  # Gift Nifty session 2
GIFT_SESSION2_END = 2645    # 02:45 next day -> represented as 26:45

_cached_nse_spot = 0.0


# ── Gift Nifty Price Feed ──────────────────────────────────────────

def get_gift_nifty_price() -> dict:
    """Fetch Gift Nifty price from niftytrader.in via __NEXT_DATA__."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(
            "https://www.niftytrader.in/gift-nifty-live",
            headers=headers, timeout=15,
        )
        if r.status_code != 200:
            log.warning(f"Gift Nifty fetch HTTP {r.status_code}")
            return {}

        match = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text
        )
        if not match:
            log.warning("Gift Nifty: no __NEXT_DATA__ found")
            return {}

        data = json.loads(match.group(1))
        gift = data["props"]["pageProps"]["initialGiftData"]
        return {
            "ltp": float(gift["last_trade_price"]),
            "open": float(gift["open"]),
            "high": float(gift["high"]),
            "low": float(gift["low"]),
            "prev_close": float(gift["close"]),
            "timestamp": gift.get("created_at", ""),
        }
    except Exception as e:
        log.error(f"Gift Nifty fetch failed: {e}")
        return {}


# ── NSE Options Chain (via nsepython) ──────────────────────────────

def scrape_options_chain() -> pd.DataFrame:
    """Fetch NSE options chain via nsepython (bypasses Akamai)."""
    global _cached_nse_spot
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            from nsepython import option_chain
            raw = option_chain("NIFTY")

            if raw is None or not isinstance(raw, dict) or "data" not in raw:
                log.error("nsepython returned empty data")
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                return pd.DataFrame()

            records = raw["data"]
            timestamp = raw.get("timestamp", "")

            expiry_dates = set()
            for rec in records:
                ce = rec.get("CE") or {}
                pe = rec.get("PE") or {}
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

            for rec in records:
                ce = rec.get("CE") or {}
                pe = rec.get("PE") or {}
                uv = ce.get("underlyingValue") or pe.get("underlyingValue")
                if uv and float(uv) > 0:
                    _cached_nse_spot = float(uv)
                    break

            rows = []
            for rec in records:
                ce = rec.get("CE") or {}
                pe = rec.get("PE") or {}
                if not ce and not pe:
                    continue

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
                         f"spot={_cached_nse_spot:.1f}, ts={timestamp})")
                today = date.today().strftime("%Y%m%d")
                save_path = PROJECT / "data" / f"nifty_options_chain_{today}.csv"
                df.to_csv(save_path, index=False)
            return df

        except Exception as e:
            log.error(f"Chain scrape attempt {attempt+1} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)

    return pd.DataFrame()


def load_chain_from_file() -> pd.DataFrame:
    """Fallback: load most recent chain CSV."""
    today = date.today().strftime("%Y%m%d")
    path = PROJECT / "data" / f"nifty_options_chain_{today}.csv"
    if path.exists():
        return pd.read_csv(path)
    from datetime import timedelta
    for days_back in range(1, 5):
        prev = (date.today() - timedelta(days=days_back)).strftime("%Y%m%d")
        path2 = PROJECT / "data" / f"nifty_options_chain_{prev}.csv"
        if path2.exists():
            log.warning(f"Using older chain: {path2.name}")
            return pd.read_csv(path2)
    return pd.DataFrame()


def compute_dte() -> int:
    """Days to expiry for current Nifty weekly (Monday expiry)."""
    today = date.today()
    weekday = today.weekday()
    if weekday == 0:
        return 0
    return (7 - weekday) % 7


# ── Session Detection ──────────────────────────────────────────────

def get_session(hhmm: int) -> str:
    """Determine current trading session."""
    if GIFT_SESSION1_START <= hhmm < NSE_OPEN:
        return "pre_market"
    elif NSE_OPEN <= hhmm <= NSE_CLOSE:
        return "market"
    elif GIFT_SESSION2_START <= hhmm <= 2359:
        return "post_market"
    elif 0 <= hhmm <= 245:  # 00:00 - 02:45 (session 2 spillover)
        return "post_market"
    else:
        return "closed"


def format_telegram_signal(signal, spot, session):
    """Format signal for Telegram HTML message."""
    action = signal["action"]
    if action == "NO_TRADE":
        return None

    lvl = signal["levels"]
    m = signal["metrics"]
    now = datetime.now().strftime("%H:%M")

    emoji = "🟢" if "CE" in action else "🔴" if "PE" in action else "🔵"

    session_tag = {
        "pre_market": "PRE-MKT (Gift Nifty)",
        "market": "MARKET",
        "post_market": "POST-MKT (Gift Nifty)",
    }.get(session, session.upper())

    lines = [
        f"{emoji} <b>GEX SIGNAL — {action}</b>",
        f"<b>[{session_tag}]</b>",
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


# ── Main Loop ──────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  GEX Live Scanner — S18 Options Strategy")
    log.info("  Extended hours via Gift Nifty")
    log.info("=" * 60)

    tg = TelegramNotifier(
        enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )

    if tg.enabled:
        tg.send_message(
            "🚀 <b>GEX Scanner Started (Extended Hours)</b>\n\n"
            "Pre-market: Gift Nifty + prev chain\n"
            "Market: NSE live chain\n"
            "Post-market: Gift Nifty + day's chain\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')} IST"
        )
        log.info("Telegram connected and startup message sent")
    else:
        log.warning("Telegram NOT enabled — signals will print to console only")

    strat = GEXOptionsStrategy()
    last_signal_time = 0
    last_signal_action = ""
    day_open = 0.0
    day_high = 0.0
    day_low = 99999.0
    chain_df = pd.DataFrame()
    chain_fetch_count = 0
    last_session = ""

    log.info(f"DTE: {compute_dte()} | Scan interval: {SCAN_INTERVAL}s")

    while True:
        now = datetime.now()
        hhmm = now.hour * 100 + now.minute
        session = get_session(hhmm)

        if session == "closed":
            if last_session and last_session != "closed":
                log.info("All sessions closed. Waiting...")
            last_session = session
            time.sleep(60)
            continue

        # Log session transitions
        if session != last_session:
            log.info(f">>> Session: {session.upper()}")
            if tg.enabled and last_session:
                tg.send_message(
                    f"🔄 <b>Session switch: {session.replace('_', ' ').upper()}</b>\n"
                    f"⏰ {now.strftime('%H:%M')} IST"
                )
            # Reset day tracking on pre-market start
            if session == "pre_market" and last_session in ("closed", "post_market", ""):
                day_open = 0.0
                day_high = 0.0
                day_low = 99999.0
                last_signal_time = 0
                last_signal_action = ""
                chain_fetch_count = 0
            last_session = session

        try:
            spot = 0.0

            if session == "market":
                # NSE hours: fetch live chain + NSE spot
                new_chain = scrape_options_chain()
                if not new_chain.empty:
                    chain_df = new_chain
                    chain_fetch_count += 1
                    spot = _cached_nse_spot
                elif chain_df.empty:
                    chain_df = load_chain_from_file()

            elif session in ("pre_market", "post_market"):
                # Gift Nifty hours: use Gift Nifty spot + stored/prev chain
                gift = get_gift_nifty_price()
                if gift:
                    spot = gift["ltp"]
                    log.info(f"Gift Nifty: {spot:.1f} (O={gift['open']:.1f} "
                             f"H={gift['high']:.1f} L={gift['low']:.1f})")

                # During market hours we already have chain_df from NSE.
                # Pre/post market: try to refresh chain (NSE may still serve stale data)
                # but primarily rely on the last saved chain.
                if chain_df.empty:
                    chain_df = load_chain_from_file()
                    if not chain_df.empty:
                        chain_fetch_count += 1

            if chain_df.empty:
                log.warning("No chain data available, retrying...")
                time.sleep(SCAN_INTERVAL)
                continue

            # If spot still 0, estimate from chain
            if spot == 0:
                mid_strike = chain_df["strike"].median()
                atm = chain_df.iloc[(chain_df["strike"] - mid_strike).abs().argsort()[:1]]
                if not atm.empty:
                    call_ltp = float(atm.iloc[0].get("call_ltp", 0))
                    put_ltp = float(atm.iloc[0].get("put_ltp", 0))
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
            gex = compute_gex_from_chain(chain_df, spot, dte)

            # For Gift Nifty sessions, use a virtual time within NSE's
            # allowed window so the strategy's time filters don't block.
            strat_hhmm = hhmm if session == "market" else 1000

            signal = strat.analyze(
                chain_df, spot, gex, dte,
                day_open, day_high, day_low, strat_hhmm
            )

            regime_short = signal["regime"][0].upper()
            flip = signal["levels"]["gamma_flip"]
            m = signal["metrics"]
            action = signal["action"]
            src = "GIFT" if session != "market" else "NSE"

            log.info(
                f"[{src}] Spot={spot:.1f} GEX={regime_short} Flip={flip:.0f}({spot-flip:+.0f}) "
                f"Flow={m['flow']} Move={m['intraday_move']:+.0f}({m['move_pct_of_range']:.0%}) "
                f"-> {action}"
            )

            if action != "NO_TRADE":
                minutes_since = (hhmm - last_signal_time) if last_signal_time else 999
                h_diff = minutes_since // 100
                m_diff = minutes_since % 100
                actual_minutes = h_diff * 60 + m_diff

                is_same = (action == last_signal_action)
                cooldown_ok = actual_minutes >= COOLDOWN_MINUTES or not is_same

                if cooldown_ok:
                    log.info(f"*** SIGNAL: {action} {signal['strike']:.0f} "
                             f"conf={signal['confidence']}% [{src}] ***")

                    print("\n" + format_signal(signal) + "\n")

                    msg = format_telegram_signal(signal, spot, session)
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
