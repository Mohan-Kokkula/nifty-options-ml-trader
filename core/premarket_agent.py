"""
premarket_agent.py — Daily pre-market regime briefing.

Runs once per day at 08:30 IST (configurable). Gathers overnight market data,
asks a free LLM (Groq Llama 3.3 70B) to classify the trading regime, and
writes a structured plan that the pilot reads at session start.

Output JSON written to data/premarket_brief_YYYY-MM-DD.json:
{
    "date": "2026-04-25",
    "regime": "trend_up" | "trend_down" | "choppy" | "event_day" | "low_vol",
    "bias": "bullish" | "bearish" | "neutral",
    "risk_appetite": 0.0..1.0,
    "max_trades_today": int,
    "min_confidence_boost": int (-10..+15),  # add to pilot.min_confidence
    "avoid_otm2_otm3": bool,
    "key_levels": {"support": float, "resistance": float},
    "rationale": "short text"
}

Pilot reads this at start of trading day and adapts:
    - effective_min_conf  += min_confidence_boost
    - max_trades_per_day   = max_trades_today
    - allowed_strikes      = drop OTM2/3 if avoid_otm2_otm3

Fail-safe: any error → no brief written → pilot uses default config.

LLM provider: NEWS_LLM_PROVIDER env var also drives this module.
    groq      → free, fast (recommended, default)
    gemini    → free, generous
    heuristic → no LLM, rule-based regime tagging
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


BRIEF_DIR = Path("data")
BRIEF_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_BRIEF_TIME = "08:30"   # IST


# ──────────────────────────────────────────────────────────────────────
# Data collection (deterministic — no LLM)
# ──────────────────────────────────────────────────────────────────────

def _safe_yf(ticker: str, period: str = "5d") -> dict:
    """Return last close + change% for a yfinance ticker. {} on failure."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        h = t.history(period=period, interval="1d")
        if len(h) < 2:
            return {}
        last = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2])
        chg_pct = (last - prev) / prev * 100.0
        return {"last": round(last, 2), "change_pct": round(chg_pct, 2)}
    except Exception as e:
        logger.debug("yfinance %s failed: %s", ticker, e)
        return {}


def _fetch_global_indices() -> dict:
    """Overnight global market snapshot."""
    return {
        "dow":         _safe_yf("^DJI"),
        "nasdaq":      _safe_yf("^IXIC"),
        "sp500":       _safe_yf("^GSPC"),
        "nikkei":      _safe_yf("^N225"),
        "hangseng":    _safe_yf("^HSI"),
        "ftse":        _safe_yf("^FTSE"),
        "dax":         _safe_yf("^GDAXI"),
        "crude":       _safe_yf("CL=F"),
        "dxy":         _safe_yf("DX-Y.NYB"),
        "us10y":       _safe_yf("^TNX"),
        "gift_nifty":  _safe_yf("^NSEI"),
        "india_vix":   _safe_yf("^INDIAVIX"),
    }


def _fetch_fii_dii() -> dict:
    """NSE FII/DII activity for previous trading day."""
    try:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (OpenClawBot/1.0)"
        s.headers["Accept"] = "application/json"
        s.headers["Referer"] = "https://www.nseindia.com/"
        s.get("https://www.nseindia.com/", timeout=8)
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        out = {}
        for row in data:
            cat = row.get("category", "").upper()
            if cat in ("FII", "DII", "FII**", "DII**"):
                out[cat.replace("**", "")] = {
                    "buy": float(row.get("buyValue", 0)),
                    "sell": float(row.get("sellValue", 0)),
                    "net": float(row.get("netValue", 0)),
                }
        return out
    except Exception as e:
        logger.debug("FII/DII fetch failed: %s", e)
        return {}


def _fetch_news_bias() -> dict:
    """Read overnight bias from already-running news_agent."""
    try:
        from core.news_agent import get_news_agent
        b = get_news_agent().current_bias() or {}
        # Strip non-serializable hot_event object
        return {
            "bias_score": b.get("bias", 0.0),
            "confidence": b.get("confidence", 0.0),
            "event_count": b.get("event_count", 0),
            "dominant_type": b.get("dominant_type", ""),
            "hot_title": (getattr(b.get("hot_event"), "title", "") or "")[:140],
        }
    except Exception as e:
        logger.debug("news_agent bias fetch failed: %s", e)
        return {}


def _fetch_event_calendar() -> list[str]:
    """Hardcoded high-impact event tags for today (cheap, reliable)."""
    today = datetime.now().date()
    flags = []
    # Indian expiry — Tuesday is current weekly Nifty expiry (since Aug 2024)
    if today.weekday() == 1:  # Tue
        flags.append("nifty_weekly_expiry")
    if today.weekday() == 3:  # Thu
        flags.append("banknifty_weekly_expiry")
    # Last Thursday of month → monthly expiry
    nxt = today + timedelta(days=7)
    if today.weekday() == 3 and nxt.month != today.month:
        flags.append("monthly_expiry")
    return flags


# ──────────────────────────────────────────────────────────────────────
# LLM scorer (Groq default, free)
# ──────────────────────────────────────────────────────────────────────

def _classify_with_groq(payload: dict) -> Optional[dict]:
    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq SDK not installed — pip install groq")
        return None
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — pre-market agent falling back")
        return None

    prompt = f"""You are a senior NSE-derivatives strategist briefing a Nifty options
trading bot. Analyze this overnight snapshot and classify today's likely regime.

OVERNIGHT DATA:
{json.dumps(payload, indent=2)}

Return STRICT JSON (no prose, no markdown fences) with these exact keys:
{{
  "regime": "trend_up" | "trend_down" | "choppy" | "event_day" | "low_vol",
  "bias": "bullish" | "bearish" | "neutral",
  "risk_appetite": 0.0 to 1.0,
  "max_trades_today": integer 1-8,
  "min_confidence_boost": integer -5 to +5,
  "avoid_otm2_otm3": true | false,
  "key_levels": {{"support": float, "resistance": float}},
  "rationale": "one paragraph explaining your call"
}}

Rules (CONSERVATIVE — boost is added on top of VIX adjustments, keep it small):
- If global indices flat/mixed AND VIX<14 → "low_vol", risk_appetite 0.4, max_trades 3, boost 0
- If global indices strongly down (-1%+) AND DXY up → "trend_down" "bearish", boost +2
- If global indices strongly up (+0.8%+) AND VIX falling → "trend_up" "bullish", boost 0
- If FII net sold > 2000 cr AND news bearish → "bearish", avoid_otm2_otm3=true, boost +2
- If event_calendar has expiry → boost +3 (theta day = trap risk)
- If VIX > 18 → "event_day", risk_appetite 0.5, max_trades 3, boost +3
- key_levels: estimate from gift_nifty last + ATR-like ±100/200 pts
- DEFAULT boost: 0 (zero) — only add positive boost for clear bearish/event setups
- NEVER set boost above +5 — VIX gate already handles high vol
"""
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        logger.warning("Groq classification failed: %s", e)
        return None


def _classify_heuristic(payload: dict) -> dict:
    """Pure-rules fallback. No LLM."""
    g = payload.get("global", {})
    vix = (g.get("india_vix") or {}).get("last", 15.0)
    dow_chg = (g.get("dow") or {}).get("change_pct", 0.0)
    nas_chg = (g.get("nasdaq") or {}).get("change_pct", 0.0)
    asia_avg = sum(
        (g.get(k) or {}).get("change_pct", 0.0) for k in ("nikkei", "hangseng")
    ) / 2.0
    overnight = (dow_chg + nas_chg) / 2.0

    fii_net = (payload.get("fii_dii", {}).get("FII") or {}).get("net", 0.0)
    news_bias = (payload.get("news") or {}).get("bias_score", 0.0)
    events = payload.get("events", [])

    # Regime — conservative boosts (capped +5; +3 typical)
    if vix >= 18:
        regime, bias, risk, max_t, boost = "event_day", "neutral", 0.5, 3, 3
    elif vix < 12 and abs(overnight) < 0.3:
        regime, bias, risk, max_t, boost = "low_vol", "neutral", 0.4, 3, 0
    elif overnight <= -1.0 or (asia_avg <= -1.0 and news_bias <= -0.3):
        regime, bias, risk, max_t, boost = "trend_down", "bearish", 0.7, 4, 0
    elif overnight >= 0.8 and news_bias >= 0.0:
        regime, bias, risk, max_t, boost = "trend_up", "bullish", 0.7, 4, 0
    else:
        regime, bias, risk, max_t, boost = "choppy", "neutral", 0.5, 3, 2

    # FII override
    if fii_net <= -2000:
        bias = "bearish"
        boost = min(5, max(boost, 2))
    elif fii_net >= 2000:
        bias = "bullish"

    avoid_otm = regime in ("low_vol", "event_day", "choppy") or "expiry" in str(events)

    gift = (g.get("gift_nifty") or {}).get("last", 0.0)
    return {
        "regime": regime,
        "bias": bias,
        "risk_appetite": risk,
        "max_trades_today": max_t,
        "min_confidence_boost": boost,
        "avoid_otm2_otm3": avoid_otm,
        "key_levels": {
            "support":    round(gift - 150, 2) if gift else 0.0,
            "resistance": round(gift + 150, 2) if gift else 0.0,
        },
        "rationale": (
            f"Heuristic: VIX={vix:.1f}, US overnight={overnight:+.2f}%, "
            f"Asia={asia_avg:+.2f}%, FII net={fii_net:+.0f}cr, "
            f"news_bias={news_bias:+.2f}, events={events}"
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def generate_brief(force: bool = False) -> dict:
    """
    Build today's pre-market brief and write to data/premarket_brief_YYYY-MM-DD.json.
    Returns the brief dict. Cached — re-running same day returns cached version
    unless force=True.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = BRIEF_DIR / f"premarket_brief_{today}.json"
    if out_path.exists() and not force:
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass

    logger.info("📋 Pre-market agent: collecting overnight data…")
    t0 = time.time()
    payload = {
        "date": today,
        "global":   _fetch_global_indices(),
        "fii_dii":  _fetch_fii_dii(),
        "news":     _fetch_news_bias(),
        "events":   _fetch_event_calendar(),
    }
    logger.info("📋 Data collected in %.1fs", time.time() - t0)

    provider = os.getenv("PREMARKET_PROVIDER", os.getenv("NEWS_LLM_PROVIDER", "groq")).lower()
    classification = None
    if provider == "groq":
        classification = _classify_with_groq(payload)
    if classification is None:
        logger.info("📋 Using heuristic regime classifier")
        classification = _classify_heuristic(payload)

    brief = {"date": today, **classification, "_payload": payload}
    out_path.write_text(json.dumps(brief, indent=2, default=str))
    logger.warning(
        "📋 PRE-MARKET BRIEF [%s] regime=%s bias=%s risk=%.1f max_trades=%d "
        "boost=%+d avoid_otm23=%s",
        today, brief.get("regime"), brief.get("bias"),
        brief.get("risk_appetite", 0), brief.get("max_trades_today", 0),
        brief.get("min_confidence_boost", 0), brief.get("avoid_otm2_otm3"),
    )
    logger.info("📋 Rationale: %s", brief.get("rationale", "")[:300])
    return brief


def get_today_brief() -> Optional[dict]:
    """Return today's brief if it exists. None otherwise. Pilot calls this."""
    today = datetime.now().strftime("%Y-%m-%d")
    p = BRIEF_DIR / f"premarket_brief_{today}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def schedule_daily_brief():
    """
    Background daemon. Generates brief at PREMARKET_BRIEF_TIME each weekday.
    Also generates immediately on startup if the brief for today doesn't exist
    and we're past the configured brief time.
    """
    import threading

    def loop():
        while True:
            try:
                now = datetime.now()
                # Skip weekends
                if now.weekday() >= 5:
                    time.sleep(3600)
                    continue
                brief_hhmm = os.getenv("PREMARKET_BRIEF_TIME", DEFAULT_BRIEF_TIME)
                target_h, target_m = map(int, brief_hhmm.split(":"))
                target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)

                # If past target and no brief yet → generate now
                if now >= target and get_today_brief() is None:
                    generate_brief()
                    time.sleep(60)
                    continue

                # If before target → sleep until target
                if now < target:
                    wait = (target - now).total_seconds()
                    logger.info("📋 Pre-market agent sleeping %.0fs until %s",
                                wait, brief_hhmm)
                    time.sleep(min(wait, 1800))
                    continue

                # Brief already done for today → sleep until tomorrow morning
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=target_h, minute=target_m, second=0, microsecond=0
                )
                wait = (tomorrow - now).total_seconds()
                time.sleep(min(wait, 3600))
            except Exception as e:
                logger.warning("PreMarket scheduler error: %s", e)
                time.sleep(300)

    t = threading.Thread(target=loop, daemon=True, name="PreMarketAgent")
    t.start()
    logger.info("📋 PreMarketAgent scheduler started")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    brief = generate_brief(force=True)
    print(json.dumps(brief, indent=2, default=str))
