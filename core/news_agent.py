"""
news_agent.py — 24/7 market news monitor for Indian equity & derivatives.

Polls RSS feeds (NSE, SEBI, MoneyControl, Economic Times) + RBI/BSE press
releases, scores each headline with Claude for:
    - sentiment (-1.0 bearish → +1.0 bullish)
    - market_impact (NIFTY, BANKNIFTY, sector, stock-specific, macro)
    - urgency (breaking / material / routine)
    - event_type (results, policy, SEBI-action, block-deal, etc.)

Maintains a rolling in-memory state that the bot queries via
`news_agent.current_bias()` before each entry decision.

Design goals:
  • Fail-safe: agent down → pilot unaffected (bias=0, confidence=0)
  • Low LLM cost: cheap sentiment model, dedup by URL, cache 24h
  • Low latency: background thread, pilot never blocks on LLM call

Run:  started automatically by main.py when NEWS_AGENT_ENABLED=true
Test: python -m core.news_agent
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

FEEDS: dict[str, str] = {
    # Active RSS feeds (no auth required)
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "moneycontrol_latest":   "https://www.moneycontrol.com/rss/latestnews.xml",
    "et_markets":            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks":             "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets":      "https://www.livemint.com/rss/markets",
    "bs_markets":            "https://www.business-standard.com/rss/markets-106.rss",
    # NSE announcements (corporate + regulatory) — JSON endpoint, handled separately
    # SEBI press releases — scraped separately (no RSS)
}

POLL_INTERVAL_SECS = int(os.getenv("NEWS_POLL_INTERVAL", "180"))  # 3 min
ROLLING_WINDOW_MIN = int(os.getenv("NEWS_WINDOW_MIN", "60"))      # keep 60 min of events
MAX_EVENTS_RETAINED = 400
HIGH_IMPACT_MINUTES = 10   # within N min of event, apply strong bias


# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    uid: str
    source: str
    title: str
    url: str
    published_ts: float
    ingested_ts: float
    # Scored by LLM (None until scored)
    sentiment: Optional[float] = None       # -1.0 .. +1.0
    impact_scope: Optional[str]  = None     # "nifty" | "banknifty" | "sector" | "stock" | "macro" | "none"
    urgency: Optional[str]       = None     # "breaking" | "material" | "routine"
    event_type: Optional[str]    = None     # "results", "sebi-action", "policy", "block-deal", ...
    confidence: Optional[float]  = None     # 0..1 LLM's own confidence

    def age_minutes(self) -> float:
        return (time.time() - self.ingested_ts) / 60.0


# ──────────────────────────────────────────────────────────────────────
# NewsAgent
# ──────────────────────────────────────────────────────────────────────

class NewsAgent:
    """
    Singleton background news-monitor.

    Public API (called by pilot):
        current_bias()  → dict with direction bias, confidence, hot events
        recent_events() → last N scored events
        is_healthy()    → agent running OK
    """

    _instance: Optional["NewsAgent"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "NewsAgent":
        with cls._lock:
            if cls._instance is None:
                cls._instance = NewsAgent()
            return cls._instance

    def __init__(self):
        self._events: dict[str, NewsEvent] = {}   # uid → event
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_ts: float = 0.0
        self._consecutive_errors: int = 0
        self._total_fetched: int = 0
        self._total_scored: int = 0
        # Scoring backends — all lazy
        self._claude = None
        self._finbert = None       # transformers pipeline
        self._finbert_failed = False
        self.provider = os.getenv("NEWS_LLM_PROVIDER", "finbert").lower()
        self.poll_interval = POLL_INTERVAL_SECS
        # Optional: notification manager for Telegram breaking-news alerts
        self._notifier = None
        self._alerted_uids: set[str] = set()   # avoid duplicate alerts
        # Set up dedicated log file
        self._setup_log_file()

    @staticmethod
    def _setup_log_file():
        """Send all news_agent logs to logs/news_agent.log (separate from bot log)."""
        try:
            from logging.handlers import RotatingFileHandler
            from pathlib import Path
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True, parents=True)
            log_path = log_dir / "news_agent.log"

            # Avoid adding duplicate handler on re-init
            if any(
                isinstance(h, RotatingFileHandler) and
                getattr(h, "baseFilename", "").endswith("news_agent.log")
                for h in logger.handlers
            ):
                return

            handler = RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,   # 5 MB per file
                backupCount=7,               # keep 7 rotated files
                encoding="utf-8",
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(handler)
            # Stop news_agent logs from also going to root/main log
            logger.propagate = False
            logger.info(f"📝 NewsAgent logs → {log_path}")
        except Exception as e:
            # If file logging fails, fall back to console (don't crash agent)
            logger.warning(f"NewsAgent file logger setup failed: {e}")

    def set_notifier(self, notifier) -> None:
        """Attach a NotificationManager (or any object with notify_news())."""
        self._notifier = notifier

    # ──────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="NewsAgent"
        )
        self._thread.start()
        logger.info("📰 NewsAgent started — polling every %ds | %d feeds",
                    POLL_INTERVAL_SECS, len(FEEDS))

    def stop(self):
        self._running = False

    def is_healthy(self) -> bool:
        if not self._running:
            return False
        if self._consecutive_errors >= 5:
            return False
        if self._last_poll_ts and (time.time() - self._last_poll_ts) > 3 * POLL_INTERVAL_SECS:
            return False
        return True

    # ──────────────────────────────────────────────────────────────
    # Public query API (pilot calls these)
    # ──────────────────────────────────────────────────────────────

    def current_bias(self) -> dict:
        """
        Aggregate bias from scored events within the rolling window.

        Returns:
            {
                "bias": float,           # -1.0 (bearish) .. +1.0 (bullish)
                "confidence": float,     # 0..1 — how loaded the signal is
                "hot_event": NewsEvent or None,  # strongest recent event
                "event_count": int,
                "dominant_type": str,    # most common event_type in window
            }
        """
        empty = {"bias": 0.0, "confidence": 0.0, "hot_event": None,
                 "event_count": 0, "dominant_type": ""}
        try:
            with self._state_lock:
                window_start = time.time() - ROLLING_WINDOW_MIN * 60
                # Only consider scored events with NIFTY/macro impact
                relevant = [
                    e for e in self._events.values()
                    if e.ingested_ts >= window_start
                    and e.sentiment is not None
                    and e.impact_scope in ("nifty", "banknifty", "macro", "sector")
                ]
                if not relevant:
                    return empty

                # Weight: recency (linear decay) × urgency × impact
                total_weight = 0.0
                weighted_sum = 0.0
                for e in relevant:
                    age_min = e.age_minutes()
                    recency_w = max(0.0, 1.0 - age_min / ROLLING_WINDOW_MIN)
                    urgency_w = {"breaking": 1.0, "material": 0.6, "routine": 0.25}.get(
                        e.urgency or "routine", 0.25
                    )
                    impact_w = {"nifty": 1.0, "banknifty": 0.8, "macro": 0.9,
                                "sector": 0.4}.get(e.impact_scope, 0.2)
                    w = recency_w * urgency_w * impact_w
                    weighted_sum += (e.sentiment or 0.0) * w
                    total_weight += w

                bias = weighted_sum / total_weight if total_weight > 0 else 0.0
                # Confidence = fraction of max possible weight
                max_weight = len(relevant) * 1.0  # if every event were breaking+nifty+fresh
                confidence = min(1.0, total_weight / max(1.0, max_weight))

                # Hot event = strongest |sentiment| × urgency × fresh
                hot = max(
                    relevant,
                    key=lambda e: abs(e.sentiment or 0.0) *
                                   {"breaking": 1.0, "material": 0.6, "routine": 0.2}.get(e.urgency or "routine", 0.2) *
                                   max(0.1, 1.0 - e.age_minutes() / ROLLING_WINDOW_MIN),
                )

                # Dominant event type
                type_counts: dict[str, int] = {}
                for e in relevant:
                    t = e.event_type or "unknown"
                    type_counts[t] = type_counts.get(t, 0) + 1
                dominant = max(type_counts, key=type_counts.get) if type_counts else ""

                return {
                    "bias": round(bias, 3),
                    "confidence": round(confidence, 3),
                    "hot_event": hot,
                    "event_count": len(relevant),
                    "dominant_type": dominant,
                }
        except Exception as e:
            logger.debug("news_agent.current_bias() failed: %s", e)
            return empty

    def recent_events(self, limit: int = 20) -> list[NewsEvent]:
        with self._state_lock:
            events = sorted(
                [e for e in self._events.values() if e.sentiment is not None],
                key=lambda e: e.ingested_ts,
                reverse=True,
            )
            return events[:limit]

    def stats(self) -> dict:
        return {
            "running": self._running,
            "healthy": self.is_healthy(),
            "total_fetched": self._total_fetched,
            "total_scored": self._total_scored,
            "retained": len(self._events),
            "consecutive_errors": self._consecutive_errors,
            "last_poll_age_sec": round(time.time() - self._last_poll_ts, 1) if self._last_poll_ts else -1,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal: polling loop
    # ──────────────────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            try:
                self._poll_once()
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                logger.warning("NewsAgent poll failed (%d): %s",
                               self._consecutive_errors, e)
            time.sleep(POLL_INTERVAL_SECS)

    def _poll_once(self):
        new_items: list[NewsEvent] = []

        # 1) RSS feeds
        for src, url in FEEDS.items():
            try:
                items = self._fetch_rss(src, url)
                new_items.extend(items)
            except Exception as e:
                logger.debug("RSS %s failed: %s", src, e)

        # 2) NSE corporate announcements
        try:
            new_items.extend(self._fetch_nse_announcements())
        except Exception as e:
            logger.debug("NSE announcements failed: %s", e)

        # 3) SEBI press releases (light scrape)
        try:
            new_items.extend(self._fetch_sebi_press())
        except Exception as e:
            logger.debug("SEBI press failed: %s", e)

        # Dedup + store
        added = 0
        with self._state_lock:
            for ev in new_items:
                if ev.uid in self._events:
                    continue
                self._events[ev.uid] = ev
                added += 1
            # Evict old
            if len(self._events) > MAX_EVENTS_RETAINED:
                cutoff = time.time() - ROLLING_WINDOW_MIN * 60 * 2  # 2× window
                self._events = {
                    uid: e for uid, e in self._events.items()
                    if e.ingested_ts >= cutoff
                }

        self._total_fetched += added
        self._last_poll_ts = time.time()
        if added > 0:
            logger.info("📰 NewsAgent: +%d new items (retained=%d)",
                        added, len(self._events))
            # Score in background so we don't hold the poll loop
            threading.Thread(
                target=self._score_unscored, daemon=True,
                name="NewsAgent-Scorer",
            ).start()

    def _fetch_rss(self, source: str, url: str) -> list[NewsEvent]:
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed — RSS disabled. pip install feedparser")
            return []
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (OpenClawBot/1.0)"}
            r = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(r.content)
        except Exception:
            feed = feedparser.parse(url)

        out: list[NewsEvent] = []
        for entry in feed.entries[:30]:
            title = (getattr(entry, "title", "") or "").strip()
            link  = (getattr(entry, "link", "") or "").strip()
            if not title or not link:
                continue
            pub_ts = self._entry_time(entry)
            uid = hashlib.sha1((link or title).encode("utf-8")).hexdigest()[:16]
            out.append(NewsEvent(
                uid=uid, source=source, title=title, url=link,
                published_ts=pub_ts, ingested_ts=time.time(),
            ))
        return out

    def _fetch_nse_announcements(self) -> list[NewsEvent]:
        """NSE corporate announcements JSON endpoint."""
        try:
            import requests
        except ImportError:
            return []
        out: list[NewsEvent] = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            }
            sess = requests.Session()
            sess.get("https://www.nseindia.com/", headers=headers, timeout=8)
            r = sess.get(
                "https://www.nseindia.com/api/corporate-announcements?index=equities",
                headers=headers, timeout=10,
            )
            if r.status_code != 200:
                return []
            data = r.json() if r.content else []
            if not isinstance(data, list):
                data = data.get("data", []) if isinstance(data, dict) else []
            for row in data[:40]:
                title = row.get("subject") or row.get("desc") or ""
                symbol = row.get("symbol", "")
                link = row.get("attchmntFile", "") or ""
                if not title:
                    continue
                full_title = f"[{symbol}] {title}" if symbol else title
                uid = hashlib.sha1((link or full_title).encode("utf-8")).hexdigest()[:16]
                out.append(NewsEvent(
                    uid=uid, source="nse_announcements", title=full_title,
                    url=link, published_ts=time.time(), ingested_ts=time.time(),
                ))
        except Exception as e:
            logger.debug("NSE announcement fetch: %s", e)
        return out

    def _fetch_sebi_press(self) -> list[NewsEvent]:
        """SEBI press releases — lightweight scrape of listing page."""
        try:
            import requests
            from html.parser import HTMLParser
        except ImportError:
            return []
        out: list[NewsEvent] = []
        try:
            url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7"
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return []
            # Very light extraction: look for <a href=...>TITLE</a> patterns
            # around <td class="tdvalue"> blocks
            import re
            pattern = re.compile(
                r'<a[^>]*href="([^"]+)"[^>]*>([^<]{20,250})</a>', re.IGNORECASE
            )
            for m in pattern.finditer(r.text):
                href, title = m.group(1), m.group(2).strip()
                if any(kw in title.lower() for kw in (
                    "order", "circular", "press release", "notice", "penalty",
                    "direction", "settlement", "consent", "regulation", "amendment",
                )):
                    if href.startswith("/"):
                        href = "https://www.sebi.gov.in" + href
                    uid = hashlib.sha1((href + title).encode("utf-8")).hexdigest()[:16]
                    out.append(NewsEvent(
                        uid=uid, source="sebi", title=title, url=href,
                        published_ts=time.time(), ingested_ts=time.time(),
                    ))
                if len(out) >= 20:
                    break
        except Exception as e:
            logger.debug("SEBI scrape: %s", e)
        return out

    @staticmethod
    def _entry_time(entry) -> float:
        for attr in ("published_parsed", "updated_parsed"):
            t = getattr(entry, attr, None)
            if t:
                try:
                    return time.mktime(t)
                except Exception:
                    pass
        return time.time()

    # ──────────────────────────────────────────────────────────────
    # LLM scoring
    # ──────────────────────────────────────────────────────────────

    # ── FinBERT (local, free, offline) ────────────────────────────
    def _get_finbert(self):
        """Lazy-load ProsusAI/finbert. Downloads ~420MB on first call."""
        if self._finbert is not None or self._finbert_failed:
            return self._finbert
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification, pipeline,
            )
            import torch  # noqa: F401 — required by transformers
            model_name = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
            logger.info("📰 Loading FinBERT (%s) — first run downloads ~420MB…", model_name)
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._finbert = pipeline(
                "text-classification",
                model=mdl, tokenizer=tok,
                top_k=None,   # return all 3 class probs
                truncation=True, max_length=256,
            )
            logger.info("📰 FinBERT ready")
            return self._finbert
        except Exception as e:
            self._finbert_failed = True
            logger.warning("FinBERT load failed — falling back to heuristic. %s", e)
            return None

    def _score_with_finbert(self, pending: list[NewsEvent]) -> bool:
        """Score each event using FinBERT + keyword meta-tagging. Returns True on success."""
        pipe = self._get_finbert()
        if pipe is None:
            return False
        try:
            texts = [e.title[:250] for e in pending]
            results = pipe(texts)
            # pipeline returns list-of-list-of-dicts with top_k=None
            with self._state_lock:
                for ev, scores in zip(pending, results):
                    # scores = [{'label':'positive','score':0.xx}, ...]
                    pos = neg = neu = 0.0
                    for s in scores:
                        lbl = s["label"].lower()
                        if lbl == "positive": pos = s["score"]
                        elif lbl == "negative": neg = s["score"]
                        elif lbl == "neutral": neu = s["score"]
                    sentiment = pos - neg                         # [-1..+1]
                    fb_conf = max(pos, neg, neu)
                    # Meta-tags from keyword heuristic
                    impact, urgency, etype = self._classify_meta(ev.title)
                    ev.sentiment    = round(sentiment, 3)
                    ev.impact_scope = impact
                    ev.urgency      = urgency
                    ev.event_type   = etype
                    ev.confidence   = round(float(fb_conf), 3)
                    self._total_scored += 1
            # Breaking alerts (log + Telegram)
            for ev in pending:
                self._maybe_alert(ev, threshold=0.4)
            return True
        except Exception as e:
            logger.warning("FinBERT scoring failed (%s) — using heuristic", e)
            return False

    @staticmethod
    def _classify_meta(title: str) -> tuple[str, str, str]:
        """Keyword-based impact_scope / urgency / event_type tagging."""
        t = (title or "").lower()

        # impact_scope
        if any(k in t for k in ("banknifty", "bank nifty", "hdfc bank", "icici bank",
                                 "sbi ", "axis bank", "kotak bank", "rbl bank")):
            impact = "banknifty"
        elif any(k in t for k in ("nifty", "sensex", "nse ", "bse ", "fii", "dii",
                                   "sgx nifty", "gift nifty")):
            impact = "nifty"
        elif any(k in t for k in ("rbi", "sebi", "fed ", "fomc", "inflation", "cpi",
                                   "wpi", "gdp", "budget", "crude", "rupee", "dollar",
                                   "repo rate", "monetary policy", "iip", "pmi",
                                   "ukraine", "china", "israel", "gaza", "war")):
            impact = "macro"
        elif any(k in t for k in ("pharma", "it sector", "metals", "auto", "banking",
                                   "fmcg", "realty", "psu", "oil & gas", "telecom")):
            impact = "sector"
        elif any(k in t for k in ("ltd", "limited", " q1 ", " q2 ", " q3 ", " q4 ",
                                   "results", "block deal", "bulk deal")):
            impact = "stock"
        else:
            impact = "none"

        # urgency
        breaking_kw = ("breaking", "emergency", "just in", "crash", "plunge",
                       "halts", "circuit", "war declared", "attack", "bombing",
                       "resigns", "bankruptcy", "default", "fraud", "raid",
                       "freezes", "suspend", "fed cuts", "fed hikes", "rbi cuts",
                       "rbi hikes", "surprise")
        material_kw = ("rbi", "sebi", "fed", "fomc", "budget", "policy",
                       "downgrade", "upgrade", "probe", "penalty", "block deal",
                       "bulk deal", "qip", "ipo", "results", "guidance",
                       "order", "circular", "fii sell", "fii buy")
        if any(k in t for k in breaking_kw):
            urgency = "breaking"
        elif any(k in t for k in material_kw):
            urgency = "material"
        else:
            urgency = "routine"

        # event_type
        if any(k in t for k in ("rbi", "repo", "monetary policy", "mpc")):
            etype = "rbi-policy"
        elif "sebi" in t or "penalty" in t or "probe" in t:
            etype = "sebi-action"
        elif any(k in t for k in ("fed", "fomc", "us cpi", "nonfarm")):
            etype = "fed-macro"
        elif any(k in t for k in ("results", " q1 ", " q2 ", " q3 ", " q4 ")):
            etype = "results"
        elif "block deal" in t or "bulk deal" in t:
            etype = "block-deal"
        elif "downgrade" in t:
            etype = "downgrade"
        elif "upgrade" in t:
            etype = "upgrade"
        elif any(k in t for k in ("war", "attack", "strike", "geopolitic")):
            etype = "geopolitics"
        elif any(k in t for k in ("inflation", "cpi", "wpi", "gdp", "iip", "pmi")):
            etype = "macro-data"
        elif any(k in t for k in ("crude", "oil", "opec")):
            etype = "commodity"
        elif any(k in t for k in ("fii", "dii")):
            etype = "flows"
        else:
            etype = "other"

        return impact, urgency, etype

    def _maybe_alert(self, ev: NewsEvent, threshold: float = 0.4) -> None:
        """Log + Telegram-push high-impact events. De-duped by uid."""
        try:
            if (ev.urgency != "breaking" or
                ev.impact_scope not in ("nifty", "banknifty", "macro") or
                abs(ev.sentiment or 0) < threshold):
                return
            if ev.uid in self._alerted_uids:
                return
            self._alerted_uids.add(ev.uid)
            # Cap memory: drop oldest 100 once we exceed 500
            if len(self._alerted_uids) > 500:
                self._alerted_uids = set(list(self._alerted_uids)[-400:])

            logger.warning(
                "📰🚨 BREAKING %s: [%s] sent=%+.2f | %s",
                ev.impact_scope.upper(), ev.source, ev.sentiment, ev.title[:120]
            )
            if self._notifier and hasattr(self._notifier, "notify_news"):
                try:
                    self._notifier.notify_news(
                        title=ev.title,
                        source=ev.source,
                        sentiment=ev.sentiment or 0.0,
                        impact=ev.impact_scope or "macro",
                        urgency=ev.urgency or "breaking",
                        url=ev.url or "",
                    )
                except Exception as e:
                    logger.debug("notify_news failed: %s", e)
        except Exception as e:
            logger.debug("_maybe_alert error: %s", e)

    def _get_claude(self):
        if self._claude is not None:
            return self._claude
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                return None
            self._claude = Anthropic(api_key=api_key)
            return self._claude
        except Exception as e:
            logger.debug("Anthropic SDK unavailable: %s", e)
            return None

    def _score_unscored(self):
        """Batch-score up to 20 unscored events per call. Safe to re-enter."""
        with self._state_lock:
            pending = [
                e for e in self._events.values()
                if e.sentiment is None
            ][:20]
        if not pending:
            return

        # ── Provider dispatch ────────────────────────────────────────
        # NEWS_LLM_PROVIDER: finbert (default, local) | anthropic | heuristic
        provider = (self.provider or "finbert").lower()

        if provider == "finbert":
            if self._score_with_finbert(pending):
                return
            # FinBERT failed → heuristic (never touches Claude unless configured)
            for ev in pending:
                self._heuristic_score(ev)
            return

        if provider == "heuristic":
            for ev in pending:
                self._heuristic_score(ev)
            return

        # provider == "anthropic" (legacy)
        client = self._get_claude()
        if client is None:
            for ev in pending:
                self._heuristic_score(ev)
            return

        # Build batch prompt
        lines = "\n".join(
            f"{i+1}. [{e.source}] {e.title[:250]}"
            for i, e in enumerate(pending)
        )
        prompt = f"""You are a professional market-news classifier for Indian equity
derivatives (NIFTY, BANKNIFTY). Score each headline below with:

- sentiment: float -1.0 (very bearish for NIFTY) to +1.0 (very bullish)
- impact_scope: one of "nifty", "banknifty", "sector", "stock", "macro", "none"
- urgency: one of "breaking", "material", "routine"
- event_type: short tag e.g. "results", "policy", "sebi-action", "block-deal",
  "downgrade", "upgrade", "macro-data", "geopolitics", "other"
- confidence: 0..1 your confidence in this classification

Return STRICT JSON array, one object per headline in order, with only these keys.
No prose, no markdown fences.

Headlines:
{lines}
"""
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Strip fences if any
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            scored = json.loads(text)
            if not isinstance(scored, list):
                raise ValueError("LLM did not return a list")
        except Exception as e:
            logger.debug("LLM scoring failed (%s) — falling back to heuristic", e)
            for ev in pending:
                self._heuristic_score(ev)
            return

        with self._state_lock:
            for ev, s in zip(pending, scored):
                if not isinstance(s, dict):
                    continue
                try:
                    ev.sentiment    = float(s.get("sentiment", 0.0))
                    ev.impact_scope = str(s.get("impact_scope", "none")).lower()
                    ev.urgency      = str(s.get("urgency", "routine")).lower()
                    ev.event_type   = str(s.get("event_type", "other")).lower()
                    ev.confidence   = float(s.get("confidence", 0.5))
                    self._total_scored += 1
                except Exception:
                    self._heuristic_score(ev)

        # Alert on breaking events
        for ev in pending:
            self._maybe_alert(ev, threshold=0.5)

    def _heuristic_score(self, ev: NewsEvent):
        """Keyword fallback when LLM/FinBERT unavailable."""
        title_l = ev.title.lower()
        bullish = ("surge", "rally", "gain", "rise", "jump", "upgrade", "beat",
                   "record high", "strong", "positive", "rate cut", "stimulus",
                   "boost", "outperform", "bullish", "robust")
        bearish = ("crash", "slump", "fall", "decline", "downgrade", "miss",
                   "loss", "probe", "fraud", "penalty", "sebi order",
                   "ban", "raid", "cut estimate", "warning", "bearish",
                   "plunge", "tumble", "rout", "default", "bankruptcy")
        score = 0.0
        for k in bullish:
            if k in title_l: score += 0.3
        for k in bearish:
            if k in title_l: score -= 0.4
        ev.sentiment = round(max(-1.0, min(1.0, score)), 3)
        impact, urgency, etype = self._classify_meta(ev.title)
        ev.impact_scope = impact
        ev.urgency      = urgency
        ev.event_type   = etype
        ev.confidence   = 0.35


# ──────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────

def get_news_agent() -> NewsAgent:
    return NewsAgent.get()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    agent = get_news_agent()
    agent.start()
    try:
        while True:
            time.sleep(30)
            print("\n=== bias ===")
            print(json.dumps(agent.current_bias(), default=lambda o: asdict(o) if hasattr(o, "__dict__") else str(o), indent=2))
            print("\n=== stats ===", agent.stats())
    except KeyboardInterrupt:
        agent.stop()
