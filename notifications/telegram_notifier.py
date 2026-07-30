"""
Telegram notification module — sends trade alerts and breaking news to a
Telegram chat via the Bot API.

Uses HTML parse mode (more forgiving than Markdown — only <, >, & need escaping).

Setup:
1. Talk to @BotFather on Telegram → /newbot → grab the token
2. Send any message to your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
   Find the "chat":{"id": ...} value — that's your chat_id
3. Add to settings.env:
     TELEGRAM_ENABLED=true
     TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
     TELEGRAM_CHAT_ID=123456789
"""

import html
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


def _esc(s) -> str:
    """HTML-escape any value for safe insertion into a Telegram HTML message."""
    return html.escape(str(s if s is not None else ""), quote=False)


class TelegramNotifier:
    """Send messages to a Telegram chat via the Bot API."""

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        enabled: bool,
        bot_token: str,
        chat_id: str,
    ):
        self.enabled = bool(enabled and bot_token and chat_id)
        self.bot_token = bot_token
        self.chat_id = str(chat_id)

    # ── Generic send (used by news + custom alerts) ────────────────
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a free-form message. Returns True on success."""
        if not self.enabled:
            return False
        try:
            url = self.API_URL.format(token=self.bot_token)
            payload = {
                "chat_id": self.chat_id,
                "text": text[:4090],   # Telegram hard limit is 4096
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            r = requests.post(url, data=payload, timeout=10)
            if r.status_code == 200:
                return True
            logger.error(
                f"📨 Telegram send failed: HTTP {r.status_code} — {r.text[:200]}"
            )
            # If parse failed, retry plain
            if r.status_code == 400 and parse_mode:
                payload.pop("parse_mode", None)
                r2 = requests.post(url, data=payload, timeout=10)
                if r2.status_code == 200:
                    return True
                logger.error(
                    f"📨 Telegram plain-text retry failed: HTTP {r2.status_code} — {r2.text[:200]}"
                )
            return False
        except Exception as e:
            logger.error(f"📨 Telegram send exception: {e}")
            return False

    # ── Trade alerts (called by NotificationManager) ───────────────
    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_id: str,
        status: str,
        details: str = "",
    ):
        if not self.enabled:
            return
        msg = self._build_trade_message(
            action, symbol, side, qty, price, order_id, status, details
        )
        if self.send_message(msg):
            logger.info(f"📨 Telegram trade alert sent: {action} {symbol}")

    @staticmethod
    def _build_trade_message(
        action: str, symbol: str, side: str, qty: int, price: float,
        order_id: str, status: str, details: str,
    ) -> str:
        a = (action or "").upper()
        is_exit = a.startswith("EXIT")
        is_entry = (side or "").upper() == "BUY" and not is_exit
        if is_entry:
            head = "🟢 <b>TRADE ENTRY</b>"
        elif is_exit:
            head = "🔴 <b>TRADE EXIT</b>"
        else:
            head = "🔔 <b>TRADE UPDATE</b>"
        if (status or "").lower() in ("failed", "rejected", "error"):
            head = "❌ <b>TRADE FAILED</b>"

        side_emoji = "🟢" if (side or "").upper() == "BUY" else "🔴"
        now = datetime.now().strftime("%d %b, %H:%M:%S")

        lines = [
            head,
            "",
            f"<b>Action:</b> {_esc(a)}",
            f"<b>Symbol:</b> <code>{_esc(symbol)}</code>",
            f"<b>Side:</b> {side_emoji} {_esc((side or '').upper())}",
            f"<b>Qty:</b> {_esc(qty)}",
            f"<b>Price:</b> ₹{price:.2f}",
        ]
        if order_id:
            lines.append(f"<b>Order ID:</b> <code>{_esc(order_id)}</code>")
        lines.append(f"<b>Status:</b> {_esc((status or '').upper())}")
        if details:
            lines.append("")
            lines.append(f"📝 {_esc(details)}")
        lines.append("")
        lines.append(f"⏰ {now} IST")
        return "\n".join(lines)

    # ── News alerts (called by news_agent) ─────────────────────────
    def send_news_alert(
        self,
        title: str,
        source: str,
        sentiment: float,
        impact: str,
        urgency: str,
        url: str = "",
    ):
        if not self.enabled:
            return
        if sentiment is None:
            return
        s = float(sentiment)
        if s >= 0.4:
            emoji, dir_label = "🟢📰", "BULLISH"
        elif s <= -0.4:
            emoji, dir_label = "🔴📰", "BEARISH"
        else:
            emoji, dir_label = "⚪📰", "NEUTRAL"
        now = datetime.now().strftime("%d %b, %H:%M")
        safe_title = _esc((title or "")[:300])
        lines = [
            f"{emoji} <b>{_esc((urgency or '').upper())} NEWS — {dir_label}</b>",
            "",
            f"<b>Impact:</b> {_esc((impact or '').upper())}",
            f"<b>Sentiment:</b> {s:+.2f}",
            f"<b>Source:</b> {_esc(source)}",
            "",
            f"<i>{safe_title}</i>",
        ]
        if url:
            lines.append("")
            lines.append(f'<a href="{_esc(url)}">Read more</a>')
        lines.append("")
        lines.append(f"⏰ {now} IST")
        if self.send_message("\n".join(lines)):
            logger.info(f"📨 Telegram news alert sent: [{urgency}] {(title or '')[:60]}")

    # ── Health check ──────────────────────────────────────────────
    def test_connection(self) -> bool:
        return self.send_message(
            "✅ <b>OpenClaw Telegram Connected</b>\n\n"
            "Bot is now sending trade &amp; news alerts.\n\n"
            f"⏰ {datetime.now().strftime('%d %b %Y, %H:%M:%S')} IST"
        )
