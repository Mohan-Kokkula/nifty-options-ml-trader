"""
Unified notification dispatcher — sends alerts to all enabled channels.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .email_notifier import EmailNotifier
from .whatsapp_notifier import WhatsAppNotifier
from .telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Dispatches trade alerts to all enabled notification channels
    in parallel (non-blocking).
    """

    def __init__(
        self,
        email_notifier: Optional[EmailNotifier] = None,
        whatsapp_notifier: Optional[WhatsAppNotifier] = None,
        telegram_notifier: Optional[TelegramNotifier] = None,
    ):
        self.email = email_notifier
        self.whatsapp = whatsapp_notifier
        self.telegram = telegram_notifier
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="notif")

    def notify_trade(
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
        """
        Fire-and-forget notifications to all enabled channels.
        Runs in background threads so trading isn't blocked.
        """
        kwargs = dict(
            action=action,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_id=order_id,
            status=status,
            details=details,
        )

        if self.email and self.email.enabled:
            self._executor.submit(self._safe_send, self.email.send_trade_alert, kwargs)

        if self.whatsapp and self.whatsapp.enabled:
            self._executor.submit(
                self._safe_send, self.whatsapp.send_trade_alert, kwargs
            )

        if self.telegram and self.telegram.enabled:
            self._executor.submit(
                self._safe_send, self.telegram.send_trade_alert, kwargs
            )

    def notify_news(
        self,
        title: str,
        source: str,
        sentiment: float,
        impact: str,
        urgency: str,
        url: str = "",
    ):
        """Send a news alert (currently Telegram-only — others are too noisy)."""
        if self.telegram and self.telegram.enabled:
            self._executor.submit(
                self._safe_send,
                self.telegram.send_news_alert,
                dict(
                    title=title, source=source, sentiment=sentiment,
                    impact=impact, urgency=urgency, url=url,
                ),
            )

    def notify(self, subject: str, message: str):
        """Send a free-form system alert (currently Telegram-only — same
        rationale as notify_news: other channels are too noisy for this)."""
        if self.telegram and self.telegram.enabled:
            self._executor.submit(
                self._safe_send,
                self.telegram.send_message,
                dict(text=f"<b>{subject}</b>\n\n{message}"),
            )

    def _safe_send(self, func, kwargs):
        try:
            func(**kwargs)
        except Exception as e:
            logger.error(f"Notification send failed: {e}")

    def shutdown(self):
        self._executor.shutdown(wait=True)
