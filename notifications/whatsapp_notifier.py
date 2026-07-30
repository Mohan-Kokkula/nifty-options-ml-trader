"""
WhatsApp notification module — sends trade alerts via Twilio WhatsApp API.

Setup:
1. Create a Twilio account at https://www.twilio.com
2. Enable the WhatsApp Sandbox (or register a WhatsApp Business number)
3. For sandbox: send "join <your-sandbox-keyword>" from your WhatsApp to
   the Twilio sandbox number (+1 415 523 8886)
4. Copy your Account SID, Auth Token, and the 'from' number into settings.env
"""

import logging
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WhatsAppNotifier:
    """Send trade alerts to WhatsApp via Twilio API."""

    TWILIO_API_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

    def __init__(
        self,
        enabled: bool,
        twilio_sid: str,
        twilio_token: str,
        whatsapp_from: str,
        whatsapp_to: str,
    ):
        self.enabled = enabled
        self.twilio_sid = twilio_sid
        self.twilio_token = twilio_token
        self.whatsapp_from = whatsapp_from
        self.whatsapp_to = whatsapp_to

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
        """Send a trade notification to WhatsApp."""
        if not self.enabled:
            return

        try:
            message = self._build_message(
                action, symbol, side, qty, price, order_id, status, details
            )

            url = self.TWILIO_API_URL.format(sid=self.twilio_sid)
            payload = {
                "From": self.whatsapp_from,
                "To": self.whatsapp_to,
                "Body": message,
            }

            resp = requests.post(
                url,
                data=payload,
                auth=(self.twilio_sid, self.twilio_token),
                timeout=15,
            )

            if resp.status_code in (200, 201):
                logger.info(f"📱 WhatsApp sent: {action} {side} {symbol}")
            else:
                logger.error(
                    f"📱 WhatsApp failed: HTTP {resp.status_code} — {resp.text}"
                )

        except Exception as e:
            logger.error(f"📱 WhatsApp failed: {e}")

    def _build_message(
        self,
        action: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_id: str,
        status: str,
        details: str,
    ) -> str:
        emoji = "✅" if status == "success" else "❌"
        side_emoji = "🟢" if side.upper() == "BUY" else "🔴"
        now = datetime.now().strftime("%d %b %Y, %I:%M %p")

        lines = [
            f"{emoji} *TRADE ALERT*",
            "",
            f"Action: *{action.upper()}*",
            f"Symbol: *{symbol}*",
            f"Side: {side_emoji} *{side.upper()}*",
            f"Qty: *{qty}*",
            f"Price: *₹{price:.2f}*",
            f"Order ID: `{order_id}`",
            f"Status: *{status.upper()}*",
        ]

        if details:
            lines.append(f"\n📝 {details}")

        lines.append(f"\n⏰ {now}")
        lines.append("_OpenClaw Nifty Trader · Kotak Neo_")

        return "\n".join(lines)
