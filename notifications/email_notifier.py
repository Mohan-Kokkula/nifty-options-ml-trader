"""
Email notification module — sends trade alerts via SMTP.
Uses Gmail App Password or any SMTP provider.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send HTML-formatted trade alert emails."""

    def __init__(
        self,
        enabled: bool,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        email_from: str,
        email_to: str,
    ):
        self.enabled = enabled
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.email_from = email_from
        self.email_to = email_to

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
        """Send a trade notification email."""
        if not self.enabled:
            return

        try:
            subject = self._build_subject(action, symbol, side, status)
            body = self._build_html_body(
                action, symbol, side, qty, price, order_id, status, details
            )

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_from
            msg["To"] = self.email_to
            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())

            logger.info(f"📧 Email sent: {subject}")

        except Exception as e:
            logger.error(f"📧 Email failed: {e}")

    def _build_subject(
        self, action: str, symbol: str, side: str, status: str
    ) -> str:
        emoji = "✅" if status == "success" else "❌"
        return f"{emoji} Trade Alert: {action.upper()} {side} {symbol}"

    def _build_html_body(
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
        status_color = "#22c55e" if status == "success" else "#ef4444"
        side_color = "#22c55e" if side.upper() == "BUY" else "#ef4444"
        now = datetime.now().strftime("%d %b %Y, %I:%M:%S %p")

        return f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; padding: 20px; background: #f8f9fa;">
            <div style="background: #ffffff; border-radius: 12px; padding: 24px; border: 1px solid #e5e7eb;">
                <h2 style="margin: 0 0 16px; color: #111827; font-size: 18px;">
                    Nifty Options Trade Alert
                </h2>

                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Action</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right;">{action.upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Symbol</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right;">{symbol}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Side</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right; color: {side_color};">{side.upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Quantity</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right;">{qty}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Price</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right;">₹{price:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Order ID</td>
                        <td style="padding: 8px 0; font-family: monospace; text-align: right; font-size: 12px;">{order_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280;">Status</td>
                        <td style="padding: 8px 0; font-weight: 600; text-align: right; color: {status_color};">{status.upper()}</td>
                    </tr>
                </table>

                {"<p style='margin: 12px 0 0; padding: 12px; background: #f3f4f6; border-radius: 8px; font-size: 13px; color: #374151;'>" + details + "</p>" if details else ""}

                <p style="margin: 16px 0 0; font-size: 11px; color: #9ca3af;">
                    {now} · OpenClaw Nifty Trader via Kotak Neo
                </p>
            </div>
        </body>
        </html>
        """
