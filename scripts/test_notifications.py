#!/usr/bin/env python3
"""
Send a test notification to Email and WhatsApp.
Use this to verify your notification setup before live trading.

Usage:
    export $(cat config/settings.env | grep -v '^#' | xargs)
    python scripts/test_notifications.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications.email_notifier import EmailNotifier
from notifications.whatsapp_notifier import WhatsAppNotifier


def main():
    print("=" * 50)
    print("  Notification Test — Email + WhatsApp")
    print("=" * 50)

    # --- Email Test ---
    email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    if email_enabled:
        print("\n📧 Sending test email...")
        notifier = EmailNotifier(
            enabled=True,
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            email_from=os.getenv("EMAIL_FROM", ""),
            email_to=os.getenv("EMAIL_TO", ""),
        )
        try:
            notifier.send_trade_alert(
                action="TEST",
                symbol="NIFTY25MAR2524500CE",
                side="BUY",
                qty=50,
                price=185.50,
                order_id="TEST-001",
                status="success",
                details="This is a test notification. If you see this, email alerts are working!",
            )
            print("   ✅ Email sent! Check your inbox.")
        except Exception as e:
            print(f"   ❌ Email failed: {e}")
    else:
        print("\n📧 Email disabled (EMAIL_ENABLED=false). Skipping.")

    # --- WhatsApp Test ---
    wa_enabled = os.getenv("WHATSAPP_ENABLED", "false").lower() == "true"
    if wa_enabled:
        print("\n📱 Sending test WhatsApp message...")
        notifier = WhatsAppNotifier(
            enabled=True,
            twilio_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM", ""),
            whatsapp_to=os.getenv("WHATSAPP_TO", ""),
        )
        try:
            notifier.send_trade_alert(
                action="TEST",
                symbol="NIFTY25MAR2524500CE",
                side="BUY",
                qty=50,
                price=185.50,
                order_id="TEST-001",
                status="success",
                details="This is a test notification. If you see this, WhatsApp alerts are working!",
            )
            print("   ✅ WhatsApp sent! Check your phone.")
        except Exception as e:
            print(f"   ❌ WhatsApp failed: {e}")
    else:
        print("\n📱 WhatsApp disabled (WHATSAPP_ENABLED=false). Skipping.")

    print("\n" + "=" * 50)
    print("  Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
