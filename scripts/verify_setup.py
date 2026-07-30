#!/usr/bin/env python3
"""
Verify setup — tests all connections and security settings before going live.
Run this BEFORE your first trade.

Usage:
    # Set environment first
    export $(cat config/settings.env | grep -v '^#' | xargs)
    python scripts/verify_setup.py
"""

import os
import sys
import smtplib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Colors for terminal
G = "\033[92m"  # Green
R = "\033[91m"  # Red
Y = "\033[93m"  # Yellow
B = "\033[94m"  # Blue
N = "\033[0m"   # Reset


def check(label: str, ok: bool, detail: str = ""):
    status = f"{G}✅ PASS{N}" if ok else f"{R}❌ FAIL{N}"
    print(f"  {status}  {label}")
    if detail:
        print(f"           {Y}{detail}{N}")
    return ok


def main():
    print(f"\n{B}{'='*60}")
    print("  OpenClaw Nifty Trader — Setup Verification (Kotak Neo)")
    print(f"{'='*60}{N}\n")

    all_ok = True

    # ------------------------------------------------------------------
    # 1. Environment Variables
    # ------------------------------------------------------------------
    print(f"{B}[1/6] Kotak Neo Credentials{N}")

    consumer_key = os.getenv("KOTAK_CONSUMER_KEY", "")
    environment  = os.getenv("KOTAK_ENVIRONMENT", "prod")

    all_ok &= check(
        "KOTAK_CONSUMER_KEY is set", bool(consumer_key),
        "" if consumer_key else
        "Get from: Kotak Neo App → Invest tab → Trade API → copy token"
    )

    perms = os.getenv("API_KEY_PERMISSIONS", "")
    all_ok &= check("API_KEY_PERMISSIONS = READ_TRADE_ONLY",
                     perms == "READ_TRADE_ONLY",
                     f"Current: '{perms}'" if perms != "READ_TRADE_ONLY" else "")

    # ------------------------------------------------------------------
    # 2. Kotak Neo Connection Test
    # ------------------------------------------------------------------
    print(f"\n{B}[2/6] Kotak Neo Connection{N}")

    if consumer_key:
        try:
            from neo_api_client import NeoAPI
            # Single token used as both consumer_key and access_token
            neo = NeoAPI(
                consumer_key=consumer_key,
                access_token=consumer_key,
                environment=environment,
                neo_fin_key=None,
            )
            all_ok &= check("Kotak Neo: NeoAPI initialized", True,
                             f"env={environment}")

            # Test funds/limits fetch
            try:
                limits = neo.limits(segment="FO")
                funds_ok = isinstance(limits, dict) and "Error" not in limits
                all_ok &= check("Kotak Neo: funds/limits fetch", funds_ok,
                                 str(limits)[:120] if not funds_ok else "")
            except Exception as e:
                all_ok &= check("Kotak Neo: funds/limits fetch", False, str(e))

        except ImportError:
            all_ok &= check("neo-api-client installed", False,
                             "Run: pip install neo-api-client")
        except Exception as e:
            all_ok &= check("Kotak Neo connection", False, str(e))
    else:
        check("Kotak Neo connection skipped (token missing)", False,
              "Set KOTAK_CONSUMER_KEY in settings.env first")
        all_ok = False

    # ------------------------------------------------------------------
    # 3. Withdrawal Guard
    # ------------------------------------------------------------------
    print(f"\n{B}[3/6] Security — Withdrawal Guard{N}")

    from security import WithdrawalGuard

    blocked_cases = [
        ("/funds/withdrawal", {}, "Withdrawal endpoint blocked"),
        ("/api/v1/withdraw", {}, "Withdraw endpoint blocked"),
        ("/payout", {}, "Payout endpoint blocked"),
        ("/api/v1/placeorder", {"action": "withdraw"}, "Withdraw keyword in payload blocked"),
    ]
    safe_cases = [
        ("/placeorder", {"action": "BUY"}, "Normal BUY order allowed"),
        ("/funds", {}, "Funds check allowed"),
        ("/positions", {}, "Positions check allowed"),
    ]

    for endpoint, payload, label in blocked_cases:
        result = WithdrawalGuard.check(endpoint, payload)
        all_ok &= check(label, result is False)

    for endpoint, payload, label in safe_cases:
        result = WithdrawalGuard.check(endpoint, payload)
        all_ok &= check(label, result is True)

    # ------------------------------------------------------------------
    # 4. Rate Limiter
    # ------------------------------------------------------------------
    print(f"\n{B}[4/6] Security — Rate Limiter{N}")

    from security import RateLimiter

    rl = RateLimiter(max_per_sec=8)
    acquired = sum(1 for _ in range(12) if rl.acquire())
    all_ok &= check(f"Rate limiter working (got {acquired}/12 tokens, expected ~8)",
                     6 <= acquired <= 10)

    # ------------------------------------------------------------------
    # 5. Email
    # ------------------------------------------------------------------
    print(f"\n{B}[5/6] Email Notifications{N}")

    email_enabled = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    if email_enabled:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASSWORD", "")

        all_ok &= check("SMTP_USER is set", bool(smtp_user))
        all_ok &= check("SMTP_PASSWORD is set", bool(smtp_pass),
                         "" if smtp_pass else "Use a Gmail App Password, not your main password")

        if smtp_user and smtp_pass:
            try:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                all_ok &= check(f"SMTP login successful ({smtp_host}:{smtp_port})", True)
            except smtplib.SMTPAuthenticationError:
                all_ok &= check("SMTP login", False,
                                "Auth failed. For Gmail, use an App Password")
            except Exception as e:
                all_ok &= check("SMTP connection", False, str(e))
    else:
        check("Email disabled (EMAIL_ENABLED=false)", True, "Enable in settings.env")

    # ------------------------------------------------------------------
    # 6. WhatsApp
    # ------------------------------------------------------------------
    print(f"\n{B}[6/6] WhatsApp Notifications{N}")

    wa_enabled = os.getenv("WHATSAPP_ENABLED", "false").lower() == "true"
    if wa_enabled:
        import requests
        sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        wa_from = os.getenv("TWILIO_WHATSAPP_FROM", "")
        wa_to   = os.getenv("WHATSAPP_TO", "")

        all_ok &= check("TWILIO_ACCOUNT_SID is set", bool(sid))
        all_ok &= check("TWILIO_AUTH_TOKEN is set", bool(token))
        all_ok &= check("TWILIO_WHATSAPP_FROM is set", bool(wa_from))
        all_ok &= check("WHATSAPP_TO is set", bool(wa_to))

        if sid and token:
            try:
                resp = requests.get(
                    f"https://api.twilio.com/2010-04-01/Accounts/{sid}.json",
                    auth=(sid, token), timeout=10,
                )
                tw_ok = resp.status_code == 200
                all_ok &= check("Twilio credentials valid", tw_ok,
                                "" if tw_ok else f"HTTP {resp.status_code}")
            except Exception as e:
                all_ok &= check("Twilio connection", False, str(e))
    else:
        check("WhatsApp disabled (WHATSAPP_ENABLED=false)", True, "Enable in settings.env")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{B}{'='*60}{N}")
    if all_ok:
        print(f"  {G}✅ All checks passed — you're ready to trade!{N}")
        print(f"  Start the server: python main.py")
    else:
        print(f"  {R}❌ Some checks failed — fix the issues above first.{N}")
    print(f"{B}{'='*60}{N}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
