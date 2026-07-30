"""
Configuration loader — reads from environment variables only.
Never stores secrets in code or on disk.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TradingConfig:
    symbol: str = "NIFTY"
    exchange: str = "NFO"
    product: str = "MIS"
    default_qty: int = 65   # NIFTY lot size — confirmed 65 on this account (was 50, bug)
    max_open_positions: int = 3
    max_daily_loss: float = 5000.0
    max_loss_per_trade: float = 2000.0
    strategy_name: str = "OpenClawNifty"
    account_capital: float = 500000.0
    risk_pct_per_trade: float = 0.02
    # Automatic strategy-health pause (see security.RiskManager)
    drawdown_pause_pct: float = 0.70
    consecutive_loss_pause_threshold: int = 4
    pause_cooldown_min: int = 60


@dataclass
class KotakNeoConfig:
    # Login-mode credentials (recommended) — fully automatic daily login
    # using TOTP. The consumer_key here is the long-lived app token from
    # Kotak Neo App → Invest → Trade API. Login is then completed with
    # mobile + password + TOTP (computed on the fly from totp_secret).
    consumer_key: str = ""
    environment: str = "prod"
    neo_fin_key: str = ""     # optional, leave blank for most accounts

    # Auto-login fields (Neo v2 flow) — required for unattended operation
    mobile: str       = ""    # e.g. +919XXXXXXXXX
    password: str     = ""    # kept for back-compat; used as MPIN if mpin empty
    totp_secret: str  = ""    # base32 seed from Kotak's TOTP/Authenticator setup
    ucc: str          = ""    # Unique Client Code (Profile → in Kotak Neo app)
    mpin: str         = ""    # 6-digit MPIN (app-unlock PIN). Prefer this over password.


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""


@dataclass
class WhatsAppConfig:
    enabled: bool = False
    twilio_sid: str = ""
    twilio_token: str = ""
    whatsapp_from: str = ""
    whatsapp_to: str = ""


@dataclass
class SecurityConfig:
    audit_log_enabled: bool = True
    audit_log_path: str = "/tmp/trade_audit.csv"
    rate_limit_per_sec: int = 8
    api_key_permissions: str = "READ_TRADE_ONLY"


@dataclass
class ClaudeConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "claude-sonnet-4-6"
    auto_execute: bool = False
    min_confidence: int = 60
    # OpenAI fallback
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"


@dataclass
class AppConfig:
    trading: TradingConfig = field(default_factory=TradingConfig)
    kotak: KotakNeoConfig = field(default_factory=KotakNeoConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    whatsapp: WhatsAppConfig = field(default_factory=WhatsAppConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    log_level: str = "INFO"
    dry_run: bool = False


def _safe_int(env_name: str, default: str, errors: list) -> int:
    """Parse an int env var; on failure, record a clean error and fall back
    to the default instead of letting a raw ValueError/traceback escape."""
    raw = os.getenv(env_name, default)
    try:
        return int(raw)
    except ValueError:
        errors.append(f"{env_name}={raw!r} is not a valid integer")
        return int(default)


def _safe_float(env_name: str, default: str, errors: list) -> float:
    """Parse a float env var; on failure, record a clean error and fall
    back to the default instead of letting a raw ValueError/traceback escape."""
    raw = os.getenv(env_name, default)
    try:
        return float(raw)
    except ValueError:
        errors.append(f"{env_name}={raw!r} is not a valid number")
        return float(default)


def load_config() -> AppConfig:
    """Load configuration from environment variables with validation."""

    # Collected up front so malformed numeric env vars produce a clean
    # "Configuration errors" message instead of an uncaught traceback.
    errors: list = []

    config = AppConfig(
        trading=TradingConfig(
            symbol=os.getenv("TRADING_SYMBOL", "NIFTY"),
            exchange=os.getenv("EXCHANGE", "NFO"),
            product=os.getenv("DEFAULT_PRODUCT", "MIS"),
            default_qty=_safe_int("DEFAULT_QTY", "65", errors),
            max_open_positions=_safe_int("MAX_OPEN_POSITIONS", "3", errors),
            max_daily_loss=_safe_float("MAX_DAILY_LOSS", "5000", errors),
            max_loss_per_trade=_safe_float("MAX_LOSS_PER_TRADE", "2000", errors),
            strategy_name=os.getenv("STRATEGY_NAME", "OpenClawNifty"),
            account_capital=_safe_float("ACCOUNT_CAPITAL", "500000", errors),
            risk_pct_per_trade=_safe_float("RISK_PCT_PER_TRADE", "0.02", errors),
            drawdown_pause_pct=_safe_float("STRATEGY_DRAWDOWN_PAUSE_PCT", "0.70", errors),
            consecutive_loss_pause_threshold=_safe_int("STRATEGY_CONSECUTIVE_LOSS_PAUSE", "4", errors),
            pause_cooldown_min=_safe_int("STRATEGY_PAUSE_COOLDOWN_MIN", "60", errors),
        ),
        kotak=KotakNeoConfig(
            consumer_key=os.getenv("KOTAK_CONSUMER_KEY", ""),
            environment=os.getenv("KOTAK_ENVIRONMENT", "prod"),
            neo_fin_key=os.getenv("KOTAK_NEO_FIN_KEY", ""),
            mobile=os.getenv("KOTAK_MOBILE", ""),
            password=os.getenv("KOTAK_PASSWORD", ""),
            totp_secret=os.getenv("KOTAK_TOTP_SECRET", ""),
            ucc=os.getenv("KOTAK_UCC", ""),
            mpin=os.getenv("KOTAK_MPIN", ""),
        ),
        email=EmailConfig(
            enabled=os.getenv("EMAIL_ENABLED", "false").lower() == "true",
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=_safe_int("SMTP_PORT", "587", errors),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            email_from=os.getenv("EMAIL_FROM", ""),
            email_to=os.getenv("EMAIL_TO", ""),
        ),
        whatsapp=WhatsAppConfig(
            enabled=os.getenv("WHATSAPP_ENABLED", "false").lower() == "true",
            twilio_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM", ""),
            whatsapp_to=os.getenv("WHATSAPP_TO", ""),
        ),
        security=SecurityConfig(
            audit_log_enabled=os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true",
            audit_log_path=os.getenv("AUDIT_LOG_PATH", "/tmp/trade_audit.csv"),
            rate_limit_per_sec=_safe_int("RATE_LIMIT_PER_SEC", "8", errors),
            api_key_permissions=os.getenv("API_KEY_PERMISSIONS", "READ_TRADE_ONLY"),
        ),
        claude=ClaudeConfig(
            enabled=os.getenv("CLAUDE_ENABLED", "false").lower() == "true",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            auto_execute=os.getenv("CLAUDE_AUTO_EXECUTE", "false").lower() == "true",
            min_confidence=_safe_int("CLAUDE_MIN_CONFIDENCE", "60", errors),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        dry_run=os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes"),
    )

    # --- Validate critical fields ---
    # NOTE: `errors` was already populated above by _safe_int/_safe_float for
    # any malformed numeric env vars — do not reset it here.

    k = config.kotak
    if not k.consumer_key:
        errors.append(
            "KOTAK_CONSUMER_KEY is required. "
            "Get it from: Kotak Neo App → Invest tab → Trade API → copy token"
        )
    # TOTP auto-login requires: mobile + ucc + mpin (or password) + totp_secret
    effective_mpin = k.mpin or k.password
    missing_login = [name for name, val in (
        ("KOTAK_MOBILE",       k.mobile),
        ("KOTAK_UCC",          k.ucc),
        ("KOTAK_MPIN",         effective_mpin),
        ("KOTAK_TOTP_SECRET",  k.totp_secret),
    ) if not val]
    if missing_login:
        errors.append(
            "Auto-login requires: " + ", ".join(missing_login) +
            ". Add them to config/settings.env.\n"
            "   • KOTAK_UCC        = Unique Client Code (Kotak Neo app → Profile)\n"
            "   • KOTAK_MPIN       = 6-digit PIN you use to unlock the app\n"
            "   • KOTAK_TOTP_SECRET= base32 string from Authenticator-app QR setup"
        )

    if config.email.enabled:
        for field_name in ["smtp_user", "smtp_password", "email_from", "email_to"]:
            if not getattr(config.email, field_name):
                errors.append(f"EMAIL_ENABLED=true but {field_name.upper()} is missing")

    if config.whatsapp.enabled:
        for field_name in ["twilio_sid", "twilio_token", "whatsapp_from", "whatsapp_to"]:
            if not getattr(config.whatsapp, field_name):
                errors.append(
                    f"WHATSAPP_ENABLED=true but {field_name.upper()} is missing"
                )

    if config.security.api_key_permissions != "READ_TRADE_ONLY":
        errors.append(
            "SECURITY RISK: API_KEY_PERMISSIONS must be READ_TRADE_ONLY. "
            "Withdrawal-enabled keys are blocked by design."
        )

    if config.claude.enabled and not config.claude.api_key and not config.claude.openai_api_key:
        errors.append(
            "CLAUDE_ENABLED=true but neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"
        )

    if k.environment not in ("prod", "uat"):
        errors.append(
            f"KOTAK_ENVIRONMENT={k.environment!r} is invalid — must be 'prod' or 'uat'"
        )

    t = config.trading
    for _name, _val in (
        ("DEFAULT_QTY", t.default_qty),
        ("MAX_OPEN_POSITIONS", t.max_open_positions),
        ("MAX_DAILY_LOSS", t.max_daily_loss),
        ("MAX_LOSS_PER_TRADE", t.max_loss_per_trade),
        ("RATE_LIMIT_PER_SEC", config.security.rate_limit_per_sec),
        ("ACCOUNT_CAPITAL", t.account_capital),
    ):
        if _val <= 0:
            errors.append(f"{_name}={_val} must be greater than 0")

    if not (0 < t.risk_pct_per_trade <= 0.10):
        errors.append(
            f"RISK_PCT_PER_TRADE={t.risk_pct_per_trade} must be between 0 and 0.10 "
            f"(0-10% of capital per trade) — this directly sizes live orders"
        )

    if errors:
        print("\n❌ Configuration errors:")
        for e in errors:
            print(f"   • {e}")
        print("\nFix these in your settings.env file and retry.\n")
        sys.exit(1)

    return config
