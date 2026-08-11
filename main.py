"""
Main entry point for the OpenClaw Nifty Options Trader.

Starts an HTTP server that OpenClaw (or any client) can call with
natural-language-like JSON commands. Also usable as a Python library.

Usage:
    python main.py                  # Start HTTP API server on port 8080
    python main.py --port 9090      # Custom port
"""

import argparse
import json
import logging
import os
import secrets
import signal
import sys
import threading
import time
from datetime import date, datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load settings.env BEFORE importing config (which reads os.environ at import-time)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent / "config" / "settings.env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # python-dotenv optional — fall back to pre-exported env vars

from config import load_config
from core.kotak_neo_client import KotakNeoClient
from core.trader import NiftyOptionsTrader
from core.strike_selector import round_to_strike
from core.claude_analyzer import ClaudeAnalyzer
from core.claude_pilot import ClaudePilot, PilotConfig
from core.strategy_monitor import StrategyMonitor, LevelSetup, build_todays_setups
from core import ml_engine
from core.openclaw_agent import OpenClawAgent, init_agent, get_agent
from notifications import NotificationManager
from notifications.email_notifier import EmailNotifier
from notifications.whatsapp_notifier import WhatsAppNotifier
from notifications.telegram_notifier import TelegramNotifier
from security import RateLimiter, RiskManager, AuditLogger
import queue


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

def _force_ipv4_if_enabled():
    """
    Force Python's requests/urllib3 to use IPv4 only. Belt-and-suspenders
    against IPv6 sneaking back (caused 100008 unauthorized on 2026-04-29
    because Kotak Neo's registered IP was IPv4 but VPS was using IPv6).

    Set REQUESTS_PREFER_IPV4=true in settings.env to enable.
    """
    import os as _os
    if _os.getenv("REQUESTS_PREFER_IPV4", "false").lower() != "true":
        return
    try:
        import socket
        # Patch socket.getaddrinfo to filter out AAAA (IPv6) records
        _orig_getaddrinfo = socket.getaddrinfo
        def _ipv4_only_getaddrinfo(*args, **kwargs):
            results = _orig_getaddrinfo(*args, **kwargs)
            return [r for r in results if r[0] == socket.AF_INET]
        socket.getaddrinfo = _ipv4_only_getaddrinfo
        print("🌐 IPv4-only mode active — IPv6 disabled at socket level")
    except Exception as e:
        print(f"⚠️ IPv4 enforcement failed: {e}")


class _DailyFileHandler(logging.Handler):
    """
    Writes to logs/YYYY-MM-DD.log, rolling over to a fresh file at midnight
    IST — checked on every emit(), not just once at process start.

    A plain logging.FileHandler opens its file ONCE when constructed and
    keeps writing to that same file for as long as the process lives, even
    across a date rollover — confirmed happening in production: the bot ran
    continuously past midnight (stop/start cron cycle didn't fire as
    expected) and kept appending everything to yesterday's dated file,
    silently, with no new file ever created for the new day.
    """

    def __init__(self, log_dir: Path, encoding: str = "utf-8"):
        super().__init__()
        self._log_dir = log_dir
        self._encoding = encoding
        self._current_date = None
        self._inner = None
        self._open_for_today()

    def _open_for_today(self):
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y-%m-%d")
        if today == self._current_date:
            return
        if self._inner is not None:
            self._inner.close()
        self._inner = logging.FileHandler(
            self._log_dir / f"{today}.log", encoding=self._encoding
        )
        if self.formatter is not None:
            self._inner.setFormatter(self.formatter)
        self._current_date = today

    def setFormatter(self, fmt):
        super().setFormatter(fmt)
        if self._inner is not None:
            self._inner.setFormatter(fmt)

    def emit(self, record):
        self._open_for_today()
        self._inner.emit(record)

    def close(self):
        if self._inner is not None:
            self._inner.close()
        super().close()


def bootstrap():
    """Initialize all components from config."""
    _force_ipv4_if_enabled()
    config = load_config()

    # Logging
    # ── Logging — terminal + daily file (auto-rolls at midnight IST) ──
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt = "%H:%M:%S"

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
    root_logger.addHandler(console_handler)

    # File handler — new file every day, even without a process restart
    file_handler = _DailyFileHandler(log_dir, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("main")
    logger.info(f"📝 Logging to {log_dir}/ (rolls to a new dated file automatically at midnight IST)")
    logger.info("🚀 Starting OpenClaw Nifty Options Trader")

    # Loud, impossible-to-miss confirmation of trading mode — resolves the
    # historically ambiguous "is this bot paper or live?" question at boot.
    if config.dry_run:
        logger.info("🟢 PAPER TRADING — DRY_RUN=true — no real orders will be placed")
    else:
        logger.warning("🔴 LIVE TRADING — DRY_RUN=false — real orders WILL be placed")

    if not os.getenv("API_AUTH_TOKEN", ""):
        logger.warning(
            "🔓 API_AUTH_TOKEN not set — the HTTP API (/trade, /close, "
            "/cancel_all, etc.) has NO authentication until either "
            "API_AUTH_TOKEN is set in settings.env, or a caller obtains a "
            "session token via POST /login {\"mpin\": \"<KOTAK_MPIN>\"}."
        )

    # Security
    rate_limiter = RateLimiter(config.security.rate_limit_per_sec)
    risk_manager = RiskManager(
        max_daily_loss=config.trading.max_daily_loss,
        max_loss_per_trade=config.trading.max_loss_per_trade,
        max_open_positions=config.trading.max_open_positions,
        drawdown_pause_pct=config.trading.drawdown_pause_pct,
        consecutive_loss_pause_threshold=config.trading.consecutive_loss_pause_threshold,
        pause_cooldown_min=config.trading.pause_cooldown_min,
    )
    audit = AuditLogger(
        log_path=config.security.audit_log_path,
        enabled=config.security.audit_log_enabled,
    )
    logger.info(f"🔒 Security: {config.security.api_key_permissions} | "
                f"Rate limit: {config.security.rate_limit_per_sec}/sec | "
                f"Max positions: {config.trading.max_open_positions} | "
                f"Daily loss cap: ₹{config.trading.max_daily_loss}")

    # Notifications
    email_notifier = EmailNotifier(
        enabled=config.email.enabled,
        smtp_host=config.email.smtp_host,
        smtp_port=config.email.smtp_port,
        smtp_user=config.email.smtp_user,
        smtp_password=config.email.smtp_password,
        email_from=config.email.email_from,
        email_to=config.email.email_to,
    )
    whatsapp_notifier = WhatsAppNotifier(
        enabled=config.whatsapp.enabled,
        twilio_sid=config.whatsapp.twilio_sid,
        twilio_token=config.whatsapp.twilio_token,
        whatsapp_from=config.whatsapp.whatsapp_from,
        whatsapp_to=config.whatsapp.whatsapp_to,
    )
    telegram_notifier = TelegramNotifier(
        enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )
    # Telegram isn't part of AppConfig's validated fields (read raw above),
    # so it can silently half-configure — warn rather than fail startup,
    # since it's a non-critical notification channel.
    _tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if bool(_tg_token) != bool(_tg_chat):
        logger.warning(
            "⚠️ Telegram partially configured — "
            f"TELEGRAM_BOT_TOKEN={'set' if _tg_token else 'MISSING'}, "
            f"TELEGRAM_CHAT_ID={'set' if _tg_chat else 'MISSING'}. "
            "Both are required for Telegram alerts to work."
        )
    notifications = NotificationManager(
        email_notifier=email_notifier,
        whatsapp_notifier=whatsapp_notifier,
        telegram_notifier=telegram_notifier,
    )
    risk_manager.set_notifier(notifications)   # deferred injection — see RiskManager.set_notifier
    # Send a startup ping so you know the bot is alive
    if telegram_notifier.enabled:
        try:
            telegram_notifier.test_connection()
        except Exception as _e:
            logger.warning(f"Telegram startup ping failed: {_e}")
    channels = []
    if config.email.enabled:
        channels.append("Email")
    if config.whatsapp.enabled:
        channels.append("WhatsApp")
    if telegram_notifier.enabled:
        channels.append("Telegram")
    logger.info(f"📬 Notifications: {', '.join(channels) or 'None'}")

    # Kotak Neo Client — auto-login via mobile + password + TOTP
    client = KotakNeoClient(
        consumer_key=config.kotak.consumer_key,
        environment=config.kotak.environment,
        neo_fin_key=config.kotak.neo_fin_key or None,
        rate_limiter=rate_limiter,
        mobile=config.kotak.mobile,
        password=config.kotak.password,
        totp_secret=config.kotak.totp_secret,
        ucc=config.kotak.ucc,
        mpin=config.kotak.mpin,
    )
    # Wire notifier into client for session expiry alerts (WhatsApp/email)
    client.set_notifier(notifications)
    # Reflect actual auth state (was login deferred or performed?)
    _kotak_logged_in = getattr(client, "_session_valid", False)
    logger.info(
        f"🔗 Kotak Neo: {config.kotak.environment} | "
        f"{'logged in (TOTP)' if _kotak_logged_in else 'login DEFERRED until 09:10 IST'}"
    )

    # 2026-04-29: Verify outbound IP matches registered Kotak IP
    # (IPv6 routing caused 100008 unauthorized — surfaced at order time).
    try:
        import ipaddress as _ipaddress
        import requests as _requests
        ipv4 = _requests.get("https://api.ipify.org", timeout=5).text.strip()
        try:
            _ipaddress.IPv4Address(ipv4)
            ip_is_valid = True
        except ValueError:
            ip_is_valid = False

        if not ip_is_valid:
            # ipify.org returns plain-text errors (e.g. "error code: 1200",
            # usually rate-limiting) instead of an IP on failure. Without
            # this check that error text gets treated as if it WERE the
            # outbound IP, producing a false "MISMATCH" alarm every boot.
            logger.warning(
                f"IP verification skipped — ipify.org returned a non-IP "
                f"response ({ipv4!r}), likely rate-limited or transiently "
                f"down. Outbound IP genuinely unverified this boot; check "
                f"manually with `curl -s https://api.ipify.org` if unsure."
            )
        else:
            expected = os.getenv("KOTAK_REGISTERED_IP", "").strip()
            logger.info(f"🌐 Outbound IPv4: {ipv4}")
            if expected and ipv4 != expected:
                logger.error(
                    f"⚠️ IP MISMATCH: outbound={ipv4} but registered={expected}. "
                    f"Trade APIs will FAIL with 100008 unauthorized."
                )
                try:
                    if telegram_notifier.enabled:
                        telegram_notifier.send_message(
                            f"🚨 IP MISMATCH at startup\n\n"
                            f"Outbound IP: <code>{ipv4}</code>\n"
                            f"Registered:  <code>{expected}</code>\n\n"
                            f"Update IP at Kotak portal OR fix VPS routing before 09:15."
                        )
                except Exception:
                    pass
            else:
                logger.info(
                    f"✅ Outbound IP {'matches Kotak registration' if expected else '(verify against Kotak portal)'}"
                )
    except Exception as _e:
        logger.warning(f"IP verification failed: {_e}")

    # 2026-05-05: EOD Telegram summary — daily P&L digest at 15:30 IST
    try:
        from core.eod_summary import schedule_eod_summary
        risk_cap_for_summary = float(getattr(config.trading, "max_daily_loss", 3000) or 3000)
        schedule_eod_summary(notifier=notifications, risk_cap=risk_cap_for_summary)
        logger.info(f"📊 EOD summary scheduled — fires at {os.getenv('EOD_REPORT_TIME', '15:30')} IST")
    except Exception as _e:
        logger.warning(f"EOD summary scheduler failed: {_e}")

    # 2026-06-13: Production shadow framework — daily report/cleanup/health
    # for the HTF36 comparison shadow (data/shadow/). No-op if disabled.
    try:
        from core.shadow_scheduler import schedule_shadow_tasks
        schedule_shadow_tasks(notifier=notifications)
        logger.info(f"👥 Shadow framework scheduled — fires at {os.getenv('SHADOW_REPORT_TIME', '15:35')} IST")
    except Exception as _e:
        logger.warning(f"Shadow framework scheduler failed: {_e}")

    # 2026-04-29: Deadman switch — fail-loud monitoring.
    # Catches silent hangs (the kind that cost 200pts on 2026-04-29).
    # Sends Telegram CRITICAL alerts at 09:11 (Kotak login check) and
    # 09:30 (pilot heartbeat check) + continuous heartbeat monitoring.
    try:
        from core.deadman_switch import start_deadman
        start_deadman(client, notifier=notifications)
        logger.info(
            "🐕 Deadman switch armed — Telegram CRITICAL alerts on any silent failure"
        )
    except Exception as _e:
        logger.warning(f"Deadman switch start failed: {_e}")

    # 2026-04-29: Kotak login scheduler is OPT-IN now (eager login is default).
    # The scheduler caused deadlocks before — only enable if you're using
    # lazy login (KOTAK_LAZY_LOGIN=true) AND understand the trade-offs.
    if os.getenv("KOTAK_LOGIN_SCHEDULER", "false").lower() == "true":
        try:
            from core.kotak_login_scheduler import schedule_daily_login
            schedule_daily_login(client, notifier=notifications)
            logger.info("⏰ Kotak login scheduler started (09:05 IST daily, separate log)")
        except Exception as _e:
            logger.warning(f"Kotak login scheduler failed to start: {_e}")

    # Trading Engine
    trader = NiftyOptionsTrader(
        client=client,
        risk_manager=risk_manager,
        audit=audit,
        notifications=notifications,
        exchange=config.trading.exchange,
        product=config.trading.product,
        default_qty=config.trading.default_qty,
        strategy=config.trading.strategy_name,
    )
    logger.info(f"📊 Trading: {config.trading.symbol} on {config.trading.exchange} "
                f"| Qty: {config.trading.default_qty} | Product: {config.trading.product}")

    # Claude Analyzer (with OpenAI fallback)
    analyzer = ClaudeAnalyzer(
        api_key=config.claude.api_key,
        model=config.claude.model,
        auto_execute=config.claude.auto_execute,
        openai_api_key=config.claude.openai_api_key,
        openai_model=config.claude.openai_model,
    )
    if config.claude.enabled:
        providers = []
        if config.claude.api_key:
            providers.append(f"Anthropic ({config.claude.model})")
        if config.claude.openai_api_key:
            providers.append(f"OpenAI ({config.claude.openai_model})")
        logger.info(f"🧠 AI: {' → fallback → '.join(providers)} | "
                    f"Auto-execute: {config.claude.auto_execute} | "
                    f"Min confidence: {config.claude.min_confidence}%")
    else:
        logger.info("🧠 AI: Disabled (set CLAUDE_ENABLED=true to activate)")

    # Strategy Monitor
    monitor = StrategyMonitor(
        trader=trader,
        notifier=notifications,
        poll_interval=15,
    )

    # Load ML Model
    _ml_load_t0 = time.time()
    ml_loaded = ml_engine.load_model(".")
    logger.info(f"⏱️  Model load time: {time.time() - _ml_load_t0:.2f}s")
    if ml_loaded:
        logger.info("🤖 ML model loaded — dual-brain mode active")
    else:
        logger.warning("🤖 ML model not found — Claude-only mode")

    # 2026-06-12: in-app intraday OI archiver (1-min ATM snapshots for the
    # OI research program). Daemon thread, passive, fail-safe; disable with
    # OI_ARCHIVER_ENABLED=false.
    try:
        from scripts.oi_archiver import start_in_app as _oi_start
        _oi_start(kotak_client=client)   # bot's Kotak client → real IV in full-chain
    except Exception as _e:
        logger.warning(f"OI archiver not started (non-fatal): {_e}")

    # 2026-06-16: in-app futures + Level-5 depth archiver (1-min) for the
    # futures-microstructure research program (basis/VWAP/CVD/depth). Daemon
    # thread, passive, fail-safe; disable with FUTURES_ARCHIVER_ENABLED=false.
    try:
        from scripts.futures_archiver import start_in_app as _fut_start
        _fut_start(kotak_client=client)  # reuse bot's session (no 2nd login)
    except Exception as _e:
        logger.warning(f"futures archiver not started (non-fatal): {_e}")

    # PSAR session brain (new, opt-in, paper-mode only): independent
    # trend-following signal validated by an 8-fold walk-forward backtest
    # (PSAR flip + opening-gap agreement + opening-range breakout with a
    # 2-bar hold confirmation + VIX regime band). Runs as a fully separate
    # daemon loop from ClaudePilot's ML-driven position management -- own
    # position state, own journal, own Telegram alerts, never places a
    # real broker order. Disabled by default; enable with
    # PSAR_BRAIN_ENABLED=true in settings.env.
    try:
        from core.psar_session_brain import start_in_app as _psar_start
        _psar_start()
    except Exception as _e:
        logger.warning(f"PSAR session brain not started (non-fatal): {_e}")

    # Trend-day brain (new, opt-in, paper-mode only, FUTURES ONLY): joins an
    # established session trend (VWAP-proxy deficit + 3 sequential closes +
    # below/above day open) once per day, structural stop at the day extreme
    # capped at 60pts, 2R target. Params selected on walk-forward VAL folds
    # 1-5 and confirmed once on TEST folds 6-8 (mean PF 1.113, 3/3 folds
    # > 1.0, n=924). Refuses to arm unless EXECUTION_MODE=futures -- the same
    # trades score PF 0.666 under options friction. Own state/journal/Telegram,
    # never touches LivePosition, never places a real broker order. Disabled by
    # default; enable with TREND_DAY_BRAIN_ENABLED=true in settings.env.
    try:
        from core.trend_day_brain import start_in_app as _td_start
        _td_start()
    except Exception as _e:
        logger.warning(f"Trend-day brain not started (non-fatal): {_e}")

    # ML-EOD brain (new, opt-in, paper-mode only, FUTURES ONLY): the SAME
    # ml_engine signal the pilot uses, but held to the close instead of
    # exited on ATR targets. Rationale: the model predicts destination
    # (65% dir_acc) far better than route (0.52 triple-barrier AUC), so a
    # short-hold exit bets on the half it cannot forecast. Replayed on
    # frozen TEST under live constraints (one position, max 3/day, no
    # entry after 15:00): pilot's short-hold structure PF 0.943, this
    # 2R-stop + EOD structure PF 1.053. Marginal -- the trend-day brain
    # is still better at 1.225 -- so this exists to generate forward
    # evidence, not to replace anything. Own state/journal/Telegram,
    # never touches LivePosition, never places a real order. Disabled by
    # default; enable with ML_EOD_BRAIN_ENABLED=true.
    try:
        from core.ml_eod_brain import start_in_app as _mle_start
        _mle_start()
    except Exception as _e:
        logger.warning(f"ML-EOD brain not started (non-fatal): {_e}")

    # Dual-Brain Pilot (ML + Claude)
    pilot_config = PilotConfig(
        ml_only_mode=os.getenv("ML_ONLY_MODE", "false").lower() == "true",
        # Base LOW_CONF gate for the ML pilot loop -- distinct from
        # CLAUDE_MIN_CONFIDENCE (which only gates the separate /analyze
        # endpoint). Defaults to PilotConfig's existing 45 if unset, so
        # behavior is unchanged unless this is explicitly configured.
        min_confidence=int(os.getenv("PILOT_MIN_CONFIDENCE", "45")),
        account_equity=config.trading.account_capital,
        max_risk_pct=config.trading.risk_pct_per_trade,
        lot_size=config.trading.default_qty,
        strategy_name=config.trading.strategy_name,
        # Futures execution port (new, opt-in). "options" preserves today's
        # behavior exactly. Set EXECUTION_MODE=futures in settings.env to
        # test the futures paper-trading path -- affects PAPER-mode signal
        # display, SL/TP delta handling, and position sizing only; live
        # order placement/symbol routing for futures is not wired yet, so
        # this must stay paired with DRY_RUN=true.
        execution_mode=os.getenv("EXECUTION_MODE", "options").strip().lower(),
        futures_margin_pct_estimate=float(os.getenv("FUTURES_MARGIN_PCT_ESTIMATE", "0.15")),
        # 0 = disabled (no extra filtering beyond the existing confidence
        # gate). Only set this once you have real `confidence` values from
        # actual futures paper-mode runs to calibrate against -- see the
        # comment at PilotConfig.futures_min_confidence_pct.
        futures_min_confidence_pct=float(os.getenv("FUTURES_MIN_CONFIDENCE_PCT", "0")),
        # PCR-alignment confidence boost (new, opt-in) -- see the comment at
        # PilotConfig.pcr_alignment_boost_enabled for the backtest evidence
        # and its caveats. Disabled by default; set
        # PCR_ALIGNMENT_BOOST_ENABLED=true to turn on.
        pcr_alignment_boost_enabled=os.getenv("PCR_ALIGNMENT_BOOST_ENABLED", "false").lower() == "true",
        pcr_alignment_boost_pct=float(os.getenv("PCR_ALIGNMENT_BOOST_PCT", "3")),
        # HH/HL structure feature (parallel output only -- logged every cycle,
        # never read by any gate/threshold). Defaults to enabled since it has
        # no effect on trade decisions; set false to disable the computation
        # and its log fields entirely.
        enable_hh_hl_feature=os.getenv("FEATURE_HH_HL_ENABLED", "true").lower() == "true",
        # Frozen validated ATR exit (opt-in) -- implements EXACTLY the
        # ATR(10)/TP=2.0x/SL=6.0x/7-bar-hold exit validated by the 2026-07
        # walk-forward research. Defaults to disabled so behavior is
        # unchanged unless explicitly enabled. See PilotConfig.__post_init__
        # in core/claude_pilot.py for exactly what this does and does not
        # touch (exit mechanics only -- no entry-side changes).
        use_frozen_atr_exit=os.getenv("USE_FROZEN_ATR_EXIT", "false").lower() == "true",
    )
    logger.info(
        f"💰 Position sizing: capital=₹{pilot_config.account_equity:,.0f} | "
        f"risk/trade={pilot_config.max_risk_pct*100:.1f}% | "
        f"lot_size={pilot_config.lot_size}"
    )
    pilot = ClaudePilot(
        trader=trader,
        analyzer=analyzer,
        ml_engine=ml_engine if ml_loaded else None,
        notifier=notifications,
        config=pilot_config,
    )

    # OpenClaw Agent (autonomous AI -- searches web, controls pilot)
    agent = init_agent(
        anthropic_key=config.claude.api_key,
        openai_key=config.claude.openai_api_key,
        model=config.claude.model,
        openai_model=config.claude.openai_model,
    )
    if agent.enabled:
        logger.info("🤖 OpenClaw Agent: ready (autonomous mode available)")
    else:
        logger.info("🤖 OpenClaw Agent: disabled (no API keys)")

    return trader, analyzer, monitor, pilot, config, logger


# ------------------------------------------------------------------
# HTTP API Server (for OpenClaw to call)
# ------------------------------------------------------------------

# Global reference set by main()
_trader: NiftyOptionsTrader = None
_analyzer: ClaudeAnalyzer = None
_monitor: StrategyMonitor = None
_pilot: ClaudePilot = None
_config = None
_logger = None

# API login-session state (see TradeHandler._check_auth/_handle_login).
# Single-threaded HTTPServer (not ThreadingHTTPServer) — no lock needed.
_login_activated = False       # once True, auth is enforced from then on
_active_token = {"value": None, "expires_at": 0.0}
_login_failed_attempts = 0
_login_locked_until = 0.0


def _parse_expiry(expiry_str: str) -> date:
    """Parse expiry from various formats."""
    if not expiry_str:
        # Use NIFTY_EXPIRY env var (actual expiry date, handles holiday shifts)
        try:
            from core.expiry_utils import get_expiry_date
            exp = get_expiry_date()
            if exp is not None:
                return exp
        except Exception:
            pass
        # Last resort fallback: next Tuesday
        today = date.today()
        days_ahead = (1 - today.weekday()) % 7 or 7
        return today + timedelta(days=days_ahead)

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d%b%y", "%d%b%Y"):
        try:
            return datetime.strptime(expiry_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse expiry: {expiry_str}")


def _build_dashboard() -> dict:
    """
    Aggregate a live risk/health snapshot from existing components —
    RiskManager (daily P&L, exposure, pause state), PerformanceTracker
    (win rate/profit factor/drawdown), KotakNeoClient (broker connectivity),
    and the deadman-switch heartbeat file (Telegram heartbeat freshness).
    No new state is created here; this is purely a read-only aggregation
    of signals that already exist, following the same pattern as the
    existing /health and /performance endpoints.
    """
    result: dict = {}

    # ── RiskManager: daily P&L, open positions, strategy-health pause ──
    if _trader is not None and getattr(_trader, "risk", None) is not None:
        risk = _trader.risk
        now = time.time()
        result["daily_pnl"] = risk.daily_pnl
        result["open_positions"] = risk.open_positions
        result["consecutive_losses"] = risk.consecutive_losses
        result["strategy_paused"] = bool(risk.paused_until and now < risk.paused_until)
        result["pause_reason"] = risk.pause_reason if result["strategy_paused"] else ""
    else:
        result["risk_manager"] = "unavailable"

    # ── Open exposure: this bot holds at most ONE position at a time ──
    # (_pilot._live_position is a singular field, not a list), so "total
    # risk" here means "current position's risk, if any" — not a sum
    # across positions.
    open_exposure = None
    if _pilot is not None and getattr(_pilot, "_live_position", None) is not None:
        pos = _pilot._live_position
        risk_pts = abs(pos.entry_price - pos.sl_price)
        delta = pos.entry_delta if pos.entry_delta > 0 else None
        open_exposure = {
            "direction": pos.direction,
            "symbol": pos.symbol,
            "risk_points": round(risk_pts, 1),
            # Rupee conversion needs a live delta; omit rather than fabricate
            # a misleading number when delta hasn't been captured yet.
            "risk_rupees": round(risk_pts * delta * pos.original_qty, 0) if delta else None,
            "qty": pos.original_qty,
        }
    result["open_exposure"] = open_exposure

    # ── Performance: reuse the existing PerformanceTracker via the pilot ──
    if _pilot is not None:
        try:
            perf = _pilot.get_performance(days=30)
            s = perf.get("summary", {})
            result["win_rate_pct"] = s.get("win_rate_pct")
            result["profit_factor"] = s.get("profit_factor")
            result["max_drawdown_rupees"] = s.get("max_drawdown_rupees")
            result["avg_r_multiple"] = s.get("avg_r_multiple")
        except Exception as e:
            result["performance_error"] = str(e)

    # ── Broker connectivity ──────────────────────────────────────────
    if _trader is not None and getattr(_trader, "client", None) is not None:
        client = _trader.client
        result["broker_session_valid"] = getattr(client, "_session_valid", None)
        result["broker_healthy"] = client.is_healthy() if hasattr(client, "is_healthy") else None
    else:
        result["broker_session_valid"] = None
        result["broker_healthy"] = None

    # ── Telegram heartbeat freshness (reuses deadman_switch.read_heartbeat) ──
    try:
        from core.deadman_switch import read_heartbeat, PILOT_STALE_SEC
        hb = read_heartbeat()
        if hb:
            age_sec = int(time.time()) - hb["ts"]
            result["heartbeat"] = {
                "age_sec": age_sec,
                "cycle": hb["cycle"],
                "status": hb["status"],
                "stale": age_sec > PILOT_STALE_SEC,
            }
        else:
            result["heartbeat"] = None
    except Exception as e:
        result["heartbeat"] = None
        result["heartbeat_error"] = str(e)

    result["telegram_configured"] = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"

    # ── Archiver health (OI + futures research collectors) ────────────
    # Purely observational -- these read-only health snapshots are never
    # consulted by any trading/signal-generation code path.
    try:
        from core.openalgo_client import get_archive_health
        result["oi_archive_hook_health"] = get_archive_health()
    except Exception as e:
        result["oi_archive_hook_health"] = {"error": str(e)}
    try:
        from scripts.oi_archiver import get_health as _oi_daemon_health
        result["oi_archiver_daemon_health"] = _oi_daemon_health()
    except Exception as e:
        result["oi_archiver_daemon_health"] = {"error": str(e)}
    try:
        from scripts.futures_archiver import get_health as _fut_daemon_health
        result["futures_archiver_health"] = _fut_daemon_health()
    except Exception as e:
        result["futures_archiver_health"] = {"error": str(e)}

    # Which gates blocked entries today, biggest first. Five of the twelve
    # gates (IV, OI FILTER, PCR, BRIEF bias, STRUCTURE PENALTY) have too
    # little history to backtest, so this live tally is the only way they
    # ever get measured. Read-only, same as everything else here.
    try:
        from core.claude_pilot import get_gate_block_tally
        result["gate_blocks_today"] = get_gate_block_tally()
    except Exception as e:
        result["gate_blocks_today"] = {"error": str(e)}

    return result


class TradeHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for trade commands from OpenClaw.

    Endpoints:
        POST /login       — {"mpin": "..."} -> session token (see _handle_login)
        POST /trade       — Execute a trade command
        POST /quote       — Get option quote
        POST /positions   — Get current positions
        POST /funds       — Get available funds
        POST /cancel_all  — Cancel all open orders
        GET  /health      — Health check
        GET  /performance — Aggregate win rate / P&L / drawdown (see PerformanceTracker)
        GET  /dashboard   — Combined live risk/health snapshot (P&L, exposure,
                             win rate, broker connectivity, Telegram heartbeat)
    """

    def _check_auth(self) -> bool:
        """
        Two independent, coexisting auth mechanisms:
          1. Static shared secret via API_AUTH_TOKEN (always checked if set).
          2. A session token obtained via POST /login (see _handle_login).

        If neither API_AUTH_TOKEN is set NOR has any /login ever succeeded
        in this process's lifetime, auth is not enforced — backward
        compatible with existing deployments (logged loudly at startup,
        see main()). The moment a /login succeeds, enforcement turns on
        and stays on: a valid token (either mechanism) is required from
        then on, even after that session token itself expires.
        """
        global _login_activated
        static_token = os.getenv("API_AUTH_TOKEN", "")
        if not static_token and not _login_activated:
            return True

        auth_header = self.headers.get("Authorization", "")
        provided = auth_header[len("Bearer "):] if auth_header.startswith("Bearer ") else ""

        if static_token and provided == static_token:
            return True
        if (
            _login_activated
            and provided
            and provided == _active_token["value"]
            and time.time() < _active_token["expires_at"]
        ):
            return True

        self._respond(401, {"error": "Unauthorized"})
        return False

    def _handle_login(self, body: dict):
        """
        POST /login {"mpin": "..."} -> {"token": "...", "expires_in_sec": N}

        Checked against KOTAK_MPIN (already required for the bot's own
        broker login). 5 failed attempts locks out further attempts for
        5 minutes, mirroring RiskManager's consecutive-failure pause
        pattern (security/__init__.py) — a 4-6 digit PIN is otherwise
        trivially brute-forceable on an internet-reachable port.
        """
        global _login_activated, _login_failed_attempts, _login_locked_until

        now = time.time()
        if now < _login_locked_until:
            self._respond(429, {
                "error": f"Too many failed attempts — locked out for "
                         f"{int(_login_locked_until - now)}s"
            })
            return

        mpin = str(body.get("mpin", ""))
        real_mpin = os.getenv("KOTAK_MPIN", "")
        if not real_mpin or not secrets.compare_digest(mpin, real_mpin):
            _login_failed_attempts += 1
            if _login_failed_attempts >= 5:
                _login_locked_until = now + 300  # 5 min lockout
                _login_failed_attempts = 0
                _logger.warning("🔒 /login: 5 failed attempts — locked out for 5 minutes")
            self._respond(401, {"error": "Invalid credentials"})
            return

        _login_failed_attempts = 0
        ttl_min = float(os.getenv("API_TOKEN_TTL_MIN", "60"))
        token = secrets.token_urlsafe(32)
        _active_token["value"] = token
        _active_token["expires_at"] = now + ttl_min * 60
        _login_activated = True
        _logger.info(f"🔑 /login succeeded — session token issued (TTL={ttl_min:.0f}min)")
        self._respond(200, {"token": token, "expires_in_sec": int(ttl_min * 60)})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "service": "nifty-trader"})
            return
        if not self._check_auth():
            return
        if self.path.startswith("/performance"):
            # /performance or /performance?days=7
            days = 30
            if "?" in self.path:
                try:
                    qs = self.path.split("?", 1)[1]
                    for kv in qs.split("&"):
                        if kv.startswith("days="):
                            days = int(kv.split("=", 1)[1])
                except Exception:
                    pass
            global _pilot
            if _pilot is None:
                self._respond(503, {"error": "Pilot not running"})
                return
            self._respond(200, _pilot.get_performance(days=days))
        elif self.path == "/dashboard":
            self._respond(200, _build_dashboard())
        else:
            self._respond(404, {"error": "Not found"})

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        if self.path == "/login":
            self._handle_login(body)
            return

        if not self._check_auth():
            return

        handlers = {
            "/trade": self._handle_trade,
            "/analyze": self._handle_analyze,
            "/pilot": self._handle_pilot,
            "/agent": self._handle_agent,
            "/monitor": self._handle_monitor,
            "/quote": self._handle_quote,
            "/strikes": self._handle_strikes,
            "/positions": self._handle_positions,
            "/funds": self._handle_funds,
            "/cancel_all": self._handle_cancel_all,
            "/close": self._handle_close,
            "/emergency_stop": self._handle_emergency_stop,
        }

        handler = handlers.get(self.path)
        if handler:
            handler(body)
        else:
            self._respond(404, {"error": f"Unknown endpoint: {self.path}"})

    def _handle_trade(self, body: dict):
        """
        Execute a trade with smart auto-strike selection.

        Body:
        {
            "action": "BUY" | "SELL",
            "option_type": "CE" | "PE",
            "strike": 24500,              (optional — overrides strike_mode)
            "strike_mode": "ATM",         (optional — default: ATM)
                Modes: ATM, OTM_1, OTM_2, OTM_3, ITM_1, ITM_2, PREMIUM
            "target_premium": 150.0,      (optional — required for PREMIUM mode)
            "expiry": "2025-03-20",       (optional — defaults to next Thursday)
            "qty": 50,                    (optional — defaults to config)
            "price_type": "MARKET",       (optional)
            "price": 0.0                  (optional — for LIMIT orders)
        }

        When "strike" is omitted, the system auto-selects using strike_mode:
          ATM   → nearest strike to live Nifty spot
          OTM_1 → 1 strike out of the money (cheaper premium)
          OTM_2 → 2 strikes OTM
          ITM_1 → 1 strike in the money (higher premium)
          PREMIUM → scans chain, finds strike nearest to target_premium
        """
        try:
            action = body.get("action", "").upper()
            option_type = body.get("option_type", "").upper()
            strike = body.get("strike")
            strike_mode = body.get("strike_mode", "ATM").upper().strip()
            target_premium = body.get("target_premium")
            expiry_str = body.get("expiry", "")
            qty = body.get("qty")
            price_type = body.get("price_type", "MARKET")
            price = float(body.get("price", 0))

            if action not in ("BUY", "SELL"):
                self._respond(400, {"error": "action must be BUY or SELL"})
                return
            if option_type not in ("CE", "PE"):
                self._respond(400, {"error": "option_type must be CE or PE"})
                return

            if target_premium is not None:
                target_premium = float(target_premium)

            expiry = _parse_expiry(expiry_str)

            # Use smart_trade — handles both manual strike and auto-selection
            result = _trader.smart_trade(
                action=action,
                option_type=option_type,
                strike_mode=strike_mode,
                strike=int(strike) if strike else None,
                expiry=expiry,
                qty=qty,
                price_type=price_type,
                price=price,
                target_premium=target_premium,
            )

            self._respond(200, result)

        except Exception as e:
            _logger.error(f"Trade error: {e}")
            self._respond(500, {"error": str(e)})

    def _handle_strikes(self, body: dict):
        """
        Get smart strike recommendation or full option chain.

        Body (recommendation):
        {
            "option_type": "CE" | "PE",
            "strike_mode": "ATM",            (default)
            "target_premium": 150.0           (optional, for PREMIUM mode)
        }

        Body (full chain):
        {
            "chain": true,
            "width": 5                        (optional, default 10)
        }
        """
        try:
            if body.get("chain"):
                width = int(body.get("width", 10))
                expiry = _parse_expiry(body.get("expiry", ""))
                result = _trader.get_option_chain(expiry=expiry, width=width)
                self._respond(200, result)
            else:
                option_type = body.get("option_type", "CE").upper()
                mode = body.get("strike_mode", "ATM").upper().strip()
                target_premium = body.get("target_premium")
                expiry = _parse_expiry(body.get("expiry", ""))

                if target_premium is not None:
                    target_premium = float(target_premium)

                result = _trader.strikes.select_strike(
                    option_type=option_type,
                    mode=mode,
                    expiry=expiry,
                    target_premium=target_premium,
                )
                self._respond(200, result)

        except Exception as e:
            _logger.error(f"Strikes error: {e}")
            self._respond(500, {"error": str(e)})

    def _handle_analyze(self, body: dict):
        """
        Ask Claude to analyze current market and recommend a trade.

        Body (optional):
        {
            "context": "I'm bullish today",     (optional extra context)
            "execute": false                     (if true AND confidence >= min, auto-trade)
        }

        Returns Claude's recommendation with:
        - action (BUY/SELL/WAIT)
        - option_type (CE/PE)
        - strike_mode, confidence, analysis, key_levels
        - If execute=true and confidence is high enough, also executes the trade
        """
        try:
            if not _analyzer or not _analyzer.enabled:
                self._respond(400, {
                    "error": "Claude analyzer is not enabled. "
                             "Set CLAUDE_ENABLED=true and ANTHROPIC_API_KEY in settings.env"
                })
                return

            extra_context = body.get("context", "")
            should_execute = body.get("execute", False)

            # Gap override removed — gap magnitude is passed as ML features only.
            # The pilot no longer hard-overrides direction from gap context.

            # Run Claude analysis with context from morning brief / OpenClaw
            recommendation = _analyzer.analyze_with_trader(
                _trader, extra_context=extra_context)

            # Auto-execute if requested and confidence is high enough
            if (
                should_execute
                and recommendation.get("action") in ("BUY", "SELL")
                and recommendation.get("confidence", 0) >= _config.claude.min_confidence
            ):
                _logger.info(
                    f"🧠→🚀 Auto-executing Claude's recommendation: "
                    f"{recommendation['action']} {recommendation.get('option_type', '')}"
                )

                expiry = _parse_expiry("")
                trade_result = _trader.smart_trade(
                    action=recommendation["action"],
                    option_type=recommendation.get("option_type", "CE"),
                    strike_mode=recommendation.get("strike_mode", "ATM"),
                    strike=recommendation.get("strike"),
                    expiry=expiry,
                    price_type=recommendation.get("price_type", "MARKET"),
                )
                recommendation["executed"] = True
                recommendation["trade_result"] = trade_result
            else:
                recommendation["executed"] = False
                if should_execute and recommendation.get("action") != "WAIT":
                    recommendation["execution_note"] = (
                        f"Not executed: confidence {recommendation.get('confidence', 0)}% "
                        f"< minimum {_config.claude.min_confidence}%"
                    )

            self._respond(200, recommendation)

        except Exception as e:
            _logger.error(f"Analyze error: {e}")
            self._respond(500, {"error": str(e)})

    def _handle_pilot(self, body: dict):
        """Control dual-brain auto-pilot (ML + Claude)."""
        try:
            action = body.get("action", "status").lower()

            if action == "start":
                # Apply overrides to pilot config
                if "interval" in body:
                    _pilot.config.analyze_interval = int(body["interval"])
                if "min_confidence" in body:
                    _pilot.config.min_confidence = int(body["min_confidence"])
                if "max_trades" in body:
                    _pilot.config.max_trades_per_day = int(body["max_trades"])
                if "cooldown" in body:
                    _pilot.config.cooldown_after_trade = int(body["cooldown"])
                if "ml_only" in body:
                    _pilot.config.ml_only_mode = str(body["ml_only"]).lower() == "true"

                _pilot.start()
                self._respond(200, {
                    "status": "started",
                    "details": _pilot.get_status(),
                })

            elif action == "stop":
                _pilot.stop()
                self._respond(200, {
                    "status": "stopped",
                    "details": _pilot.get_status(),
                })

            elif action == "status":
                self._respond(200, _pilot.get_status())

            else:
                self._respond(400, {"error": f"Unknown action: {action}"})

        except Exception as e:
            _logger.error(f"Pilot error: {e}")
            self._respond(500, {"error": str(e)})

    def _handle_agent(self, body: dict):
        """
        Control the OpenClaw autonomous agent.

        Body:
        {"action": "start"}             -- Start agent scheduler (morning/intraday/evening)
        {"action": "stop"}              -- Stop agent
        {"action": "status"}            -- Get agent status
        {"action": "morning"}           -- Run morning routine NOW
        {"action": "check"}             -- Run intraday check NOW
        {"action": "squareoff"}         -- Run evening square-off NOW
        {"action": "run", "task": "..."} -- Run any custom task
        """
        try:
            agent = get_agent()
            if not agent or not agent.enabled:
                self._respond(400, {
                    "error": "OpenClaw agent not available. "
                             "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in settings.env"
                })
                return

            action = body.get("action", "status").lower()

            if action == "start":
                agent.start()
                self._respond(200, {"status": "started", "details": agent.get_status()})

            elif action == "stop":
                agent.stop()
                self._respond(200, {"status": "stopped"})

            elif action == "status":
                self._respond(200, agent.get_status())

            elif action == "morning":
                result = agent.morning_routine()
                self._respond(200, {"status": "done", "result": result})

            elif action == "check":
                result = agent.intraday_check()
                self._respond(200, {"status": "done", "result": result})

            elif action == "squareoff":
                result = agent.evening_squareoff()
                self._respond(200, {"status": "done", "result": result})

            elif action == "run":
                task = body.get("task", "")
                if not task:
                    self._respond(400, {"error": "Missing 'task' field"})
                    return
                result = agent.run_task(task, max_turns=body.get("max_turns", 10))
                self._respond(200, {"status": "done", "result": result})

            else:
                self._respond(400, {"error": f"Unknown action: {action}"})

        except Exception as e:
            _logger.error(f"Agent error: {e}")
            self._respond(500, {"error": str(e)})

    def _handle_monitor(self, body: dict):
        """
        Control the strategy monitor.

        Body:
        {"action": "start"}                    — Start monitoring with today's setups
        {"action": "start", "auto_execute": true}  — Start + auto-trade on triggers
        {"action": "stop"}                     — Stop monitoring
        {"action": "status"}                   — Get current status
        {"action": "add", "setup": {...}}      — Add a custom setup

        Custom setup fields:
        {
            "name": "MY_SETUP",
            "direction": "LONG" | "SHORT",
            "option_type": "CE" | "PE",
            "strike_mode": "ATM",
            "watch_zone_low": 23400,
            "watch_zone_high": 23500,
            "trigger_type": "BOUNCE" | "REJECTION" | "TOUCH",
            "stop_loss": 23300,
            "target": 23800,
            "auto_execute": false,
            "qty": 50
        }
        """
        try:
            action = body.get("action", "status").lower()

            if action == "start":
                auto = body.get("auto_execute", False)
                setups = build_todays_setups()
                if auto:
                    for s in setups:
                        s.auto_execute = True
                _monitor.load_setups(setups)
                _monitor.start()
                self._respond(200, {
                    "status": "started",
                    "setups": len(setups),
                    "auto_execute": auto,
                    "details": _monitor.status,
                })

            elif action == "stop":
                _monitor.stop()
                self._respond(200, {"status": "stopped"})

            elif action == "status":
                self._respond(200, _monitor.status)

            elif action == "add":
                setup_data = body.get("setup", {})
                if not setup_data.get("name"):
                    self._respond(400, {"error": "setup.name is required"})
                    return

                setup = LevelSetup(
                    name=setup_data["name"],
                    direction=setup_data.get("direction", "LONG").upper(),
                    option_type=setup_data.get("option_type", "CE").upper(),
                    strike_mode=setup_data.get("strike_mode", "ATM"),
                    watch_zone_low=float(setup_data.get("watch_zone_low", 0)),
                    watch_zone_high=float(setup_data.get("watch_zone_high", 0)),
                    trigger_type=setup_data.get("trigger_type", "TOUCH").upper(),
                    confirm_candles=int(setup_data.get("confirm_candles", 2)),
                    confirm_direction=setup_data.get("confirm_direction", "UP").upper(),
                    qty=int(setup_data.get("qty", 50)),
                    stop_loss=float(setup_data.get("stop_loss", 0)),
                    target=float(setup_data.get("target", 0)),
                    auto_execute=bool(setup_data.get("auto_execute", False)),
                    max_triggers=int(setup_data.get("max_triggers", 1)),
                )
                _monitor.add_setup(setup)
                self._respond(200, {
                    "status": "added",
                    "setup": setup.name,
                    "details": _monitor.status,
                })

            else:
                self._respond(400, {"error": f"Unknown action: {action}"})

        except Exception as e:
            _logger.error(f"Monitor error: {e}")
            self._respond(500, {"error": str(e)})

    def _handle_quote(self, body: dict):
        try:
            symbol = body.get("symbol", "")
            exchange = body.get("exchange", "")

            if not symbol:
                # Default: return Nifty spot quote
                symbol = "NIFTY"
                exchange = "NSE_INDEX"

            if not exchange:
                exchange = "NSE_INDEX" if symbol.upper() in ("NIFTY", "BANKNIFTY") else "NFO"

            quote = _trader.client.get_quote(symbol, exchange)
            self._respond(200, {"symbol": symbol, "exchange": exchange, "quote": quote})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_positions(self, body: dict):
        try:
            positions = _trader.get_positions_summary()
            self._respond(200, {"positions": positions})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_funds(self, body: dict):
        try:
            funds = _trader.get_funds_available()
            self._respond(200, {"funds": funds})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_cancel_all(self, body: dict):
        try:
            result = _trader.cancel_all()
            self._respond(200, {"result": result})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_close(self, body: dict):
        """
        Independent-verification fix #4 (2026-07-21): routed through
        ClaudePilot.manual_close() -- which uses the same
        _attempt_protected_close() flow as monitor exits and
        emergency_stop() -- instead of calling trader.close_position()
        directly. The old direct call never cancelled the exchange SL and
        never updated _live_position, silently desyncing the pilot.
        `symbol` is now optional: manual close acts on whatever position
        the pilot is currently tracking (this bot holds at most one
        position at a time), validated against `symbol` if provided.
        """
        try:
            symbol = body.get("symbol", "")
            result = _pilot.manual_close(symbol=symbol)
            status_code = 200 if result.get("status") == "success" else 400
            self._respond(status_code, {"result": result})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_emergency_stop(self, body: dict):
        """
        Operational-safety fix #3 (2026-07-21): stop new entries, cancel
        pending orders, flatten any open position with broker confirmation,
        verify the book is flat, then stop the pilot's monitoring threads —
        in that order. Unlike POST /pilot {"action":"stop"}, this actually
        flattens open positions rather than only stopping the loop.
        """
        try:
            reason = body.get("reason", "manual_api")
            result = _pilot.emergency_stop(reason=reason)
            self._respond(200, {"result": result})
        except Exception as e:
            _logger.critical(f"Emergency stop endpoint failed: {e}")
            self._respond(500, {"error": str(e)})

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        _logger.debug(f"HTTP: {args[0]}")


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

def main():
    global _trader, _analyzer, _monitor, _pilot, _config, _logger

    parser = argparse.ArgumentParser(description="OpenClaw Nifty Options Trader")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    args = parser.parse_args()

    _trader, _analyzer, _monitor, _pilot, _config, _logger = bootstrap()

    # Fetch live spot at startup to show real examples.
    # Retry a few times so the WebSocket has a moment to deliver the first
    # NIFTY tick — at app launch, OHLCVBuilder may be empty for ~2-5 seconds
    # and the Kotak quotes REST endpoint sometimes returns ltp=0 right at
    # market-open seconds.
    import time as _t
    spot = 0.0
    atm = "???"
    otm1_ce = "???"
    otm2_pe = "???"
    last_err: Exception | None = None

    # Skip the cosmetic spot fetch outside market hours — no point hitting
    # Kotak before 09:15 (lazy-login defers anyway, and NSE/TV are noisy at
    # 04:00 when bot starts via cron). Pilot fetches its own spot at first
    # cycle once the market is open.
    from datetime import datetime as _dt
    _now = _dt.now()
    _in_market = (
        _now.weekday() < 5 and (
            (_now.hour == 9 and _now.minute >= 10) or
            (10 <= _now.hour <= 14) or
            (_now.hour == 15 and _now.minute <= 30)
        )
    )
    if not _in_market:
        _logger.info(
            "📈 Skipping startup spot fetch — market closed. "
            "Live data fires at first pilot cycle (09:15)."
        )
    else:
        for attempt in range(1, 6):  # 5 tries × 2s = ~10s grace
            try:
                spot = _trader.get_nifty_spot()
                atm = _trader.get_atm_strike()
                otm1_ce = atm + 50
                otm2_pe = atm - 100
                _logger.info(f"📈 Nifty spot: ₹{spot:.2f} → ATM strike: {atm}")
                break
            except Exception as e:
                last_err = e
                if attempt < 5:
                    _logger.info(f"   spot fetch attempt {attempt}/5 — waiting for WebSocket tick...")
                    _t.sleep(2.0)
        else:
            _logger.warning(f"⚠️ Could not fetch live spot at startup: {last_err}")
            _logger.warning("   Auto-strike will still work when you place a trade "
                            "(OHLCVBuilder will have ticks by then).")

    server = HTTPServer(("0.0.0.0", args.port), TradeHandler)
    _logger.info(f"🌐 HTTP API server running on http://0.0.0.0:{args.port}")
    _logger.info("   Endpoints: POST /trade, /analyze, /pilot, /agent, /monitor, /strikes, /quote, /positions, /funds, /cancel_all, /close")
    _logger.info("              GET  /health")
    _logger.info("")
    _logger.info("   Auto-strike examples (live):")
    _logger.info(f'     ATM CE:     {{"action":"BUY","option_type":"CE"}}  → strike {atm}')
    _logger.info(f'     OTM_1 CE:   {{"action":"BUY","option_type":"CE","strike_mode":"OTM_1"}}  → strike {otm1_ce}')
    _logger.info(f'     OTM_2 PE:   {{"action":"BUY","option_type":"PE","strike_mode":"OTM_2"}}  → strike {otm2_pe}')
    _logger.info( '     By premium: {"action":"SELL","option_type":"CE","strike_mode":"PREMIUM","target_premium":150}')
    _logger.info(f'     Manual:     {{"action":"BUY","option_type":"CE","strike":{atm}}}')
    _logger.info( '     Chain:      POST /strikes {"chain":true,"width":5}')
    if _analyzer and _analyzer.enabled:
        _logger.info("")
        _logger.info("   Claude AI analysis:")
        _logger.info( '     Analyze:    POST /analyze {}')
        _logger.info( '     + Execute:  POST /analyze {"execute":true}')
        _logger.info( '     + Context:  POST /analyze {"context":"market looks bullish"}')
    _logger.info("")
    _logger.info("   Level monitor (watches support/resistance, alerts on trigger):")
    _logger.info( '     Start:      POST /monitor {"action":"start"}')
    _logger.info( '     Auto-trade: POST /monitor {"action":"start","auto_execute":true}')
    _logger.info( '     Status:     POST /monitor {"action":"status"}')
    _logger.info( '     Stop:       POST /monitor {"action":"stop"}')
    if _analyzer and _analyzer.enabled:
        _logger.info("")
        _logger.info("   Claude Auto-Pilot (fully autonomous trading):")
        _logger.info( '     Start:      POST /pilot {"action":"start"}')
        _logger.info( '     Custom:     POST /pilot {"action":"start","interval":180,"min_confidence":70,"max_trades":3}')
        _logger.info( '     Status:     POST /pilot {"action":"status"}')
        _logger.info( '     Stop:       POST /pilot {"action":"stop"}')
    _logger.info("")
    _logger.info("Ready for OpenClaw commands. Press Ctrl+C to stop.")

    # Auto-start pilot if configured (always ML-only — Claude moved to PostTradeAnalyzer)
    if os.getenv("PILOT_AUTO_START", "false").lower() == "true":
        has_ml = ml_engine.is_ready()
        if has_ml:
            _pilot.config.ml_only_mode = True   # Claude removed from live path
            _logger.info("🤖 PILOT_AUTO_START — ML-direct mode (zero LLM latency)")
            _pilot.start()
        else:
            _logger.warning("🤖 PILOT_AUTO_START=true but no ML model found")

    # Start PostTradeAnalyzer — async Claude coaching (does NOT block live trading)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            from core.post_trade_analyzer import PostTradeAnalyzer
            _post_analyzer = PostTradeAnalyzer(api_key=anthropic_key)
            _post_analyzer.start()
            _logger.info(
                "📊 PostTradeAnalyzer started — Claude reviews trades every 30min "
                "(async, zero impact on live execution)"
            )
        except Exception as _e:
            _logger.warning(f"PostTradeAnalyzer start failed: {_e}")

    # ── Pre-market Briefing Agent (2026-04-25) — daily 08:30 IST regime call
    if os.getenv("PREMARKET_AGENT_ENABLED", "true").lower() == "true":
        try:
            from core.premarket_agent import schedule_daily_brief
            schedule_daily_brief()
            _logger.info(
                "📋 PreMarketAgent enabled — daily brief at "
                f"{os.getenv('PREMARKET_BRIEF_TIME', '08:30')} IST "
                f"(provider={os.getenv('PREMARKET_PROVIDER', 'groq')})"
            )
        except Exception as _e:
            _logger.warning(f"PreMarketAgent start failed (fail-open): {_e}")

    # ── 24/7 News Agent (2026-04-24) — advisory sentiment source ──────
    if os.getenv("NEWS_AGENT_ENABLED", "true").lower() == "true":
        try:
            from core.news_agent import get_news_agent
            _news_agent = get_news_agent()
            # Wire notifier so breaking events fan out to Telegram.
            # `notifications` is constructed inside bootstrap() — reach it via
            # the trader (it was passed in as `notifications=` kwarg).
            _notifier = getattr(_trader, "notif", None) or getattr(_trader, "notifications", None)
            if _notifier is not None:
                _news_agent.set_notifier(_notifier)
            else:
                _logger.warning(
                    "NewsAgent: no notifier on trader — Telegram news alerts disabled"
                )
            _news_agent.start()
            _logger.info(
                "📰 NewsAgent started — polling NSE/SEBI/MoneyControl/ET every "
                f"{getattr(_news_agent, 'poll_interval', 180)}s "
                "(advisory gate in claude_pilot._decide_entry)"
            )
        except Exception as _e:
            _logger.warning(f"NewsAgent start failed (fail-open): {_e}")
    else:
        _logger.info("📰 NewsAgent disabled (NEWS_AGENT_ENABLED=false)")

    # ── Native WebSocket OHLCV (Fix #2) — DISABLED BY DEFAULT ──────────
    # Live market data comes from TradingView (premium subscription). The
    # native WS→OHLCVBuilder pipeline below is kept intact as a backup —
    # set NATIVE_OHLCV=true in settings.env to switch over.
    _tick_queue    = None
    _tick_feed     = None
    _ohlcv_builder = None
    if os.getenv("NATIVE_OHLCV", "false").lower() == "true":
        try:
            from core.tick_feed    import TickFeed
            from core.ohlcv_builder import OHLCVBuilder, set_ohlcv_builder

            _tick_queue    = queue.Queue(maxsize=20_000)
            _ohlcv_builder = OHLCVBuilder(tick_queue=_tick_queue)
            _ohlcv_builder.start()
            set_ohlcv_builder(_ohlcv_builder)

            _tick_feed = TickFeed(
                neo_client=_trader.client,
                tick_queue=_tick_queue,
                spot_callback=_pilot.update_tick_spot,  # 1s position monitor
            )
            _tick_feed.start()
            _logger.info(
                "📡 NATIVE_OHLCV=true → WebSocket tick feed + OHLCVBuilder active"
            )

            # BUG FIX 2026-05-25: at startup (~04:00) Kotak session is not yet
            # authenticated (KOTAK_LAZY_LOGIN=true), so _resolve_futures_token()
            # silently fails and the WebSocket launches with futures=[].
            # Register a post-login callback so futures subscription is added
            # to the live WebSocket immediately after the 09:05 login completes.
            _trader.client.register_post_login_callback(
                _tick_feed.retry_futures_subscription
            )
            _logger.info(
                "📡 TickFeed: futures subscription wired to Kotak post-login "
                "callback (will activate at 09:05 login)"
            )
        except Exception as _e:
            _logger.warning(
                f"⚠️  WebSocket OHLCV pipeline failed to start: {_e} "
                f"— falling back to TradingView"
            )
    else:
        _logger.info(
            "📊 OHLCV source: TradingView (premium). "
            "Set NATIVE_OHLCV=true to use the WS-tick builder instead."
        )

    # Auto-start OpenClaw agent if configured
    agent = get_agent()
    if os.getenv("AGENT_AUTO_START", "false").lower() == "true" and agent and agent.enabled:
        agent.start()
        _logger.info("🧠 AGENT_AUTO_START — autonomous mode (searches web, controls pilot)")
    elif agent and agent.enabled:
        _logger.info("🧠 OpenClaw Agent: ready (POST /agent {\"action\":\"start\"} to activate)")
        _logger.info("   Manual:  POST /agent {\"action\":\"morning\"} — run morning routine now")
        _logger.info("   Custom:  POST /agent {\"action\":\"run\",\"task\":\"search Nifty outlook and start pilot\"}")

    def _graceful_shutdown():
        _logger.info("\n🛑 Shutting down...")
        # Independent-verification fix #5 (2026-07-21): a plain process
        # shutdown (SIGTERM/SIGINT -- VPS reboot, systemd restart, deploy)
        # previously only called _pilot.stop(), which halts the monitoring
        # threads WITHOUT flattening an open position. If a live position
        # exists, invoke the full emergency_stop() sequence instead (it
        # also stops the threads as its own last step, so stop() would be
        # redundant here). If already flat, stop() alone is unchanged.
        try:
            _has_live_position = _pilot._live_position is not None
        except Exception:
            _has_live_position = False
        if _has_live_position:
            _logger.critical(
                "🚨 Live position open at shutdown — invoking emergency_stop() "
                "to flatten it before exiting"
            )
            try:
                _pilot.emergency_stop(reason="sigterm_shutdown")
            except Exception as e:
                _logger.critical(
                    f"emergency_stop() during shutdown failed ({e}) — "
                    f"falling back to stop() (position may remain open)"
                )
                _pilot.stop()
        else:
            _pilot.stop()
        _monitor.stop()
        if agent:
            agent.stop()
        if _tick_feed is not None:
            _tick_feed.stop()
        if _ohlcv_builder is not None:
            _ohlcv_builder.stop()
        _trader.notif.shutdown()

    def _signal_handler(signum, frame):
        # Signal handlers run on the main thread, which is the same thread
        # blocked inside server.serve_forever(). HTTPServer.shutdown() blocks
        # until serve_forever()'s loop notices the shutdown request and exits
        # — calling it directly here would deadlock (the loop can never
        # notice, since this handler itself is running on that same thread).
        # It's documented as safe to call from another thread, so we do that.
        _logger.info(f"\n🛑 Received signal {signum} — shutting down gracefully...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _signal_handler)
        except (ValueError, OSError, AttributeError) as _e:
            _logger.debug(f"Could not register handler for {_sig}: {_e}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Fallback in case SIGINT reached Python's default handler instead
        # of ours (e.g. registration above failed) — still shut down cleanly.
        threading.Thread(target=server.shutdown, daemon=True).start()
    finally:
        _graceful_shutdown()


if __name__ == "__main__":
    main()
