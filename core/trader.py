"""
Nifty Options Trading Engine.
Handles option symbol construction, strike selection, and trade execution
with integrated notifications and security checks.
"""

import logging
import math
from datetime import datetime, date
from typing import Optional

from core.kotak_neo_client import KotakNeoClient as OpenAlgoClient  # drop-in replacement
from core.strike_selector import StrikeSelector, round_to_strike
from notifications import NotificationManager
from security import RiskManager, AuditLogger

logger = logging.getLogger(__name__)


def build_option_symbol(
    underlying: str,
    expiry: date,
    strike: int,
    option_type: str,
) -> str:
    """
    Build an OpenAlgo-compatible option symbol.
    Format: NIFTY25MAR24500CE  (example)

    Note: OpenAlgo symbol format may vary by broker — verify with
    OpenAlgo's /search endpoint for the exact format Kotak Neo expects.
    """
    # Expiry format: DDMMMYY (e.g., 20MAR25)
    expiry_str = expiry.strftime("%d%b%y").upper()
    ot = option_type.upper()
    if ot not in ("CE", "PE"):
        raise ValueError(f"option_type must be 'CE' or 'PE', got '{ot}'")
    return f"{underlying}{expiry_str}{strike}{ot}"


class NiftyOptionsTrader:
    """
    High-level trading interface for Nifty options.
    Every trade goes through:
      1. Risk check
      2. Withdrawal guard (in the client)
      3. Rate limiter (in the client)
      4. Order placement
      5. Audit log
      6. Email + WhatsApp notification
    """

    def __init__(
        self,
        client: OpenAlgoClient,
        risk_manager: RiskManager,
        audit: AuditLogger,
        notifications: NotificationManager,
        exchange: str = "NFO",
        product: str = "MIS",
        default_qty: int = 50,
        strategy: str = "OpenClawNifty",
    ):
        self.client = client
        self.risk = risk_manager
        self.audit = audit
        self.notif = notifications
        self.exchange = exchange
        self.product = product
        self.default_qty = default_qty
        self.strategy = strategy
        self.strikes = StrikeSelector(client, exchange)

    # ------------------------------------------------------------------
    # Core Trade Actions
    # ------------------------------------------------------------------

    def buy_option(
        self,
        symbol: str,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
    ) -> dict:
        """Buy an option (CE or PE)."""
        return self._execute_trade(
            action="BUY",
            symbol=symbol,
            qty=qty or self.default_qty,
            price_type=price_type,
            price=price,
            trigger_price=trigger_price,
        )

    def sell_option(
        self,
        symbol: str,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
    ) -> dict:
        """Sell an option (CE or PE)."""
        return self._execute_trade(
            action="SELL",
            symbol=symbol,
            qty=qty or self.default_qty,
            price_type=price_type,
            price=price,
            trigger_price=trigger_price,
        )

    def buy_ce(
        self,
        strike: int,
        expiry: date,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
    ) -> dict:
        """Buy a Nifty Call option."""
        symbol = build_option_symbol("NIFTY", expiry, strike, "CE")
        return self.buy_option(symbol, qty, price_type)

    def buy_pe(
        self,
        strike: int,
        expiry: date,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
    ) -> dict:
        """Buy a Nifty Put option."""
        symbol = build_option_symbol("NIFTY", expiry, strike, "PE")
        return self.buy_option(symbol, qty, price_type)

    def sell_ce(
        self,
        strike: int,
        expiry: date,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
    ) -> dict:
        """Sell a Nifty Call option."""
        symbol = build_option_symbol("NIFTY", expiry, strike, "CE")
        return self.sell_option(symbol, qty, price_type)

    def sell_pe(
        self,
        strike: int,
        expiry: date,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
    ) -> dict:
        """Sell a Nifty Put option."""
        symbol = build_option_symbol("NIFTY", expiry, strike, "PE")
        return self.sell_option(symbol, qty, price_type)

    def close_position(self, symbol: str) -> dict:
        """Square off an open option position."""
        import os
        if os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes"):
            logger.info(f"[DRY RUN] close_position skipped broker call: {symbol}")
            return {"status": "success", "message": "DRY_RUN — no broker call made"}
        try:
            result = self.client.close_position(
                symbol=symbol,
                exchange=self.exchange,
                product=self.product,
                strategy=self.strategy,
            )
            status = "success" if result.get("status") == "success" else "failed"
            order_id = result.get("orderid", "N/A")

            # P&L is now recorded by claude_pilot position monitor with real value
            # Do NOT call record_trade_close here to avoid double-counting

            self.audit.log(
                action="CLOSE_POSITION",
                symbol=symbol,
                side="SQUAREOFF",
                order_id=order_id,
                status=status,
            )
            self.notif.notify_trade(
                action="CLOSE_POSITION",
                symbol=symbol,
                side="SQUAREOFF",
                qty=0,
                price=0.0,
                order_id=order_id,
                status=status,
                details="Position squared off" if status == "success" else "Broker did not confirm close — position remains open",
            )
            return result

        except Exception as e:
            self.audit.log(
                action="CLOSE_POSITION",
                symbol=symbol,
                status="error",
                details=str(e),
            )
            self.notif.notify_trade(
                action="CLOSE_POSITION",
                symbol=symbol,
                side="SQUAREOFF",
                qty=0,
                price=0.0,
                order_id="",
                status="error",
                details=str(e),
            )
            raise

    def cancel_all(self) -> dict:
        """Cancel all open orders."""
        import os
        if os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes"):
            logger.info("[DRY RUN] cancel_all skipped broker call")
            return {"status": "success", "cancelled": [], "errors": [], "message": "DRY_RUN — no broker call made"}
        result = self.client.cancel_all_orders(strategy=self.strategy)
        status = "success" if result.get("status") == "success" else "failed"
        self.audit.log(action="CANCEL_ALL", status=status)
        self.notif.notify_trade(
            action="CANCEL_ALL",
            symbol="ALL",
            side="-",
            qty=0,
            price=0.0,
            order_id="",
            status=status,
            details="All open orders cancelled",
        )
        return result

    # ------------------------------------------------------------------
    # Market Data Helpers
    # ------------------------------------------------------------------

    def get_nifty_spot(self) -> float:
        """Fetch current Nifty spot price."""
        return self.strikes.get_spot()

    def get_atm_strike(self) -> int:
        """Get the at-the-money strike for Nifty."""
        return self.strikes.get_atm()

    def get_option_quote(self, symbol: str) -> dict:
        """Get quote for a specific option symbol."""
        return self.client.get_quote(symbol, self.exchange)

    def get_positions_summary(self) -> dict:
        """Get current positions."""
        return self.client.get_positions()

    def get_funds_available(self) -> dict:
        """Get available margin/funds."""
        return self.client.get_funds()

    # ------------------------------------------------------------------
    # Smart Trade — auto-selects strike based on mode
    # ------------------------------------------------------------------

    def smart_trade(
        self,
        action: str,
        option_type: str,
        strike_mode: str = "ATM",
        strike: Optional[int] = None,
        expiry: Optional[date] = None,
        qty: Optional[int] = None,
        price_type: str = "MARKET",
        price: float = 0.0,
        target_premium: Optional[float] = None,
    ) -> dict:
        """
        Execute a trade using OpenAlgo's optionsorder endpoint.
        OpenAlgo auto-resolves the correct symbol.
        """
        qty = qty or self.default_qty

        # Convert strike_mode to OpenAlgo offset (remove underscore)
        offset = strike_mode.upper().replace("_", "")
        ot = option_type.upper()

        # Get real expiry from OpenAlgo (not calculated)
        expiry_str = self.strikes.get_next_expiry()

        import os
        if os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes"):
            logger.info(f"[DRY RUN] smart_trade skipped broker call: {action} NIFTY {ot} {offset}")
            return {
                "trade": {"orderid": "DRY_RUN", "status": "simulated", "symbol": f"NIFTY {ot}"},
                "strike_selection": {
                    "offset": offset, "expiry": expiry_str, "option_type": ot,
                    "resolved_symbol": f"NIFTY {ot}", "underlying_ltp": 0,
                },
                "symbol": f"NIFTY {ot}",
            }

        # --- Risk check ---
        allowed, reason = self.risk.can_open_trade()
        if not allowed:
            self.audit.log(
                action=action, symbol=f"NIFTY {ot}", side=action,
                qty=qty, status="RISK_BLOCKED", details=reason,
            )
            self.notif.notify_trade(
                action=action, symbol=f"NIFTY {ot}", side=action,
                qty=qty, price=0, order_id="", status="blocked",
                details=f"Risk limit: {reason}",
            )
            return {"status": "blocked", "reason": reason}

        logger.info(
            f"🎯 Smart trade: {action} NIFTY {expiry_str} {offset} {ot} x{qty}"
        )

        try:
            result = self.client.place_option_order(
                underlying="NIFTY",
                exchange="NSE_INDEX",
                expiry_date=expiry_str,
                offset=offset,
                option_type=ot,
                action=action.upper(),
                quantity=qty,
                price_type=price_type,
                product=self.product,
                price=price,
                strategy=self.strategy,
            )

            order_id = result.get("orderid", "N/A")
            resolved_symbol = result.get("symbol", f"NIFTY {ot}")
            underlying_ltp = result.get("underlying_ltp", 0)
            status = "success" if result.get("status") == "success" else "failed"

            if status == "success" and action.upper() == "BUY":
                self.risk.record_trade_open()

            self.audit.log(
                action="PLACE_ORDER", symbol=resolved_symbol,
                side=action, qty=qty, price=underlying_ltp,
                order_id=order_id, status=status,
                details=f"offset={offset} expiry={expiry_str} product={self.product}",
            )

            self.notif.notify_trade(
                action="PLACE_ORDER", symbol=resolved_symbol,
                side=action, qty=qty, price=underlying_ltp,
                order_id=order_id, status=status,
                details=(
                    f"Offset: {offset} | Expiry: {expiry_str} | "
                    f"Spot: ₹{underlying_ltp} | Product: {self.product}"
                ),
            )

            logger.info(
                f"{'✅' if status == 'success' else '❌'} "
                f"Order {order_id}: {action} {resolved_symbol} — {status}"
            )

            return {
                "trade": result,
                "strike_selection": {
                    "offset": offset,
                    "expiry": expiry_str,
                    "option_type": ot,
                    "resolved_symbol": resolved_symbol,
                    "underlying_ltp": underlying_ltp,
                },
                "symbol": resolved_symbol,
            }

        except Exception as e:
            self.audit.log(
                action="PLACE_ORDER", symbol=f"NIFTY {offset} {ot}",
                side=action, qty=qty, status="error", details=str(e),
            )
            self.notif.notify_trade(
                action="PLACE_ORDER", symbol=f"NIFTY {offset} {ot}",
                side=action, qty=qty, price=0, order_id="",
                status="error", details=str(e),
            )
            raise

    def get_option_chain(self, width: int = 10) -> dict:
        """Fetch live option chain with real premiums from OpenAlgo."""
        return self.strikes.get_option_chain(width=width)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_trade(
        self,
        action: str,
        symbol: str,
        qty: int,
        price_type: str,
        price: float = 0.0,
        trigger_price: float = 0.0,
    ) -> dict:
        """Execute a trade with full security + notification pipeline."""

        # Step 1: Risk check
        allowed, reason = self.risk.can_open_trade()
        if not allowed:
            self.audit.log(
                action=action,
                symbol=symbol,
                side=action,
                qty=qty,
                status="RISK_BLOCKED",
                details=reason,
            )
            self.notif.notify_trade(
                action=action,
                symbol=symbol,
                side=action,
                qty=qty,
                price=price,
                order_id="",
                status="blocked",
                details=f"Risk limit: {reason}",
            )
            return {"status": "blocked", "reason": reason}

        # Step 2: Place order (rate limiter + withdrawal guard are in the client)
        try:
            result = self.client.place_order(
                symbol=symbol,
                exchange=self.exchange,
                action=action,
                quantity=qty,
                price_type=price_type,
                product=self.product,
                price=price,
                trigger_price=trigger_price,
                strategy=self.strategy,
            )

            order_id = result.get("orderid", "N/A")
            status = "success" if result.get("status") == "success" else "failed"

            if status == "success" and action == "BUY":
                self.risk.record_trade_open()

            # Step 3: Audit log
            self.audit.log(
                action="PLACE_ORDER",
                symbol=symbol,
                side=action,
                qty=qty,
                price=price,
                order_id=order_id,
                status=status,
                details=f"type={price_type} product={self.product}",
            )

            # Step 4: Notify all channels
            self.notif.notify_trade(
                action="PLACE_ORDER",
                symbol=symbol,
                side=action,
                qty=qty,
                price=price,
                order_id=order_id,
                status=status,
                details=f"Type: {price_type} | Product: {self.product}",
            )

            logger.info(
                f"{'✅' if status == 'success' else '❌'} "
                f"Order {order_id}: {action} {qty}x {symbol} — {status}"
            )
            return result

        except PermissionError as e:
            # Withdrawal guard triggered
            self.audit.log(
                action="BLOCKED",
                symbol=symbol,
                side=action,
                qty=qty,
                status="SECURITY_BLOCK",
                details=str(e),
            )
            self.notif.notify_trade(
                action="SECURITY_BLOCK",
                symbol=symbol,
                side=action,
                qty=qty,
                price=price,
                order_id="",
                status="blocked",
                details=str(e),
            )
            raise

        except Exception as e:
            self.audit.log(
                action="PLACE_ORDER",
                symbol=symbol,
                side=action,
                qty=qty,
                status="error",
                details=str(e),
            )
            self.notif.notify_trade(
                action="PLACE_ORDER",
                symbol=symbol,
                side=action,
                qty=qty,
                price=price,
                order_id="",
                status="error",
                details=str(e),
            )
            raise
