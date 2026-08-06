"""
claude_pilot.py — Dual-Brain Auto-Pilot
========================================
Architecture:
  1. ML model runs every 5 min → CALL/PUT/SKIP (free, instant)
  2. Only when ML says CALL or PUT → Claude confirms (paid, 5-6s)
  3. Claude confirmed + confidence >= threshold → execute trade
  4. Position check before execution to avoid duplicates

This saves ~95% of Claude API calls (most cycles ML says SKIP).
"""

import logging
import time
import threading
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional
from dataclasses import dataclass, field

from core.vix_regime import classify_vix, apply_vix_to_sl_tp, format_regime, VIXRegime
from core.iv_engine  import IVEngine, IVSnapshot, init_iv_engine

logger = logging.getLogger(__name__)


@dataclass
class PilotConfig:
    analyze_interval: int = 300
    market_open: str = "09:15"
    market_close: str = "15:15"
    min_confidence: int = 45    # RECONCILED 2026-05-24: 70->45 to match walk-forward-validated config
    max_trades_per_day: int = 6   # RECONCILED 2026-05-24: 3->6 to match validated flood cap
    cooldown_after_trade: int = 30  # just enough for broker order fill to reflect
    notify_on_wait: bool = False
    ml_only_mode: bool = False  # True = skip Claude, trade on ML alone
    # Dynamic SL/TP based on ATR — tuned for 90%+ win rate
    use_dynamic_sl_tp: bool = True
    sl_atr_multiplier: float = 2.2    # SL = 2.2x ATR (wide = absorb noise)
    tp_atr_multiplier: float = 2.2    # RECONCILED 2026-05-24: 2.0->2.2 (walk-forward-validated)
    # V10 (15-yr-veteran tuning): tighter SL + wider TP for asymmetric R:R.
    # Options-buyers lose to theta — minimum 1:2 realized R:R needed for edge.
    min_sl_points: float = 30.0       # Floor: 25→30 (avoid noise stop-outs)
    max_sl_points: float = 80.0      # RECONCILED 2026-05-24: 100->80 (walk-forward-validated)
    min_tp_points: float = 30.0       # RECONCILED 2026-05-24: 60->30 (walk-forward-validated)
    max_tp_points: float = 130.0      # RECONCILED 2026-05-24: 150->130 (walk-forward-validated)
    min_rr_ratio: float = 1.5         # 2026-05-07: 2.0→1.5 (with realistic TPs, 1.5 R:R is enough)
    # Breakeven stop — zero-risk once in profit
    # V10: trigger at 0.5R (half stop distance), NOT a fixed 15pts.
    # Fixed 15pts on a 50pt ATR day is noise; you stop on normal pullbacks.
    use_breakeven_stop: bool = True
    breakeven_trigger_r: float = 0.5     # Trigger BE when profit >= 0.5 × initial_sl_distance
    breakeven_trigger_pts: float = 15.0  # Hard floor — used only if 0.5R < this (kept for legacy)
    # 2026-06-10 FIX: was 3.0pts. walkforward_options_trades.csv shows EVERY
    # BE_SL exit (240 trades, 32% of all trades) nets ~-141 (WR=2.1%, total
    # -33,763) because the locked premium gain (lock_pts * lot_size * 0.5delta
    # = 3*65*0.5 = Rs.97.5) was LESS than round-trip cost (Rs.85-150) — i.e.
    # every BE_SL exit was a guaranteed small loss by construction, regardless
    # of what price did next. Raised to 10pts -> locked gain = 10*65*0.5 =
    # Rs.325, comfortably covering cost -> BE_SL exits should now net roughly
    # +175 to +240 instead of -141. Trigger timing (0.5R) is unchanged.
    breakeven_lock_pts: float = 10.0     # Lock 10pts profit at breakeven (was 3.0)
    # Trailing stop — aggressive profit protection
    use_trailing_stop: bool = True
    trail_activation_r: float = 0.8   # Activate trail after 0.8x SL profit (early)
    trail_step_pct: float = 0.60      # Trail SL to lock 60% of unrealized profit
    # ATR-multiple trailing stop (additive, opt-in) — takes whichever of
    # {profit-lock trail above, ATR-multiple trail} is more protective.
    # Disabled by default: no behavior change until explicitly turned on.
    use_atr_trailing_stop: bool = False
    atr_trail_multiplier: float = 1.5   # SL trails this many ATRs behind spot
    # Premium trailing stop (additive, opt-in) — exits if current premium
    # retraces more than premium_trail_giveback_pct from its own peak,
    # distinct from the existing hard drop-from-entry premium stop.
    use_premium_trailing_stop: bool = False
    premium_trail_giveback_pct: float = 0.35
    # Per-position max hold duration (additive, opt-in). 0 = disabled
    # (today's behavior — only the fixed 15:14:30 daily square-off applies).
    max_hold_minutes: int = 0
    # Opening-range SL widening (additive, opt-in). 0 = disabled. This bot
    # never holds overnight (MIS, forced same-day square-off), so classic
    # gap protection doesn't apply — this instead widens SL for entries
    # taken shortly after market open, when opening-range volatility is
    # highest ("gap protection where possible").
    gap_protection_widen_pct: float = 0.0
    gap_protection_window_min: int = 5
    # Trailing target — extend TP when momentum is strong
    use_trailing_tp: bool = True
    trail_tp_activation_pct: float = 0.75  # When price reaches 75% of TP, start trailing TP
    trail_tp_extend_pct: float = 0.50      # Extend TP by 50% of remaining momentum each poll
    # Entry cutoff — no new trades after 15:00 (15 min buffer before square-off)
    entry_cutoff_hour: int = 15
    entry_cutoff_min: int = 0
    # Anti-reversal — no flip within N bars of entry
    anti_reversal_bars: int = 6
    # Position monitoring
    position_poll_interval: int = 1  # Check open positions every 1 second (high-freq monitor)
    # Volatility-adjusted position sizing (fractional Kelly criterion)
    account_equity: float = 500000.0    # Current account equity (Rs.)
    max_risk_pct: float = 0.02          # Max 2% of equity risked per trade
    kelly_fraction: float = 0.25        # Quarter-Kelly (conservative)
    min_lots: int = 1                   # Floor: always trade at least 1 lot
    max_lots: int = 5                   # Ceiling: never exceed 5 lots
    lot_size: int = 65                  # Nifty lot size
    # Futures execution port (new, opt-in — see core/futures_selector.py).
    # "options" preserves today's behavior exactly; nothing below is read
    # unless this is explicitly set to "futures".
    execution_mode: str = "options"
    # Conservative ESTIMATE of NIFTY futures SPAN+exposure margin as a % of
    # notional (spot * lot_size). Real margin varies day-to-day with NSE's
    # published margin files; this is deliberately on the high/safe side
    # for offline sizing. The authoritative check is the live available-funds
    # query (KotakNeoClient.get_funds()) done right before order placement —
    # this estimate only pre-shapes the Kelly sizing, it never overrides
    # the real broker-reported capital check.
    futures_margin_pct_estimate: float = 0.15
    # Optional extra selectivity floor for futures mode (0 = disabled, no
    # behavior change). See the futures selectivity floor comment at the
    # confidence-gate check for why this needs live calibration before use
    # -- it's on the same 0-100 scale as `confidence`/effective_min_conf,
    # NOT the raw backtest quantile that motivated adding this knob.
    futures_min_confidence_pct: float = 0.0
    # PCR-alignment confidence boost (new, opt-in). Backtested on the PSAR
    # signal + 2+ years of daily PCR data (2024-05 to 2026-07, the only
    # history available -- no proper VAL/TEST split possible with this
    # short a window): PCR-aligned trades scored PF 1.81 vs 1.29 baseline,
    # consistent across two separate chronological chunks (PF 1.795 and
    # 1.824) -- promising, not proven the way other filters in this
    # project are. Mirrors the existing OI-buildup penalty (oi_penalty)
    # in shape, but as a small, capped REDUCTION to effective_min_conf
    # for trades PCR agrees with, not a requirement. Disabled by default;
    # enable with PCR_ALIGNMENT_BOOST_ENABLED=true once you want to start
    # validating it against live paper-trading data.
    pcr_alignment_boost_enabled: bool = False
    pcr_alignment_boost_pct: float = 3.0
    strategy_name: str = "OpenClawNifty"  # written into journal records for per-strategy analytics
    # HH/HL structure feature (parallel output only — logged every cycle via
    # the Feature Engineering Directive, NOT read by any gate or threshold).
    # Default True: unlike the stop-management toggles above, this only adds
    # a logged diagnostic value with zero effect on trade decisions, so there
    # is no behavior-change risk in leaving it on. See core/structure_features.py.
    enable_hh_hl_feature: bool = True
    # ------------------------------------------------------------------
    # V10 — 15-yr-veteran improvements (#5..#10)
    # ------------------------------------------------------------------
    # #6 — VIX-aware confidence floor: in expansion (>28) require strong conviction
    vix_high_threshold: float = 999.0  # RECONCILED 2026-05-24: disabled (was 28.0) - extra VIX gate not in validated config
    vix_high_min_conf: int = 65
    # #7 — Liquidity / spread filter: reject when bid-ask spread is wide
    max_spread_pct: float = 0.02        # >2% of mid = skip trade
    block_otm2_after_hour: int = 24     # RECONCILED 2026-05-24: disabled (was 13) - strike-selection patch
    block_otm2_after_min: int = 30
    # #8 — Consecutive-loss size scaling (replaces blunt flood-exit)
    loss_streak_halve_after: int = 2    # After N consecutive losers
    loss_streak_recovery_trades: int = 2  # Halve size for next M trades, then reset
    # #9 — Hard block on first 10 minutes (pure noise / opening auction artefacts)
    morning_hard_block_min: int = 0   # RECONCILED 2026-05-24: 105->0. The 13-day basis was noise;
                                        # 2yr walk-forward: 09:15-10:59 was the BEST window
                                        # (+Rs.2.99L, 56% of total profit). Block removed.
    # #5 — Delta-based strike preference
    use_delta_strike_override: bool = False  # RECONCILED 2026-05-24: disabled - backtest prices ATM only
    intraday_itm_after_hour: int = 14   # After 14:00, prefer ITM1 (low theta exposure)
    # IV gate — skip entry when options are priced dangerously above VIX
    iv_vs_vix_block_threshold: float = 999.0  # RECONCILED 2026-05-24: disabled (was 6.0) - not in validated config
    iv_rank_block_threshold:   float = 999.0  # RECONCILED 2026-05-24: disabled (was 80.0) - not in validated config
    iv_crush_block:            bool  = False  # RECONCILED 2026-05-24: disabled - not in validated config
    # #10 — Slippage tracking: log expected vs actual fill diff
    track_slippage: bool = True

    # RECONCILED 2026-05-24: master switch for the 5 hardcoded V10 signal
    # patches (V-RECOVERY, RSI-divergence, VWAP-structure, CHOP-DAY,
    # early-reversal). False = run the walk-forward-validated path only.
    enable_v10_signal_patches: bool = False

    # ------------------------------------------------------------------
    # FROZEN VALIDATED ATR EXIT (additive, opt-in) — implements EXACTLY the
    # ATR(10)/TP=2.0x/SL=6.0x/7-bar-hold exit validated by the 2026-07
    # walk-forward research (8-year yearly walk-forward, 75-config parameter
    # grid + robustness scoring, adversarial false-discovery/perturbation
    # testing). Default False: zero behavior change to any existing
    # deployment unless explicitly enabled. When True, this single flag
    # (via __post_init__ below, plus the short-circuit in
    # _get_dynamic_sl_tp and the skip-guards around the Claude-suggested/
    # theta-time/expiry-day/OI-magnetic-TP adjustment blocks) overrides
    # every other exit-adjustment mechanism so live exits match the
    # validated backtest exactly: fixed TP/SL computed once at entry from a
    # plain ATR(10), held unchanged, exit at whichever of {TP, SL, 7-bar
    # max hold, daily square-off} comes first. This flag changes ONLY exit
    # mechanics — no entry-side filter, confidence-threshold, or signal-
    # generation behavior is touched.
    use_frozen_atr_exit: bool = False
    frozen_atr_period: int = 10
    frozen_atr_tp_multiplier: float = 2.0
    frozen_atr_sl_multiplier: float = 6.0
    frozen_atr_max_hold_bars: int = 7
    frozen_atr_bar_minutes: int = 5   # bar duration used by the validated research

    def __post_init__(self):
        if self.use_frozen_atr_exit:
            # Force off every OTHER exit-adjustment mechanism so the frozen
            # config is exact, not "frozen exit plus whatever else was on."
            self.use_breakeven_stop = False
            self.use_trailing_stop = False
            self.use_atr_trailing_stop = False
            self.use_premium_trailing_stop = False
            self.use_trailing_tp = False
            self.max_hold_minutes = self.frozen_atr_max_hold_bars * self.frozen_atr_bar_minutes
            logger.info(
                "FROZEN ATR EXIT enabled: breakeven/trailing/ATR-trailing/"
                "premium-trailing/trailing-TP forced OFF; "
                f"max_hold_minutes forced to {self.max_hold_minutes} "
                f"({self.frozen_atr_max_hold_bars} bars x {self.frozen_atr_bar_minutes}min)"
            )


class PositionState:
    """Position lifecycle states to prevent race conditions."""
    OPEN = "OPEN"         # Position is active, being monitored
    CLOSING = "CLOSING"   # Close order sent, waiting for broker fill
    CLOSED = "CLOSED"     # Broker confirmed, safe to clear


@dataclass
class LivePosition:
    """Tracks an open position for SL/TP/trailing stop monitoring."""
    direction: str          # "CALL" or "PUT"
    entry_price: float      # Nifty spot at entry (for Index SL/TP tracking)
    entry_time: float       # monotonic time
    sl_price: float         # Current stop loss level (may trail)
    tp_price: float         # Target price
    initial_sl: float       # Original SL (never changes)
    symbol: str = ""        # Option symbol for closing
    atr_at_entry: float = 0.0
    breakeven_activated: bool = False
    trail_activated: bool = False
    tp_trailing: bool = False       # Whether TP is being trailed
    original_tp: float = 0.0       # Original TP before trailing
    peak_unrealized: float = 0.0   # Highest unrealized profit seen
    entry_cycle: int = 0           # Cycle number at entry (for anti-reversal)
    # Premium tracking — protects against Theta/IV crush
    entry_premium: float = 0.0     # Option premium at entry (what we paid)
    current_premium: float = 0.0   # Latest option premium (polled)
    peak_premium: float = 0.0      # Highest premium seen since entry (for premium trailing stop)
    premium_hard_stop_pct: float = 0.50  # V10: 50% (was 30 — too tight, fired on IV crush noise).
                                          # Also gated by directional confirmation in monitor loop.
    premium_history: list = field(default_factory=list)  # rolling premium prints for ATR
    # 2026-05-15: Partial profit-taking + Greeks-based exit fields
    partial_exited: bool = False        # True after 50% exit at 1R
    original_qty: int = 0               # Original qty for partial-exit calc
    entry_delta: float = 0.0            # Delta at entry (for Greeks-based exit)
    # Position lifecycle state (prevents race condition)
    state: str = "OPEN"            # OPEN → CLOSING → CLOSED
    # Round-3 independent-verification critical fix (2026-07-21): identifies
    # which caller (monitor/manual_close/emergency_stop/reconciliation) won
    # the atomic OPEN->CLOSING transition via _try_acquire_close_ownership().
    # "" means no one currently owns closing this position.
    close_owner: str = ""
    # Independent-verification fix #6 (2026-07-21): once the 15:14:30 (or
    # expiry-day 14:30) square-off condition has fired for this position,
    # it latches here and stays true for the rest of the position's life —
    # see the sticky time_exit logic in _position_monitor_loop. Prevents a
    # close retry that runs past an hour boundary (e.g. 15:59->16:00) from
    # silently un-triggering the forced square-off mid-retry.
    square_off_latched: bool = False

    # Exchange-resident SL order (Phase 2)
    # order_id of the SL SELL order placed at the exchange after entry.
    # "" means no exchange SL is active (paper trade, or placement failed).
    sl_order_id: str = ""
    # OPTION-PREMIUM (₹) trigger/limit for the exchange SL order. <= 0 means
    # no premium-based exchange SL can be computed (skip placement).
    exchange_sl_trigger: float = 0.0
    exchange_sl_limit: float = 0.0


# ── Pure stop-management helpers ───────────────────────────────────────
# Extracted as standalone functions (rather than inlined in the monitor
# loop) specifically so they're unit-testable without constructing a full
# ClaudePilot/broker/thread stack. See scripts/test_stop_management.py.

def _atr_trail_sl_candidate(direction: str, current_price: float, atr: float, multiplier: float) -> float:
    """Candidate SL level trailing `multiplier` × ATR behind current price."""
    if direction == "CALL":
        return current_price - multiplier * atr
    return current_price + multiplier * atr


def _premium_trail_stop_hit(peak_premium: float, current_premium: float, giveback_pct: float) -> bool:
    """True if current premium has retraced >= giveback_pct from its own peak."""
    if peak_premium <= 0 or current_premium <= 0 or giveback_pct <= 0:
        return False
    retrace_pct = (peak_premium - current_premium) / peak_premium
    return retrace_pct >= giveback_pct


def _max_hold_exceeded(entry_time_monotonic: float, max_hold_minutes: int, now_monotonic: float = None) -> bool:
    """True if a position has been held longer than max_hold_minutes (<=0 disables the check)."""
    if max_hold_minutes <= 0:
        return False
    now = now_monotonic if now_monotonic is not None else time.monotonic()
    return (now - entry_time_monotonic) / 60.0 >= max_hold_minutes


def _pcr_mood(pcr: float) -> Optional[str]:
    """PCR-implied direction using core/market_intel.py's own pcr_score formula
    ((pcr-0.5)/0.8*100), matching what the PCR-alignment backtest used.
    Returns "CALL" (bullish-leaning), "PUT" (bearish-leaning), or None (neutral)."""
    pcr_score = max(0.0, min(100.0, (pcr - 0.5) / 0.8 * 100.0))
    if pcr_score >= 60:
        return "CALL"
    if pcr_score <= 35:
        return "PUT"
    return None


def _gap_protection_widen(sl_pts: float, tp_pts: float, minutes_since_open: float,
                           window_min: int, widen_pct: float) -> tuple:
    """
    Widen SL (and TP proportionally, to preserve R:R) for entries taken
    within `window_min` minutes of market open, to absorb opening-range
    volatility. Returns (sl_pts, tp_pts) unchanged if disabled or outside
    the window. `minutes_since_open` may be negative (before open) or
    large (well after open) — both leave inputs unchanged.
    """
    if widen_pct <= 0 or window_min <= 0:
        return sl_pts, tp_pts
    if minutes_since_open < 0 or minutes_since_open >= window_min:
        return sl_pts, tp_pts
    return sl_pts * (1 + widen_pct), tp_pts * (1 + widen_pct)


def _close_confirmed(close_result: dict) -> bool:
    """
    True only if a close_position()/cancel_all() result indicates the
    broker actually confirmed the close — not merely that the API call
    didn't raise. A broker REJECTION (e.g. margin block, RMS hold) comes
    back as a normal dict with status="failed", no exception, and must be
    checked explicitly rather than assumed successful.
    """
    return bool(close_result) and close_result.get("status") == "success"


def _sticky_time_exit(raw_time_exit: bool, already_latched: bool) -> bool:
    """
    Independent-verification fix #6 (2026-07-21): pure combinator for the
    square-off "stickiness" rule. Once time_exit has been true once for a
    position, it must stay true for the rest of that position's life,
    regardless of what the wall-clock condition evaluates to on a later
    cycle (e.g. after an hour rollover from 15:59 to 16:00, where
    `now.hour == 15` stops matching). `already_latched` is the position's
    own square_off_latched flag; the caller writes the return value back
    onto it (it doubles as both "should time_exit fire this cycle" and
    "should the latch now be set").
    """
    return bool(raw_time_exit) or bool(already_latched)


class ClaudePilot:
    def __init__(self, trader, analyzer, ml_engine, notifier, config=None):
        self.trader = trader
        self.analyzer = analyzer
        self.ml_engine = ml_engine
        self.notifier = notifier
        self.config = config or PilotConfig()

        # Inject option chain client into market_intel for broker OI data
        # Priority: OpenAlgo REST API (if configured) → Kotak Neo direct
        try:
            from core.market_intel import init_openalgo_client
            from core.openalgo_client import create_openalgo_client
            oa_client = create_openalgo_client()
            if oa_client:
                init_openalgo_client(oa_client)
                logger.info("Market intel: using OpenAlgo REST API for option chain")
            elif hasattr(trader, 'client') and trader.client:
                init_openalgo_client(trader.client)
                logger.info("Market intel: using Kotak Neo direct for option chain")
        except Exception as e:
            logger.warning(f"Could not inject client into market_intel: {e}")

        # V9.3: Per-trade journal (win/loss/session/direction breakdown)
        try:
            from core.trade_journal import TradeJournal
            self._journal = TradeJournal()
            logger.info(f"Trade journal initialized: {self._journal.path}")
        except Exception as e:
            logger.warning(f"Trade journal init failed: {e}")
            self._journal = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._reconciliation_thread: Optional[threading.Thread] = None
        # Operational-safety fix #3 (2026-07-21): distinct from
        # _day_halted_after_loss -- that flag auto-clears at the next daily
        # boundary; this one requires an explicit manual reset since an
        # emergency stop is a deliberate, high-severity operator action.
        self._emergency_stopped = False
        self._lock = threading.Lock()
        self._trades_today = self._load_trade_count()  # V9.3: survives restarts
        # Track peak profit & max drawdown per position (for journal exit)
        self._pos_peak_profit_pts: float = 0.0
        self._pos_max_drawdown_pts: float = 0.0
        self._analyses_today = 0
        self._ml_signals_today = 0
        self._claude_calls_today = 0
        self._today = date.today()
        self._last_trade_time = 0
        self._last_recommendation = {}

        # V10 #8 — consecutive-loss tracker
        self._loss_streak = 0
        self._size_halved_remaining = 0
        # V10 #10 — slippage tracker (per-day rolling)
        self._slippage_history: list = []   # list of (expected, actual, diff_pts)

        # Bar buffer for ML model (needs ~55 bars)
        self._bar_buffer = None

        # WS tick spot — updated by TickFeed callback for 1s position monitoring
        self._ws_spot: float = 0.0          # last spot from WebSocket tick
        self._ws_spot_ts: float = 0.0       # monotonic time of last WS spot update

        # ── Premium cache (Phase-1 blocking-I/O fix) ──────────────────────
        # Option-LTP is polled in a separate daemon thread so the 1-second
        # SL/TP monitor loop is never blocked by REST I/O.
        # The monitor loop reads _premium_cache; the poller writes it.
        # Lock protects concurrent reads/writes to the dict.
        self._premium_cache: dict = {
            "premium": 0.0,   # last known option LTP
            "ts":      0.0,   # monotonic time of last successful poll
            "symbol":  "",    # symbol at time of last poll
        }
        self._premium_cache_lock = threading.Lock()
        self._premium_poller_thread: Optional[threading.Thread] = None
        # Maximum age (seconds) of cached premium before it is considered stale.
        # If stale: premium-stop is disabled for that cycle; spot-stop still fires.
        _PREMIUM_CACHE_MAX_AGE_SEC = 30.0
        self._PREMIUM_CACHE_MAX_AGE_SEC = _PREMIUM_CACHE_MAX_AGE_SEC
        # Monitor latency tracking (for watchdog metrics)
        self._monitor_last_cycle_ms: float = 0.0   # duration of last full cycle

        # IV engine — per-strike implied volatility (NSE chain or Kotak+BS)
        self._iv_engine: IVEngine = init_iv_engine(
            kotak_client=getattr(trader, "client", None)
        )
        self._iv_snap: Optional[IVSnapshot] = None   # latest snapshot for this cycle

        # Live position tracking for SL/TP/trailing stop
        self._live_position: Optional[LivePosition] = None
        self._current_atr: float = 35.0  # Default 5m ATR(14), updated each cycle
        self._current_atr10: float = 35.0  # Default 5m ATR(10) — frozen-exit path only
        self._current_atr_15m: float = 55.0  # Default 15m ATR, for reversal threshold
        self._current_vix: float = 15.0  # Default VIX, updated each cycle
        self._prev_vix: float = 0.0      # Previous day VIX for direction
        self._vix_regime: Optional[VIXRegime] = None
        self._prev_ml_signal: int = 2    # For 2-bar confirmation (2=SKIP)
        self._pilot_adx15: float = 0.0   # Pilot's own 15m ADX (reliable)

        # Session price tracking — for reversal detection
        self._session_high: float = 0.0
        self._session_low: float = 999999.0
        self._reversal_threshold: float = 80.0  # fallback; overridden by 1.5 × ATR_15m each cycle

        # Persistent VWAP deficit tracking — counts consecutive cycles where
        # spot is >50pts below (or above) the futures VWAP.  3+ consecutive
        # cycles in the same VWAP band = established structural bias, used to
        # block counter-VWAP entries (e.g. CALL when 5 cycles below VWAP).
        self._consec_vwap_below: int = 0   # cycles spot < futVWAP - 50pts
        self._consec_vwap_above: int = 0   # cycles spot > futVWAP + 50pts
        self._last_spot_vs_vwap: float = 0.0  # signed gap from last cycle

        # V-RECOVERY deduplication — tracks cycle# of last V-RECOVERY signal
        # to prevent the same intraday bounce from firing on consecutive cycles.
        # Cooldown = V_RECOVERY_COOLDOWN_CYCLES × analyze_interval (default 4×5min = 20min).
        self._last_v_recovery_cycle: int = 0
        self._v_recovery_fired_spot: float = 0.0  # spot price when V-RECOVERY last fired

        # SMC structure tracking — Order Blocks & Fair Value Gaps
        self._order_blocks: list = []   # [{level, type, strength, time}]
        self._fvg_zones: list = []      # [{high, low, type, time}]
        self._rsi_history: list = []    # [(spot, rsi)] for divergence detection
        self._swing_highs: list = []    # [spot] recent swing highs
        self._swing_lows: list = []     # [spot] recent swing lows

    # ── Session-state persistence ──────────────────────────────────────
    # Restoring VWAP bias counters and session high/low across restarts
    # lets the morning filters kick in immediately even after a hot-restart
    # (e.g. 09:05 login-scheduler restart) instead of needing 3 fresh cycles.
    _STATE_FILE = "data/session_state.json"   # plain string — Path imported lazily

    def _save_session_state(self):
        """Write intraday counters to disk so a same-day restart can reload them."""
        try:
            from pathlib import Path
            import json as _json
            import dataclasses as _dc
            _f = Path(self._STATE_FILE)
            today = datetime.now().strftime("%Y-%m-%d")
            state = {
                "date": today,
                "consec_vwap_below": self._consec_vwap_below,
                "consec_vwap_above": self._consec_vwap_above,
                "last_spot_vs_vwap": self._last_spot_vs_vwap,
                "session_high": self._session_high,
                "session_low": self._session_low if self._session_low < 999000 else 0,
            }
            # Persist any open PAPER (DRY_RUN) position so a restart doesn't
            # orphan it — real broker positions are separately recovered via
            # _reconcile_broker_position(), but paper trades leave no broker
            # footprint to reconcile against, so they need their own state.
            _pos = getattr(self, "_live_position", None)
            if _pos is not None and getattr(_pos, "symbol", "") == "PAPER":
                state["paper_position"] = _dc.asdict(_pos)
            _f.parent.mkdir(parents=True, exist_ok=True)
            _f.write_text(_json.dumps(state))
        except Exception as _e:
            logger.debug(f"Session state save failed: {_e}")

    def _load_session_state(self):
        """Restore intraday counters if the saved state is from today."""
        try:
            from pathlib import Path
            import json as _json
            _f = Path(self._STATE_FILE)
            if not _f.exists():
                return
            state = _json.loads(_f.read_text())
            today = datetime.now().strftime("%Y-%m-%d")
            if state.get("date") != today:
                logger.debug("Session state from different day — not restoring")
                return
            self._consec_vwap_below  = int(state.get("consec_vwap_below", 0))
            self._consec_vwap_above  = int(state.get("consec_vwap_above", 0))
            self._last_spot_vs_vwap  = float(state.get("last_spot_vs_vwap", 0.0))
            sh = float(state.get("session_high", 0.0))
            sl = float(state.get("session_low", 0.0))
            if sh > 0:
                self._session_high = sh
            if sl > 0:
                self._session_low = sl
            logger.info(
                f"Session state restored: vwap_below={self._consec_vwap_below} "
                f"vwap_above={self._consec_vwap_above} "
                f"session_high={self._session_high:.0f} "
                f"session_low={self._session_low:.0f}"
            )

            # Restore an open PAPER position, if one was persisted today and
            # nothing has already armed the monitor (e.g. a fresh trade this
            # same restart before _load_session_state ran).
            paper_data = state.get("paper_position")
            if paper_data and self._live_position is None:
                try:
                    # entry_time is a time.monotonic() value — not safe to
                    # reuse verbatim across a process restart (same reason
                    # _reconcile_broker_position resets it for live
                    # positions, claude_pilot.py:579). Reset to now.
                    paper_data = dict(paper_data)
                    paper_data["entry_time"] = time.monotonic()
                    self._live_position = LivePosition(**paper_data)
                    logger.info(
                        f"[PAPER] Position restored from session state — "
                        f"{self._live_position.direction} "
                        f"entry={self._live_position.entry_price:.0f} "
                        f"SL={self._live_position.sl_price:.0f} "
                        f"TP={self._live_position.tp_price:.0f} "
                        f"— monitor re-armed"
                    )
                except Exception as _e:
                    logger.warning(f"Paper position restore failed: {_e}")
        except Exception as _e:
            logger.debug(f"Session state load failed: {_e}")

    # ── Exchange-Resident SL helpers (Phase 2) ────────────────────────────
    # These wrap kotak_neo_client exchange-SL methods with env-var guard and
    # logging so they can be disabled instantly without redeploy.

    def _place_exchange_sl(self, pos: "LivePosition") -> None:
        """
        Place a SL (stop-loss limit) DAY order at the exchange for `pos`.

        Called immediately after entry fills. Never raises — a failure is
        logged at WARNING and software-only protection remains active.

        Disabled when:
          - EXCHANGE_SL_ENABLED=false (env var, default true)
          - DRY_RUN=true  (paper trades don't touch the broker)
          - pos.symbol == "PAPER"
          - pos.exchange_sl_trigger / exchange_sl_limit not computed
            (e.g. reconciled positions without entry_premium data)
        """
        import os as _os
        if _os.getenv("EXCHANGE_SL_ENABLED", "true").lower() not in ("true", "1", "yes"):
            logger.debug("ExchangeSL: disabled via EXCHANGE_SL_ENABLED=false")
            return
        if _os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes"):
            logger.debug("ExchangeSL: skipped (DRY_RUN mode)")
            return
        if not pos.symbol or pos.symbol == "PAPER":
            return
        if pos.exchange_sl_trigger <= 0 or pos.exchange_sl_limit <= 0:
            logger.debug(
                f"ExchangeSL: skipped for {pos.symbol} — no premium-based "
                f"trigger/limit available"
            )
            return
        try:
            client = getattr(self.trader, "client", None)
            if client is None:
                return
            # Premium-denominated trigger/limit — order is on the option
            # contract, so prices must be in premium terms (not spot points).
            # Trails handled by software monitor which is always tighter.
            order_id = client.place_exchange_sl(
                symbol=pos.symbol,
                qty=pos.original_qty,
                trigger_price=pos.exchange_sl_trigger,
                limit_price=pos.exchange_sl_limit,
                product="MIS",
            )
            if order_id:
                pos.sl_order_id = order_id
                logger.info(
                    f"ExchangeSL armed: {pos.direction} {pos.symbol} "
                    f"trigger=Rs.{pos.exchange_sl_trigger:.2f} "
                    f"limit=Rs.{pos.exchange_sl_limit:.2f} order_id={order_id}"
                )
            else:
                logger.warning(
                    f"ExchangeSL: placement returned no order_id for {pos.symbol} — "
                    f"software-only protection active"
                )
        except Exception as e:
            logger.warning(f"ExchangeSL: _place_exchange_sl failed ({e})")

    def _cancel_exchange_sl(self, pos: "LivePosition") -> None:
        """
        Cancel the exchange SL-M order for `pos`.

        Called immediately before `close_position()` to prevent a double-sell.
        Never raises — failure is logged. If cancel fails, the exchange SL-M
        may also fire; the resulting net-short will show in the order book.
        sl_order_id is cleared only on confirmed cancellation, so a failed
        attempt is retried on the next close cycle instead of being silently
        forgotten.
        """
        if not pos.sl_order_id or pos.sl_order_id == "":
            return
        try:
            client = getattr(self.trader, "client", None)
            if client is None:
                return
            if client.cancel_exchange_sl(pos.sl_order_id):
                pos.sl_order_id = ""   # clear so we don't try to cancel again
            else:
                logger.warning(
                    f"ExchangeSL: cancel not confirmed for {pos.sl_order_id} — "
                    f"retaining sl_order_id for retry on next cycle"
                )
        except Exception as e:
            logger.warning(f"ExchangeSL: _cancel_exchange_sl failed ({e})")

    def _attempt_protected_close(self, pos: "LivePosition", reason: str = "") -> tuple:
        """
        Operational-safety fix #2 (2026-07-21): ATTEMPT CLOSE -> CONFIRM
        CLOSE -> CANCEL SL, in that order. Extracted from
        _position_monitor_loop's CLOSING-state handling (identical
        statements/order — pure code motion, no logic change) specifically
        so it's directly unit-testable without mocking the whole 1s monitor
        loop (broker client, notifier, journal, WS feed, ATR trailing, ...).

        Previously the exchange SL was cancelled BEFORE attempting the
        close ("prevent double-sell"), which left the position with ZERO
        protection for the entire duration of any close failure/retry — an
        unbounded-downside risk if price gapped hard while unprotected.
        This ordering removes that: the exchange SL is only ever cancelled
        AFTER the close is confirmed, so on any close failure the SL was
        never touched and is still live.

        Known, accepted residual risk (not eliminated by this reorder, and
        not silently hidden): if the exchange SL fires at almost the exact
        same instant as this close order, both could execute and
        momentarily net-short the position. This is a narrow timing window,
        self-correctable by the periodic broker reconciliation (fix #5) and
        by the broker's own RMS/margin checks, versus the previous design's
        unbounded, systemic unprotected window on every close retry. This
        tradeoff is a deliberate choice, not an oversight.

        Returns (close_succeeded: bool, exception_or_None). Never raises.
        """
        _is_paper = (pos.symbol == "PAPER")
        if _is_paper:
            logger.info(f"[PAPER] {reason} hit — no broker close needed")
            return True, None   # PAPER positions have no broker leg
        try:
            if pos.symbol:
                _close_result = self.trader.close_position(pos.symbol)
            else:
                _close_result = self.trader.cancel_all()
            _close_succeeded = _close_confirmed(_close_result)

            if _close_succeeded and pos.sl_order_id:
                # Safe now: broker confirms the position is flat, so
                # cancelling the now-redundant exchange SL cannot create a
                # protection gap.
                self._cancel_exchange_sl(pos)
            return _close_succeeded, None
        except Exception as e:
            logger.error(f"Position close failed: {e}")
            return False, e

    def _try_acquire_close_ownership(self, pos: "LivePosition", owner: str) -> bool:
        """
        Round-3 independent-verification CRITICAL fix (2026-07-21): the
        ONLY way any code path may transition a position from OPEN to
        CLOSING. This is a true compare-and-swap — the read of pos.state
        and the write of pos.state + pos.close_owner happen as one
        indivisible unit under self._lock, so it is IMPOSSIBLE for two
        concurrent callers (monitor loop, manual_close, emergency_stop,
        reconciliation) to both observe OPEN and both proceed to close.

        Exactly one caller's transition succeeds and returns True — that
        caller is now the sole owner and is the ONLY one permitted to call
        _attempt_protected_close(), touch RiskManager, write the journal,
        or call _clear_live_position() for this position. Every other
        caller gets False back and MUST do nothing else this cycle: no
        broker call, no accounting, no journal write, no cleanup.

        This closes the exact race the round-3 audit proved: previously
        every caller checked `pos.state == OPEN` and then, independently
        and often much later (after a blocking spot/broker I/O call that
        can take up to ~1 second), assumed that check was still valid
        without ever re-verifying it under the lock — so two callers could
        both "win" and both execute a full close+accounting sequence,
        double-counting RiskManager.daily_pnl.
        """
        with self._lock:
            if pos.state == PositionState.OPEN:
                pos.state = PositionState.CLOSING
                pos.close_owner = owner
                return True
            return False

    def _release_close_ownership(self, pos: "LivePosition", owner: str) -> None:
        """
        Called ONLY by the caller that acquired ownership via
        _try_acquire_close_ownership(), when its OWN close attempt fails to
        confirm and must be retried later. Reverts CLOSING -> OPEN and
        clears close_owner so the position becomes acquirable again (by
        this same caller on its next cycle, or by any other caller).

        A no-op if this caller does not currently hold ownership (state
        isn't CLOSING, or close_owner doesn't match) — guards against a
        caller that never actually won the race accidentally releasing
        someone else's in-flight or already-completed close.
        """
        with self._lock:
            if pos.state == PositionState.CLOSING and pos.close_owner == owner:
                pos.state = PositionState.OPEN
                pos.close_owner = ""

    def _clear_live_position(self, pos: "LivePosition") -> bool:
        """
        Independent-verification critical fix (2026-07-21): THE ONE
        canonical way to finish closing a position. Atomically sets
        pos.state = CLOSED AND self._live_position = None under self._lock.

        Why this exists: emergency_stop() and _reconcile_runtime()'s Case 1
        repair previously only flipped pos.state to CLOSED without ever
        clearing self._live_position. That silently and PERMANENTLY blocked
        all future trade entries, because the real entry gate
        (_run_analysis, "Check no position is open or closing") tests
        `self._live_position is not None` — bare presence, not `.state`.
        Only the normal SL/TP/time/max-hold exit path cleared both fields
        together (previously inlined there); this extracts that exact
        behavior into one shared helper so the bug class cannot recur a
        third time via a new call site that forgets the second line.

        Guards against clobbering a DIFFERENT position: only clears
        self._live_position if it still IS `pos` (it may have already been
        replaced — e.g. a concurrent reconciliation orphan-adoption ran in
        between). The passed-in `pos` object's own .state is always set to
        CLOSED regardless, since that object is done either way.

        Deliberately does NOT record P&L / write the journal / notify —
        those differ per caller (normal exit has spot/pnl/journal fields
        emergency_stop and reconcile don't have) and stay at each call
        site, same as before this fix. This helper's only job is the two
        assignments that were the actual bug.

        Returns True if self._live_position was actually cleared here.
        """
        with self._lock:
            pos.state = PositionState.CLOSED
            pos.close_owner = ""
            if self._live_position is pos:
                self._live_position = None
                return True
            return False

    def _finish_successful_exit(
        self,
        pos: "LivePosition",
        reason: str,
        spot: float = 0.0,
        pnl_pts: Optional[float] = None,
    ) -> None:
        """
        THE canonical, single place every CONFIRMED close finishes at.
        Owns every post-close side effect: RiskManager accounting,
        analytics (loss-streak, probability calibrator), strategy-lockout
        statistics (anti-whipsaw, day-halt-after-loss), the journal exit
        record, cleanup (_clear_live_position + paper session-state save),
        and the Telegram exit notification.

        Callers MUST already hold close ownership (via
        _try_acquire_close_ownership) and have a broker-confirmed close
        (via _attempt_protected_close) before calling this — it does not
        re-verify either. Called by exactly four places: the normal
        SL/TP/TIME_EXIT/MAX_HOLD/PREMIUM_*/TRAIL_SL/BE_SL monitor-loop
        path, manual_close(), emergency_stop(), and the 15s reconciler's
        Case-1 repair.

        pnl_pts=None means an UNKNOWN outcome — used ONLY by reconciliation
        Case 1, where the broker already reports FLAT but no close order
        was ever sent by this process, so there is no confirmed fill price
        to compute a real P&L from. In that mode:
          - RiskManager.record_trade_close(pnl=0.0) still runs, so
            open_positions is correctly decremented (previously this
            never happened for a reconciliation-detected close, leaving
            open_positions permanently elevated) — but daily_pnl is
            deliberately NOT touched with a guessed number: fabricating a
            P&L estimate from current spot (which may have moved
            significantly since the real, unknown-timing external close)
            risks silently corrupting the real-money risk ledger with
            false precision, which is worse than a known gap.
          - Analytics (loss-streak, calibrator) and the day-halt/
            anti-whipsaw strategy-lockout flags are SKIPPED entirely —
            feeding a guessed win/loss into the ML calibrator or halting
            trading for an unconfirmed loss would poison real learned
            signals with fabricated data.
          - The journal gets the existing, deliberately-imprecise
            find_open_entry()+mark_entry_closed() synthetic-EXIT path
            (pnl=0, result="UNKNOWN"), never record_exit() with invented
            numbers — mark_entry_closed already exists specifically for
            "we don't know what really happened" (see its docstring).
          - Notification still fires, worded to say the outcome is unknown.

        When pnl_pts is a real, known number (every other caller — the
        position WAS actually closed by a broker call this process made
        and confirmed), the full sequence runs identically for all of
        them: this is what fixes emergency_stop() never having touched
        RiskManager/journal at all, and manual_close()/normal-exit having
        three independently-written (and therefore driftable) copies of
        the same bookkeeping.
        """
        _known_outcome = pnl_pts is not None
        _is_paper = (pos.symbol == "PAPER")
        trade_qty = getattr(self.trader, "default_qty", self.config.lot_size)
        pnl_rupees = (pnl_pts * trade_qty) if _known_outcome else 0.0

        # ── Accounting (RiskManager) ────────────────────────────────────
        try:
            self.trader.risk.record_trade_close(pnl=pnl_rupees)
            logger.info(
                f"RiskManager updated: P&L={pnl_rupees:+.0f} "
                f"({(pnl_pts if _known_outcome else 0.0):+.0f}pts × {trade_qty}qty) | "
                f"Daily P&L: {self.trader.risk.daily_pnl:+.0f}"
            )
        except Exception as e:
            logger.warning(f"RiskManager P&L update failed: {e}")

        # ── Analytics + strategy-lockout statistics (known outcome only) ─
        if _known_outcome:
            if reason == "SL" and pnl_pts < 0:
                self._last_loss_time = time.monotonic()
                self._last_loss_dir = pos.direction
                cooldown_min = getattr(self.config, "whipsaw_cooldown_min", 30)
                logger.warning(
                    f"📉 SL hit on {pos.direction} — opposite direction "
                    f"BLOCKED for {cooldown_min} min (anti-whipsaw)"
                )
            if pnl_pts < 0:
                self._day_halted_after_loss = True
                logger.warning(
                    f"🛑 DAY HALTED — loss on {pos.direction} (-{abs(pnl_pts):.0f}pts). "
                    f"No more trades today. Resumes tomorrow 09:15."
                )
            try:
                if pnl_pts < 0:
                    self._loss_streak += 1
                    if self._loss_streak >= self.config.loss_streak_halve_after \
                       and self._size_halved_remaining == 0:
                        self._size_halved_remaining = self.config.loss_streak_recovery_trades
                        logger.warning(
                            f"LOSS STREAK = {self._loss_streak} → halving size for "
                            f"next {self._size_halved_remaining} trades (recovery mode)"
                        )
                else:
                    if self._loss_streak > 0:
                        logger.info(f"WIN — loss streak reset (was {self._loss_streak})")
                    self._loss_streak = 0

                # 2026-04-27: feed outcome into probability calibrator
                try:
                    from core.probability_calibrator import get_calibrator
                    raw = float(getattr(pos, "entry_confidence", 0) or 0)
                    if raw > 0:
                        direction = pos.direction or "ANY"
                        regime = getattr(pos, "entry_regime", "ANY")
                        get_calibrator().record(
                            raw_conf=raw,
                            won=(pnl_pts > 0),
                            direction=direction,
                            regime=regime,
                        )
                except Exception as _e:
                    logger.debug(f"Calibrator record failed: {_e}")
            except Exception as _e:
                logger.debug(f"Loss-streak update failed: {_e}")

        # ── Journal ────────────────────────────────────────────────────
        if self._journal:
            if _known_outcome:
                try:
                    self._journal.record_exit(
                        exit_spot=spot,
                        exit_reason=reason,
                        pnl_pts=pnl_pts,
                        pnl_rupees=pnl_rupees,
                        peak_profit_pts=self._pos_peak_profit_pts,
                        max_drawdown_pts=self._pos_max_drawdown_pts,
                        breakeven_hit=pos.breakeven_activated,
                        trail_activated=pos.trail_activated,
                    )
                except Exception as e:
                    logger.warning(f"Journal exit record failed: {e}")
            else:
                try:
                    je = self._journal.find_open_entry()
                    if je:
                        self._journal.mark_entry_closed(je, reason=reason)
                except Exception as e:
                    logger.warning(f"Journal synthetic-exit record failed: {e}")

        # ── Cleanup / state transition ────────────────────────────────
        self._clear_live_position(pos)
        if _is_paper:
            # Drop the now-closed paper position from the persisted
            # session state so a restart doesn't resurrect an
            # already-exited trade.
            self._save_session_state()

        # ── Notification ─────────────────────────────────────────────
        try:
            if _known_outcome:
                _exit_prefix = "PAPER_" if _is_paper else ""
                self.notifier.notify_trade(
                    action=f"{_exit_prefix}EXIT_{reason}",
                    symbol="PAPER" if _is_paper else (pos.symbol or f"NIFTY {pos.direction}"),
                    side="SELL",
                    qty=trade_qty,
                    price=spot,
                    order_id="PAPER" if _is_paper else "",
                    status="simulated" if _is_paper else "executed",
                    details=(
                        f"{'[PAPER] ' if _is_paper else ''}"
                        f"EXIT: {reason} | Dir={pos.direction} "
                        f"Entry={pos.entry_price:.0f} Exit={spot:.0f} "
                        f"P&L={pnl_pts:+.0f}pts | "
                        f"Trail={'ON' if pos.trail_activated else 'OFF'}"
                    ),
                )
            else:
                self.notifier.notify_trade(
                    action="RECONCILE_REPAIR", symbol=pos.symbol or "", side="-",
                    qty=0, price=0.0, order_id="", status="repaired",
                    details=(
                        f"🚨 15s reconcile: internal state showed an OPEN "
                        f"position ({pos.symbol}) but broker reports FLAT. "
                        f"Internal state corrected to CLOSED. P&L is "
                        f"UNKNOWN (no confirmed fill data) — verify no "
                        f"real loss/profit was missed."
                    ),
                )
        except Exception as _e:
            logger.debug(f"Exit notification failed: {_e}")

    def _reconcile_broker_position(self) -> None:
        """
        Query the broker on startup and rebuild _live_position if an open
        NIFTY option position exists that the in-memory state doesn't know about.

        Three outcomes:
          A. Broker has open position + journal has matching open entry
             → Rebuild LivePosition from journal data; monitor resumes immediately.

          B. Broker has open position + no journal entry (entry was lost)
             → Rebuild with conservative defaults (wide SL/TP from current spot).
             Operator is alerted via Telegram.

          C. Broker is FLAT + journal shows an un-closed entry
             → Write synthetic EXIT to close the journal record.
             (Position was closed outside the bot — manual, auto-square-off.)

        Fail-open: any exception is caught and logged. The bot starts normally.
        Monitoring may be missed for this startup if reconciliation fails, but
        startup is never blocked.
        """
        try:
            client = getattr(self.trader, "client", None)
            if client is None or not getattr(client, "_session_valid", False):
                logger.debug("Reconcile: Kotak session not ready — skipping (normal at 04:00)")
                return

            broker_positions = client.get_open_nifty_positions()
            journal_entry    = self._journal.find_open_entry() if self._journal else None

            # ── Outcome C: broker flat, journal stale ─────────────────────
            if not broker_positions and journal_entry:
                logger.warning(
                    "Reconcile: journal has un-closed entry but broker is FLAT "
                    f"({journal_entry.get('direction')} {journal_entry.get('option_type')} "
                    f"@ {journal_entry.get('entry_time')}) — writing synthetic EXIT"
                )
                if self._journal:
                    self._journal.mark_entry_closed(journal_entry, reason="RECONCILE_FLAT")
                return

            # ── No broker position at all — nothing to reconcile ──────────
            if not broker_positions:
                logger.debug("Reconcile: broker FLAT, journal clean — nothing to restore")
                return

            # Use the first open NIFTY option position
            bp = broker_positions[0]
            if len(broker_positions) > 1:
                logger.warning(
                    f"Reconcile: found {len(broker_positions)} open NIFTY positions — "
                    f"restoring the first one ({bp['symbol']})"
                )

            symbol    = bp["symbol"]
            direction = bp["direction"]
            net_qty   = bp["net_qty"]
            ltp_now   = bp.get("ltp", 0.0)

            # Get current spot price for SL/TP reconstruction
            try:
                spot_now = self.trader.get_nifty_spot()
            except Exception:
                spot_now = 0.0

            # ── Outcome A: broker open + journal entry matched ────────────
            if journal_entry and journal_entry.get("direction", "").upper() == direction:
                entry_price  = float(journal_entry.get("entry_spot", spot_now or 0))
                sl_price     = float(journal_entry.get("sl_price",  0))
                tp_price     = float(journal_entry.get("tp_price",  0))
                entry_prem   = float(journal_entry.get("entry_premium", 0))
                orig_qty     = int(journal_entry.get("qty", net_qty))

                # Validate the SL/TP are still sensible (not expired levels)
                # If spot has already blown through SL, use a conservative reset
                if spot_now > 0 and sl_price > 0:
                    if direction == "CALL" and spot_now <= sl_price:
                        logger.warning(
                            f"Reconcile: spot {spot_now:.0f} already at/below "
                            f"journal SL {sl_price:.0f} — widening SL to current"
                        )
                        sl_price = spot_now - self._current_atr * 1.5
                    elif direction == "PUT" and spot_now >= sl_price:
                        logger.warning(
                            f"Reconcile: spot {spot_now:.0f} already at/above "
                            f"journal SL {sl_price:.0f} — widening SL to current"
                        )
                        sl_price = spot_now + self._current_atr * 1.5

                source = "JOURNAL"

            # ── Outcome B: broker open + no matching journal entry ─────────
            else:
                if journal_entry:
                    logger.warning(
                        f"Reconcile: journal entry direction {journal_entry.get('direction')} "
                        f"does not match broker {direction} — using conservative defaults"
                    )
                else:
                    logger.warning(
                        "Reconcile: broker has open position but no journal entry found "
                        "— using conservative defaults (wide SL/TP)"
                    )

                if spot_now <= 0:
                    logger.error("Reconcile: cannot get spot price — skipping reconciliation")
                    return

                # Conservative fallback: 1.5× ATR SL each side, 2× ATR TP
                atr = max(self._current_atr, 25.0)   # floor at 25 pts
                if direction == "CALL":
                    sl_price = spot_now - atr * 1.5
                    tp_price = spot_now + atr * 2.0
                else:
                    sl_price = spot_now + atr * 1.5
                    tp_price = spot_now - atr * 2.0

                entry_price = spot_now
                entry_prem  = ltp_now
                orig_qty    = abs(net_qty)
                source      = "CONSERVATIVE_DEFAULT"

            with self._lock:
                if self._live_position is not None:
                    logger.info("Reconcile: _live_position already set — skip")
                    return

                self._live_position = LivePosition(
                    direction    = direction,
                    entry_price  = entry_price,
                    entry_time   = time.monotonic(),   # best we can do post-restart
                    sl_price     = sl_price,
                    tp_price     = tp_price,
                    initial_sl   = sl_price,
                    symbol       = symbol,
                    atr_at_entry = self._current_atr,
                    entry_premium= entry_prem,
                    state        = PositionState.OPEN,
                    original_qty = orig_qty,
                )

            logger.warning(
                f"RECONCILED orphan position: {direction} {symbol} "
                f"qty={net_qty} entry~{entry_price:.0f} "
                f"SL={sl_price:.0f} TP={tp_price:.0f} "
                f"[source={source}]"
            )

            # ── Phase 2: check for / restore exchange SL on reconciled position ──
            # After a VPS crash and restart, the exchange SL may have been
            # cancelled by the exchange (if it already fired) or may still be
            # active (if crash happened before SL was hit).
            # Check the order book: if no open exchange SL SELL exists for this
            # symbol, log it — reconciled positions lack entry_premium /
            # premium_sl_pts, so _place_exchange_sl() will no-op (no premium
            # trigger/limit to compute); software SL remains the protection.
            try:
                existing_sl_orders = client.get_open_exchange_sl_orders()
                sl_exists = any(
                    o["symbol"] == symbol and o.get("trigger_price", 0) > 0
                    for o in existing_sl_orders
                )
                if sl_exists:
                    matched = next(
                        o for o in existing_sl_orders if o["symbol"] == symbol
                    )
                    logger.info(
                        f"Reconcile: found existing exchange SL order "
                        f"{matched['order_id']} trigger={matched['trigger_price']:.2f}"
                    )
                    with self._lock:
                        if self._live_position:
                            self._live_position.sl_order_id = matched["order_id"]
                else:
                    logger.warning(
                        f"Reconcile: no exchange SL order found for {symbol} — "
                        f"software-only protection active (premium-based "
                        f"re-arm not possible after restart)"
                    )
            except Exception as _esl_e:
                logger.warning(f"Reconcile: exchange SL check failed ({_esl_e})")

            try:
                self.notifier.notify_trade(
                    action="RECONCILED",
                    symbol=symbol,
                    side="HOLD",
                    qty=abs(net_qty),
                    price=entry_price,
                    order_id="RECONCILE",
                    status="monitoring_restored",
                    details=(
                        f"[RECONCILE] Bot restarted with open position.\n"
                        f"Dir={direction} Entry~{entry_price:.0f} "
                        f"SL={sl_price:.0f} TP={tp_price:.0f}\n"
                        f"Source={source} | Monitoring ACTIVE | ExchangeSL checked"
                    ),
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(
                f"Reconcile: unexpected error ({e}) — starting without reconciliation. "
                f"Check for orphaned broker positions manually."
            )

    def _reconciliation_loop(self):
        """
        Operational-safety fix #5 (2026-07-21): background daemon that calls
        _reconcile_runtime() every _RECONCILE_INTERVAL seconds for the
        lifetime of the pilot. Runs on its own thread, independent of the 1s
        SL/TP monitor loop, so a broker API stall during reconciliation can
        never delay SL/TP execution.
        """
        while self._running:
            time.sleep(self._RECONCILE_INTERVAL)
            if not self._running:
                break
            try:
                self._reconcile_runtime()
            except Exception as e:
                logger.error(f"Reconciliation loop: unhandled error ({e}) — continuing")

    def _reconcile_runtime(self) -> None:
        """
        Operational-safety fix #5: periodic comparison of broker position vs
        internal _live_position vs the exchange SL order book, with
        automatic repair of the inconsistencies that can be corrected
        without guessing at unknown broker-side details.

        Repaired automatically:
          - Internal OPEN, broker FLAT -> the position was closed outside
            the bot (manual square-off, RMS liquidation, or a fill the bot
            missed). Internal state is corrected to CLOSED, the journal
            entry is closed out, and a Telegram alert is sent so a human
            can verify no P&L was missed.
          - Internal FLAT, broker OPEN -> an orphaned broker position the
            bot isn't monitoring (e.g. an entry order filled but the
            response that would have set _live_position was lost to a
            broker timeout). Adopted via the exact same conservative-default
            logic _reconcile_broker_position() already uses at startup, so
            SL/TP monitoring resumes; Telegram alert sent.

        Alerted only (not auto-repaired — an automatic action here risks
        being wrong in a way that's worse than a human check):
          - Internal OPEN with a tracked sl_order_id, but no matching order
            in the broker's exchange-SL order book. Software SL/TP
            monitoring keeps running and remains the protection; no
            automatic re-placement is attempted.

        Fail-open: never raises — the caller (_reconciliation_loop) also
        wraps this, but every internal step is defensive so one bad cycle
        can't take down the reconciliation thread or corrupt state.
        """
        client = getattr(self.trader, "client", None)
        if client is None or not getattr(client, "_session_valid", False):
            return

        try:
            broker_positions = client.get_open_nifty_positions()
        except Exception as e:
            logger.debug(f"Reconcile(15s): broker position query failed ({e}) — skip this cycle")
            return

        with self._lock:
            pos = self._live_position
            pos_symbol = pos.symbol if pos else None
            pos_state = pos.state if pos else None

        broker_open = bool(broker_positions)

        # ── Case 1: internal thinks OPEN, broker is FLAT ──────────────────
        # PAPER positions (DRY_RUN) never touch the broker by design -- the
        # broker is always flat for them, which is not a mismatch. Without
        # this guard every paper trade was being flagged as a "mismatch"
        # and force-closed within one 15s cycle of entry (observed live:
        # 23/23 occurrences across three days of logs were paper trades,
        # confirmed by the immediately-preceding "[PAPER] Position tracker
        # armed" log line each time), making paper trading unable to ever
        # hold a position long enough to reach SL/TP.
        if (pos is not None and pos_state == PositionState.OPEN and not broker_open
                and pos_symbol != "PAPER"):
            # Round-3 independent-verification CRITICAL fix (2026-07-21):
            # acquire ownership before touching this position. pos_state
            # was captured before the broker query above and may be stale
            # — if another caller (monitor exit, manual_close,
            # emergency_stop) has since taken ownership of a legitimate
            # in-flight close, do NOT interfere with it (no premature
            # state clear, no synthetic journal entry, no spurious alert).
            # Skip this repair entirely; the true owner finishes the close
            # correctly, and any genuine mismatch is re-evaluated fresh on
            # the next 15s cycle.
            if not self._try_acquire_close_ownership(pos, "reconciliation"):
                logger.debug(
                    "Reconcile(15s): position is already owned by an "
                    "in-flight close — skipping Case-1 repair this cycle"
                )
                return
            logger.critical(
                "RECONCILE_MISMATCH_INTERNAL_OPEN_BROKER_FLAT",
                extra={"event": "reconcile_mismatch", "case": "internal_open_broker_flat",
                       "symbol": pos_symbol or ""},
            )
            # Round-3 canonicalization fix (2026-07-21): hand off to
            # _finish_successful_exit() with pnl_pts=None (UNKNOWN outcome
            # — see that method's docstring for the full A-vs-B reasoning).
            # This position was closed entirely outside this process, so
            # there is no confirmed fill price: RiskManager.open_positions
            # is still correctly decremented (record_trade_close(pnl=0.0)),
            # but daily_pnl is never touched with a guessed number,
            # analytics/loss-streak/calibrator are skipped (a fabricated
            # win/loss would poison real learned signals), and the journal
            # gets the existing soft mark_entry_closed() synthetic-EXIT
            # (pnl=0, result="UNKNOWN") instead of invented numbers.
            self._finish_successful_exit(pos, "RECONCILE_15S_FLAT", pnl_pts=None)
            return   # one repair per cycle — re-evaluate the rest next cycle

        # ── Case 2: internal thinks FLAT, broker is OPEN (orphan) ─────────
        if pos is None and broker_open:
            _orphan_symbol = broker_positions[0].get("symbol", "")
            logger.critical(
                "RECONCILE_MISMATCH_INTERNAL_FLAT_BROKER_OPEN",
                extra={"event": "reconcile_mismatch", "case": "internal_flat_broker_open",
                       "symbol": _orphan_symbol},
            )
            try:
                self._reconcile_broker_position()   # reuses startup adoption logic verbatim
            except Exception as e:
                logger.error(f"Reconcile(15s): adoption of orphan position failed: {e}")
            try:
                self.notifier.notify_trade(
                    action="RECONCILE_REPAIR", symbol=_orphan_symbol, side="-",
                    qty=0, price=0.0, order_id="", status="repaired",
                    details=(
                        "🚨 15s reconcile: broker reports an OPEN position the "
                        "bot wasn't monitoring. Adopted with conservative "
                        "SL/TP; verify entry price/levels are sane."
                    ),
                )
            except Exception:
                pass
            return

        # ── Case 3: both OPEN, exchange SL missing ─────────────────────────
        if pos is not None and pos_state == PositionState.OPEN and broker_open and pos.sl_order_id:
            try:
                existing_sl_orders = client.get_open_exchange_sl_orders()
                sl_exists = any(
                    o.get("symbol") == pos_symbol and o.get("trigger_price", 0) > 0
                    for o in existing_sl_orders
                )
            except Exception as e:
                logger.debug(f"Reconcile(15s): exchange SL check failed ({e}) — skip")
                sl_exists = True   # fail-open: don't false-alarm on a query failure

            if not sl_exists:
                logger.critical(
                    "RECONCILE_MISMATCH_EXCHANGE_SL_MISSING",
                    extra={"event": "reconcile_mismatch", "case": "exchange_sl_missing",
                           "symbol": pos_symbol or "", "sl_order_id": pos.sl_order_id},
                )
                try:
                    self.notifier.notify_trade(
                        action="RECONCILE_ALERT", symbol=pos_symbol or "", side="-",
                        qty=0, price=0.0, order_id=pos.sl_order_id, status="alert",
                        details=(
                            f"🚨 15s reconcile: exchange SL order "
                            f"{pos.sl_order_id} for {pos_symbol} is no longer "
                            f"in the broker order book (cancelled/rejected/"
                            f"expired outside the bot). Software SL/TP "
                            f"monitoring is still ACTIVE and remains the "
                            f"protection for this position — no automatic "
                            f"re-placement attempted. Verify manually."
                        ),
                    )
                except Exception:
                    pass

    _RECONCILE_INTERVAL = 15   # seconds — Operational-safety fix #5

    def start(self):
        if self._running:
            logger.warning("Pilot already running")
            return
        self._load_session_state()   # restore same-day counters on restart
        self._reconcile_broker_position()   # restore monitoring for any open position
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Start position monitor thread for SL/TP/trailing stop
        self._monitor_thread = threading.Thread(
            target=self._position_monitor_loop, daemon=True
        )
        self._monitor_thread.start()
        # Start premium poller — decoupled from monitor loop (Phase-1 fix)
        self._premium_poller_thread = threading.Thread(
            target=self._premium_poller_loop, daemon=True, name="PremiumPoller"
        )
        self._premium_poller_thread.start()
        # Operational-safety fix #5 (2026-07-21): periodic broker
        # reconciliation, decoupled from the 1s SL/TP monitor loop so a
        # broker/API hiccup here never delays SL/TP execution.
        self._reconciliation_thread = threading.Thread(
            target=self._reconciliation_loop, daemon=True, name="Reconciliation"
        )
        self._reconciliation_thread.start()
        logger.info(
            f"Pilot STARTED | interval={self.config.analyze_interval}s "
            f"| confidence>={self.config.min_confidence}% "
            f"| ML-only={self.config.ml_only_mode} "
            f"| dynamic_sl_tp={self.config.use_dynamic_sl_tp} "
            f"| trailing_stop={self.config.use_trailing_stop}"
        )
        # Retroactively resolve any shadow trades whose outcomes were never
        # calculated (happens because _pending is empty on restart).
        # Runs in a daemon thread — non-blocking, uses nifty_5min.csv.
        def _resolve_shadows():
            try:
                from core.shadow_logger import resolve_historical_outcomes
                result = resolve_historical_outcomes()
                if result.get("resolved", 0) > 0:
                    logger.info(
                        f"Shadow outcomes retroactively resolved: "
                        f"{result['resolved']} new | {result['already_done']} existing | "
                        f"{result['no_data']} no-data"
                    )
            except Exception as e:
                logger.debug(f"Shadow outcome resolution failed: {e}")

        threading.Thread(target=_resolve_shadows, daemon=True,
                         name="ShadowOutcomeResolver").start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        if self._premium_poller_thread:
            self._premium_poller_thread.join(timeout=10)
        if getattr(self, "_reconciliation_thread", None):
            self._reconciliation_thread.join(timeout=10)
        logger.info("Pilot STOPPED")

    def emergency_stop(self, reason: str = "manual") -> dict:
        """
        Operational-safety fix #3 (2026-07-21): full emergency-shutdown
        sequence. Steps run strictly in order -- each is guaranteed to have
        completed (or failed loudly, without aborting the remaining steps)
        before the next begins:

          1. Stop new entries    -- self._emergency_stopped, checked at the
             same two entry gates as _day_halted_after_loss. Requires manual
             reset (no auto-clear at the next day boundary).
          2. Cancel pending entry orders (broker cancel-all).
          3. Immediately flatten any open position. Uses the same
             close-then-confirm ordering as the fix #2 CLOSING-state logic
             (never cancels the exchange SL until the broker confirms the
             close), retried until confirmed. Ownership of the close is
             acquired via the round-3 _try_acquire_close_ownership()
             compare-and-swap (still running -- thread shutdown is step 5,
             deliberately last) -- if another caller (the monitor loop's
             own SL/TP exit, or a concurrent manual close) already owns
             closing this exact position, emergency_stop does NOT attempt
             a second broker close; it defers to step 4 to confirm the
             book ends up flat regardless of who performed the close. On a
             confirmed close this process itself performed, hands off to
             the canonical _finish_successful_exit() (round-3
             canonicalization fix, 2026-07-21) for RiskManager/analytics/
             journal/cleanup/notification -- the same helper the normal
             exit path and manual_close() use.
          4. Verify broker reports zero open positions, with bounded retry.
          5. Stop monitoring threads -- last, so monitoring/retry stays
             alive through steps 2-4.

        Never raises. A failure at any step is recorded in the returned
        dict and alerted via Telegram rather than aborting the remaining
        steps -- e.g. threads must still be stopped even if the broker
        close call itself failed.
        """
        logger.critical(f"🚨 EMERGENCY STOP requested (reason={reason})")
        report = {"reason": reason, "steps": {}}

        # 1. Stop new entries.
        self._emergency_stopped = True
        report["steps"]["stop_new_entries"] = "ok"

        # 2. Cancel pending entry orders.
        try:
            cancel_result = self.trader.cancel_all()
            report["steps"]["cancel_pending_orders"] = cancel_result.get("status", "unknown")
        except Exception as e:
            logger.error(f"Emergency stop: cancel_all failed: {e}")
            report["steps"]["cancel_pending_orders"] = f"error: {e}"

        # 3. Immediately flatten any open position.
        with self._lock:
            pos = self._live_position

        _flatten_ok = True
        _owns_close = False
        if pos is not None:
            # Round-3 independent-verification CRITICAL fix (2026-07-21):
            # atomic ownership acquisition instead of an unconditional
            # `pos.state = CLOSING`. If another caller (the monitor loop's
            # own SL/TP/time-exit handling, or a concurrent manual_close)
            # already owns closing this position, do NOT attempt a second
            # broker close, do NOT touch RiskManager/journal -- defer
            # entirely to step 4 below to confirm the book is flat.
            _owns_close = self._try_acquire_close_ownership(pos, "emergency_stop")
            if not _owns_close:
                logger.info(
                    "Emergency stop: another close is already in flight for "
                    "this position (monitor exit or manual close) — not "
                    "attempting a duplicate close; step 4 will verify the "
                    "broker book is flat regardless of who closes it."
                )

        if pos is not None and _owns_close:
            # Reuses the exact same close-then-confirm-then-cancel-SL helper
            # as the CLOSING-state monitor loop (fix #2) -- one source of
            # truth for "how to safely close a position" instead of a
            # second, divergence-prone copy of the same logic.
            _flatten_ok = False
            _MAX_FLATTEN_ATTEMPTS = 5
            for attempt in range(1, _MAX_FLATTEN_ATTEMPTS + 1):
                _confirmed, _exc = self._attempt_protected_close(pos, "EMERGENCY_STOP")
                if _confirmed:
                    # Round-3 canonicalization fix (2026-07-21): hand off to
                    # _finish_successful_exit() for RiskManager/analytics/
                    # journal/cleanup/notification -- previously
                    # emergency_stop() only cleared the position pointer and
                    # touched NONE of the accounting, leaving
                    # RiskManager.open_positions permanently elevated (and
                    # no journal record at all) after every emergency stop
                    # that flattened a real position. Best-effort spot
                    # lookup mirrors manual_close()'s approach -- the close
                    # itself is confirmed, only the P&L needs computing.
                    _spot = 0.0
                    _pnl_pts = 0.0
                    try:
                        _spot = self.trader.get_nifty_spot() or 0.0
                        if _spot:
                            _pnl_pts = (_spot - pos.entry_price) if pos.direction == "CALL" \
                                else (pos.entry_price - _spot)
                    except Exception as _e:
                        logger.warning(f"Emergency stop: spot lookup for P&L failed ({_e}) — recording pnl=0")
                    self._finish_successful_exit(pos, "EMERGENCY_STOP", _spot, _pnl_pts)
                    _flatten_ok = True
                    break
                logger.critical(
                    f"Emergency stop: close attempt {attempt}/{_MAX_FLATTEN_ATTEMPTS} "
                    f"not confirmed by broker (error={_exc}) — retrying"
                )
                time.sleep(1)
            if not _flatten_ok:
                # Exhausted every attempt -- release ownership so a later
                # cycle (monitor loop, a subsequent manual close) can retry
                # instead of leaving this position permanently un-owned.
                self._release_close_ownership(pos, "emergency_stop")

        if pos is None:
            report["steps"]["flatten_position"] = "ok"   # nothing to flatten
        elif not _owns_close:
            report["steps"]["flatten_position"] = "skipped — already owned by another in-flight close"
        else:
            report["steps"]["flatten_position"] = (
                "ok" if _flatten_ok else "FAILED — position may still be open"
            )

        # 4. Verify broker reports zero open positions.
        _zero_confirmed = False
        client = getattr(self.trader, "client", None)
        for attempt in range(1, 6):
            try:
                remaining = client.get_open_nifty_positions() if client else []
            except Exception as e:
                logger.error(f"Emergency stop: position verification query failed: {e}")
                remaining = None
            if remaining is not None and len(remaining) == 0:
                _zero_confirmed = True
                break
            time.sleep(1)
        report["steps"]["verify_zero_positions"] = (
            "ok" if _zero_confirmed else "FAILED — broker still reports open position(s)"
        )

        if not _flatten_ok or not _zero_confirmed:
            logger.critical(
                "🚨🚨 EMERGENCY STOP INCOMPLETE — manual intervention required. "
                f"flatten_ok={_flatten_ok} zero_confirmed={_zero_confirmed}"
            )
            try:
                self.notifier.notify_trade(
                    action="EMERGENCY_STOP", symbol=(pos.symbol if pos else ""), side="-",
                    qty=0, price=0.0, order_id="", status="INCOMPLETE",
                    details=(
                        f"🚨 EMERGENCY STOP could not fully confirm a flat book "
                        f"(flatten_ok={_flatten_ok}, zero_confirmed={_zero_confirmed}). "
                        f"New entries are blocked and threads will still stop, but "
                        f"MANUALLY VERIFY broker positions/orders immediately."
                    ),
                )
            except Exception:
                pass
        else:
            try:
                self.notifier.notify_trade(
                    action="EMERGENCY_STOP", symbol="", side="-", qty=0, price=0.0,
                    order_id="", status="ok",
                    details="Emergency stop completed: entries blocked, orders cancelled, book confirmed flat.",
                )
            except Exception:
                pass

        # 5. Stop monitoring threads -- last.
        self.stop()
        report["steps"]["stop_monitoring_threads"] = "ok"
        return report

    def manual_close(self, symbol: str = "") -> dict:
        """
        Independent-verification fix #4 (2026-07-21): manual close (POST
        /close) now routes through the exact same _attempt_protected_close()
        flow used by monitor exits and emergency_stop, instead of calling
        the broker directly.

        Previously POST /close called trader.close_position() directly:
        it never cancelled the exchange SL (orphaning a resting sell order
        against a position that no longer existed) and never touched
        _live_position (silently desyncing the pilot until the next 15s
        reconciliation cycle happened to notice).

        Now: confirm broker close -> cancel exchange SL (only after
        confirmation, same ordering as fix #2) -> hand off to the
        canonical _finish_successful_exit() for RiskManager/analytics/
        journal/cleanup/notification (round-3 canonicalization fix,
        2026-07-21 -- the same helper the normal exit path and
        emergency_stop() use, so this is no longer a separately-maintained
        copy of that bookkeeping). Behaves like a real exit, not a
        side-channel bypass.

        If there is no pilot-tracked open position, this deliberately does
        NOT fall back to a raw untracked broker close by symbol -- doing
        so would silently reintroduce the exact orphaning bug this fix
        removes. An untracked broker position is a job for the 15s
        reconciliation (fix #5), not manual close.

        Round-3 independent-verification CRITICAL fix (2026-07-21): the
        OPEN->CLOSING transition is now performed by the same atomic
        _try_acquire_close_ownership() compare-and-swap used by the
        monitor loop and emergency_stop(). If another caller already owns
        closing this position, this call does NOT attempt a broker close,
        does NOT touch RiskManager/journal, and returns immediately.

        Returns a status dict; never raises.
        """
        with self._lock:
            pos = self._live_position
            if pos is None:
                return {
                    "status": "error",
                    "reason": "no tracked open position — nothing to close "
                              "(an untracked broker position will self-heal "
                              "via the 15s reconciliation cycle)",
                }
            if symbol and pos.symbol and symbol != pos.symbol:
                return {
                    "status": "error",
                    "reason": f"symbol mismatch: tracked position is "
                              f"{pos.symbol!r}, requested {symbol!r}",
                }

        if not self._try_acquire_close_ownership(pos, "manual_close"):
            return {
                "status": "error",
                "reason": "another close is already in flight for this "
                          "position (monitor exit, another manual close, "
                          "or emergency stop) — not attempting a duplicate "
                          "close",
            }

        _MAX_ATTEMPTS = 5
        _confirmed = False
        _last_exc = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _confirmed, _last_exc = self._attempt_protected_close(pos, "MANUAL_CLOSE")
            if _confirmed:
                break
            logger.critical(
                f"Manual close: attempt {attempt}/{_MAX_ATTEMPTS} not "
                f"confirmed by broker (error={_last_exc}) — retrying"
            )
            time.sleep(1)

        if not _confirmed:
            self._release_close_ownership(pos, "manual_close")
            logger.critical(
                f"🚨 MANUAL CLOSE FAILED — broker did not confirm the close "
                f"after {_MAX_ATTEMPTS} attempts. Position remains OPEN and "
                f"monitored normally; exchange SL (if any) was never "
                f"cancelled, so it is still protected."
            )
            return {
                "status": "error",
                "reason": f"broker did not confirm close after "
                          f"{_MAX_ATTEMPTS} attempts (last error: {_last_exc})",
            }

        # Broker-confirmed close: compute best-effort P&L (only the input
        # this caller alone can supply -- the confirmed close itself
        # happened via _attempt_protected_close above) and hand off to the
        # canonical _finish_successful_exit() for every post-close side
        # effect (RiskManager, analytics, journal, cleanup, notification)
        # -- see that method's docstring; it's the ONE place this logic
        # lives, no longer duplicated here.
        spot = 0.0
        pnl_pts = 0.0
        try:
            spot = self.trader.get_nifty_spot() or 0.0
            if spot:
                pnl_pts = (spot - pos.entry_price) if pos.direction == "CALL" \
                    else (pos.entry_price - spot)
        except Exception as e:
            logger.warning(f"Manual close: spot lookup for P&L failed ({e}) — recording pnl=0")

        trade_qty = getattr(self.trader, "default_qty", self.config.lot_size)
        pnl_rupees = pnl_pts * trade_qty
        _closed_symbol = pos.symbol

        self._finish_successful_exit(pos, "MANUAL_CLOSE", spot, pnl_pts)

        return {
            "status": "success", "symbol": _closed_symbol,
            "pnl_pts": pnl_pts, "pnl_rupees": pnl_rupees,
        }

    @property
    def is_running(self):
        return self._running

    def get_status(self):
        with self._lock:
            return {
                "running": self._running,
                "cycle": self._analyses_today,
                "ml_signals": self._ml_signals_today,
                "claude_calls": self._claude_calls_today,
                "trades": self._trades_today,
                "last": self._last_recommendation,
            }

    def get_performance(self, days: int = 30) -> dict:
        """Get performance summary + breakdowns from trade journal."""
        try:
            from core.performance_tracker import PerformanceTracker
            tracker = PerformanceTracker(self._journal)
            return {
                "summary": tracker.summary(days),
                "by_session": tracker.breakdown_by_session(days),
                "by_direction": tracker.breakdown_by_direction(days),
                "by_vix_regime": tracker.breakdown_by_vix_regime(days),
                "by_exit_reason": tracker.breakdown_by_exit_reason(days),
                "report": tracker.format_report(days),
            }
        except Exception as e:
            logger.error(f"get_performance failed: {e}")
            return {"error": str(e)}

    def update_tick_spot(self, price: float):
        """
        Called by TickFeed (or main.py) whenever a live Nifty WS tick arrives.
        Position monitor prefers this price (sub-second) over REST API polling.
        """
        import time as _t
        self._ws_spot = float(price)
        self._ws_spot_ts = _t.monotonic()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_daily(self):
        today = date.today()
        if today != self._today:
            with self._lock:
                logger.info("New trading day - resetting counters")
                self._trades_today = 0
                self._analyses_today = 0
                self._ml_signals_today = 0
                self._claude_calls_today = 0
                self._today = today
                self._prev_ml_signal = 2  # track last signal for logging
                # Reset session high/low and SMC structures
                self._session_high = 0.0
                self._session_low = 999999.0
                self._consec_vwap_below = 0
                self._consec_vwap_above = 0
                self._last_spot_vs_vwap = 0.0
                self._order_blocks = []
                self._fvg_zones = []
                self._rsi_history = []
                self._swing_highs = []
                self._swing_lows = []
                # Reset V-RECOVERY deduplication
                self._last_v_recovery_cycle = 0
                self._v_recovery_fired_spot = 0.0
                # 2026-05-15 STRATEGY K: reset halt-after-loss flag at new day
                self._day_halted_after_loss = False
                # ── Apply pre-market brief (if available) ────────────
                self._apply_premarket_brief()

    def _apply_premarket_brief(self):
        """Read today's pre-market brief and adapt config for the session."""
        try:
            from core.premarket_agent import get_today_brief
            brief = get_today_brief()
            if not brief:
                logger.info("📋 No pre-market brief for today — using default config")
                self._premarket_brief = None
                return
            self._premarket_brief = brief
            # Apply max_trades_today
            mt = int(brief.get("max_trades_today", 0) or 0)
            if 1 <= mt <= 8:
                old = self.config.max_trades_per_day
                self.config.max_trades_per_day = mt
                logger.warning(
                    f"📋 BRIEF: max_trades {old}→{mt} (regime={brief.get('regime')} "
                    f"bias={brief.get('bias')} risk={brief.get('risk_appetite')})"
                )
            # Stash boost / avoid_otm flags for _decide_entry to read
            self._brief_conf_boost = int(brief.get("min_confidence_boost", 0) or 0)
            self._brief_avoid_otm23 = bool(brief.get("avoid_otm2_otm3", False))
            self._brief_bias = str(brief.get("bias", "neutral")).lower()
            logger.info(
                f"📋 BRIEF applied: conf_boost={self._brief_conf_boost:+d} "
                f"avoid_OTM2/3={self._brief_avoid_otm23} bias={self._brief_bias}"
            )
        except Exception as e:
            logger.debug(f"premarket brief apply failed (fail-open): {e}")
            self._premarket_brief = None

    def _in_cooldown(self):
        if self._last_trade_time == 0:
            return False
        return time.monotonic() - self._last_trade_time < self.config.cooldown_after_trade

    # NSE trading holidays — published annually by NSE circular. VERIFY against
    # the official NSE holiday list before each calendar year:
    #   https://www.nseindia.com/resources/exchange-communication-holidays
    # Format: "YYYY-MM-DD". Muhurat / special trading sessions are excluded
    # (the bot stays idle on those anyway — the 3:30pm cutoff wins).
    NSE_HOLIDAYS: set[str] = {
        # 2026 — cross-check with NSE circular; lunar-calendar dates approximate
        "2026-01-26",  # Republic Day (Mon)
        "2026-02-19",  # Mahashivratri
        "2026-03-03",  # Holi
        "2026-03-31",  # Id-Ul-Fitr (Ramzan)
        "2026-04-03",  # Good Friday
        "2026-04-14",  # Dr. Ambedkar Jayanti
        "2026-05-01",  # Maharashtra Day
        "2026-05-28",  # Bakri Id (Eid ul-Adha) — NSE confirmed holiday
        "2026-08-15",  # Independence Day (Sat — already weekend)
        "2026-08-26",  # Ganesh Chaturthi (approx)
        "2026-10-02",  # Gandhi Jayanti
        "2026-10-21",  # Diwali – Laxmi Pujan
        "2026-11-04",  # Guru Nanak Jayanti (approx)
        "2026-12-25",  # Christmas
    }

    def _is_market_hours(self):
        now_dt = datetime.now()
        # Weekend block: Monday=0 ... Sunday=6 → skip Sat(5) + Sun(6)
        if now_dt.weekday() >= 5:
            return False
        # NSE holiday block
        if now_dt.strftime("%Y-%m-%d") in self.NSE_HOLIDAYS:
            return False
        now = now_dt.time()
        h_o, m_o = map(int, self.config.market_open.split(":"))
        h_c, m_c = map(int, self.config.market_close.split(":"))
        return dtime(h_o, m_o) <= now <= dtime(h_c, m_c)

    def _wait_until_next_slot(self):
        now = datetime.now()
        interval = self.config.analyze_interval
        secs = now.hour * 3600 + now.minute * 60 + now.second
        remainder = secs % interval
        wait = 0 if remainder == 0 else interval - remainder
        if wait > 0:
            nxt = now + timedelta(seconds=wait)
            logger.debug(f"Next slot at {nxt.strftime('%H:%M:%S')} ({wait}s)")
            end = time.monotonic() + wait
            while time.monotonic() < end and self._running:
                time.sleep(min(5, end - time.monotonic()))

    def _run_loop(self):
        self._wait_until_next_slot()

        while self._running:
            try:
                self._reset_daily()

                if not self._is_market_hours():
                    # Keep heartbeat alive so deadman doesn't fire false STALE
                    # alerts during non-market hours, weekends, and NSE holidays.
                    # (Heartbeat at line ~474 is skipped by this continue — write
                    # it here explicitly so the file timestamp stays fresh.)
                    try:
                        from core.deadman_switch import write_heartbeat
                        write_heartbeat(
                            cycle=(self._analyses_today or 0),
                            status="non_market",
                        )
                    except Exception:
                        pass
                    time.sleep(30)
                    self._wait_until_next_slot()
                    continue

                # Update shadow trade outcomes (every cycle, very fast).
                # Use ws_spot if available (fresher), otherwise skip — retroactive
                # resolver will pick up remaining NULLs on the next restart.
                try:
                    from core.shadow_logger import get_shadow_logger
                    _shadow_spot = self._ws_spot or 0.0
                    if _shadow_spot > 0:
                        get_shadow_logger().update_outcomes(_shadow_spot)
                except Exception:
                    pass

                # 2026-04-29: Heartbeat for deadman switch — write BEFORE
                # any work so a hang in next steps still leaves a fresh beat
                # 2026-05-18 BUG FIX: was using undefined `cycle` → NameError
                # silently caught → heartbeat never updated from this point.
                try:
                    from core.deadman_switch import write_heartbeat
                    _next_cycle = (self._analyses_today or 0) + 1
                    write_heartbeat(cycle=_next_cycle, status="loop_enter")
                except Exception as _e:
                    logger.debug(f"Heartbeat write failed: {_e}")

                # Persist intraday session state so a same-day restart can
                # restore VWAP bias counters and session high/low immediately.
                self._save_session_state()

                # DAY HALT GATE — top-of-loop check so no analysis runs after a loss.
                # The execute-path gate at line ~2758 is too deep — cycles blocked
                # by OI filter never reach it, and when one slips through the bot
                # fires a second trade. Check here guarantees a hard stop.
                _estopped = getattr(self, "_emergency_stopped", False)
                if getattr(self, "_day_halted_after_loss", False) or _estopped:
                    try:
                        from core.deadman_switch import write_heartbeat
                        write_heartbeat(
                            cycle=(self._analyses_today or 0),
                            status="emergency_stopped" if _estopped else "day_halted",
                        )
                    except Exception:
                        pass
                    time.sleep(60)
                    continue

                # Market is open — ensure broker session is alive (lazy login).
                # First call of the day at ~09:15 triggers TOTP login. Subsequent
                # calls are no-ops since _session_valid stays True.
                try:
                    if hasattr(self.trader.client, "ensure_logged_in"):
                        if not self.trader.client.ensure_logged_in():
                            logger.error(
                                "Cycle skipped: Kotak login failed — will retry next cycle"
                            )
                            time.sleep(30)
                            continue
                except Exception as _e:
                    logger.error(f"Kotak ensure_logged_in error: {_e}")
                    time.sleep(30)
                    continue

                with self._lock:
                    max_td = self.config.max_trades_per_day
                    if self._vix_regime:
                        max_td = min(max_td, self._vix_regime.max_trades_per_day)
                    trades_so_far = self._trades_today
                    has_position = self._live_position is not None
                if trades_so_far >= max_td:
                    logger.info(f"Daily limit ({trades_so_far}/{max_td})")
                    # Heartbeat before the long sleep so deadman stays quiet
                    try:
                        from core.deadman_switch import write_heartbeat
                        write_heartbeat(
                            cycle=(self._analyses_today or 0) + 1,
                            status="daily_limit",
                        )
                    except Exception:
                        pass
                    time.sleep(300)
                    continue

                # Entry cutoff — no new analysis after 15:00 (only manage existing positions)
                now = datetime.now()
                cutoff_h = self.config.entry_cutoff_hour
                cutoff_m = self.config.entry_cutoff_min
                if now.hour > cutoff_h or (now.hour == cutoff_h and now.minute >= cutoff_m):
                    if not has_position:
                        logger.debug(f"Past entry cutoff ({cutoff_h}:{cutoff_m:02d})")
                        try:
                            from core.deadman_switch import write_heartbeat
                            write_heartbeat(
                                cycle=(self._analyses_today or 0) + 1,
                                status="entry_cutoff",
                            )
                        except Exception:
                            pass
                        time.sleep(60)
                        continue

                # V10: Expiry day — no NEW entries after 13:00 (was 14:30).
                # After 13:00 on expiry, theta is brutal (~₹3-5/min on ATM) AND
                # gamma swings make any new directional bet a coin flip.
                # Existing positions still managed normally (force-close at 14:30 below).
                try:
                    from core.expiry_utils import is_expiry_day
                    if is_expiry_day() and not has_position:
                        if now.hour >= 13:
                            logger.info("EXPIRY DAY: Past 13:00 cutoff — no new entries (theta+gamma)")
                            # ↓ Explicit heartbeat so deadman doesn't fire false STALE alerts
                            # on weekly expiry days (pilot is alive, just intentionally idle).
                            # The loop-enter heartbeat at top covers most paths, but the
                            # 60-second sleep here occasionally races the 6-minute stale window.
                            try:
                                from core.deadman_switch import write_heartbeat
                                write_heartbeat(
                                    cycle=(self._analyses_today or 0) + 1,
                                    status="expiry_cutoff",
                                )
                            except Exception:
                                pass
                            time.sleep(60)
                            continue
                except Exception:
                    pass

                # CHOP-DAY GATE: after 11:00, if day range < 60pts AND
                # 15m ADX < 20 → no trend → block all new entries
                try:
                    import pandas as pd
                    import numpy as np
                    if not has_position and (now.hour > 11 or (now.hour == 11 and now.minute >= 0)):
                        df5_today = self.trader.get_intraday_5min()
                        if df5_today is not None and len(df5_today) > 0:
                            today_only = df5_today[df5_today.index.date == now.date()]
                            if len(today_only) >= 5:
                                day_range = float(today_only["high"].max() - today_only["low"].min())
                                df15 = self.trader.get_intraday_15min() if hasattr(self.trader, "get_intraday_15min") else None
                                adx15 = 25.0
                                if df15 is not None and len(df15) >= 14:
                                    h, l, c = df15["high"], df15["low"], df15["close"]
                                    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
                                    atr = tr.rolling(14).mean()
                                    up = h.diff(); dn = -l.diff()
                                    plus_dm  = ((up > dn) & (up > 0)) * up
                                    minus_dm = ((dn > up) & (dn > 0)) * dn
                                    plus_di  = 100 * (plus_dm.rolling(14).mean() / atr)
                                    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
                                    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
                                    adx15 = float(dx.rolling(14).mean().iloc[-1])
                                if day_range < 60 and adx15 < 20 and self.config.enable_v10_signal_patches:
                                    logger.info(f"CHOP-DAY GATE: range={day_range:.0f}pts ADX15={adx15:.0f} → no trend, skip cycle")
                                    time.sleep(60)
                                    continue
                except Exception as _e:
                    logger.debug(f"chop gate skipped: {_e}")

                if self._in_cooldown():
                    remaining = int(self.config.cooldown_after_trade -
                                    (time.monotonic() - self._last_trade_time))
                    logger.debug(f"Cooldown: {remaining}s")
                    self._wait_until_next_slot()
                    continue

                logger.info(f"--- Analysis at {now.strftime('%H:%M')} ---")
                self._run_analysis()

            except Exception as e:
                logger.error(f"Pilot error: {e}")
                time.sleep(30)

            self._wait_until_next_slot()

    # ------------------------------------------------------------------
    # Core analysis: ML first, then Claude if needed
    # ------------------------------------------------------------------

    def _run_analysis(self):
        with self._lock:
            self._analyses_today += 1
            cycle = self._analyses_today

        # ── Issue #8: per-filter shadow logging helper ─────────────────────
        # Called by every filter that rejects a signal so all skips appear in
        # shadow_trades.jsonl for the filter audit.  Reason strings use the
        # convention "bucket_name[:detail]" matching shadow_logger's bucketing.
        # Fail-open: any exception is swallowed — never block trading.
        def _shadow_skip(reason: str, signal: str = "", conf: float = 0.0) -> None:
            try:
                from core.shadow_logger import get_shadow_logger
                _sig = signal or ml_direction
                if _sig in ("SKIP", ""):
                    return   # no directional signal to measure
                get_shadow_logger().log_skip(
                    cycle=cycle,
                    signal=_sig,
                    conf_pct=conf or confidence if "confidence" in dir() else conf,
                    reason=reason,
                    spot=spot,
                    ml_proba=list(ml_proba) if hasattr(ml_proba, "__iter__") else [],
                    regime=str(ml_indicators.get("regime", "")),
                )
            except Exception:
                pass

        # 2026-05-18 BUG FIX: write heartbeat at START of analysis cycle.
        # Previously only wrote heartbeat AFTER ML completed — so if ML/feature
        # engineering hung, heartbeat would stay stuck on previous cycle.
        try:
            from core.deadman_switch import write_heartbeat
            write_heartbeat(cycle=cycle, status="analysis_start")
        except Exception as _e:
            logger.debug(f"Heartbeat start failed: {_e}")

        # Step 0: Update VIX regime (determines SL/TP scaling + trade eligibility)
        regime = self._update_vix_regime()
        if regime and not regime.tradeable:
            logger.info(
                f"Cycle #{cycle}: VIX regime {regime.name} (VIX={regime.vix_level:.1f} "
                f"{regime.vix_direction}) → NOT TRADEABLE"
            )
            with self._lock:
                self._last_recommendation = {
                    "cycle": cycle, "ml_signal": "SKIP",
                    "action": "WAIT", "confidence": 0,
                    "reason": f"VIX regime {regime.name} not tradeable",
                    "time": datetime.now().isoformat(),
                }
            return

        # Override max trades per day from VIX regime
        if regime:
            effective_max_trades = min(self.config.max_trades_per_day, regime.max_trades_per_day)
        else:
            effective_max_trades = self.config.max_trades_per_day

        # Step 1: Get spot price + track session high/low
        try:
            spot = self.trader.get_nifty_spot()
        except Exception as e:
            logger.error(f"Cycle #{cycle}: No spot price: {e}")
            return

        # Update session high/low for reversal detection
        if spot > self._session_high:
            self._session_high = spot
        if spot < self._session_low:
            self._session_low = spot

        # Step 1.5: Fetch live IV snapshot (60s cached — near-zero runtime cost)
        vix_now = self._current_vix or 15.0
        try:
            self._iv_snap = self._iv_engine.get_snapshot(spot=spot, vix_level=vix_now)
            if self._iv_snap.available:
                logger.info(f"Cycle #{cycle}: IV → {self._iv_snap.summary}")
        except Exception as _e:
            logger.debug(f"IV fetch failed (non-fatal): {_e}")
            self._iv_snap = None

        # ── ADX context log (V8 model handles sideways internally) ──
        # No hard block — V8 has tf15_adx, tf15_trending, tf15_sideways
        # as features and learned when to skip. Hard gate was causing
        # missed signals at borderline ADX values (20-25).
        try:
            from core.tv_fetcher import get_tv_fetcher
            import pandas as pd
            df15 = get_tv_fetcher().get_nifty_15min(n_bars=30)
            if not df15.empty and len(df15) >= 15:
                h15, l15, c15 = df15["high"], df15["low"], df15["close"]
                tr15 = pd.concat([
                    h15 - l15,
                    (h15 - c15.shift()).abs(),
                    (l15 - c15.shift()).abs()
                ], axis=1).max(axis=1)
                up15  = h15 - h15.shift(1)
                dn15  = l15.shift(1) - l15
                pdm15 = up15.where((up15 > dn15) & (up15 > 0), 0.0)
                ndm15 = dn15.where((dn15 > up15) & (dn15 > 0), 0.0)
                atr15 = tr15.ewm(span=14, adjust=False).mean()
                pdi15 = 100 * pdm15.ewm(span=14, adjust=False).mean() / (atr15 + 1e-9)
                ndi15 = 100 * ndm15.ewm(span=14, adjust=False).mean() / (atr15 + 1e-9)
                dx15  = 100 * (pdi15 - ndi15).abs() / (pdi15 + ndi15 + 1e-9)
                adx15 = float(dx15.ewm(span=14, adjust=False).mean().iloc[-1])
                self._pilot_adx15 = adx15  # Store for bypass logic
                state = "TRENDING" if adx15 > 25 else ("WEAK" if adx15 > 20 else "SIDEWAYS")
                logger.info(f"Cycle #{cycle}: 15m ADX={adx15:.1f} ({state}) → passing to ML")
        except Exception as e:
            logger.debug(f"ADX context check failed: {e}")

        # Step 2: Run ML model
        ml_signal = 2
        ml_proba = [0.0, 0.0, 1.0]
        ml_indicators = {}
        ml_direction = "SKIP"

        if self.ml_engine and self.ml_engine.is_ready():
            try:
                ml_signal, ml_proba_arr, ml_conf, ml_indicators = \
                    self._run_ml_prediction(spot)
                ml_proba = list(ml_proba_arr)
                ml_direction = "CALL" if ml_signal == 0 else ("PUT" if ml_signal == 1 else "SKIP")
                self._ml_signals_today += (1 if ml_signal != 2 else 0)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        logger.info(
            f"Cycle #{cycle}: ML={ml_direction} "
            f"(C={ml_proba[0]:.3f} P={ml_proba[1]:.3f} S={ml_proba[2]:.3f})"
        )

        # Gap override removed — gap magnitude is passed as ML features only.
        # The ML model weights gap size via fut_dist_vwap_pct and gap feature inputs.

        # ══════════════════════════════════════════════════════════════
        # V9.3: INTRADAY V-RECOVERY OVERRIDE
        # ══════════════════════════════════════════════════════════════
        # Problem: On V-recovery days (drop 300pts then rally back),
        # ML keeps saying SKIP because all 156 features are lagging —
        # RSI(14) takes 70+ minutes to recover from extreme lows.
        # Even if ML says CALL, VWAP + Structure filters block it
        # because spot is below session VWAP during the recovery.
        #
        # Detection:
        #   1. Session dropped > 1.5 × ATR_15m from session high
        #   2. Price has recovered > 50% of that drop
        #   3. Last 3 closes making higher lows (structure shifting)
        #
        # Actions when recovery mode active:
        #   - If ML=SKIP → force CALL with moderate confidence
        #   - Set recovery_mode flag → bypasses VWAP filter, structure
        #     filter, and 2-bar confirmation (price is moving fast)
        #
        recovery_mode = False
        try:
            session_drop = self._session_high - self._session_low
            atr_15m_rec = self._compute_atr_15m()
            drop_threshold = 1.5 * atr_15m_rec

            if session_drop > drop_threshold and self._session_low < self._session_high and self.config.enable_v10_signal_patches:
                recovery_pct = (spot - self._session_low) / session_drop
                # Check last 3 bars making higher lows (micro structure shift)
                higher_lows = False
                try:
                    from core.tv_fetcher import get_tv_fetcher
                    df5_rec = get_tv_fetcher().get_nifty_5min(n_bars=10)
                    if not df5_rec.empty and len(df5_rec) >= 4:
                        lows_3 = df5_rec["low"].astype(float).tail(3).values
                        higher_lows = (lows_3[-1] > lows_3[-2] > lows_3[-3])
                except Exception:
                    pass

                if recovery_pct >= 0.50 and higher_lows:
                    # ── Deduplication: suppress if same V-recovery already fired ──
                    # Cooldown = 4 cycles (~20 min) so the same intraday bounce
                    # doesn't produce two BUY signals on consecutive cycles.
                    # Also check that spot moved at least 20pts since last fire
                    # (allows a genuine new leg to break through the cooldown).
                    V_RECOVERY_COOLDOWN = 4  # cycles
                    cycles_since_last = cycle - self._last_v_recovery_cycle
                    spot_moved = abs(spot - self._v_recovery_fired_spot) >= 20.0
                    if self._last_v_recovery_cycle > 0 and cycles_since_last < V_RECOVERY_COOLDOWN and not spot_moved:
                        logger.info(
                            f"Cycle #{cycle}: V-RECOVERY conditions met but COOLDOWN active "
                            f"(last fired cycle #{self._last_v_recovery_cycle}, "
                            f"{cycles_since_last} cycles ago, spot moved {abs(spot - self._v_recovery_fired_spot):.0f}pts < 20) — suppressed"
                        )
                    else:
                        recovery_mode = True
                        logger.info(
                            f"Cycle #{cycle}: V-RECOVERY detected: "
                            f"session drop {session_drop:.0f}pts (threshold={drop_threshold:.0f}), "
                            f"recovered {recovery_pct:.0%}, higher lows confirmed"
                        )

                        # Phase 1 fix (2026-06-09): Do NOT override ML SKIP with
                        # hardcoded [0.60, 0.15, 0.25] probabilities.
                        # Previously this injected fake 60% CALL confidence when the
                        # model said SKIP, causing T5(−₹4,891) and T14(−₹4,144) on
                        # 2026-04-30 and 2026-05-07. ML SKIP is now respected.
                        # V-RECOVERY detection is recorded as a shadow observation so
                        # the next model retrain can learn the pattern properly.
                        if ml_signal == 2:
                            self._v_recovery_skip_count = getattr(
                                self, "_v_recovery_skip_count", 0) + 1
                            logger.warning(
                                f"Cycle #{cycle}: ⚠️ V-RECOVERY detected — "
                                f"ML=SKIP MAINTAINED (NOT overriding). "
                                f"recovery={recovery_pct:.0%} | "
                                f"suppressed_total=#{self._v_recovery_skip_count}"
                            )
                            try:
                                _notif = getattr(self.trader, "notif", None)
                                if (_notif and getattr(_notif, "telegram", None)
                                        and _notif.telegram.enabled):
                                    _notif.telegram.send_message(
                                        f"⚠️ <b>V-RECOVERY signal — ML SKIP maintained</b>\n"
                                        f"recovery={recovery_pct:.0%} | cycle=#{cycle}\n"
                                        f"No trade (ML SKIP respected per Phase 1 fix)."
                                    )
                            except Exception:
                                pass
                            ml_indicators["recovery_override"] = False
                            ml_indicators["v_recovery_suppressed"] = True
                            ml_indicators["recovery_pct"] = round(recovery_pct * 100, 1)
                            # Still record cycle/spot so cooldown applies next detection
                            self._last_v_recovery_cycle = cycle
                            self._v_recovery_fired_spot = spot

                elif recovery_pct >= 0.30:
                    # Recovering but not enough yet — log for awareness
                    logger.debug(
                        f"Cycle #{cycle}: Partial recovery {recovery_pct:.0%} "
                        f"(need 50%+, higher_lows={higher_lows})"
                    )
        except Exception as e:
            logger.debug(f"Recovery detection failed: {e}")

        # ── FEATURE ENGINEERING DIRECTIVE ──────────────────────────────
        # Instead of overriding ML SKIP, compute micro-structure features
        # and inject them into ml_indicators for the NEXT training pipeline.
        # The ML model learns these patterns → no more manual overrides.
        #
        # Features computed:
        #   pa_vwap_distance:     spot - session_VWAP (signed, pts)
        #   pa_ema3_slope:        (EMA3 - EMA8) / EMA8 * 100 (% spread)
        #   pa_sequential_closes: consecutive same-dir 5m closes (+up, -down)
        #   pa_rsi_momentum:      RSI(14) value
        #   pa_vwap_cross:        1 if spot crossed VWAP in last 3 bars, else 0
        #
        try:
            from core.tv_fetcher import get_tv_fetcher
            import pandas as pd

            df5_fe = get_tv_fetcher().get_nifty_5min(n_bars=80)
            df3_fe = get_tv_fetcher().get_ohlcv("NIFTY", "NSE", 3, n_bars=20)

            if not df5_fe.empty and len(df5_fe) >= 10 and not df3_fe.empty and len(df3_fe) >= 10:

                # --- pa_vwap_distance: spot - session VWAP ---
                df_v = df5_fe.copy()
                df_v["typical"] = (
                    df_v["high"].astype(float) +
                    df_v["low"].astype(float) +
                    df_v["close"].astype(float)
                ) / 3.0
                vol = df_v["volume"].astype(float) if "volume" in df_v.columns \
                    else pd.Series(1.0, index=df_v.index)
                df_v["date"] = df_v.index.date
                df_v["cum_tpv"] = df_v.groupby("date").apply(
                    lambda g: (g["typical"] * vol.loc[g.index]).cumsum()
                ).droplevel(0)
                df_v["cum_vol"] = df_v.groupby("date").apply(
                    lambda g: vol.loc[g.index].cumsum()
                ).droplevel(0)
                df_v["vwap"] = df_v["cum_tpv"] / df_v["cum_vol"].replace(0, float('nan'))
                vwap_now = float(df_v["vwap"].iloc[-1])
                pa_vwap_distance = round(spot - vwap_now, 2)

                # --- pa_ema3_slope: (EMA3 - EMA8) / EMA8 * 100 ---
                close_3m = df3_fe["close"].astype(float)
                ema3 = float(close_3m.ewm(span=3, adjust=False).mean().iloc[-1])
                ema8 = float(close_3m.ewm(span=8, adjust=False).mean().iloc[-1])
                pa_ema3_slope = round((ema3 - ema8) / (ema8 + 1e-9) * 100, 4)

                # --- pa_sequential_closes: consecutive same-direction closes ---
                closes_5m = df5_fe["close"].astype(float).tail(10).values
                seq_count = 0
                if len(closes_5m) >= 2:
                    last_dir = 1 if closes_5m[-1] > closes_5m[-2] else -1
                    seq_count = 1
                    for i in range(len(closes_5m) - 2, 0, -1):
                        bar_dir = 1 if closes_5m[i] > closes_5m[i - 1] else -1
                        if bar_dir == last_dir:
                            seq_count += 1
                        else:
                            break
                    seq_count *= last_dir  # positive=up, negative=down

                # --- pa_rsi_momentum ---
                pa_rsi = ml_indicators.get("rsi14", 50)

                # --- pa_vwap_cross: did spot cross VWAP in last 3 bars? ---
                last3_closes = df5_fe["close"].astype(float).tail(3).values
                vwap_vals = df_v["vwap"].tail(3).values
                pa_vwap_cross = 0
                if len(last3_closes) >= 3 and len(vwap_vals) >= 3:
                    for i in range(1, len(last3_closes)):
                        prev_side = last3_closes[i - 1] - vwap_vals[i - 1]
                        curr_side = last3_closes[i] - vwap_vals[i]
                        if prev_side * curr_side < 0:  # sign change = cross
                            pa_vwap_cross = 1
                            break

                # --- struct_hh_hl: causal HH/HL price-structure feature ---
                # Parallel output only (see core/structure_features.py) — computed
                # and logged every cycle, never read by any gate/threshold/trade
                # decision. Reuses df5_fe (already fetched above) so this adds no
                # extra network/data cost.
                hh_hl_str = ""
                if self.config.enable_hh_hl_feature:
                    from core.structure_features import compute_hh_hl_structure
                    closes_for_struct = df5_fe["close"].astype(float).tolist()
                    struct = compute_hh_hl_structure(closes_for_struct)
                    ml_indicators["struct_hh_hl_up"] = struct["hh_hl_up"]
                    ml_indicators["struct_hh_hl_down"] = struct["hh_hl_down"]
                    ml_indicators["struct_n_swing_highs"] = struct["n_swing_highs"]
                    ml_indicators["struct_n_swing_lows"] = struct["n_swing_lows"]
                    hh_hl_str = (
                        f" HH_HL_up={struct['hh_hl_up']} HH_HL_down={struct['hh_hl_down']}"
                    )

                # Inject into ml_indicators (for logging + next training pipeline)
                ml_indicators["pa_vwap_distance"] = pa_vwap_distance
                ml_indicators["pa_ema3_slope"] = pa_ema3_slope
                ml_indicators["pa_sequential_closes"] = seq_count
                ml_indicators["pa_rsi_momentum"] = pa_rsi
                ml_indicators["pa_vwap_cross"] = pa_vwap_cross

                logger.info(
                    f"Cycle #{cycle}: FE Directive: VWAP_dist={pa_vwap_distance:+.0f} "
                    f"EMA3_slope={pa_ema3_slope:+.3f}% seq_closes={seq_count} "
                    f"RSI={pa_rsi:.0f} VWAP_cross={pa_vwap_cross}{hh_hl_str}"
                )

        except Exception as e:
            logger.debug(f"Feature engineering directive failed: {e}")

        # Step 3: If ML says SKIP → done (no Claude call needed)
        if ml_signal == 2:
            logger.info(f"Cycle #{cycle}: ML SKIP - no trade")
            self._prev_ml_signal = 2
            with self._lock:
                self._last_recommendation = {
                    "cycle": cycle, "ml_signal": "SKIP",
                    "action": "WAIT", "confidence": 0,
                    "time": datetime.now().isoformat(),
                }
            return

        # ══════════════════════════════════════════════════════════════
        # Step 3.05: RSI DIVERGENCE + MULTI-TF MOMENTUM FILTER (SMC v2)
        # ══════════════════════════════════════════════════════════════
        # OLD: Rigid RSI > 75 block. Missed continuation moves.
        # NEW: 3-layer check:
        #   Layer 1: Bearish divergence (price HH but RSI LH) → block CALL
        #   Layer 2: Bullish divergence (price LL but RSI HL) → block PUT
        #   Layer 3: Multi-TF RSI confirmation (5m + 15m must agree)
        #   Layer 4: Extreme RSI (>85 / <15) still hard-blocked (exhaustion)
        rsi_val = ml_indicators.get("rsi14", 50)
        rsi_block = False
        rsi_reason = ""

        try:
            from core.tv_fetcher import get_tv_fetcher
            import pandas as pd

            # --- Layer 1 & 2: RSI Divergence Detection ---
            # Track (spot, rsi) pairs for swing detection
            self._rsi_history.append((spot, rsi_val))
            if len(self._rsi_history) > 30:
                self._rsi_history = self._rsi_history[-30:]

            if len(self._rsi_history) >= 6:
                spots = [h[0] for h in self._rsi_history]
                rsis = [h[1] for h in self._rsi_history]

                # Find last two local peaks (for bearish divergence)
                # Peak = higher than both neighbors in a 3-bar window
                peaks_idx = []
                for j in range(1, len(spots) - 1):
                    if spots[j] > spots[j-1] and spots[j] > spots[j+1]:
                        peaks_idx.append(j)

                # Bearish divergence: price makes Higher High, RSI makes Lower High
                if ml_signal == 0 and len(peaks_idx) >= 2 and rsi_val > 65:
                    p1, p2 = peaks_idx[-2], peaks_idx[-1]
                    if spots[p2] > spots[p1] and rsis[p2] < rsis[p1]:
                        rsi_block = True
                        rsi_reason = (
                            f"Bearish RSI divergence: price {spots[p1]:.0f}→{spots[p2]:.0f} (HH) "
                            f"but RSI {rsis[p1]:.0f}→{rsis[p2]:.0f} (LH)"
                        )

                # Find last two local troughs (for bullish divergence)
                troughs_idx = []
                for j in range(1, len(spots) - 1):
                    if spots[j] < spots[j-1] and spots[j] < spots[j+1]:
                        troughs_idx.append(j)

                # Bullish divergence: price makes Lower Low, RSI makes Higher Low
                if ml_signal == 1 and len(troughs_idx) >= 2 and rsi_val < 35:
                    t1, t2 = troughs_idx[-2], troughs_idx[-1]
                    if spots[t2] < spots[t1] and rsis[t2] > rsis[t1]:
                        rsi_block = True
                        rsi_reason = (
                            f"Bullish RSI divergence: price {spots[t1]:.0f}→{spots[t2]:.0f} (LL) "
                            f"but RSI {rsis[t1]:.0f}→{rsis[t2]:.0f} (HL)"
                        )

            # --- Layer 3: Multi-TF RSI confirmation (5m + 15m) ---
            if not rsi_block:
                df15_rsi = get_tv_fetcher().get_nifty_15min(n_bars=20)
                if not df15_rsi.empty and len(df15_rsi) >= 14:
                    close_15 = df15_rsi["close"].astype(float)
                    delta_15 = close_15.diff()
                    gain_15 = delta_15.clip(lower=0).rolling(14).mean()
                    loss_15 = (-delta_15.clip(upper=0)).rolling(14).mean()
                    rs_15 = gain_15 / loss_15.replace(0, float('nan'))
                    rsi_15m = float((100 - (100 / (1 + rs_15))).iloc[-1])

                    # CALL needs bullish momentum on both TFs
                    if ml_signal == 0 and rsi_val > 70 and rsi_15m < 45:
                        rsi_block = True
                        rsi_reason = (
                            f"RSI TF conflict: 5m RSI={rsi_val:.0f} (OB) but "
                            f"15m RSI={rsi_15m:.0f} (bearish) — momentum exhausting"
                        )
                    # PUT needs bearish momentum on both TFs
                    if ml_signal == 1 and rsi_val < 30 and rsi_15m > 55:
                        rsi_block = True
                        rsi_reason = (
                            f"RSI TF conflict: 5m RSI={rsi_val:.0f} (OS) but "
                            f"15m RSI={rsi_15m:.0f} (bullish) — selling exhausting"
                        )

            # --- Layer 4: Extreme exhaustion hard block (RSI > 85 or < 15) ---
            if not rsi_block:
                if ml_signal == 0 and rsi_val > 85:
                    rsi_block = True
                    rsi_reason = f"RSI extreme exhaustion: {rsi_val:.0f} > 85"
                if ml_signal == 1 and rsi_val < 15:
                    rsi_block = True
                    rsi_reason = f"RSI extreme exhaustion: {rsi_val:.0f} < 15"

        except Exception as e:
            # Fallback: use simple OB/OS if TV fetch fails
            if ml_signal == 0 and rsi_val > 78:
                rsi_block = True
                rsi_reason = f"RSI overbought fallback: {rsi_val:.1f} > 78"
            if ml_signal == 1 and rsi_val < 22:
                rsi_block = True
                rsi_reason = f"RSI oversold fallback: {rsi_val:.1f} < 22"
            logger.debug(f"RSI divergence check failed, using fallback: {e}")

        if rsi_block and self.config.enable_v10_signal_patches:
            # V9.3: Recovery mode bypasses RSI multi-TF conflict (not extreme exhaustion)
            if recovery_mode and "exhaustion" not in rsi_reason:
                logger.info(
                    f"Cycle #{cycle}: RSI FILTER would block ({rsi_reason}) "
                    f"but V-RECOVERY MODE active → bypassing"
                )
                rsi_block = False
            else:
                logger.info(f"Cycle #{cycle}: ⛔ RSI FILTER: {rsi_reason} → SKIP")
                self._prev_ml_signal = ml_signal
                with self._lock:
                    self._last_recommendation = {
                        "cycle": cycle, "ml_signal": ml_direction,
                        "action": "WAIT", "confidence": 0,
                        "reason": rsi_reason,
                        "time": datetime.now().isoformat(),
                    }
                _shadow_skip(f"rsi_filter:{rsi_reason[:40]}")
                return

        # ══════════════════════════════════════════════════════════════
        # Step 3.06: VWAP + FAST EMA STRUCTURE FILTER (SMC v2)
        # ══════════════════════════════════════════════════════════════
        # OLD: 5m EMA9/EMA21 — too slow for scalping (45-105 min lag).
        # NEW: 3-layer structure check:
        #   Layer 1: Session VWAP — institutional fair value anchor
        #            CALL must be above VWAP, PUT must be below VWAP
        #   Layer 2: Fast EMA3/EMA8 on 3-minute chart — execution-TF trend
        #            Confirms micro-structure direction
        #   Layer 3: 5m EMA9/EMA21 as secondary confirmation (existing logic)
        #            Only blocks when BOTH VWAP and fast EMA disagree
        structure_block = False
        structure_reason = ""
        vwap_bullish = None  # None = unknown
        fast_ema_bullish = None

        try:
            from core.tv_fetcher import get_tv_fetcher
            import pandas as pd
            import numpy as np

            # --- Layer 1: Session VWAP (from Nifty FUTURES — has real volume) ---
            # Spot Nifty has NO volume → VWAP on spot is meaningless.
            # VWAPEngine fetches current-month futures bars via TradingView/Kotak.
            from core.vwap_engine import VWAPEngine
            from core.tv_fetcher import get_tv_fetcher as _get_tv
            _vwap_engine = VWAPEngine(
                tv_fetcher=_get_tv(),
                kotak_client=getattr(self.trader, "client", None),
            )
            vwap_now = _vwap_engine.get_current_vwap()
            if vwap_now > 0:
                spot_vs_vwap = spot - vwap_now
                vwap_bullish = spot > vwap_now
                logger.info(
                    f"Cycle #{cycle}: Futures VWAP={vwap_now:.0f} spot={spot:.0f} "
                    f"({'ABOVE' if vwap_bullish else 'BELOW'} "
                    f"by {abs(spot_vs_vwap):.0f}pts)"
                )
                # Track persistent VWAP bias across cycles (used to block
                # counter-VWAP entries when structural bias is established)
                _vwap_threshold = 50.0
                if spot_vs_vwap < -_vwap_threshold:
                    self._consec_vwap_below += 1
                    self._consec_vwap_above = 0
                elif spot_vs_vwap > _vwap_threshold:
                    self._consec_vwap_above += 1
                    self._consec_vwap_below = 0
                else:
                    # Neutral zone — decay both counters (don't hard-reset so a
                    # brief chop period doesn't erase a genuine structural bias)
                    self._consec_vwap_below = max(0, self._consec_vwap_below - 1)
                    self._consec_vwap_above = max(0, self._consec_vwap_above - 1)
                self._last_spot_vs_vwap = spot_vs_vwap
            else:
                logger.warning(f"Cycle #{cycle}: VWAP unavailable — skipping VWAP gate")

            # --- Layer 2: Fast EMA3/EMA8 on 3-minute chart ---
            df3 = get_tv_fetcher().get_ohlcv("NIFTY", "NSE", 3, n_bars=30)
            if not df3.empty and len(df3) >= 10:
                close_3m = df3["close"].astype(float)
                ema3 = float(close_3m.ewm(span=3, adjust=False).mean().iloc[-1])
                ema8 = float(close_3m.ewm(span=8, adjust=False).mean().iloc[-1])
                fast_ema_bullish = ema3 > ema8

                logger.info(
                    f"Cycle #{cycle}: 3m EMA3={ema3:.0f} EMA8={ema8:.0f} "
                    f"({'BULL' if fast_ema_bullish else 'BEAR'})"
                )

            # --- Decision Matrix ---
            # Both VWAP and fast EMA must agree to BLOCK
            # Single disagreement = warning only (reduce confidence later)
            if ml_signal == 0:  # CALL
                vwap_against = (vwap_bullish is False)
                ema_against = (fast_ema_bullish is False)
                if vwap_against and ema_against:
                    structure_block = True
                    structure_reason = (
                        f"Structure bearish: below VWAP + 3m EMA3<EMA8 — "
                        f"CALL against institutional flow"
                    )
                elif vwap_against or ema_against:
                    # Partial conflict — add warning, don't block
                    conflict = "below VWAP" if vwap_against else "3m EMA bearish"
                    ml_indicators["structure_warning"] = conflict
                    logger.info(
                        f"Cycle #{cycle}: ⚠️ Partial structure conflict: {conflict} "
                        f"(not blocking, reducing confidence)"
                    )

            if ml_signal == 1:  # PUT
                vwap_against = (vwap_bullish is True)
                ema_against = (fast_ema_bullish is True)
                if vwap_against and ema_against:
                    structure_block = True
                    structure_reason = (
                        f"Structure bullish: above VWAP + 3m EMA3>EMA8 — "
                        f"PUT against institutional flow"
                    )
                elif vwap_against or ema_against:
                    conflict = "above VWAP" if vwap_against else "3m EMA bullish"
                    ml_indicators["structure_warning"] = conflict
                    logger.info(
                        f"Cycle #{cycle}: ⚠️ Partial structure conflict: {conflict} "
                        f"(not blocking, reducing confidence)"
                    )

        except Exception as e:
            # Fallback: use original EMA9/21 on 5m
            try:
                df5_fb = get_tv_fetcher().get_nifty_5min(n_bars=30)
                if not df5_fb.empty and len(df5_fb) >= 21:
                    close_s = df5_fb["close"].astype(float)
                    ema9 = float(close_s.ewm(span=9, adjust=False).mean().iloc[-1])
                    ema21 = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
                    if ml_signal == 0 and ema9 < ema21:
                        structure_block = True
                        structure_reason = f"EMA fallback: EMA9={ema9:.0f} < EMA21={ema21:.0f}"
                    if ml_signal == 1 and ema9 > ema21:
                        structure_block = True
                        structure_reason = f"EMA fallback: EMA9={ema9:.0f} > EMA21={ema21:.0f}"
            except Exception:
                pass
            logger.debug(f"VWAP+EMA filter failed, using fallback: {e}")

        # ══════════════════════════════════════════════════════════════
        # 2026-04-27 FIX: EARLY REVERSAL OVERRIDE
        # ══════════════════════════════════════════════════════════════
        # Problem from today's chart:
        #   • 11:15 NIFTY bottomed at 23960 (down 180pts from 24140 high)
        #   • 11:30-11:40 ML fired 6 CALL signals exactly at the V-bottom
        #   • STRUCTURE FILTER killed all 6 (spot below VWAP, 3m EMA bear)
        #   • Price then ran +160pts to 24120 by 14:00
        #   • Bot missed the BEST trade of the day
        #
        # V-RECOVERY mode only triggers AFTER 50% of bounce → too late
        # Need to catch reversals AT the extreme, not after.
        #
        # Override fires when ALL true:
        #   ML signals CALL/PUT AND
        #   RSI is at counter-trend extreme (< 35 for CALL / > 65 for PUT) AND
        #   Spot is within 0.4% of day-low (CALL) / day-high (PUT) AND
        #   Last 5m candle shows reversal shape (close > open for CALL)
        #
        # This is the classic "buy oversold at support" / "short overbought
        # at resistance" setup that trains well in backtest but gets killed
        # by structure filters in live.
        early_reversal = False
        early_reversal_reason = ""
        try:
            from core.tv_fetcher import get_tv_fetcher as _get_tv2
            df5_rev = _get_tv2().get_nifty_5min(n_bars=5)
            # 2026-06-11 BUG FIX: key is "rsi14" (ml_engine._build_summary) —
            # the old "rsi_14"/"rsi" lookups never matched, so this always
            # read 50 and the early-reversal detector could never fire.
            rsi_now = float(ml_indicators.get("rsi14", ml_indicators.get("rsi_14", 50)) or 50)
            day_low = self._session_low if self._session_low < 999000 else spot
            day_high = self._session_high if self._session_high > 0 else spot

            # Need a reasonable session range to consider this a reversal
            session_range = day_high - day_low
            atr_for_rev = self._compute_atr_15m() or 30.0

            if not df5_rev.empty and session_range >= atr_for_rev:
                last_open = float(df5_rev["open"].iloc[-1])
                last_close = float(df5_rev["close"].iloc[-1])
                last_low = float(df5_rev["low"].iloc[-1])
                last_high = float(df5_rev["high"].iloc[-1])
                bullish_candle = last_close > last_open
                bearish_candle = last_close < last_open

                # CALL reversal at oversold day-low
                if ml_signal == 0:
                    near_low = day_low > 0 and ((spot - day_low) / day_low) <= 0.004  # 0.4%
                    if rsi_now <= 35 and near_low and bullish_candle and self.config.enable_v10_signal_patches:
                        early_reversal = True
                        early_reversal_reason = (
                            f"CALL reversal: RSI={rsi_now:.0f}<=35, "
                            f"spot {spot:.0f} within 0.4% of day_low {day_low:.0f}, "
                            f"5m close {last_close:.0f} > open {last_open:.0f}"
                        )

                # PUT reversal at overbought day-high
                if ml_signal == 1:
                    near_high = day_high > 0 and ((day_high - spot) / day_high) <= 0.004
                    if rsi_now >= 65 and near_high and bearish_candle and self.config.enable_v10_signal_patches:
                        early_reversal = True
                        early_reversal_reason = (
                            f"PUT reversal: RSI={rsi_now:.0f}>=65, "
                            f"spot {spot:.0f} within 0.4% of day_high {day_high:.0f}, "
                            f"5m close {last_close:.0f} < open {last_open:.0f}"
                        )
        except Exception as _e:
            logger.debug(f"Early reversal detector failed: {_e}")

        # 2026-04-27 EXPERT FIX (López de Prado, Ch. 3):
        # Filters should ADJUST probability, not gate entry. Convert
        # structure filter from binary block → confidence penalty.
        # Then let the probability gate (effective_min_conf) decide.
        structure_penalty_pct = 0
        if structure_block and self.config.enable_v10_signal_patches:
            if recovery_mode and ml_signal == 0:
                logger.info(
                    f"Cycle #{cycle}: STRUCTURE conflict ({structure_reason}) "
                    f"but V-RECOVERY MODE → no penalty for CALL"
                )
            elif early_reversal:
                logger.warning(
                    f"Cycle #{cycle}: STRUCTURE conflict ({structure_reason}) "
                    f"but EARLY REVERSAL → no penalty. {early_reversal_reason}"
                )
                ml_indicators["early_reversal"] = True
            else:
                # Apply scoring penalty rather than binary skip.
                # 12% penalty matches historical underperformance of
                # counter-structure trades vs aligned trades.
                structure_penalty_pct = 12
                ml_indicators["structure_penalty"] = structure_penalty_pct
                ml_indicators["structure_reason"] = structure_reason
                logger.info(
                    f"Cycle #{cycle}: STRUCTURE PENALTY -{structure_penalty_pct}% "
                    f"({structure_reason}) → conf threshold raised"
                )
            # Reset block to false in all cases — we score, not gate.
            # Decision deferred to confidence threshold below.
            structure_block = False

        # Step 3.1: Opening session higher confidence bar (09:15-09:30)
        # 2026-04-30 BUG FIX: Don't overwrite ml_conf — it's the DIRECTIONAL
        # confidence from ml_engine. Use a local variable for the raw check.
        ml_conf_raw = max(ml_proba[0], ml_proba[1])   # for opening-bar check only
        now_t = datetime.now()
        if now_t.hour == 9 and now_t.minute < 30 and ml_conf_raw < 0.60:
            logger.info(
                f"Cycle #{cycle}: ML={ml_direction} raw_p={ml_conf_raw:.2f} "
                f"< 0.60 opening threshold → SKIP"
            )
            _shadow_skip(f"opening_threshold:raw_p={ml_conf_raw:.2f}")
            self._prev_ml_signal = ml_signal
            with self._lock:
                self._last_recommendation = {
                    "cycle": cycle, "ml_signal": ml_direction,
                    "action": "WAIT", "confidence": 0,
                    "reason": f"Opening conf {ml_conf:.2f} < 0.60",
                    "time": datetime.now().isoformat(),
                }
            return

        # Step 3.2: 2-bar confirmation REMOVED — trigger on first qualifying bar.
        # Gap between signal and execution was causing missed moves on trending days.
        self._prev_ml_signal = ml_signal

        # ══════════════════════════════════════════════════════════════
        # Step 3.3: SMC REVERSAL FILTER — FVG + Order Block Validation
        # ══════════════════════════════════════════════════════════════
        # OLD: Hard skip if price dropped/rallied 80pts. Missed valid
        #      bounces off support/demand zones.
        # NEW: 3-step logic:
        #   1. Detect if price moved 80+ pts from session extreme
        #   2. Scan for Fair Value Gaps (FVG) and Order Blocks (OB) in the zone
        #   3. If FVG/OB found at current level → ALLOW signal (smart money zone)
        #      If NO structure support → BLOCK signal (naked reversal = trap)
        #
        # FVG (Fair Value Gap):
        #   Bullish FVG: bar[i-2].high < bar[i].low (gap up, unfilled)
        #   Bearish FVG: bar[i-2].low > bar[i].high (gap down, unfilled)
        #
        # Order Block (OB):
        #   Bullish OB: last bearish candle before a strong bullish impulse
        #   Bearish OB: last bullish candle before a strong bearish impulse
        #
        drop_from_high = self._session_high - spot
        rise_from_low = spot - self._session_low
        # V9.3: Dynamic reversal threshold = 1.5 × ATR_15m (adapts to volatility)
        atr_15m = self._compute_atr_15m()
        threshold = max(40.0, round(1.5 * atr_15m, 1))  # floor 40pts
        self._reversal_threshold = threshold  # update for logging
        logger.debug(f"Reversal threshold: 1.5×ATR_15m({atr_15m:.1f}) = {threshold:.0f}pts")

        reversal_detected = (
            (ml_signal == 0 and drop_from_high >= threshold) or
            (ml_signal == 1 and rise_from_low >= threshold)
        )

        # V9.3: Recovery mode bypasses reversal filter — the recovery IS the reversal
        if recovery_mode and reversal_detected and ml_signal == 0:
            logger.info(
                f"Cycle #{cycle}: Reversal filter triggered (drop={drop_from_high:.0f}pts) "
                f"but V-RECOVERY MODE active → bypassing for CALL"
            )
            reversal_detected = False

        if reversal_detected:
            # Scan for SMC structure at current price level
            smc_support = False
            smc_reason = ""
            zone_no_choch = False   # zone present but CHoCH not yet confirmed

            try:
                from core.tv_fetcher import get_tv_fetcher
                import pandas as pd
                import numpy as np

                df5_smc = get_tv_fetcher().get_nifty_5min(n_bars=60)
                if not df5_smc.empty and len(df5_smc) >= 10:
                    highs = df5_smc["high"].astype(float).values
                    lows = df5_smc["low"].astype(float).values
                    opens = df5_smc["open"].astype(float).values
                    closes = df5_smc["close"].astype(float).values

                    # --- Detect Fair Value Gaps (FVG) ---
                    fvg_zones = []
                    for k in range(2, len(df5_smc)):
                        # Bullish FVG: gap between bar[k-2].high and bar[k].low
                        if lows[k] > highs[k-2]:
                            fvg_zones.append({
                                "type": "BULL",
                                "high": float(lows[k]),
                                "low": float(highs[k-2]),
                                "mid": float((lows[k] + highs[k-2]) / 2),
                            })
                        # Bearish FVG: gap between bar[k-2].low and bar[k].high
                        if highs[k] < lows[k-2]:
                            fvg_zones.append({
                                "type": "BEAR",
                                "high": float(lows[k-2]),
                                "low": float(highs[k]),
                                "mid": float((lows[k-2] + highs[k]) / 2),
                            })

                    # --- Detect Order Blocks (OB) ---
                    # OB = last opposite candle before an impulsive move
                    ob_zones = []
                    impulse_threshold = 30.0  # 30pts = significant move

                    for k in range(1, len(df5_smc) - 2):
                        body_k = closes[k] - opens[k]
                        move_after = closes[k+2] - closes[k]

                        # Bullish OB: bearish candle (body<0) before strong up-move
                        if body_k < -5 and move_after > impulse_threshold:
                            ob_zones.append({
                                "type": "BULL_OB",
                                "high": float(opens[k]),   # OB top = open of bearish candle
                                "low": float(closes[k]),   # OB bottom = close of bearish candle
                                "strength": float(move_after),
                            })
                        # Bearish OB: bullish candle (body>0) before strong down-move
                        if body_k > 5 and move_after < -impulse_threshold:
                            ob_zones.append({
                                "type": "BEAR_OB",
                                "high": float(closes[k]),  # OB top = close of bullish candle
                                "low": float(opens[k]),    # OB bottom = open of bullish candle
                                "strength": float(abs(move_after)),
                            })

                    self._fvg_zones = fvg_zones[-10:]  # keep last 10
                    self._order_blocks = ob_zones[-10:]

                    # --- Check if current price sits in a support/resistance zone ---
                    zone_tolerance = 25.0  # pts tolerance for zone match
                    zone_found = False
                    zone_desc = ""

                    if ml_signal == 0:  # CALL after big drop → check for demand zone
                        for zone in fvg_zones:
                            if zone["type"] == "BULL":
                                if zone["low"] - zone_tolerance <= spot <= zone["high"] + zone_tolerance:
                                    zone_found = True
                                    zone_desc = (
                                        f"Bullish FVG {zone['low']:.0f}-{zone['high']:.0f}"
                                    )
                                    break
                        if not zone_found:
                            for ob in ob_zones:
                                if ob["type"] == "BULL_OB":
                                    if ob["low"] - zone_tolerance <= spot <= ob["high"] + zone_tolerance:
                                        zone_found = True
                                        zone_desc = (
                                            f"Bullish OB {ob['low']:.0f}-{ob['high']:.0f} "
                                            f"(str={ob['strength']:.0f}pts)"
                                        )
                                        break

                    if ml_signal == 1:  # PUT after big rally → check for supply zone
                        for zone in fvg_zones:
                            if zone["type"] == "BEAR":
                                if zone["low"] - zone_tolerance <= spot <= zone["high"] + zone_tolerance:
                                    zone_found = True
                                    zone_desc = (
                                        f"Bearish FVG {zone['low']:.0f}-{zone['high']:.0f}"
                                    )
                                    break
                        if not zone_found:
                            for ob in ob_zones:
                                if ob["type"] == "BEAR_OB":
                                    if ob["low"] - zone_tolerance <= spot <= ob["high"] + zone_tolerance:
                                        zone_found = True
                                        zone_desc = (
                                            f"Bearish OB {ob['low']:.0f}-{ob['high']:.0f} "
                                            f"(str={ob['strength']:.0f}pts)"
                                        )
                                        break

                    # ── V9.3: Micro-CHoCH confirmation ────────────────────
                    # FVG/OB alone is not enough — price must ALSO break the
                    # previous 3-bar swing high (for CALL) or swing low (for PUT).
                    # CHoCH = Change of Character = first sign structure is shifting.
                    #
                    # CALL CHoCH: current close > max(high of last 3 bars before current)
                    # PUT  CHoCH: current close < min(low of last 3 bars before current)
                    choch_confirmed = False
                    choch_desc = ""

                    if zone_found and len(highs) >= 5 and len(lows) >= 5:
                        recent_close = float(closes[-1])
                        # Last 3 completed bars (exclude current)
                        swing_high_3 = float(max(highs[-4:-1]))  # bars [-4,-3,-2]
                        swing_low_3 = float(min(lows[-4:-1]))

                        if ml_signal == 0 and recent_close > swing_high_3:
                            choch_confirmed = True
                            choch_desc = (
                                f"CHoCH: close {recent_close:.0f} > "
                                f"3-bar swing high {swing_high_3:.0f}"
                            )
                        elif ml_signal == 1 and recent_close < swing_low_3:
                            choch_confirmed = True
                            choch_desc = (
                                f"CHoCH: close {recent_close:.0f} < "
                                f"3-bar swing low {swing_low_3:.0f}"
                            )

                    # Final SMC verdict: zone + CHoCH both required
                    # EXCEPTION (RSI extreme): when RSI is at extreme levels
                    # (<= 30 for CALL / >= 70 for PUT) AND spot is within 0.5%
                    # of the session low/high, the RSI extreme itself acts as
                    # the structural confirmation — CHoCH requirement is relaxed.
                    # Rationale: at RSI=17-30, the oversold condition is so extreme
                    # that an FVG zone alone is sufficient for a reversal trade.
                    # The CHoCH typically confirms 5 minutes AFTER the actual bottom
                    # (the next bar), causing the bot to miss the entry entirely.
                    # This exception restores those V-recovery entries.
                    # 2026-06-11 BUG FIX: key is "rsi14" — the old "rsi_14"/
                    # "rsi" lookups never matched, so this always read 50 and
                    # the RSI-extreme CHoCH relaxation below was dead code
                    # (Jun 11: blocked the 12:40/12:45 PUTs at RSI=66 near
                    # day-high — exactly the case this exception was built for).
                    rsi_now_smc = float(
                        ml_indicators.get("rsi14",
                        ml_indicators.get("rsi_14", 50)) or 50
                    )
                    day_low_smc  = self._session_low  if self._session_low  < 999000 else spot
                    day_high_smc = self._session_high if self._session_high > 0       else spot
                    rsi_extreme_call = (
                        ml_signal == 0 and rsi_now_smc <= 35   # oversold extreme
                        and day_low_smc > 0
                        and (spot - day_low_smc) / day_low_smc <= 0.005  # within 0.5% of day low
                    )
                    rsi_extreme_put = (
                        ml_signal == 1 and rsi_now_smc >= 65   # overbought extreme
                        and day_high_smc > 0
                        and (day_high_smc - spot) / day_high_smc <= 0.005  # within 0.5% of day high
                    )

                    if zone_found and choch_confirmed:
                        smc_support = True
                        smc_reason = f"{zone_desc} + {choch_desc}"
                    elif zone_found and not choch_confirmed and (rsi_extreme_call or rsi_extreme_put):
                        # RSI extreme + FVG zone → allow without CHoCH
                        smc_support = True
                        smc_reason = (
                            f"{zone_desc} + RSI_extreme={rsi_now_smc:.0f} "
                            f"(CHoCH relaxed — V-recovery at extreme)"
                        )
                        logger.info(
                            f"Cycle #{cycle}: SMC zone + RSI extreme "
                            f"(RSI={rsi_now_smc:.0f}) → CHoCH relaxed, "
                            f"allowing {'CALL' if ml_signal==0 else 'PUT'}"
                        )
                    elif zone_found and not choch_confirmed:
                        # Zone present but no CHoCH and RSI not extreme
                        smc_support = False
                        smc_reason = ""
                        zone_no_choch = True
                        ml_indicators["smc_zone_pending"] = zone_desc
                        logger.info(
                            f"Cycle #{cycle}: SMC zone found ({zone_desc}) but "
                            f"NO micro-CHoCH — structure not yet confirmed"
                        )

            except Exception as e:
                logger.debug(f"SMC structure scan failed: {e}")

            if smc_support:
                # Price dropped/rallied into a valid SMC zone — ALLOW the signal
                ml_indicators["smc_zone"] = smc_reason
                if ml_signal == 0:
                    logger.info(
                        f"Cycle #{cycle}: ✅ REVERSAL+SMC: Dropped {drop_from_high:.0f}pts "
                        f"BUT found {smc_reason} → ALLOWING CALL"
                    )
                else:
                    logger.info(
                        f"Cycle #{cycle}: ✅ REVERSAL+SMC: Rallied {rise_from_low:.0f}pts "
                        f"BUT found {smc_reason} → ALLOWING PUT"
                    )
            elif zone_no_choch:
                # 2026-06-11: zone found but CHoCH unconfirmed — penalty, not
                # block. Jun 11 audit: this hard block killed all 8 PUT signals
                # (incl. 87%-conf #43 at the day high). A valid FVG/OB at price
                # is partial confirmation; raise the Gate-2 bar instead of
                # vetoing outright. Truly naked reversals still hard-block below.
                ml_indicators["structure_penalty"] = (
                    int(ml_indicators.get("structure_penalty", 0) or 0) + 8
                )
                logger.info(
                    f"Cycle #{cycle}: REVERSAL FILTER: zone present, CHoCH "
                    f"pending → +8pp confidence penalty (was hard SKIP)"
                )
            else:
                # No SMC structure at all — naked reversal, BLOCK the signal
                if ml_signal == 0:
                    block_reason = (
                        f"Dropped {drop_from_high:.0f}pts from session high "
                        f"{self._session_high:.0f} — NO demand zone (FVG/OB) found"
                    )
                else:
                    block_reason = (
                        f"Rallied {rise_from_low:.0f}pts from session low "
                        f"{self._session_low:.0f} — NO supply zone (FVG/OB) found"
                    )

                logger.info(f"Cycle #{cycle}: ⛔ REVERSAL FILTER: {block_reason} → SKIP")
                _shadow_skip("reversal_smc")
                with self._lock:
                    self._last_recommendation = {
                        "cycle": cycle, "ml_signal": ml_direction,
                        "action": "WAIT", "confidence": 0,
                        "reason": f"Reversal: {block_reason}",
                        "time": datetime.now().isoformat(),
                    }
                return

        # Step 3.5: Market intel check (sentiment + OI + events)
        intel = None
        try:
            from core.market_intel import pre_trade_check, is_signal_aligned
            intel = pre_trade_check()

            # Block trade if market intel says unsafe
            if not intel.safe:
                logger.info(
                    f"Cycle #{cycle}: ML={ml_direction} BLOCKED by market intel: "
                    f"{intel.context_str}"
                )
                with self._lock:
                    self._last_recommendation = {
                        "cycle": cycle, "ml_signal": ml_direction,
                        "action": "WAIT", "confidence": 0,
                        "reason": f"Market intel unsafe: {intel.context_str}",
                        "time": datetime.now().isoformat(),
                    }
                return

            # Check OI alignment — extreme PCR stays a HARD GATE; OI-buildup
            # contradiction is now a confidence penalty (2026-06-11).
            # Jun 11 #24: SHORT_BUILD hard-blocked a 90%-conf CALL at 11:10 and
            # the market rallied 130pts after — buildup classification flips
            # cycle-to-cycle (NEUTRAL at 10:51 and 11:15) — too noisy to veto.
            aligned, align_reason = is_signal_aligned(ml_direction, intel)
            if not aligned and align_reason.startswith("HARD BLOCK: OI="):
                ml_indicators["oi_penalty"] = 6
                logger.info(
                    f"Cycle #{cycle}: OI FILTER: ML={ml_direction} {align_reason} "
                    f"→ +6pp conf penalty (was hard SKIP)"
                )
            elif not aligned:
                logger.info(
                    f"Cycle #{cycle}: ⛔ OI FILTER: ML={ml_direction} {align_reason} → SKIP"
                )
                _shadow_skip(f"oi_filter:{align_reason[:30]}")
                with self._lock:
                    self._last_recommendation = {
                        "cycle": cycle, "ml_signal": ml_direction,
                        "action": "WAIT", "confidence": 0,
                        "reason": f"OI conflict: {align_reason}",
                        "time": datetime.now().isoformat(),
                    }
                return

            # Soft warning (aligned=True but reason contains WARNING)
            if "WARNING" in align_reason:
                ml_indicators["oi_warning"] = align_reason
                logger.info(f"Cycle #{cycle}: ⚠️ OI soft warning: {align_reason}")

            # 2026-08-06: PCR-alignment confidence boost (opt-in, see
            # PilotConfig.pcr_alignment_boost_enabled for backtest evidence
            # and its "promising, not proven" caveat).
            if self.config.pcr_alignment_boost_enabled and intel.pcr_available:
                pcr_mood = _pcr_mood(intel.pcr)
                if pcr_mood is not None and pcr_mood == ml_direction:
                    ml_indicators["pcr_boost"] = -int(self.config.pcr_alignment_boost_pct)
                    logger.info(
                        f"Cycle #{cycle}: PCR ALIGNED (pcr={intel.pcr:.2f} mood={pcr_mood}) "
                        f"→ -{self.config.pcr_alignment_boost_pct:.0f}pp conf boost"
                    )

            # Add market intel + OI features to ml_indicators
            ml_indicators["market_intel"] = intel.context_str
            ml_indicators["sentiment_score"] = intel.sentiment_score
            ml_indicators["max_pain"] = intel.max_pain
            ml_indicators["oi_support"] = intel.oi_support
            ml_indicators["oi_resistance"] = intel.oi_resistance

            # ── Institutional Feature Engineering Pipeline ─────────────
            # These features are logged every cycle and available for the
            # next ML training run. They capture OI microstructure that
            # pure price-action features miss.

            # 1. PCR and PCR momentum (current - 5min ago)
            ml_indicators["oi_pcr"] = intel.pcr
            ml_indicators["pcr_5m_momentum"] = intel.pcr_5m_momentum

            # 2. OI buildup classification + raw changes
            ml_indicators["oi_buildup"] = intel.oi_buildup
            ml_indicators["oi_ce_change"] = intel.ce_oi_change
            ml_indicators["oi_pe_change"] = intel.pe_oi_change

            # 3. Distance from spot to Max OI strikes (pts)
            #    oi_ce_distance_pts = Max_CE_OI_Strike - Spot (positive = room to run up)
            #    oi_pe_distance_pts = Spot - Max_PE_OI_Strike (positive = room to run down)
            ml_indicators["oi_ce_distance_pts"] = round(intel.oi_resistance - spot, 1) if intel.oi_resistance > 0 else 0
            ml_indicators["oi_pe_distance_pts"] = round(spot - intel.oi_support, 1) if intel.oi_support > 0 else 0

            # 4. Max Pain distance (signed: positive = spot above max pain)
            ml_indicators["oi_max_pain_dist"] = round(spot - intel.max_pain, 1) if intel.max_pain > 0 else 0

            # 5. ATM OI data + gamma squeeze metrics
            ml_indicators["atm_ce_oi"] = intel.atm_ce_oi
            ml_indicators["atm_pe_oi"] = intel.atm_pe_oi
            ml_indicators["atm_ce_oi_change"] = intel.atm_ce_oi_change
            ml_indicators["atm_pe_oi_change"] = intel.atm_pe_oi_change

            # 6. ATM unwind rate (% change in opposing ATM OI over 5 min)
            #    For CALL signal: atm_oi_unwind_rate = ATM CE unwind % (negative = squeeze)
            #    For PUT signal:  atm_oi_unwind_rate = ATM PE unwind % (negative = squeeze)
            ml_indicators["atm_ce_unwind_pct"] = intel.atm_ce_unwind_pct
            ml_indicators["atm_pe_unwind_pct"] = intel.atm_pe_unwind_pct

            # 7. OI walls (top resistance/support strikes)
            ml_indicators["oi_wall_above"] = intel.oi_walls_above[0][0] if intel.oi_walls_above else 0
            ml_indicators["oi_wall_below"] = intel.oi_walls_below[0][0] if intel.oi_walls_below else 0

            logger.info(
                f"Cycle #{cycle}: Intel OK — PCR={intel.pcr:.2f}(mom={intel.pcr_5m_momentum:+.3f}) "
                f"OI={intel.oi_buildup} MaxPain={intel.max_pain:.0f} "
                f"CE_dist={intel.oi_resistance - spot:+.0f} PE_dist={spot - intel.oi_support:+.0f} "
                f"ATM_CE_unwind={intel.atm_ce_unwind_pct:+.1f}% "
                f"ATM_PE_unwind={intel.atm_pe_unwind_pct:+.1f}%"
            )

        except Exception as e:
            logger.debug(f"Market intel check failed (non-fatal): {e}")

        # Step 3.7: Inject VIX regime into ML indicators for Claude
        if regime:
            ml_indicators["vix_regime"] = regime.name
            ml_indicators["vix_level"] = regime.vix_level
            ml_indicators["vix_direction"] = regime.vix_direction
            ml_indicators["vix_sl_multiplier"] = regime.sl_multiplier
            ml_indicators["vix_tp_multiplier"] = regime.tp_multiplier
            ml_indicators["vix_preferred_strike"] = regime.preferred_strike

            # VIX trend-only filter: in VERY_HIGH/EXTREME, only trade with 15m trend
            if regime.trend_only:
                try:
                    from core.tv_fetcher import get_tv_fetcher
                    import pandas as pd
                    df15t = get_tv_fetcher().get_nifty_15min(n_bars=5)
                    if not df15t.empty and len(df15t) >= 3:
                        trend_up = float(df15t["close"].iloc[-1]) > float(df15t["close"].iloc[-3])
                        signal_up = (ml_signal == 0)  # CALL
                        if trend_up != signal_up:
                            logger.info(
                                f"Cycle #{cycle}: VIX {regime.name} TREND-ONLY filter: "
                                f"ML={ml_direction} vs 15m trend={'UP' if trend_up else 'DOWN'} → SKIP"
                            )
                            _shadow_skip("vix_trend_only")
                            with self._lock:
                                self._last_recommendation = {
                                    "cycle": cycle, "ml_signal": ml_direction,
                                    "action": "WAIT", "confidence": 0,
                                    "reason": f"VIX {regime.name}: signal against 15m trend",
                                    "time": datetime.now().isoformat(),
                                }
                            return
                except Exception as e:
                    logger.debug(f"Trend-only filter check failed: {e}")

        # Step 4: ML triggers trade directly — Claude removed from live path.
        # Calling Claude (2-8s latency) in a 5-min cycle ruins options pricing.
        # Claude now runs ASYNC in PostTradeAnalyzer (post_trade_analyzer.py)
        # for retrospective journaling and EOD coaching — zero live latency.
        logger.info(f"Cycle #{cycle}: ML={ml_direction} -> direct execution (no LLM latency)")
        # Heartbeat update: ML processed successfully
        try:
            from core.deadman_switch import write_heartbeat
            write_heartbeat(cycle=cycle, status=f"ml_{ml_direction}")
        except Exception:
            pass
        # 2026-04-30 FIX: Use ml_conf (directional probability) NOT raw proba.max().
        # ml_engine.predict_precomputed returns conf = call/(call+put) so the
        # 3-class SKIP doesn't bleed probability mass from CALL/PUT.
        # Without this fix, today's CALL=0.566 PUT=0.430 reports 56% (loses
        # to 60% threshold). With it, reports 90% (passes easily).
        recommendation = {
            "action":      "BUY",
            "option_type": "CE" if ml_signal == 0 else "PE",
            "strike_mode": "ATM",
            "confidence":  int(round(ml_conf * 100)),
            "analysis":    (
                f"ML direct: {ml_direction} "
                f"dir_conf={ml_conf:.1%} raw_p={max(ml_proba[0],ml_proba[1]):.1%}"
            ),
            "market_bias": "BULLISH" if ml_signal == 0 else "BEARISH",
            "_market_data": {"spot": spot},
        }

        with self._lock:
            self._last_recommendation = {
                "cycle": cycle,
                "ml_signal": ml_direction,
                "time": datetime.now().isoformat(),
                **recommendation,
            }

        action = recommendation.get("action", "WAIT")
        confidence = recommendation.get("confidence", 0)
        option_type = recommendation.get("option_type", "")
        strike_mode = recommendation.get("strike_mode", "ATM")
        analysis = recommendation.get("analysis", "")
        bias = recommendation.get("market_bias", "")

        # Apply VIX regime adjustments to confidence and strike
        if regime:
            original_conf = confidence
            confidence = max(0, min(100, confidence + regime.confidence_adj))
            if regime.preferred_strike != "ATM" and strike_mode == "ATM":
                strike_mode = regime.preferred_strike
            if original_conf != confidence:
                logger.info(
                    f"Cycle #{cycle}: VIX {regime.name} adj: "
                    f"conf {original_conf}→{confidence} strike→{strike_mode}"
                )

        # V10 #5 — Delta-based strike override (theta/gamma-aware)
        # After 14:00 OR on expiry day → switch to ITM1 (delta ≈0.6, less theta exposure).
        # In high-VIX regimes, ITM is also safer (gamma-driven swings hurt OTM more).
        if getattr(self.config, "use_delta_strike_override", True):
            try:
                _now = datetime.now()
                _is_late = _now.hour >= self.config.intraday_itm_after_hour
                _is_expiry = False
                try:
                    from core.expiry_utils import is_expiry_day
                    _is_expiry = is_expiry_day()
                except Exception:
                    pass
                if (_is_late or _is_expiry) and strike_mode in ("ATM", "OTM1", "OTM2", "OTM3"):
                    old_strike = strike_mode
                    strike_mode = "ITM1"
                    logger.info(
                        f"Cycle #{cycle}: DELTA STRIKE OVERRIDE "
                        f"({'EXPIRY' if _is_expiry else f'>={self.config.intraday_itm_after_hour}:00'}): "
                        f"{old_strike}→{strike_mode} (lower theta, higher delta)"
                    )
            except Exception as _e:
                logger.debug(f"Delta strike override failed: {_e}")

        # V10 #7 — Ban OTM2/OTM3 after 13:30 (illiquid, wide spread, theta-toxic)
        try:
            _now = datetime.now()
            _ban_h = self.config.block_otm2_after_hour
            _ban_m = self.config.block_otm2_after_min
            past_ban = (_now.hour > _ban_h or
                        (_now.hour == _ban_h and _now.minute >= _ban_m))
            if past_ban and strike_mode in ("OTM2", "OTM3"):
                old = strike_mode
                strike_mode = "OTM1"
                logger.info(
                    f"Cycle #{cycle}: OTM2/3 BAN past {_ban_h}:{_ban_m:02d} — "
                    f"{old}→{strike_mode} (illiquid + theta toxic late-day)"
                )
        except Exception:
            pass

        logger.info(
            f"Cycle #{cycle}: ML={ml_direction} -> Final={action} {option_type} "
            f"conf={confidence}% bias={bias}"
        )

        # Step 5: WAIT if low confidence or vetoed
        if action == "WAIT":
            logger.info(f"Cycle #{cycle}: WAIT. {analysis[:80]}")
            return

        # V9.3: Raised ML-only threshold 45→60%. Trading at 45% was coin-flip
        # territory — April 2 had 17 wrong PE trades all at 45-55% confidence.
        # 60% matches Claude-confirmed threshold. Better to miss trades than
        # take bad ones. For 94%+ accuracy, high conviction only.
        #
        # 2026-06-11 ASYMMETRIC GATE: live CALL WR=22.2% vs PUT near backtest.
        # The shared 70% bar over-blocked PUTs (Jun 10: market weakened all
        # afternoon, zero trades) while still admitting losing CALLs.
        # CALL 70→72 (tighten the weak side), PUT 70→58 (loosen the side
        # tracking backtest). Pairs with CONFIDENCE_CALL/PUT in ml_engine.py.
        if self.config.ml_only_mode:
            effective_min_conf = 72 if option_type == "CE" else 58
        else:
            effective_min_conf = self.config.min_confidence

        # V9.4 (2026-04-16): MOMENTUM OVERRIDE — lower threshold to 55% when
        # price structure strongly confirms the ML direction. Fixes the April
        # 16 miss where Nifty dropped 250pts below VWAP with 4 consecutive
        # bearish 5m closes but every PUT signal sat at 57-59% conf.
        #
        # Trigger requires BOTH:
        #   1. Price > 40pts on the signal side of VWAP (pa_vwap_distance)
        #   2. >=3 same-direction 5m closes in a row (pa_sequential_closes)
        # These features come from the Feature Engineering Directive upstream.
        try:
            pa_vd = float(ml_indicators.get("pa_vwap_distance", 0.0) or 0.0)
            pa_sc = int(ml_indicators.get("pa_sequential_closes", 0) or 0)
            momentum_trigger = False
            if ml_signal == 1 and pa_vd <= -40 and pa_sc <= -3:
                momentum_trigger = "PUT trend-confirm"
            elif ml_signal == 0 and pa_vd >= +40 and pa_sc >= +3:
                momentum_trigger = "CALL trend-confirm"
            if momentum_trigger and effective_min_conf > 55:
                logger.info(
                    f"Cycle #{cycle}: MOMENTUM OVERRIDE ({momentum_trigger}): "
                    f"VWAP_dist={pa_vd:+.0f}pts seq_closes={pa_sc} "
                    f"→ threshold {effective_min_conf}%→55%"
                )
                effective_min_conf = 55
        except Exception as _e:
            logger.debug(f"Momentum override check failed: {_e}")

        # ── 60m TREND CONTINUATION THRESHOLD ────────────────────────────────
        # When the 60-minute chart confirms a sustained trend (2+ consecutive
        # bearish 60m bars + steep negative 3h slope + spot well below VWAP),
        # lower the PUT threshold from 70% to 55%.
        #
        # This catches afternoon trend days like Jun 5 2026 where ML had low
        # PUT confidence (0.05-0.14) due to RSI oversold on 5m, but the
        # hourly structure clearly confirmed a downtrend.
        #
        # After retraining with the new 60m features (tf60_consec_bear etc.),
        # the model should generate higher PUT confidence natively. This gate
        # acts as a pre-retrain helper and post-retrain safety net.
        try:
            _tf60_cb   = float(ml_indicators.get("tf60_consec_bear", 0) or 0)
            _tf60_sl   = float(ml_indicators.get("tf60_trend_slope_3h", 0) or 0)
            _tf60_3hh  = float(ml_indicators.get("tf60_dist_from_3h_high", 0) or 0)
            _pa_vd     = float(ml_indicators.get("pa_vwap_distance", 0) or 0)
            _trend_put = (
                ml_signal == 1               # ML says PUT (any confidence)
                and _tf60_cb  >= 0.33        # 2+ consecutive bearish 60m bars
                and _tf60_sl  < -0.5         # steep negative 3h slope
                and _tf60_3hh < -0.004       # >0.4% below 3h high
                and _pa_vd    <= -60         # spot well below VWAP
            )
            _trend_call = (
                ml_signal == 0
                and float(ml_indicators.get("tf60_consec_bull", 0) or 0) >= 0.33
                and _tf60_sl  > 0.5
                and float(ml_indicators.get("tf60_dist_from_3h_low", 0) or 0) > 0.004
                and _pa_vd    >= 60
            )
            if (_trend_put or _trend_call) and effective_min_conf > 55:
                _dir = "PUT" if _trend_put else "CALL"
                logger.warning(
                    f"Cycle #{cycle}: 60M TREND CONTINUATION — {_dir} confirmed "
                    f"(consec={'bear' if _trend_put else 'bull'}={_tf60_cb:.2f} "
                    f"slope={_tf60_sl:.2f} 3h_dist={_tf60_3hh:.3f} "
                    f"VWAP={_pa_vd:+.0f}pts) → min_conf {effective_min_conf}%→55%"
                )
                effective_min_conf = 55
        except Exception as _tc_e:
            logger.debug(f"60m trend continuation check failed: {_tc_e}")

        # 2026-04-27: EARLY REVERSAL OVERRIDE — lower threshold to 50% when
        # we have a high-quality reversal setup. RSI-extreme + day-extreme +
        # reversal candle is a textbook setup that ML often scores 50-60%.
        # The 3 conditions for early_reversal already act as a filter —
        # confidence threshold can be lower.
        if ml_indicators.get("early_reversal") and effective_min_conf > 50:
            logger.warning(
                f"Cycle #{cycle}: EARLY REVERSAL boost — "
                f"min_conf {effective_min_conf}%→50% (RSI extreme + day-extreme + reversal candle)"
            )
            effective_min_conf = 50

        # 2026-04-27: STRUCTURE PENALTY (replaces old binary block).
        # If trade is counter-structure, raise the bar by 12% — high-quality
        # reversal trades will still pass; weak counter-trend chases won't.
        struct_pen = int(ml_indicators.get("structure_penalty", 0) or 0)
        # 2026-06-11: OI-buildup contradiction penalty (was a hard block)
        struct_pen += int(ml_indicators.get("oi_penalty", 0) or 0)
        # 2026-08-06: PCR-alignment boost (negative -- reduces the bar).
        # Opt-in, see PilotConfig.pcr_alignment_boost_enabled.
        struct_pen += int(ml_indicators.get("pcr_boost", 0) or 0)
        if struct_pen:
            old = effective_min_conf
            # Floor at 50 (matches the floor used elsewhere in this function,
            # e.g. the regime-alignment branch) so a PCR boost can never push
            # the bar below the same safety floor every other lowering path respects.
            effective_min_conf = min(85, max(50, effective_min_conf + struct_pen))
            logger.info(
                f"Cycle #{cycle}: STRUCTURE PENALTY {struct_pen:+d}% → "
                f"min_conf {old}%→{effective_min_conf}%"
            )

        # ══════════════════════════════════════════════════════════════
        # 2026-04-27 EXPERT FIX: REGIME-AWARE THRESHOLD
        # ══════════════════════════════════════════════════════════════
        # Carver "Systematic Trading": "Different regimes have different
        # base rates. A signal in trending market ≠ same signal in chop."
        #
        # Adjust effective_min_conf based on classified regime:
        #   - REVERSAL_UP/DOWN matching ML direction → 50% floor
        #   - TREND_UP/DOWN matching ML direction → 55% floor
        #   - Counter-regime trade → +10% penalty
        #   - CHOP / OPENING → use regime's own confidence_floor (70-85%)
        try:
            from core.regime_engine import get_regime_engine
            from core.tv_fetcher import get_tv_fetcher as _tv_for_regime
            df5_reg = _tv_for_regime().get_nifty_5min(n_bars=40)
            regime = get_regime_engine().classify(
                df5_reg, spot, self._session_high, self._session_low
            )
            ml_indicators["regime"] = regime.name
            ml_indicators["regime_confidence"] = round(regime.confidence, 2)
            logger.info(
                f"Cycle #{cycle}: REGIME={regime.name} "
                f"conf={regime.confidence:.2f} R²={regime.trend_strength:.2f} "
                f"reversal_pot={regime.reversal_potential:.2f} "
                f"favors={regime.favored_direction} "
                f"floor={regime.confidence_floor}% | {regime.notes}"
            )

            # Apply regime floor (use whichever is HIGHER — preserve existing
            # gates like VIX expansion that may have set effective_min_conf high).
            old_min = effective_min_conf
            ml_dir  = "CALL" if option_type == "CE" else "PUT"

            # EARLY REVERSAL EXCEPTION: when the trade is a textbook
            # oversold/overbought V-reversal (RSI extreme + demand/supply zone
            # + near session extreme), don't let the regime classification
            # raise the confidence floor.  The RSI extreme IS the regime
            # confirmation — the CHOP/NEITHER regime is an artefact of
            # the market structure analysis looking at the broader chop,
            # not the reversal context at the session extreme.
            #
            # Without this: early_reversal lowers floor to 50, regime
            # immediately raises it back to 65 (CHOP floor), blocking
            # valid V-recovery trades like the 10:55 CALL on Jun 5.
            if ml_indicators.get("early_reversal"):
                # Keep the floor set by early_reversal (50%) — don't escalate.
                logger.info(
                    f"Cycle #{cycle}: EARLY REVERSAL — regime floor escalation "
                    f"bypassed (floor stays at {effective_min_conf}%)"
                )
            elif regime.favored_direction == ml_dir:
                # Regime is aligned — use the lower of regime floor or current
                effective_min_conf = max(50, min(effective_min_conf, regime.confidence_floor))
                if effective_min_conf < old_min:
                    logger.warning(
                        f"Cycle #{cycle}: REGIME ALIGNED ({regime.name} favors {ml_dir}) "
                        f"→ min_conf {old_min}%→{effective_min_conf}%"
                    )
            elif regime.favored_direction in ("CALL", "PUT") and regime.favored_direction != ml_dir:
                # Counter-regime — raise the bar.
                # 2026-06-11: +10/floor-70 → +5/floor-65. Jun 11: REVERSAL_DOWN
                # was wrong 3/3 times during the V-recovery (called at 10:51,
                # 11:15, 11:21 — market rallied 130pts after each) and its +10
                # blocked a 79%-conf CALL winner at 10:51. The regime engine's
                # low-R² V-shape blindspot doesn't deserve a 10pp veto-weight.
                effective_min_conf = min(85, max(effective_min_conf + 5, 65))
                logger.info(
                    f"Cycle #{cycle}: COUNTER-REGIME ({regime.name} favors "
                    f"{regime.favored_direction}, ML wants {ml_dir}) "
                    f"→ min_conf {old_min}%→{effective_min_conf}%"
                )
            else:
                # NEITHER (chop/opening/unknown) — use regime's strict floor
                # 2026-06-11: PUT-side CHOP floor capped at 65 (lunch-chop is
                # 75). PUT signals track backtest WR; the 75 floor was blocking
                # valid afternoon PUTs (Jun 10). CALL keeps the full floor.
                _floor = regime.confidence_floor
                if option_type == "PE" and regime.name == "CHOP":
                    _floor = min(_floor, 65)
                effective_min_conf = max(effective_min_conf, _floor)
                if effective_min_conf > old_min:
                    logger.info(
                        f"Cycle #{cycle}: REGIME {regime.name} (no edge) "
                        f"→ min_conf {old_min}%→{effective_min_conf}%"
                    )
        except Exception as _e:
            logger.debug(f"Regime engine unavailable (fail-open): {_e}")

        # ── PERSISTENT VWAP BIAS BLOCK ─────────────────────────────────────
        # If spot has been >50pts below (or above) the futures VWAP for 3+
        # consecutive cycles (~15 min), the session has an established
        # structural bias.  Buying against that bias on a CHoCH micro-signal
        # is a low-conviction fade — block it.
        #
        # Jun 1 post-mortem: spot was 80–110pts BELOW VWAP for cycles 2–6
        # (09:20–09:40). The CHoCH at 09:40 looked like a local reversal but
        # the persistent VWAP deficit confirmed the broader structure was DOWN.
        # This block would have prevented the SL-hitting CE trade at 09:40.
        if self.config.enable_v10_signal_patches:
            if ml_dir == "CALL" and self._consec_vwap_below >= 3:
                if ml_indicators.get("early_reversal"):
                    # RSI extreme + demand zone at session low — this IS the
                    # V-recovery we want to catch. VWAP is far above because
                    # the market just crashed; spot being below VWAP is the
                    # setup, not a reason to skip. Allow through.
                    logger.info(
                        f"Cycle #{cycle}: VWAP BIAS bypassed (early_reversal) — "
                        f"oversold V-recovery at session low overrides VWAP deficit"
                    )
                else:
                    logger.info(
                        f"Cycle #{cycle}: ⛔ VWAP BIAS BLOCK: spot BELOW futures VWAP "
                        f"for {self._consec_vwap_below} consecutive cycles "
                        f"(last gap={self._last_spot_vs_vwap:+.0f}pts) — "
                        f"persistent bearish structure, blocking CALL → SKIP"
                    )
                    _shadow_skip(f"vwap_bias:below_{self._consec_vwap_below}c")
                    return
            if ml_dir == "PUT" and self._consec_vwap_above >= 3:
                if ml_indicators.get("early_reversal"):
                    logger.info(
                        f"Cycle #{cycle}: VWAP BIAS bypassed (early_reversal) — "
                        f"overbought V-reversal at session high overrides VWAP surplus"
                    )
                else:
                    logger.info(
                        f"Cycle #{cycle}: ⛔ VWAP BIAS BLOCK: spot ABOVE futures VWAP "
                        f"for {self._consec_vwap_above} consecutive cycles "
                        f"(last gap={self._last_spot_vs_vwap:+.0f}pts) — "
                        f"persistent bullish structure, blocking PUT → SKIP"
                    )
                    _shadow_skip(f"vwap_bias:above_{self._consec_vwap_above}c")
                    return

        # 2026-05-05: HARD BLOCK 09:15-10:00 (45 min, was 10 min).
        # Reason: 09:55 entries lost ₹7,800 across May 4-5 — bot kept entering
        # at opening-range extremes (day high/low not yet established).
        # By 10:00, 9 bars of session data exist → trap detector + regime
        # engine have proper inputs to filter bad setups.
        now_time = datetime.now()
        session_minutes = (now_time.hour - 9) * 60 + now_time.minute - 15
        hard_block_min = getattr(self.config, "morning_hard_block_min", 45)
        if 0 <= session_minutes < hard_block_min:
            logger.info(
                f"Cycle #{cycle}: MORNING HARD BLOCK (first {hard_block_min} min): "
                f"{action} {option_type} conf={confidence}% → SKIP "
                f"(opening trap window — wait until 10:00 IST)"
            )
            _shadow_skip("morning_hard_block", conf=confidence)
            return

        # 09:25-09:45 = secondary trap zone — require 75%.
        if hard_block_min <= session_minutes < 30:
            morning_min_conf = 75
            if confidence < morning_min_conf:
                logger.info(
                    f"Cycle #{cycle}: MORNING TRAP GUARD (09:25-09:45): "
                    f"{action} {option_type} conf {confidence}% < {morning_min_conf}% → SKIP"
                )
                _shadow_skip("morning_trap_guard", conf=confidence)
                return
            # BUG FIX 2026-06-01: was `= morning_min_conf` which silently
            # LOWERED effective_min_conf back to 75 after counter-regime had
            # already raised it to 80. Use max() so the stricter threshold wins.
            effective_min_conf = max(effective_min_conf, morning_min_conf)
            logger.info(f"Cycle #{cycle}: Morning session — raised min conf to {effective_min_conf}%")

        # ══════════════════════════════════════════════════════════════
        # ADVISORY GATES (2026-04-24 fixes) — NEWS BIAS + TRAP DETECTOR
        # Both are fail-safe: any exception → pass (never block on bug).
        # ══════════════════════════════════════════════════════════════
        try:
            from core.trap_detector import get_trap_detector
            trap = get_trap_detector().is_trap(
                option_type=option_type,
                spot=spot,
                ml_indicators=ml_indicators,
            )
            if trap.is_trap:
                logger.warning(
                    f"Cycle #{cycle}: TRAP GATE [{trap.reason}] — "
                    f"{action} {option_type} conf={confidence}% → SKIP. {trap.detail}"
                )
                _shadow_skip(f"trap_gate:{trap.reason}", conf=confidence)
                return
        except Exception as _e:
            logger.debug(f"Trap detector unavailable (fail-open): {_e}")

        try:
            import os
            if os.getenv("NEWS_AGENT_ENABLED", "true").lower() == "true":
                from core.news_agent import get_news_agent
                bias = get_news_agent().current_bias()
                nb = (bias or {}).get("bias", "neutral")
                nconf = float((bias or {}).get("confidence", 0.0) or 0.0)
                hot = (bias or {}).get("hot_event")
                # Only block when news is confident AND directly opposes the trade
                if nconf >= 0.55:
                    if option_type == "CE" and nb == "bearish":
                        logger.warning(
                            f"Cycle #{cycle}: NEWS GATE — CALL vs bearish news "
                            f"(conf={nconf:.2f}) → SKIP. hot={hot}"
                        )
                        _shadow_skip("news_gate:bearish_vs_call", conf=confidence)
                        return
                    if option_type == "PE" and nb == "bullish":
                        logger.warning(
                            f"Cycle #{cycle}: NEWS GATE — PUT vs bullish news "
                            f"(conf={nconf:.2f}) → SKIP. hot={hot}"
                        )
                        _shadow_skip("news_gate:bullish_vs_put", conf=confidence)
                        return
                    logger.info(
                        f"Cycle #{cycle}: NEWS OK — {option_type} aligns with "
                        f"news bias={nb} conf={nconf:.2f}"
                    )
        except Exception as _e:
            logger.debug(f"News agent unavailable (fail-open): {_e}")

        # V10 #6 — VIX-expansion confidence floor: VIX>28 → require 65%+
        try:
            _vix_now = float(getattr(self, "_current_vix", 0) or 0)
            if _vix_now >= self.config.vix_high_threshold:
                vix_floor = self.config.vix_high_min_conf
                if confidence < vix_floor:
                    logger.info(
                        f"Cycle #{cycle}: VIX EXPANSION ({_vix_now:.1f}>={self.config.vix_high_threshold}): "
                        f"{action} conf {confidence}% < {vix_floor}% → SKIP "
                        f"(directional bets unreliable in 2-way vol)"
                    )
                    _shadow_skip(f"vix_expansion:vix={_vix_now:.0f}", conf=confidence)
                    return
                effective_min_conf = max(effective_min_conf, vix_floor)
        except Exception:
            pass

        # ── Apply pre-market brief overrides (capped to avoid impossibility)
        # 2026-04-27 fix: VIX adjustment already raises bar 3-5%. Compounding
        # the brief boost on top makes ML threshold unreachable on event_days.
        # Cap brief boost to +5 max, and ignore it when VIX gate already fired.
        try:
            raw_boost = getattr(self, "_brief_conf_boost", 0)
            boost = max(-10, min(5, raw_boost))   # cap at +5
            # If VIX has already raised threshold above base 60, skip the boost
            base_min = 60 if self.config.ml_only_mode else self.config.min_confidence
            if effective_min_conf > base_min:
                if boost > 0:
                    logger.debug(
                        f"Cycle #{cycle}: BRIEF boost {raw_boost:+d} skipped — "
                        f"VIX/morning gate already at {effective_min_conf}%"
                    )
                boost = 0
            if boost:
                effective_min_conf = max(50, min(85, effective_min_conf + boost))
                logger.info(
                    f"Cycle #{cycle}: BRIEF boost {boost:+d} → min_conf {effective_min_conf}%"
                )
            avoid_otm23 = getattr(self, "_brief_avoid_otm23", False)
            if avoid_otm23 and strike_mode in ("OTM2", "OTM3"):
                old = strike_mode
                strike_mode = "OTM1"
                logger.info(
                    f"Cycle #{cycle}: BRIEF avoid_OTM23 — {old}→OTM1"
                )
            brief_bias = getattr(self, "_brief_bias", "neutral")
            # Reduced from 70% → 65% to keep gate effective without being impossible
            if brief_bias == "bearish" and option_type == "CE":
                effective_min_conf = max(effective_min_conf, 65)
            elif brief_bias == "bullish" and option_type == "PE":
                effective_min_conf = max(effective_min_conf, 65)
        except Exception:
            pass

        # ── FUTURES SELECTIVITY FLOOR (opt-in, disabled by default) ─────────
        # Backtest finding (8-fold walk-forward, futures friction): filtering
        # to only the model's top ~30%-confidence signals improved mean PF
        # from 0.982 to 1.021 (6/8 folds better, not just one lucky window).
        # That backtest measured confidence on the raw model proba.max()
        # scale, NOT the `confidence` variable checked below (which goes
        # through the call/(call+put) blend + 90% cap applied earlier in
        # this function) -- those are different scales, so the backtest's
        # ~47.6% (70th percentile) number does NOT plug in here directly.
        # Defaults to 0 (disabled, no behavior change) until calibrated
        # against real confidence values from actual futures paper-mode
        # runs -- set FUTURES_MIN_CONFIDENCE_PCT once you have that data.
        if (self.config.execution_mode == "futures"
                and self.config.futures_min_confidence_pct > 0):
            effective_min_conf = max(effective_min_conf, self.config.futures_min_confidence_pct)

        if confidence < effective_min_conf:
            logger.info(
                f"Cycle #{cycle}: {action} {option_type} but conf {confidence}% < "
                f"{effective_min_conf}% ({'ML-only' if self.config.ml_only_mode else 'Claude'})"
            )
            # Log to shadow trader for post-hoc analysis (was this gate right?)
            try:
                from core.shadow_logger import get_shadow_logger
                get_shadow_logger().log_skip(
                    cycle=cycle,
                    signal=("CALL" if option_type == "CE" else "PUT"),
                    conf_pct=confidence,
                    reason=f"low_conf<{effective_min_conf}%",
                    spot=spot,
                    ml_proba=list(ml_proba) if hasattr(ml_proba, "__iter__") else [],
                    regime=str(ml_indicators.get("regime", "")),
                    extra={"effective_min_conf": effective_min_conf},
                )
            except Exception as _e:
                logger.debug(f"Shadow logger failed: {_e}")
            return

        # ══════════════════════════════════════════════════════════════
        # STEP 5.5: IV GATE — block entry when options are priced too rich
        # ══════════════════════════════════════════════════════════════
        # India VIX only tells you *historical* vol expectation.
        # When front-week ATM IV >> VIX you are paying event-premium
        # that crushes even if spot moves in your direction.
        #
        # Three independent triggers — any one fires → SKIP:
        #   A. iv_vs_vix > 6pts: options >6 pts richer than VIX
        #   B. iv_rank > 80th pct AND iv_vs_vix > 3pts: historically expensive
        #   C. IV rising >10% intraday (active crush-risk: event / news)
        #
        # Recovery mode bypasses this — momentum outweighs IV decay.
        try:
            iv = self._iv_snap
            if iv and iv.available and not recovery_mode:
                iv_block      = False
                iv_block_why  = ""

                if iv.iv_vs_vix > self.config.iv_vs_vix_block_threshold:
                    iv_block     = True
                    iv_block_why = (
                        f"IV too rich vs VIX: ATM_IV={iv.atm_iv:.1f}% "
                        f"VIX={iv.vix_level:.1f}% spread={iv.iv_vs_vix:+.1f}pts "
                        f"(threshold={self.config.iv_vs_vix_block_threshold:.0f}pts)"
                    )

                if (not iv_block
                        and iv.iv_rank > self.config.iv_rank_block_threshold
                        and iv.iv_vs_vix > 3.0):
                    iv_block     = True
                    iv_block_why = (
                        f"IV historically expensive: rank={iv.iv_rank:.0f}th pct "
                        f"AND iv_vs_vix={iv.iv_vs_vix:+.1f}pts"
                    )

                if (not iv_block
                        and self.config.iv_crush_block
                        and iv.is_crush_risk):
                    iv_block     = True
                    iv_block_why = (
                        f"IV expansion risk: IV up {iv.iv_change_pct:+.1f}% from session open "
                        f"(potential crush after move)"
                    )

                if iv_block:
                    logger.warning(
                        f"Cycle #{cycle}: IV GATE BLOCK — {iv_block_why} → SKIP"
                    )
                    ml_indicators["iv_block"]    = True
                    ml_indicators["iv_block_why"] = iv_block_why
                    with self._lock:
                        self._last_recommendation = {
                            "cycle": cycle, "ml_signal": ml_direction,
                            "action": "WAIT", "confidence": 0,
                            "reason": f"IV gate: {iv_block_why}",
                            "time": datetime.now().isoformat(),
                        }
                    return

                # Inject IV metrics into ml_indicators for journal & logging
                ml_indicators["atm_iv"]            = iv.atm_iv
                ml_indicators["iv_skew"]           = iv.iv_skew
                ml_indicators["iv_vs_vix"]         = iv.iv_vs_vix
                ml_indicators["iv_rank"]           = iv.iv_rank
                ml_indicators["iv_change_pct"]     = iv.iv_change_pct
                ml_indicators["put_call_iv_ratio"] = iv.put_call_iv_ratio
        except Exception as _iv_e:
            logger.debug(f"IV gate check failed (non-fatal): {_iv_e}")

        # Step 6: Check BROKER positions (survives restarts — not in-memory)
        # V9.3: This is critical. On April 2 the bot restarted 3 times,
        # each restart cleared _live_position, and it stacked 19 trades.
        # Now we ALWAYS check the broker's position book before executing.
        open_positions = []
        try:
            pos_data = self.trader.get_positions_summary()
            if isinstance(pos_data, dict) and pos_data.get("data"):
                plist = pos_data["data"]
                if isinstance(plist, list):
                    open_positions = [p for p in plist if int(p.get("netqty", p.get("quantity", 0))) != 0]
        except Exception:
            pass

        # Block if ANY Nifty option position is open (CE or PE)
        for p in open_positions:
            sym = str(p.get("symbol", "")).upper()
            qty = int(p.get("netqty", p.get("quantity", 0)))
            if "NIFTY" in sym and qty != 0:
                logger.info(
                    f"Cycle #{cycle}: Broker position already open: {sym} qty={qty} → SKIP"
                )
                return

        # Also check in-memory state (belt + suspenders)
        with self._lock:
            if self._live_position is not None and self._live_position.state != "CLOSED":
                logger.info(
                    f"Cycle #{cycle}: In-memory position {self._live_position.state} → SKIP"
                )
                return

        # ══════════════════════════════════════════════════════════════
        # Step 7: DELTA-ADJUSTED DYNAMIC SL/TP (SMC v2)
        # ══════════════════════════════════════════════════════════════
        # OLD: ATR-based spot SL only. Option premium SL was disconnected.
        # NEW: Compute spot SL, then translate to option premium SL using
        #      Delta approximation. Accounts for:
        #   - ATM delta ≈ 0.50 (default)
        #   - OTM1 delta ≈ 0.40
        #   - ITM1 delta ≈ 0.60
        #   - Gamma acceleration on sharp moves
        #   - VIX regime scaling (unchanged)
        #
        # Formula:
        #   Premium_SL = Spot_SL × Delta × (1 + Gamma_Adj)
        #   Premium_TP = Spot_TP × Delta × (1 - Theta_Haircut)
        #
        #   Gamma_Adj: +10% for ATM (gamma highest), +5% for OTM/ITM
        #   Theta_Haircut: -5% for intraday (theta drag on TP)
        #
        direction = "CALL" if option_type == "CE" else "PUT"
        sl_pts, tp_pts = self._get_dynamic_sl_tp(direction)

        # Use Claude's suggested SL/TP if available and reasonable
        # (skipped under the frozen validated exit -- "no additional filters")
        if not self.config.use_frozen_atr_exit:
            claude_sl = recommendation.get("stop_loss_points", 0)
            claude_tp = recommendation.get("target_points", 0)
            if claude_sl and self.config.min_sl_points <= claude_sl <= self.config.max_sl_points:
                sl_pts = claude_sl
            if claude_tp and self.config.min_tp_points <= claude_tp <= self.config.max_tp_points:
                tp_pts = claude_tp

        # 2026-05-15 TIER-2 #6: Theta-aware TP adjustment.
        # Late-day entries have less time for TP to be hit AND more theta decay.
        # Scale TP down as session progresses (morning=full, afternoon=tighter).
        # (skipped under the frozen validated exit -- "no dynamic TP/SL")
        if not self.config.use_frozen_atr_exit:
            try:
                now_for_theta = datetime.now()
                session_min = (now_for_theta.hour - 9) * 60 + now_for_theta.minute - 15
                if session_min >= 240:        # after 13:15 (last 2h)
                    old_tp = tp_pts
                    tp_pts = max(self.config.min_tp_points, round(tp_pts * 0.70, 1))
                    logger.info(
                        f"Cycle #{cycle}: THETA-TIME adj (after 13:15): "
                        f"TP {old_tp:.0f}->{tp_pts:.0f} (-30%)"
                    )
                elif session_min >= 180:      # after 12:15 (mid-day)
                    old_tp = tp_pts
                    tp_pts = max(self.config.min_tp_points, round(tp_pts * 0.85, 1))
                    logger.info(
                        f"Cycle #{cycle}: THETA-TIME adj (12:15-13:15): "
                        f"TP {old_tp:.0f}->{tp_pts:.0f} (-15%)"
                    )
            except Exception as _e:
                logger.debug(f"Theta-time adj failed: {_e}")

        # ══════════════════════════════════════════════════════════════
        # EXPIRY DAY ADJUSTMENTS — Gamma risk + Theta decay
        # ══════════════════════════════════════════════════════════════
        # On expiry day: gamma is extreme (SL gets hit faster), theta
        # crushes premium. Widen SL by 20% to absorb gamma spikes,
        # tighten TP by 20% to lock profit before theta kills it.
        # Pre-expiry (DTE=1): moderate 10% adjustment.
        try:
            from core.expiry_utils import get_dte, is_expiry_day, is_pre_expiry
            dte = get_dte()
            # SL/TP widening skipped under the frozen validated exit -- "no
            # additional filters" / "no regime adjustments". dte/expiry_day
            # are still recorded below (informational journal fields only).
            if self.config.use_frozen_atr_exit:
                pass
            elif is_expiry_day():
                old_sl, old_tp = sl_pts, tp_pts
                sl_pts = round(sl_pts * 1.20, 1)  # wider SL for gamma
                tp_pts = round(tp_pts * 0.80, 1)  # faster TP for theta
                sl_pts = min(sl_pts, self.config.max_sl_points)
                tp_pts = max(tp_pts, self.config.min_tp_points)
                logger.info(
                    f"Cycle #{cycle}: EXPIRY DAY adj: SL {old_sl:.0f}→{sl_pts:.0f} (+20%) "
                    f"TP {old_tp:.0f}→{tp_pts:.0f} (-20%) [gamma/theta protection]"
                )
            elif is_pre_expiry():
                old_sl, old_tp = sl_pts, tp_pts
                sl_pts = round(sl_pts * 1.10, 1)
                tp_pts = round(tp_pts * 0.90, 1)
                sl_pts = min(sl_pts, self.config.max_sl_points)
                tp_pts = max(tp_pts, self.config.min_tp_points)
                logger.info(
                    f"Cycle #{cycle}: PRE-EXPIRY adj: SL {old_sl:.0f}→{sl_pts:.0f} (+10%) "
                    f"TP {old_tp:.0f}→{tp_pts:.0f} (-10%)"
                )
            ml_indicators["dte"] = dte
            ml_indicators["expiry_day"] = is_expiry_day()
        except Exception as e:
            logger.debug(f"Expiry adjustment skipped: {e}")

        # ══════════════════════════════════════════════════════════════
        # SMC FEATURES — inject into ml_indicators for live context log
        # Full computation happens in ml_engine feature pipeline.
        # Here we pull the latest snapshot for logging + journal.
        # ══════════════════════════════════════════════════════════════
        try:
            from core.smc_features import extract_latest_smc_features, get_smc_signal
            from core.tv_fetcher import get_tv_fetcher as _tv_f
            df_15m = _tv_f().get_ohlcv("NIFTY", "NSE", 15, n_bars=50)
            # Previous day high/low from market intel
            pdh = getattr(intel, "prev_day_high", 0.0) if intel else 0.0
            pdl = getattr(intel, "prev_day_low",  0.0) if intel else 0.0
            smc_feats = extract_latest_smc_features(df_15m, pdh=pdh, pdl=pdl)
            smc_signal = get_smc_signal(smc_feats)
            ml_indicators.update(smc_feats)
            ml_indicators["smc_signal"] = smc_signal
            logger.info(
                f"Cycle #{cycle}: SMC → {smc_signal} | "
                f"FVG bull={smc_feats.get('fvg_bull_recent',0)} "
                f"bear={smc_feats.get('fvg_bear_recent',0)} | "
                f"OB bull={smc_feats.get('ob_bull',0)} "
                f"bear={smc_feats.get('ob_bear',0)} | "
                f"Sweep={smc_feats.get('liq_sweep_recent',0)}"
            )
        except Exception as _smc_e:
            logger.debug(f"SMC features skipped: {_smc_e}")
            ml_indicators["smc_signal"] = "NEUTRAL"

        # ── SMC DIRECTION CONFLICT BLOCK ──────────────────────────────────
        # If the 15m SMC context is firmly against the trade direction, block.
        # SMC=BEARISH means bearish FVGs dominate on 15m — buying CE into that
        # structure is a low-probability fade even with a micro CHoCH signal.
        # SMC=BULLISH + PUT has the same problem in the other direction.
        #
        # Jun 1 post-mortem: SMC → BEARISH (FVG bull=0.0 bear=1.0) was logged
        # but the bot still executed BUY CE. This block closes that gap.
        _smc_sig = ml_indicators.get("smc_signal", "NEUTRAL")
        if self.config.enable_v10_signal_patches:
            if _smc_sig == "BEARISH" and direction == "CALL":
                logger.info(
                    f"Cycle #{cycle}: ⛔ SMC CONTEXT BLOCK: SMC=BEARISH conflicts "
                    f"with CALL direction "
                    f"(FVG bull={ml_indicators.get('fvg_bull_recent',0)} "
                    f"bear={ml_indicators.get('fvg_bear_recent',0)}) → SKIP"
                )
                _shadow_skip("smc_context:bearish_vs_call", conf=confidence)
                return
            if _smc_sig == "BULLISH" and direction == "PUT":
                logger.info(
                    f"Cycle #{cycle}: ⛔ SMC CONTEXT BLOCK: SMC=BULLISH conflicts "
                    f"with PUT direction "
                    f"(FVG bull={ml_indicators.get('fvg_bull_recent',0)} "
                    f"bear={ml_indicators.get('fvg_bear_recent',0)}) → SKIP"
                )
                _shadow_skip("smc_context:bullish_vs_put", conf=confidence)
                return

        # ══════════════════════════════════════════════════════════════
        # OI MAGNETIC TP — Cap TP 10pts before Max OI strike
        # ══════════════════════════════════════════════════════════════
        # Institutional option sellers defend the Max CE OI and Max PE OI
        # strikes. Price is magnetically attracted to these levels but
        # rarely pushes through. Front-run by exiting 10pts before.
        #
        # CALL: TP cannot exceed (Max_CE_OI_strike - 10) above spot
        # PUT:  TP cannot exceed (spot - Max_PE_OI_strike - 10)
        # Nifty strikes are 50pt intervals; Max OI = institutional wall.
        #
        # V9.3: Dynamic OI buffer = 0.25 × ATR_5m (adapts to volatility)
        # Entire block skipped under the frozen validated exit -- "no
        # additional filters". oi_tp_capped/magnetic_strike stay at their
        # defaults so the informational ml_indicators fields below are
        # still recorded either way.
        oi_tp_buffer = max(5.0, round(0.25 * self._current_atr, 1))
        oi_tp_capped = False
        magnetic_strike = 0.0

        if self.config.use_frozen_atr_exit:
            pass
        else:
            if intel and direction == "CALL" and intel.oi_resistance > 0:
                # Max CE OI strike above spot = resistance ceiling
                wall_dist = intel.oi_resistance - spot
                magnetic_tp = wall_dist - oi_tp_buffer
                if 20 < magnetic_tp < tp_pts:
                    old_tp = tp_pts
                    tp_pts = round(magnetic_tp, 1)
                    oi_tp_capped = True
                    magnetic_strike = intel.oi_resistance
                    logger.info(
                        f"Cycle #{cycle}: MAGNETIC TP: CALL TP {old_tp:.0f}→{tp_pts:.0f}pts "
                        f"(Max CE OI at {intel.oi_resistance:.0f}, buffer={oi_tp_buffer:.0f}pts [0.25×ATR])"
                    )

            if intel and direction == "PUT" and intel.oi_support > 0:
                # Max PE OI strike below spot = support floor
                wall_dist = spot - intel.oi_support
                magnetic_tp = wall_dist - oi_tp_buffer
                if 20 < magnetic_tp < tp_pts:
                    old_tp = tp_pts
                    tp_pts = round(magnetic_tp, 1)
                    oi_tp_capped = True
                    magnetic_strike = intel.oi_support
                    logger.info(
                        f"Cycle #{cycle}: MAGNETIC TP: PUT TP {old_tp:.0f}→{tp_pts:.0f}pts "
                        f"(Max PE OI at {intel.oi_support:.0f}, buffer={oi_tp_buffer:.0f}pts [0.25×ATR])"
                    )

            # Enforce minimum TP and R:R after magnetic capping
            if oi_tp_capped:
                tp_pts = max(self.config.min_tp_points, tp_pts)
                # V10: enforce 2.0 R:R floor (was 1.5 — math doesn't work for options buyers)
                min_rr = getattr(self.config, "min_rr_ratio", 2.0)
                if tp_pts < sl_pts * min_rr:
                    tp_pts = round(sl_pts * min_rr, 1)
                    logger.info(f"Cycle #{cycle}: Magnetic cap → R:R floor (1:{min_rr}) enforced → TP={tp_pts:.0f}pts")

        ml_indicators["oi_tp_capped"] = oi_tp_capped
        ml_indicators["oi_magnetic_strike"] = magnetic_strike

        # Spot-level SL/TP prices
        if direction == "CALL":
            sl_price = spot - sl_pts
            tp_price = spot + tp_pts
        else:
            sl_price = spot + sl_pts
            tp_price = spot - tp_pts

        # --- Delta-adjusted premium SL/TP ---
        if self.config.execution_mode == "futures":
            # Futures move 1:1 with spot -- no option Greeks discount, no
            # gamma/theta concept. delta=1.0 collapses the premium_sl/tp
            # formulas below to exactly sl_pts/tp_pts, which is correct:
            # futures P&L IS the spot-point move, nothing to convert.
            delta = 1.0
            gamma_adj = 0.0
            theta_haircut = 0.0
            strike_key = "FUT"
        else:
            # Determine delta — prefer live BS delta from IV snapshot over hardcoded map.
            # Live delta accounts for DTE and current volatility level (ATM delta at
            # VIX=25 is ~0.48, not 0.50; OTM1 at DTE=1 is ~0.32, not 0.40).
            _delta_map_fallback = {
                "ATM": 0.50, "OTM1": 0.40, "OTM2": 0.30, "OTM3": 0.22,
                "ITM1": 0.60, "ITM2": 0.70,
            }
            strike_key = strike_mode.upper().replace("_", "")
            iv_delta_key = f"{'CE' if direction == 'CALL' else 'PE'}_{strike_key}"
            if (self._iv_snap and self._iv_snap.available
                    and iv_delta_key in self._iv_snap.live_delta):
                delta = self._iv_snap.live_delta[iv_delta_key]
                logger.info(
                    f"Cycle #{cycle}: Live delta for {iv_delta_key} = {delta:.3f} "
                    f"(ATM_IV={self._iv_snap.atm_iv:.1f}% DTE={self._iv_snap.dte:.0f})"
                )
            else:
                delta = _delta_map_fallback.get(strike_key, 0.50)
                logger.debug(f"Cycle #{cycle}: Using fallback delta {delta} for {strike_key}")

            # Gamma adjustment: ATM has highest gamma (premium moves faster than delta implies)
            # On sharp moves, actual premium change > delta × spot_change
            gamma_adj = 0.10 if strike_key == "ATM" else 0.05

            # Theta haircut: TP takes longer to hit, theta erodes premium
            # Intraday: ~5% haircut on TP target
            now_hour = datetime.now().hour
            theta_haircut = 0.08 if now_hour >= 14 else 0.05  # worse near close

        # Premium SL = how much the option premium drops when spot hits SL
        # Premium TP = how much the option premium rises when spot hits TP
        premium_sl = sl_pts * delta * (1 + gamma_adj)
        premium_tp = tp_pts * delta * (1 - theta_haircut)

        # Store for position monitor (used in premium hard stop)
        ml_indicators["delta"] = delta
        ml_indicators["premium_sl_pts"] = round(premium_sl, 1)
        ml_indicators["premium_tp_pts"] = round(premium_tp, 1)
        ml_indicators["spot_sl_pts"] = round(sl_pts, 1)
        ml_indicators["spot_tp_pts"] = round(tp_pts, 1)

        logger.info(
            f"Cycle #{cycle}: Delta-adjusted SL/TP: "
            f"Spot SL={sl_pts:.0f}pts TP={tp_pts:.0f}pts | "
            f"Delta={delta} γ_adj={gamma_adj} θ_cut={theta_haircut} | "
            f"Premium SL=₹{premium_sl:.1f} TP=₹{premium_tp:.1f} "
            f"(R:R = 1:{premium_tp/premium_sl:.1f})"
        )

        # ══════════════════════════════════════════════════════════════
        # Step 7.5: GAMMA SQUEEZE DETECTION + KELLY POSITION SIZING
        # ══════════════════════════════════════════════════════════════
        # Gamma Squeeze: When ATM call writers are trapped and unwinding
        # (ATM CE OI dropping), delta hedging forces them to buy futures,
        # pushing price up further. This creates a high-probability
        # momentum cascade. Boost ML confidence → larger Kelly size.
        #
        # CALL squeeze: ATM CE OI dropping (unwind_pct < -2%)
        # PUT squeeze:  ATM PE OI dropping (unwind_pct < -2%)
        #
        ml_conf_raw = max(ml_proba[0], ml_proba[1])  # 0.0-1.0 scale
        gamma_squeeze = False
        squeeze_boost = 0.0

        if intel:
            # CALL signal + ATM CE OI unwinding = call writers trapped → squeeze
            if direction == "CALL" and intel.atm_ce_unwind_pct < -2.0:
                gamma_squeeze = True
                squeeze_boost = 0.10
                logger.info(
                    f"Cycle #{cycle}: 🔥 GAMMA SQUEEZE detected: ATM CE OI "
                    f"unwinding {intel.atm_ce_unwind_pct:+.1f}% "
                    f"(ATM={intel.atm_strike:.0f} CE_OI_chg={intel.atm_ce_oi_change:+,.0f}) "
                    f"→ Kelly boost +{squeeze_boost:.0%}"
                )

            # PUT signal + ATM PE OI unwinding = put writers trapped → squeeze
            if direction == "PUT" and intel.atm_pe_unwind_pct < -2.0:
                gamma_squeeze = True
                squeeze_boost = 0.10
                logger.info(
                    f"Cycle #{cycle}: 🔥 GAMMA SQUEEZE detected: ATM PE OI "
                    f"unwinding {intel.atm_pe_unwind_pct:+.1f}% "
                    f"(ATM={intel.atm_strike:.0f} PE_OI_chg={intel.atm_pe_oi_change:+,.0f}) "
                    f"→ Kelly boost +{squeeze_boost:.0%}"
                )

        # Apply squeeze boost (capped at 1.0)
        kelly_conf = min(1.0, ml_conf_raw + squeeze_boost)
        ml_indicators["gamma_squeeze"] = gamma_squeeze
        ml_indicators["kelly_conf_input"] = round(kelly_conf, 3)

        if self.config.execution_mode == "futures":
            num_lots = self._compute_futures_position_size(sl_pts, kelly_conf, spot)
        else:
            num_lots = self._compute_position_size(sl_pts, kelly_conf, delta)

        # 2026-06-11: SL-TIGHTEN FALLBACK before skipping. Jun 11 #21: the
        # day's best signal (90% conf CALL) was skipped because 1 lot lost
        # ₹2,066 vs the ₹2,000 cap — ₹66 over. If shrinking the SL by ≤15%
        # brings 1 lot under the cap, trade with the tighter stop instead.
        # Larger gaps still skip (a deeply-truncated SL is a different trade).
        if num_lots == 0 and delta > 0 and sl_pts > 0:
            try:
                import os as _os_slt
                _cap = float(_os_slt.getenv("MAX_LOSS_PER_TRADE", "2000"))
                # 0.995 margin so int() truncation in sizing can't round to 0
                fit_sl = (_cap / (delta * self.config.lot_size)) * 0.995
                if _cap > 0 and fit_sl >= sl_pts * 0.85:
                    logger.warning(
                        f"Cycle #{cycle}: MAX_LOSS SL-TIGHTEN — SL "
                        f"{sl_pts:.0f}→{fit_sl:.0f}pts to fit ₹{_cap:.0f} cap "
                        f"(1 lot was ₹{sl_pts * delta * self.config.lot_size:.0f})"
                    )
                    sl_pts = round(fit_sl, 1)
                    # Recompute everything derived from sl_pts above
                    sl_price = (spot - sl_pts) if direction == "CALL" else (spot + sl_pts)
                    premium_sl = sl_pts * delta * (1 + gamma_adj)
                    ml_indicators["premium_sl_pts"] = round(premium_sl, 1)
                    ml_indicators["spot_sl_pts"] = round(sl_pts, 1)
                    if self.config.execution_mode == "futures":
                        num_lots = self._compute_futures_position_size(sl_pts, kelly_conf, spot)
                    else:
                        num_lots = self._compute_position_size(sl_pts, kelly_conf, delta)
            except Exception as _e:
                logger.debug(f"SL-tighten fallback failed (skip stands): {_e}")

        # Phase 5 fix (2026-06-09): _compute_position_size returns 0 when
        # even 1 lot would exceed MAX_LOSS_PER_TRADE. Skip the trade.
        if num_lots == 0:
            logger.warning(
                f"Cycle #{cycle}: 🛑 MAX_LOSS_PER_TRADE exceeded for 1 lot "
                f"(sl_pts={sl_pts:.0f}, delta={delta:.2f}) — SKIP trade"
            )
            try:
                _shadow_skip(
                    f"max_loss_per_trade:1lot_exceeds_limit "
                    f"sl={sl_pts:.0f}pts delta={delta:.2f}",
                    conf=confidence,
                )
            except Exception:
                pass
            return

        trade_qty = num_lots * self.config.lot_size

        ml_indicators["position_lots"] = num_lots
        ml_indicators["position_qty"] = trade_qty

        # ── DATA FRESHNESS GATE (entries only) ─────────────────────────────
        # Refuse to OPEN a position on stale/delayed data. The model's horizon
        # is 15 min forward; entering on a 15-min-lagged yFinance bar (or a
        # >7-min-old cache bar) means trading a window that already happened.
        # Exits are unaffected — the position monitor uses ws_spot/Kotak quotes,
        # not tv_fetcher. Toggle off instantly with ENTRY_REQUIRE_FRESH_DATA=false.
        import os as _os_fresh
        if _os_fresh.getenv("ENTRY_REQUIRE_FRESH_DATA", "true").strip().lower() \
                in ("true", "1", "yes"):
            _fresh = self._entry_data_is_fresh()
            if not _fresh["ok"]:
                logger.warning(
                    f"Cycle #{cycle}: 🛑 DATA STALE GATE — {action} {option_type} "
                    f"entry BLOCKED: {_fresh['reason']}"
                )
                # Still record as a shadow skip so we capture what the model wanted
                try:
                    from core.shadow_logger import get_shadow_logger
                    get_shadow_logger().log_skip(
                        cycle=cycle,
                        signal=("CALL" if option_type == "CE" else "PUT"),
                        conf_pct=confidence,
                        reason=f"data_stale:{_fresh['source']}",
                        spot=spot,
                        ml_proba=list(ml_proba) if hasattr(ml_proba, "__iter__") else [],
                        regime=str(ml_indicators.get("regime", "")),
                    )
                except Exception:
                    pass
                return

        # Step 8: SMART EXECUTION
        logger.info(
            f"Cycle #{cycle}: EXECUTING {action} {option_type} ({strike_mode}) "
            f"conf={confidence}% SL={sl_pts:.0f}pts TP={tp_pts:.0f}pts "
            f"lots={num_lots} qty={trade_qty}"
        )

        # PAPER TRADING MODE — log signal without executing
        import os
        dry_run = os.getenv("DRY_RUN", "false").strip().lower() in ("true", "1", "yes")
        if dry_run:
            is_futures_mode = self.config.execution_mode == "futures"
            display_label = f"NIFTY FUT ({direction})" if is_futures_mode else f"NIFTY {option_type}"
            logger.info(
                f"Cycle #{cycle}: [DRY RUN] WOULD EXECUTE {action} {display_label} "
                f"({'futures' if is_futures_mode else strike_mode}) conf={confidence}% "
                f"SL={sl_pts:.0f} TP={tp_pts:.0f} lots={num_lots} spot={spot:.0f}"
            )
            self.notifier.notify_trade(
                action="DRY_RUN",
                symbol=display_label,
                side=action,
                qty=trade_qty,
                price=spot,
                order_id="PAPER",
                status="simulated",
                details=(
                    f"[PAPER] {action} {display_label} "
                    f"conf={confidence}% SL={sl_pts:.0f} TP={tp_pts:.0f} "
                    f"R:R=1:{tp_pts/sl_pts:.1f}"
                ),
            )
            self.trader.audit.log(
                action="DRY_RUN_SIGNAL",
                symbol=display_label,
                side=action,
                details=(
                    f"conf={confidence}% SL={sl_pts:.0f} TP={tp_pts:.0f} "
                    f"spot={spot:.0f} lots={num_lots}"
                ),
            )
            # Record DRY_RUN signal to trade journal
            if self._journal:
                try:
                    self._journal.record_dry_run_signal(
                        direction=direction,
                        option_type=option_type,
                        strike_mode=strike_mode,
                        spot=spot,
                        confidence=confidence,
                        sl_pts=sl_pts,
                        tp_pts=tp_pts,
                        cycle=cycle,
                        ml_proba=ml_proba,
                        atr_5m=self._current_atr,
                        atr_15m=self._current_atr_15m,
                        vix=self._current_vix,
                        vix_regime=regime.name if regime else "",
                        adx_15m=self._pilot_adx15,
                        recovery_mode=ml_indicators.get("recovery_mode", False),
                        gap_override=ml_indicators.get("gap_override", False),
                        gamma_squeeze=ml_indicators.get("gamma_squeeze", False),
                        dte=ml_indicators.get("dte", -1),
                        is_expiry=ml_indicators.get("expiry_day", False),
                        pcr=getattr(intel, "pcr", 0.0) if intel else 0.0,
                        pcr_available=getattr(intel, "pcr_available", False) if intel else False,
                        strategy_name=self.config.strategy_name,
                    )
                except Exception as e:
                    logger.warning(f"Journal dry-run record failed: {e}")

            # ── PAPER POSITION TRACKER ────────────────────────────────────
            # Set _live_position so the position monitor sends
            # PAPER_EXIT_SL / PAPER_EXIT_TP Telegram alerts — same timing as
            # live mode but without placing any real broker order.
            # symbol="PAPER" is the sentinel that tells the monitor to skip
            # the broker close call.
            with self._lock:
                if self._live_position is None:
                    self._live_position = LivePosition(
                        direction=direction,
                        entry_price=spot,
                        entry_time=time.monotonic(),
                        sl_price=sl_price,
                        tp_price=tp_price,
                        initial_sl=sl_price,
                        symbol="PAPER",
                        atr_at_entry=self._current_atr,
                        entry_premium=0.0,
                        state=PositionState.OPEN,
                        original_qty=trade_qty,
                        entry_cycle=cycle,
                    )
                    logger.info(
                        f"Cycle #{cycle}: [PAPER] Position tracker armed — "
                        f"{direction} entry={spot:.0f} "
                        f"SL={sl_price:.0f} TP={tp_price:.0f} "
                        f"→ Telegram alert fires on PAPER_EXIT_SL/TP"
                    )
            self._save_session_state()
            return

        # ── FUTURES LIVE-ORDER SAFETY GUARD ─────────────────────────────
        # Live order placement/symbol routing for execution_mode="futures"
        # is not implemented yet -- only the PAPER path above (DRY_RUN=true)
        # is wired. Everything below this point (strike resolution, real
        # place_order calls) is options-specific and would do something
        # undefined if reached in futures mode. Refuse outright rather than
        # silently fall through into options logic with a futures config.
        if self.config.execution_mode == "futures":
            logger.error(
                f"Cycle #{cycle}: 🛑 EXECUTION_MODE=futures but DRY_RUN is not "
                f"true — live futures order placement is NOT implemented. "
                f"Refusing to trade. Set DRY_RUN=true to test futures in "
                f"paper mode, or EXECUTION_MODE=options for live trading."
            )
            try:
                self.notifier.notify_trade(
                    action="BLOCKED",
                    symbol="NIFTY FUT",
                    side=action,
                    qty=0,
                    price=spot,
                    order_id="N/A",
                    status="blocked",
                    details=(
                        "EXECUTION_MODE=futures requires DRY_RUN=true — "
                        "live futures order placement is not implemented yet."
                    ),
                )
            except Exception:
                pass
            return

        try:
            # Check no position is open or closing (atomic check)
            with self._lock:
                if self._live_position is not None:
                    state = self._live_position.state
                    logger.warning(f"Cycle #{cycle}: Position already {state} — skipping")
                    return

            # 2026-05-15 BUG FIX: Heartbeat update BETWEEN gates so deadman
            # switch knows we're alive mid-cycle (previous hang at "entering"
            # status went undetected for 9+ min on 2026-05-15).
            def _hb(stage):
                try:
                    from core.deadman_switch import write_heartbeat
                    write_heartbeat(cycle=cycle, status=stage)
                except Exception:
                    pass

            _hb("decide_entry")

            # 2026-05-15 #4: MAX-PAIN gate (expiry day only).
            # Wrapped in watchdog — fetch_option_chain_oi can be slow (40-60 strikes).
            try:
                from core.expiry_utils import is_expiry_day, get_dte
                if is_expiry_day():
                    from core.max_pain import (
                        fetch_option_chain_oi, compute_max_pain, get_max_pain_bias,
                    )
                    from core.watchdog_helper import call_with_timeout
                    expiry_str = self.trader.strikes.get_next_expiry()
                    # 8 sec timeout — option chain pull can be slow
                    chain = call_with_timeout(
                        fn=lambda: fetch_option_chain_oi(
                            self.trader.client, expiry_str, spot, strikes_each_side=10,
                        ),
                        timeout_sec=8.0,
                        default=[],
                        name="fetch_option_chain_oi",
                    )
                    mp_strike = compute_max_pain(chain) if chain else None
                    if mp_strike:
                        bias = get_max_pain_bias(spot, mp_strike, threshold_pct=0.3)
                        ml_indicators["max_pain"] = mp_strike
                        ml_indicators["max_pain_bias"] = bias
                        logger.info(
                            f"Cycle #{cycle}: MAX-PAIN={mp_strike}, spot={spot:.0f}, "
                            f"bias={bias}"
                        )
                        # Block if trading AGAINST max-pain gravity on expiry day
                        if bias == "BEARISH" and option_type == "CE":
                            logger.warning(
                                f"Cycle #{cycle}: 🛑 MAX-PAIN BEARISH (spot above MP "
                                f"{mp_strike}) — blocking CE on expiry day"
                            )
                            return
                        if bias == "BULLISH" and option_type == "PE":
                            logger.warning(
                                f"Cycle #{cycle}: 🛑 MAX-PAIN BULLISH (spot below MP "
                                f"{mp_strike}) — blocking PE on expiry day"
                            )
                            return
            except Exception as _e:
                logger.debug(f"Max-pain gate failed (fail-open): {_e}")

            _hb("after_maxpain")

            # 2026-05-15 TIER-1 #1: Options-pricing gate (Sensibull replacement).
            # Compute IV + Greeks ourselves via Black-Scholes from Kotak chain.
            # Block trade if:
            #   - ATM IV > VIX + 4pts (options too expensive)
            #   - ATM delta < 0.35 (too far OTM, won't capture move)
            try:
                from core.options_chain import (
                    get_atm_greeks, is_premium_overpriced, is_delta_too_low,
                )
                from core.expiry_utils import get_dte
                from core.watchdog_helper import call_with_timeout
                dte = max(1, get_dte())
                vix_now = float(getattr(self, "_current_vix", 0) or 0)
                # 5 sec timeout — single option quote should take <1s
                greeks = call_with_timeout(
                    fn=lambda: get_atm_greeks(
                        kotak_client=self.trader.client,
                        spot=spot,
                        dte_days=dte,
                        option_type=option_type,
                    ),
                    timeout_sec=5.0,
                    default=None,
                    name="get_atm_greeks",
                )
                if greeks and greeks.get("iv") is not None:
                    iv_pct = greeks["iv"] * 100
                    ml_indicators["computed_iv"] = round(iv_pct, 1)
                    ml_indicators["computed_delta"] = greeks.get("delta")
                    ml_indicators["computed_theta_per_day"] = greeks.get("theta_per_day")
                    logger.info(
                        f"Cycle #{cycle}: GREEKS @ATM{greeks['strike']}{option_type}: "
                        f"IV={iv_pct:.1f}% Δ={greeks['delta']:.2f} "
                        f"θ={greeks['theta_per_day']:.1f}/day premium=₹{greeks['premium']:.0f}"
                    )
                    if is_premium_overpriced(greeks, vix_now):
                        logger.warning(
                            f"Cycle #{cycle}: 🛑 OPTIONS OVERPRICED — "
                            f"IV={iv_pct:.1f}% vs VIX={vix_now:.1f} (>4pts gap) → SKIP"
                        )
                        return
                    if is_delta_too_low(greeks, min_delta=0.30):
                        logger.warning(
                            f"Cycle #{cycle}: 🛑 LOW DELTA — "
                            f"|Δ|={abs(greeks['delta']):.2f}<0.30 (too far OTM) → SKIP"
                        )
                        return
            except Exception as _e:
                logger.debug(f"Greeks gate failed (fail-open): {_e}")

            _hb("after_greeks")

            # 2026-05-15 TIER-2 #4: Multi-TF agreement check.
            # 2026-06-11: hard veto → confidence penalty. Jun 11 audit: this
            # veto blocked 4 CALLs at 90% conf during a confirmed V-recovery
            # (15m/60m EMAs still carried the prior day's downtrend — on any
            # reversal day they lag by construction). Now: bypass when the
            # intraday regime engine confirms the signal direction; otherwise
            # require conf to clear effective_min_conf + 8pp instead of vetoing.
            try:
                tf_dir_5m  = "CALL" if ml_signal == 0 else ("PUT" if ml_signal == 1 else None)
                # Pull pre-computed multi-TF directional features from ml_indicators
                ema_15m_bull = (ml_indicators.get("tf15_ema9", 0) >
                                ml_indicators.get("tf15_ema21", 0))
                ema_60m_bull = (ml_indicators.get("tf60_ema9", 0) >
                                ml_indicators.get("tf60_ema21", 0))
                if tf_dir_5m is not None:
                    htf_against = (
                        (tf_dir_5m == "CALL" and not (ema_15m_bull or ema_60m_bull))
                        or (tf_dir_5m == "PUT" and (ema_15m_bull and ema_60m_bull))
                    )
                    if htf_against:
                        _regime_name = str(ml_indicators.get("regime", ""))
                        regime_confirms = (
                            (tf_dir_5m == "CALL" and _regime_name == "TREND_UP")
                            or (tf_dir_5m == "PUT" and _regime_name == "TREND_DOWN")
                        )
                        tf_bar = min(85, effective_min_conf + 8)
                        if regime_confirms:
                            logger.info(
                                f"Cycle #{cycle}: MULTI-TF DISAGREE bypassed — "
                                f"intraday regime {_regime_name} confirms {tf_dir_5m}"
                            )
                        elif confidence < tf_bar:
                            logger.warning(
                                f"Cycle #{cycle}: 🛑 MULTI-TF DISAGREE — 5m {tf_dir_5m} "
                                f"vs HTF, conf {confidence}% < {tf_bar}% "
                                f"(min_conf+8pp) → SKIP"
                            )
                            _shadow_skip(
                                f"multi_tf_disagree:{tf_dir_5m.lower()}_vs_htf",
                                conf=confidence,
                            )
                            return
                        else:
                            logger.info(
                                f"Cycle #{cycle}: MULTI-TF DISAGREE but conf "
                                f"{confidence}% ≥ {tf_bar}% (min_conf+8pp) → proceeding"
                            )
            except Exception as _e:
                logger.debug(f"Multi-TF check failed (fail-open): {_e}")

            # 2026-05-15 #5: PCR MOMENTUM gate (uses existing intel data).
            # Rapid PCR rise (>+0.05 in 5min) = sentiment shifting bearish
            # → require higher conf for CALL, allow PUT normally.
            try:
                pcr_mom = ml_indicators.get("pcr_5m_momentum", 0)
                if pcr_mom is not None and abs(pcr_mom) > 0.05:
                    if pcr_mom > 0.05 and option_type == "CE":
                        logger.warning(
                            f"Cycle #{cycle}: 🛑 PCR RISING +{pcr_mom:.3f} in 5m "
                            f"(bearish shift) → blocking CE"
                        )
                        _shadow_skip("pcr_momentum:rising_vs_call", conf=confidence)
                        return
                    if pcr_mom < -0.05 and option_type == "PE":
                        logger.warning(
                            f"Cycle #{cycle}: 🛑 PCR FALLING {pcr_mom:.3f} in 5m "
                            f"(bullish shift) → blocking PE"
                        )
                        _shadow_skip("pcr_momentum:falling_vs_put", conf=confidence)
                        return
            except Exception as _e:
                logger.debug(f"PCR-momentum gate failed: {_e}")

            _hb("after_pcr")

            # 2026-05-15 STRATEGY K — three data-driven gates from 13-day backtest:
            #   (1) Halt for day after ANY losing trade (no revenge trading)
            #   (2) CALL trades only if confidence >= 85% (CALL signals = 17% WR)
            #   (3) Already-applied: 11:00 hard block via morning_hard_block_min=105
            try:
                # Gate 1: halt-after-loss / emergency-stop
                if getattr(self, "_emergency_stopped", False):
                    logger.warning(
                        f"Cycle #{cycle}: 🛑 EMERGENCY STOPPED — "
                        f"{action} {option_type} BLOCKED until manually reset"
                    )
                    _shadow_skip("strategy_k:emergency_stopped", conf=confidence)
                    return
                if getattr(self, "_day_halted_after_loss", False):
                    logger.warning(
                        f"Cycle #{cycle}: 🛑 DAY HALTED (after loss) — "
                        f"{action} {option_type} BLOCKED until tomorrow"
                    )
                    _shadow_skip("strategy_k:day_halted", conf=confidence)
                    return
                # Gate 2: CALL-specific high-confidence threshold
                if option_type == "CE" and confidence < 85:
                    logger.warning(
                        f"Cycle #{cycle}: 🛑 CALL FILTER — conf {confidence}% < 85% "
                        f"(13-day backtest: CALL trades had 17% WR, need >=85% conf to fire)"
                    )
                    _shadow_skip("strategy_k:call_conf_filter", conf=confidence)
                    return
            except Exception as _e:
                logger.debug(f"Strategy-K gates failed: {_e}")

            # 2026-05-05 CRITICAL FIX: Risk-manager hard gate.
            # Previously _smart_execute bypassed risk.can_open_trade(), causing
            # ₹9,451 loss across 2026-05-04 and 2026-05-05 (cap was ₹3,000/day).
            try:
                allowed, reason = self.trader.risk.can_open_trade()
                if not allowed:
                    logger.warning(
                        f"Cycle #{cycle}: 🛑 RISK CAP — {action} {option_type} "
                        f"BLOCKED. {reason}"
                    )
                    # Telegram alert for visibility
                    try:
                        notif = getattr(self.trader, "notif", None)
                        if notif and getattr(notif, "telegram", None) and notif.telegram.enabled:
                            notif.telegram.send_message(
                                f"🛑 <b>RISK CAP HIT — TRADE BLOCKED</b>\n\n"
                                f"Signal: {action} {option_type}\n"
                                f"Reason: {reason}\n"
                                f"No more trades today."
                            )
                    except Exception:
                        pass
                    return
            except Exception as _e:
                logger.debug(f"Risk check failed (fail-open): {_e}")

            # Broker circuit breaker: pause NEW entries during a sustained
            # Kotak Neo outage (MAX_CONSECUTIVE_FAILURES in a row). Existing
            # open positions keep being monitored/closed regardless — this
            # only blocks opening additional ones. Fail-open if the client
            # doesn't expose is_healthy() (e.g. a different client type).
            try:
                client = getattr(self.trader, "client", None)
                if client is not None and hasattr(client, "is_healthy") and not client.is_healthy():
                    logger.warning(
                        f"Cycle #{cycle}: 🛑 BROKER UNHEALTHY — {action} {option_type} "
                        f"BLOCKED (sustained Kotak Neo call failures)"
                    )
                    return
            except Exception as _e:
                logger.debug(f"Broker health check failed (fail-open): {_e}")

            # 2026-05-05 ANTI-WHIPSAW: After a losing trade, block OPPOSITE
            # direction for `whipsaw_cooldown_min` minutes (default 30).
            # Stops the bot from chasing reversals immediately after stop-out.
            # May 4: CALL exit at 10:14 → PUT entry at 10:15 (1 min) → -₹4,079 loss.
            try:
                if hasattr(self, "_last_loss_time") and self._last_loss_time:
                    cooldown_min = getattr(self.config, "whipsaw_cooldown_min", 30)
                    age_min = (time.monotonic() - self._last_loss_time) / 60
                    if (age_min < cooldown_min and
                        getattr(self, "_last_loss_dir", None) and
                        self._last_loss_dir != ("CALL" if option_type == "CE" else "PUT")):
                        logger.warning(
                            f"Cycle #{cycle}: 🚫 WHIPSAW BLOCK — last loss was "
                            f"{self._last_loss_dir} {age_min:.1f}min ago. "
                            f"Skipping opposite {action} {option_type} "
                            f"({cooldown_min - age_min:.1f}min cooldown left)"
                        )
                        return
            except Exception as _e:
                logger.debug(f"Whipsaw check failed: {_e}")

            result = self._smart_execute(
                action=action,
                option_type=option_type,
                strike_mode=strike_mode,
                qty=trade_qty,
                timeout_sec=3,
            )

            # V9.3: Handle spread-gate abort
            if result.get("status") == "ABORTED":
                logger.warning(
                    f"Cycle #{cycle}: Entry ABORTED — {result.get('reason', 'unknown')}"
                )
                return

            order_id = result.get("trade", {}).get("orderid", "")
            symbol = result.get("symbol", f"NIFTY {option_type}")

            # Fetch entry premium (what we paid for the option)
            entry_premium = 0.0
            try:
                time.sleep(1)  # Allow broker fill to settle
                quote = self.trader.client.get_quote(symbol, self.trader.exchange)
                if isinstance(quote.get("data"), dict):
                    entry_premium = float(quote["data"].get("ltp", 0))
                elif isinstance(quote.get("ltp"), (int, float)):
                    entry_premium = float(quote["ltp"])
                if entry_premium > 0:
                    logger.info(f"Option entry premium: Rs.{entry_premium:.2f} ({symbol})")
            except Exception as e:
                logger.warning(f"Could not fetch entry premium: {e}")

            # ── Slippage computation (Issue #7) ────────────────────────────
            # Four values per trade:
            #   signal_premium  = option ask at signal time (from _smart_execute)
            #   limit_price_    = IOC limit we submitted
            #   fill_price      = actual execution price (entry_premium, post-fill REST quote)
            #   market_impact   = fill_price - signal_premium  (total cost vs fair value)
            #   exec_slippage   = fill_price - limit_price_    (IOC execution quality)
            signal_premium_   = 0.0
            limit_price_      = 0.0
            market_impact_pts = 0.0
            exec_slippage_pts = 0.0
            try:
                signal_premium_  = float(result.get("signal_premium", 0) or 0)
                limit_price_     = float(result.get("limit_price",    0) or 0)
                fill_price       = entry_premium   # what we actually paid

                if fill_price > 0 and signal_premium_ > 0:
                    market_impact_pts = round(fill_price - signal_premium_, 4)
                if fill_price > 0 and limit_price_ > 0:
                    exec_slippage_pts = round(fill_price - limit_price_, 4)

                # In-memory rolling history (5-tuple for richer analysis)
                self._slippage_history.append((
                    signal_premium_, limit_price_, fill_price,
                    market_impact_pts, exec_slippage_pts,
                ))
                if len(self._slippage_history) > 50:
                    self._slippage_history = self._slippage_history[-50:]

                # Rolling averages
                n = len(self._slippage_history)
                avg_mi = sum(s[3] for s in self._slippage_history) / n
                avg_es = sum(s[4] for s in self._slippage_history) / n

                logger.info(
                    f"SLIPPAGE: signal={signal_premium_:.2f} "
                    f"limit={limit_price_:.2f} fill={fill_price:.2f} | "
                    f"market_impact={market_impact_pts:+.2f}pts "
                    f"exec_slip={exec_slippage_pts:+.2f}pts | "
                    f"rolling avg_impact={avg_mi:+.2f}pts (n={n})"
                )
                if avg_mi > 2.0 and n >= 5:
                    logger.warning(
                        f"SLIPPAGE LEAK: avg market impact={avg_mi:.2f}pts > 2pts "
                        f"over {n} trades — consider tighter IOC pricing"
                    )
            except Exception as _e:
                logger.debug(f"Slippage tracking failed: {_e}")

            # Premium-denominated exchange-SL trigger/limit (Phase 2 fix) —
            # the exchange order is on the option contract, so the trigger
            # must be in premium (Rs.) terms, derived from entry_premium and
            # premium_sl_pts (the premium-equivalent of the spot SL distance).
            _premium_sl_pts = float(ml_indicators.get("premium_sl_pts", 0) or 0)
            _exch_sl_trigger = 0.0
            _exch_sl_limit = 0.0
            if entry_premium > 0 and _premium_sl_pts > 0:
                _exch_sl_trigger = round(max(entry_premium - _premium_sl_pts, 0.05), 2)
                _exch_sl_limit = round(_exch_sl_trigger * 0.95, 2)

            # Atomically set position and increment counter
            with self._lock:
                self._trades_today += 1
                self._save_trade_count()  # V9.3: persist across restarts
                self._last_trade_time = time.monotonic()
                self._live_position = LivePosition(
                    direction=direction,
                    entry_price=spot,
                    entry_time=time.monotonic(),
                    sl_price=sl_price,
                    tp_price=tp_price,
                    initial_sl=sl_price,
                    symbol=symbol,
                    atr_at_entry=self._current_atr,
                    entry_premium=entry_premium,
                    state=PositionState.OPEN,
                    # 2026-05-15: For partial profit-taking + Greeks-based exit
                    original_qty=trade_qty,
                    entry_delta=float(ml_indicators.get("computed_delta", 0) or 0),
                    exchange_sl_trigger=_exch_sl_trigger,
                    exchange_sl_limit=_exch_sl_limit,
                )
                # Stash diagnostic context for calibrator + post-trade analyzer
                try:
                    setattr(self._live_position, "entry_confidence", confidence)
                    setattr(self._live_position, "entry_regime",
                            ml_indicators.get("regime", "ANY"))
                except Exception:
                    pass
                # Reset peak/drawdown trackers for new position
                self._pos_peak_profit_pts = 0.0
                self._pos_max_drawdown_pts = 0.0

            # ── Exchange-resident SL (Phase 2) ────────────────────────────
            # Place SL (stop-loss limit) order at the exchange immediately
            # after entry so a VPS crash or monitor failure still closes the
            # position. Done OUTSIDE the lock to avoid holding the lock
            # during I/O.
            try:
                if self._live_position is not None:
                    self._place_exchange_sl(self._live_position)
            except Exception as _esl_e:
                logger.warning(f"ExchangeSL: post-entry placement failed: {_esl_e}")

            # Record entry to per-trade journal (incl. slippage fields from Issue #7)
            if self._journal:
                try:
                    self._journal.record_entry(
                        direction=direction,
                        option_type=option_type,
                        strike_mode=strike_mode,
                        entry_spot=spot,
                        entry_premium=entry_premium,
                        confidence=confidence,
                        sl_pts=sl_pts,
                        tp_pts=tp_pts,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        lots=num_lots,
                        qty=trade_qty,
                        cycle=cycle,
                        symbol=symbol,
                        atr_5m=self._current_atr,
                        atr_15m=self._current_atr_15m,
                        vix=self._current_vix,
                        vix_regime=regime.name if regime else "",
                        adx_15m=self._pilot_adx15,
                        ml_proba=ml_proba,
                        recovery_mode=ml_indicators.get("recovery_mode", False),
                        gap_override=ml_indicators.get("gap_override", False),
                        gamma_squeeze=ml_indicators.get("gamma_squeeze", False),
                        dte=ml_indicators.get("dte", -1),
                        is_expiry=ml_indicators.get("expiry_day", False),
                        pcr=getattr(intel, "pcr", 0.0) if intel else 0.0,
                        pcr_available=getattr(intel, "pcr_available", False) if intel else False,
                        oi_magnetic_strike=ml_indicators.get("oi_magnetic_strike", 0.0),
                        is_dry_run=False,
                        # Slippage fields (Issue #7)
                        signal_premium=signal_premium_,
                        limit_price=limit_price_,
                        fill_price=entry_premium,
                        market_impact_pts=market_impact_pts,
                        exec_slippage_pts=exec_slippage_pts,
                        strategy_name=self.config.strategy_name,
                    )
                except Exception as e:
                    logger.warning(f"Journal entry record failed: {e}")

            fill_type = result.get("fill_type", "UNKNOWN")
            self.notifier.notify_trade(
                action="PILOT_TRADE",
                symbol=symbol,
                side=action,
                qty=trade_qty,
                price=spot,
                order_id=order_id,
                status="executed",
                details=(
                    f"AUTO-PILOT TRADE #{self._trades_today}\n"
                    f"ML: {ml_direction} | Claude: {action} {option_type}\n"
                    f"Confidence: {confidence}% | Bias: {bias}\n"
                    f"Lots: {num_lots} ({trade_qty} qty) | Fill: {fill_type}\n"
                    f"SL: {sl_price:.0f} ({sl_pts:.0f}pts) | "
                    f"TP: {tp_price:.0f} ({tp_pts:.0f}pts) | "
                    f"ATR: {self._current_atr:.1f}\n"
                    f"Trail: {'ON' if self.config.use_trailing_stop else 'OFF'}\n"
                    f"Analysis: {analysis}\n"
                    f"Trades: {self._trades_today}/{self.config.max_trades_per_day}"
                ),
            )

        except Exception as e:
            logger.error(f"Cycle #{cycle}: Execution FAILED: {e}")
            self.notifier.notify_trade(
                action="PILOT_ERROR", symbol=f"NIFTY {option_type}",
                side=action, qty=0, price=spot, order_id="",
                status="error", details=str(e),
            )

    # ------------------------------------------------------------------
    # ATR-based dynamic SL/TP
    # ------------------------------------------------------------------

    def _compute_current_atr(self) -> float:
        """Fetch 5-min ATR(14) from TradingView for dynamic SL/TP sizing."""
        try:
            import pandas as pd
            from core.tv_fetcher import get_tv_fetcher
            df5 = get_tv_fetcher().get_nifty_5min(n_bars=30)
            if df5.empty or len(df5) < 15:
                return self._current_atr  # fallback to last known
            h, l, c = df5["high"], df5["low"], df5["close"]
            tr = pd.concat([
                h - l,
                (h - c.shift()).abs(),
                (l - c.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if atr > 0:
                self._current_atr = atr
            return self._current_atr
        except Exception as e:
            logger.debug(f"ATR computation failed: {e}")
            return self._current_atr

    def _compute_current_atr10(self) -> float:
        """Fetch 5-min ATR(frozen_atr_period, =10 by default) — used ONLY by
        the frozen validated exit path (use_frozen_atr_exit). Deliberately
        separate from _compute_current_atr()'s ATR(14) so enabling the
        frozen exit does not change ATR(14)'s behavior anywhere else it's
        used (reconciliation SL defaults, OI buffer, ATR-trailing-stop,
        logging)."""
        try:
            import pandas as pd
            from core.tv_fetcher import get_tv_fetcher
            period = self.config.frozen_atr_period
            df5 = get_tv_fetcher().get_nifty_5min(n_bars=30)
            if df5.empty or len(df5) < period + 1:
                return self._current_atr10  # fallback to last known
            h, l, c = df5["high"], df5["low"], df5["close"]
            tr = pd.concat([
                h - l,
                (h - c.shift()).abs(),
                (l - c.shift()).abs(),
            ], axis=1).max(axis=1)
            atr10 = float(tr.rolling(period).mean().iloc[-1])
            if atr10 > 0:
                self._current_atr10 = atr10
            return self._current_atr10
        except Exception as e:
            logger.debug(f"ATR10 computation failed: {e}")
            return self._current_atr10

    def _compute_atr_15m(self) -> float:
        """Fetch 15-min ATR(14) for volatility-adaptive reversal threshold."""
        try:
            import pandas as pd
            from core.tv_fetcher import get_tv_fetcher
            df15 = get_tv_fetcher().get_nifty_15min(n_bars=30)
            if df15.empty or len(df15) < 15:
                return self._current_atr_15m
            h, l, c = df15["high"], df15["low"], df15["close"]
            tr = pd.concat([
                h - l,
                (h - c.shift()).abs(),
                (l - c.shift()).abs(),
            ], axis=1).max(axis=1)
            atr15 = float(tr.rolling(14).mean().iloc[-1])
            if atr15 > 0:
                self._current_atr_15m = atr15
            return self._current_atr_15m
        except Exception as e:
            logger.debug(f"ATR_15m computation failed: {e}")
            return self._current_atr_15m

    def _update_vix_regime(self) -> VIXRegime:
        """Fetch live VIX and classify into regime."""
        try:
            from core.tv_fetcher import get_tv_fetcher
            tv = get_tv_fetcher()
            for sym in ["INDIAVIX", "INDIA VIX"]:
                try:
                    raw = tv.get_ohlcv(sym, "NSE", interval_minutes="D", n_bars=3)
                    if raw is not None and not raw.empty:
                        vix_now = float(raw["close"].iloc[-1])
                        vix_prev = float(raw["close"].iloc[-2]) if len(raw) >= 2 else 0.0
                        if vix_now > 0:
                            self._current_vix = vix_now
                            self._prev_vix = vix_prev
                            break
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")

        # Also try broker quote as fallback
        if self._current_vix <= 0:
            try:
                for sym in ["INDIAVIX", "INDIA VIX", "INDIA_VIX"]:
                    try:
                        q = self.trader.client.get_quote(sym, "NSE_INDEX")
                        v = float(q.get("data", {}).get("ltp", 0) or 0)
                        if v > 0:
                            self._current_vix = v
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        self._vix_regime = classify_vix(self._current_vix, self._prev_vix)
        logger.info(f"VIX regime: {format_regime(self._vix_regime)}")
        return self._vix_regime

    def _get_dynamic_sl_tp(self, direction: str) -> tuple:
        """
        Compute dynamic SL and TP based on current ATR + VIX regime.
        Returns (sl_points, tp_points) scaled by both ATR and VIX.
        """
        cfg = self.config

        # ── FROZEN VALIDATED ATR EXIT (opt-in) — bypasses everything below:
        # no Kalman ATR, no VIX regime scaling, no gap protection, no
        # min-RR floor. Exactly TP=frozen_atr_tp_multiplier x
        # ATR(frozen_atr_period), SL=frozen_atr_sl_multiplier x
        # ATR(frozen_atr_period), matching the validated research exactly. ──
        if cfg.use_frozen_atr_exit:
            atr10 = self._compute_current_atr10()
            sl_pts = atr10 * cfg.frozen_atr_sl_multiplier
            tp_pts = atr10 * cfg.frozen_atr_tp_multiplier
            logger.info(
                f"FROZEN ATR EXIT: ATR({cfg.frozen_atr_period})={atr10:.1f} -> "
                f"SL={sl_pts:.1f}pts TP={tp_pts:.1f}pts (R:R = 1:{tp_pts/sl_pts:.2f})"
            )
            return round(sl_pts, 1), round(tp_pts, 1)

        if not cfg.use_dynamic_sl_tp:
            return 35.0, 100.0  # legacy fixed values

        # 2026-05-15 TIER-4: Kalman ATR — adapts in 2-3 bars vs rolling-14 (lags 7-14 bars)
        # Use MAX of rolling ATR and Kalman ATR (don't reduce SL during regime shifts)
        atr = self._compute_current_atr()
        try:
            from core.kalman_atr import get_kalman_atr
            # Feed latest true-range to Kalman
            from core.tv_fetcher import get_tv_fetcher
            df5_tr = get_tv_fetcher().get_nifty_5min(n_bars=3)
            if not df5_tr.empty and len(df5_tr) >= 2:
                last = df5_tr.iloc[-1]
                prev_close = float(df5_tr["close"].iloc[-2])
                tr = max(
                    float(last["high"]) - float(last["low"]),
                    abs(float(last["high"]) - prev_close),
                    abs(float(last["low"]) - prev_close),
                )
                katr = get_kalman_atr()
                k_atr = katr.update(tr)
                # Use MAX (don't shrink SL when vol picks up)
                if k_atr > atr * 1.1:
                    logger.info(
                        f"Kalman ATR={k_atr:.1f} > rolling ATR={atr:.1f} "
                        f"→ using Kalman (regime shifting up)"
                    )
                    atr = max(atr, k_atr)
        except Exception as _e:
            logger.debug(f"Kalman ATR update failed: {_e}")

        sl_pts = atr * cfg.sl_atr_multiplier
        tp_pts = atr * cfg.tp_atr_multiplier

        # Apply VIX regime scaling
        regime = self._vix_regime
        if regime:
            sl_pts, tp_pts = apply_vix_to_sl_tp(
                sl_pts, tp_pts, regime,
                min_sl=cfg.min_sl_points,
                max_sl=90.0,   # Allow wider SL in high-VIX
                min_tp=cfg.min_tp_points,
                max_tp=350.0,  # Allow wider TP in high-VIX
            )
        else:
            sl_pts = max(cfg.min_sl_points, min(cfg.max_sl_points, sl_pts))
            tp_pts = max(cfg.min_tp_points, min(cfg.max_tp_points, tp_pts))
            # V10: enforce min R:R floor — TP must be >= sl × min_rr_ratio
            min_rr = getattr(cfg, "min_rr_ratio", 2.0)
            if tp_pts < sl_pts * min_rr:
                tp_pts = min(sl_pts * min_rr, cfg.max_tp_points)

        # ── Gap protection (additive, opt-in) ──────────────────────────
        # This bot never holds overnight (MIS, forced same-day square-off),
        # so classic gap-on-a-held-position risk doesn't apply. Instead,
        # widen SL/TP for entries taken shortly after market open, when
        # opening-range volatility is highest.
        if cfg.gap_protection_widen_pct > 0:
            now_dt = datetime.now()
            market_open_dt = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            minutes_since_open = (now_dt - market_open_dt).total_seconds() / 60.0
            _pre_sl, _pre_tp = sl_pts, tp_pts
            sl_pts, tp_pts = _gap_protection_widen(
                sl_pts, tp_pts, minutes_since_open,
                cfg.gap_protection_window_min, cfg.gap_protection_widen_pct,
            )
            if sl_pts != _pre_sl:
                logger.info(
                    f"GAP PROTECTION: {minutes_since_open:.1f}min since open "
                    f"(<{cfg.gap_protection_window_min}min window) — "
                    f"widened SL {_pre_sl:.0f}->{sl_pts:.0f}pts TP {_pre_tp:.0f}->{tp_pts:.0f}pts"
                )

        regime_name = regime.name if regime else "UNKNOWN"
        logger.info(
            f"Dynamic SL/TP: ATR={atr:.1f} VIX={self._current_vix:.1f}({regime_name}) "
            f"→ SL={sl_pts:.0f}pts TP={tp_pts:.0f}pts (R:R = 1:{tp_pts/sl_pts:.1f})"
        )
        return round(sl_pts, 1), round(tp_pts, 1)

    # ------------------------------------------------------------------
    # Smart Limit Execution — reduces slippage on option entry
    # ------------------------------------------------------------------

    def _entry_data_is_fresh(self) -> dict:
        """
        Check the PRIMARY decision feed (NIFTY 5m) is fresh enough to OPEN a
        position. Returns {"ok": bool, "reason": str, "source": str}.

        HARD rejection conditions (timezone-immune — both use time.time()):
          1. source in {YFINANCE, NONE} — delayed (~15min) or no data at all.
             This is the core protection: never enter on Yahoo's lagged feed
             or when every source failed.
          2. fetch age > ENTRY_MAX_FETCH_AGE_SEC (default 120s) — the feed was
             not refreshed recently (pilot normally fetches every cycle).

        bar_age is INTENTIONALLY not a hard condition. Source timezones are
        inconsistent (TV = IST-naive, Kotak-WS = server-local), so an absolute
        bar-age gate is unreliable and could wrongly block ALL trading on a
        non-IST server. The existing 10-min cache TTL already bounds TV_CACHE
        staleness. bar_age is logged for diagnostics only.

        Fail policy:
          - Detected staleness → fail-CLOSED (block entry). This is the point.
          - Internal error in the check itself → fail-OPEN (allow entry) so a
            bug here can never silently halt all trading. Logged at WARNING.
        """
        import os as _os
        try:
            from core.tv_fetcher import get_data_freshness
            f = get_data_freshness("NIFTY", "NSE", 5)

            max_fetch_age = int(_os.getenv("ENTRY_MAX_FETCH_AGE_SEC", "120"))
            bad_sources   = {"YFINANCE", "NONE"}
            bar_age_diag  = f.get("bar_age_sec", -1)

            if f["source"] in bad_sources:
                return {"ok": False, "source": f["source"],
                        "reason": (f"source={f['source']} (delayed/unavailable) "
                                   f"[bar_age~{bar_age_diag:.0f}s]")}
            if f["fetch_age_sec"] > max_fetch_age:
                return {"ok": False, "source": f["source"],
                        "reason": (f"fetch_age={f['fetch_age_sec']:.0f}s "
                                   f"> {max_fetch_age}s (src={f['source']})")}
            return {"ok": True, "source": f["source"],
                    "reason": f"fresh (src={f['source']}, bar_age~{bar_age_diag:.0f}s)"}
        except Exception as _e:
            # Fail-OPEN: never let a bug in the freshness check halt trading.
            logger.warning(f"Freshness check error (fail-open, allowing entry): {_e}")
            return {"ok": True, "source": "UNKNOWN", "reason": "check_error_failopen"}

    def _smart_execute(
        self,
        action: str,
        option_type: str,
        strike_mode: str,
        qty: int,
        timeout_sec: int = 3,   # kept for API compat, unused (IOC resolves instantly)
    ) -> dict:
        """
        IOC-only execution — no MARKET fallback.

        Attempt 1: IOC LIMIT at best_ask + 1 tick
        Attempt 2: IOC LIMIT at best_ask + 3 ticks  (if attempt 1 misses)
        ABORT if both miss — avoids chasing at any price.

        Kotak Neo LIMIT orders are IOC by default (validity="IOC") — the broker
        fills instantly or cancels; there is no open-order risk.

        Returns:
            dict with order result. status="ABORTED" means both IOC attempts missed.
        """
        _TICK = 0.05  # minimum price increment for Nifty options

        # --- Step 1: Price discovery — native Kotak Neo via get_option_quote() ---
        limit_price = 0.0
        _best_ask   = 0.0   # raw ask; reused for +3-tick retry
        try:
            offset = strike_mode.upper().replace("_", "")
            ot = option_type.upper()
            expiry_str = self.trader.strikes.get_next_expiry()

            # Compute actual strike from spot + offset
            from core.strike_selector import round_to_strike
            spot = self.trader.get_nifty_spot()
            atm_strike = round_to_strike(spot)
            offset_map = {"ATM": 0, "OTM1": 1, "OTM2": 2, "OTM3": 3,
                          "ITM1": -1, "ITM2": -2}
            offset_steps = offset_map.get(offset, 0)
            step_size = 50  # Nifty strike gap
            if ot == "CE":
                strike = atm_strike + (offset_steps * step_size)
            else:
                strike = atm_strike - (offset_steps * step_size)

            # Native Kotak depth quote — search_scrip + quotes(quote_type="depth")
            q = {}
            try:
                q = self.trader.client.get_option_quote(
                    underlying="NIFTY",
                    expiry=expiry_str,
                    strike=int(strike),
                    option_type=ot,
                    quote_type="depth",
                ) or {}
            except Exception as e:
                logger.debug(f"get_option_quote(depth) failed: {e}")

            best_ask = float(q.get("ask", 0) or 0)
            best_bid = float(q.get("bid", 0) or 0)
            ltp      = float(q.get("ltp", 0) or 0)

            # V10 #7: Spread Tolerance Gate
            MAX_SPREAD_PCT = float(
                getattr(self.config, "max_spread_pct", 0.02)
            ) * 100
            if best_bid > 0 and best_ask > 0:
                mid = (best_ask + best_bid) / 2.0
                spread_pct = ((best_ask - best_bid) / mid) * 100
                logger.info(
                    f"Smart exec: bid={best_bid:.2f} ask={best_ask:.2f} "
                    f"spread={spread_pct:.2f}% (strike={strike} {ot})"
                )
                if spread_pct > MAX_SPREAD_PCT:
                    logger.warning(
                        f"Smart exec: Spread too wide ({spread_pct:.1f}% > "
                        f"{MAX_SPREAD_PCT}%) — aborting entry"
                    )
                    return {"status": "ABORTED",
                            "reason": f"Spread {spread_pct:.1f}% > {MAX_SPREAD_PCT}%"}

            if best_ask > 0:
                _best_ask   = best_ask
                limit_price = round(best_ask + _TICK, 2)  # +1 tick
                logger.info(
                    f"Smart exec: best_ask={best_ask:.2f} → IOC LIMIT at {limit_price:.2f}"
                )
            elif ltp > 0:
                # Fallback: use LTP + 0.5% buffer
                limit_price = round(ltp * 1.005, 2)
                logger.info(
                    f"Smart exec: no depth — LTP={ltp:.2f} → LIMIT at {limit_price:.2f}"
                )
            else:
                # Last-resort LTP-only quote (no depth book)
                try:
                    q2 = self.trader.client.get_option_quote(
                        underlying="NIFTY",
                        expiry=expiry_str,
                        strike=int(strike),
                        option_type=ot,
                        quote_type="ltp",
                    ) or {}
                    ltp2 = float(q2.get("ltp", 0) or 0)
                    if ltp2 > 0:
                        limit_price = round(ltp2 * 1.005, 2)
                        logger.info(
                            f"Smart exec: LTP-only={ltp2:.2f} → LIMIT at {limit_price:.2f}"
                        )
                except Exception as e:
                    logger.debug(f"get_option_quote(ltp) failed: {e}")

        except Exception as e:
            logger.debug(f"Smart exec price discovery failed: {e}")

        # --- Step 2: First IOC LIMIT attempt at best_ask + 1 tick ---
        if limit_price <= 0:
            logger.warning(
                "Smart exec: price discovery failed — ABORT (no MARKET fallback)"
            )
            return {"status": "ABORTED", "reason": "NO_PRICE_FOR_IOC", "fill_type": "NONE"}

        def _ioc_attempt(price: float, label: str) -> dict:
            """Place a single IOC LIMIT order; return result dict."""
            try:
                r = self.trader.smart_trade(
                    action=action, option_type=option_type,
                    strike_mode=strike_mode, qty=qty,
                    price_type="LIMIT", price=price,
                )
                return r
            except Exception as exc:
                logger.error(f"Smart exec {label} order error: {exc}")
                return {"status": "ABORTED", "reason": f"ORDER_ERROR: {exc}"}

        def _ioc_filled(r: dict) -> bool:
            """True if result indicates a filled (not aborted/cancelled/blocked) order."""
            status = r.get("status") or r.get("trade", {}).get("status")
            reason = r.get("reason") or r.get("trade", {}).get("reason")
            return (
                status not in ("ABORTED", "ERROR", "CANCELLED", "blocked", "error", "failed")
                and reason not in ("IOC_NOT_FILLED",)
            )

        def _order_error_uncertain(r: dict) -> bool:
            """True only if the failure was a raised exception during the
            order call (broker outcome unknown -- it may have filled and
            only the response was lost), not an explicit broker rejection
            or local abort (spread/price gate), which are unambiguous."""
            reason = r.get("reason") or r.get("trade", {}).get("reason") or ""
            return str(reason).startswith("ORDER_ERROR:")

        # signal_premium: the option's mid/ask at the moment we're about to send the
        # order.  This is the "fair value" the model used when deciding to trade.
        # Captured here (just before IOC) so the caller can compute market impact.
        # We use best_ask as the pre-trade reference; it's already available.
        _signal_premium = float(_best_ask or limit_price)

        # Attempt 1: best_ask + 1 tick
        result = _ioc_attempt(limit_price, "IOC-1")
        if _ioc_filled(result):
            result["fill_type"]       = "IOC_LIMIT"
            result["limit_price"]     = limit_price       # what we asked to pay
            result["signal_premium"]  = _signal_premium   # market price at signal
            logger.info(f"Smart exec: IOC attempt 1 filled at {limit_price:.2f}")
            return result

        # C6: IOC-1 raised an exception (broker outcome unknown) rather than
        # an explicit rejection -- verify no stray fill exists before firing
        # a second order, to avoid a duplicate position. Uses the bot's
        # single-position-at-a-time invariant: no position should exist yet
        # at this point in the entry flow, so any open position found here
        # can only be the one IOC-1 just placed.
        if _order_error_uncertain(result):
            try:
                stray_positions = self.trader.client.get_open_nifty_positions()
            except Exception as recon_exc:
                logger.critical(
                    f"Smart exec: reconciliation check itself failed "
                    f"({recon_exc}) after IOC-1 raised {result.get('reason')} "
                    f"— broker outcome unknown, aborting rather than risk a "
                    f"duplicate order. Check the broker order book manually."
                )
                result["status"] = "ABORTED"
                result["reason"] = f"RECONCILIATION_FAILED_AFTER: {result.get('reason')}"
                return result

            if stray_positions:
                stray = stray_positions[0]
                logger.critical(
                    f"Smart exec: IOC-1 raised an exception but a position "
                    f"now exists at the broker ({stray.get('symbol')}) — "
                    f"treating as FILLED to avoid a duplicate order. "
                    f"Reason: {result.get('reason')}"
                )
                result["status"]          = "RECONCILED_FILLED"
                result["fill_type"]       = "IOC_LIMIT_RECONCILED"
                result["limit_price"]     = limit_price
                result["signal_premium"]  = _signal_premium
                result["symbol"]          = stray.get("symbol", f"NIFTY {ot}")
                result["trade"]           = {
                    "status": "success",
                    "orderid": "RECONCILED",
                    "symbol": stray.get("symbol", ""),
                }
                return result
            # else: no stray position found -- IOC-1 genuinely did not fill,
            # safe to proceed to the IOC-2 retry below exactly as before.

        logger.info(
            f"Smart exec: IOC attempt 1 ({limit_price:.2f}) not filled "
            f"-> retry at +3 ticks"
        )

        # Attempt 2: best_ask + 3 ticks  (if ask known) else limit_price + 2 ticks
        retry_price = round(
            (_best_ask + 3 * _TICK) if _best_ask > 0 else (limit_price + 2 * _TICK),
            2,
        )
        result = _ioc_attempt(retry_price, "IOC-2")
        if _ioc_filled(result):
            result["fill_type"]       = "IOC_RETRY"
            result["limit_price"]     = retry_price        # widened limit
            result["signal_premium"]  = _signal_premium    # still the original signal price
            logger.info(f"Smart exec: IOC attempt 2 filled at {retry_price:.2f}")
            return result

        logger.warning(
            f"Smart exec: IOC attempt 2 ({retry_price:.2f}) also not filled -> ABORT entry"
        )
        return {"status": "ABORTED", "reason": "IOC_RETRY_FAILED", "fill_type": "NONE"}

    # ------------------------------------------------------------------
    # Volatility-adjusted position sizing (fractional Kelly criterion)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # V9.3: Trade count persistence — survives bot restarts
    # ------------------------------------------------------------------

    _TRADE_COUNT_FILE = "logs/.trade_count"

    def _load_trade_count(self) -> int:
        """Load today's trade count from file. Returns 0 if file is stale or missing."""
        try:
            import json
            with open(self._TRADE_COUNT_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == date.today().isoformat():
                count = data.get("count", 0)
                logger.info(f"Restored trade count from file: {count} trades today")
                return count
        except Exception:
            pass
        return 0

    def _save_trade_count(self):
        """Persist today's trade count to file."""
        try:
            import json
            with open(self._TRADE_COUNT_FILE, "w") as f:
                json.dump({"date": date.today().isoformat(), "count": self._trades_today}, f)
        except Exception as e:
            logger.debug(f"Could not save trade count: {e}")

    def _compute_actual_kelly(self) -> Optional[float]:
        """
        Compute Kelly fraction from REAL historical trades in trade journal.

        Kelly formula: f = (p * b - q) / b
            p = win rate
            q = loss rate (1 - p)
            b = avg_win / avg_loss ratio

        Returns: Kelly fraction in [0, 1], capped at 0.25 (1/4-Kelly is safer)
        Returns None if insufficient data (<20 trades).
        """
        try:
            import json
            from pathlib import Path
            jpath = Path("logs/trade_journal.jsonl")
            if not jpath.exists():
                return None
            wins, losses = [], []
            with jpath.open() as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    if j.get("event") != "EXIT":
                        continue
                    pnl = j.get("pnl_rupees", 0) or 0
                    if pnl > 0:
                        wins.append(pnl)
                    elif pnl < 0:
                        losses.append(abs(pnl))
            n = len(wins) + len(losses)
            if n < 20:
                return None
            win_rate = len(wins) / n
            avg_win = sum(wins) / max(1, len(wins))
            avg_loss = sum(losses) / max(1, len(losses))
            if avg_loss <= 0 or avg_win <= 0:
                return None
            b = avg_win / avg_loss
            kelly_full = (win_rate * b - (1 - win_rate)) / b
            # Quarter-Kelly for safety (avoid over-betting on noisy estimates)
            kelly_safe = max(0.0, min(0.25, kelly_full * 0.25))
            return kelly_safe
        except Exception:
            return None

    def _compute_position_size(self, sl_pts: float, ml_confidence: float, delta: float) -> int:
        """
        Compute number of lots using volatility-adjusted fractional Kelly.

        Formula:
            risk_amount   = equity * max_risk_pct          (e.g. 500000 * 0.02 = 10000)
            loss_per_lot  = sl_pts * delta * lot_size      (e.g. 47 * 0.50 * 65 = 1527.5)
            raw_lots      = risk_amount / loss_per_lot     (e.g. 10000 / 1527.5 = 6.5)
            kelly_edge    = (ml_confidence - 0.50) * 2     (0→1 scale, 50%=no edge, 100%=max)
            kelly_lots    = raw_lots * kelly_fraction * (1 + kelly_edge)
            final_lots    = clamp(round(kelly_lots), min_lots, max_lots)

        Returns:
            Number of lots (integer), always between min_lots and max_lots.
        """
        cfg = self.config
        equity = cfg.account_equity
        risk_amount = equity * cfg.max_risk_pct

        # Premium loss per lot when SL is hit
        loss_per_lot = sl_pts * delta * cfg.lot_size
        if loss_per_lot <= 0:
            logger.warning("Position sizing: loss_per_lot <= 0, defaulting to 1 lot")
            return cfg.min_lots

        # Raw lots from pure risk sizing (how many lots can we afford to lose?)
        raw_lots = risk_amount / loss_per_lot

        # Kelly scaling: higher ML confidence → closer to full allocation
        kelly_edge = max(0.0, min(1.0, (ml_confidence - 0.50) * 2))

        # 2026-05-15 TRUE KELLY: Use actual historical WR/RR from trade journal
        # if we have enough data. Falls back to confidence-based edge when not.
        try:
            actual_kelly = self._compute_actual_kelly()
            if actual_kelly is not None:
                # Blend: 70% actual Kelly + 30% confidence edge (gradual transition)
                blended = 0.7 * actual_kelly + 0.3 * kelly_edge
                kelly_edge = max(0.0, min(1.0, blended))
                logger.info(
                    f"True Kelly: actual={actual_kelly:.2f}, conf_edge_only={kelly_edge:.2f}"
                )
        except Exception as _e:
            logger.debug(f"True Kelly calculation failed (fallback): {_e}")

        kelly_lots = raw_lots * cfg.kelly_fraction * (1 + kelly_edge)

        # Clamp to [min_lots, max_lots]
        final_lots = max(cfg.min_lots, min(cfg.max_lots, round(kelly_lots)))

        # ── Phase 5 fix (2026-06-09): MAX_LOSS_PER_TRADE hard enforcement ──
        # Previously this setting was stored in RiskManager but never checked
        # at sizing time, causing losses up to 3× the ₹2,000 limit
        # (T16=−₹6,461, T19=−₹6,019, T20=−₹4,037, T5=−₹4,891, T14=−₹4,144).
        # Now: reduce lots until loss_per_lot × lots ≤ MAX_LOSS_PER_TRADE.
        # If even 1 lot would exceed the limit, return 0 → caller skips trade.
        # Read from RiskManager (the actual source of truth for this limit)
        # instead of re-reading the env var directly — was a duplicated,
        # could-drift-out-of-sync config path.
        try:
            _max_loss = float(getattr(self.trader.risk, "max_loss_per_trade", 2000.0))
            if _max_loss > 0 and loss_per_lot > 0:
                _max_lots_safe = int(_max_loss / loss_per_lot)
                if _max_lots_safe < cfg.min_lots:
                    logger.warning(
                        f"Position sizing: MAX_LOSS_PER_TRADE=₹{_max_loss:.0f} BLOCK — "
                        f"1 lot would lose ₹{loss_per_lot:.0f} "
                        f"(sl={sl_pts:.0f}pts × delta={delta:.2f} × lot_size={cfg.lot_size}) "
                        f"> limit. Returning 0 → trade will be SKIPPED."
                    )
                    return 0   # sentinel: caller must skip this trade
                elif _max_lots_safe < final_lots:
                    logger.warning(
                        f"Position sizing: MAX_LOSS_PER_TRADE=₹{_max_loss:.0f} — "
                        f"reducing {final_lots} lots → {_max_lots_safe} lots "
                        f"(loss_per_lot=₹{loss_per_lot:.0f})"
                    )
                    final_lots = _max_lots_safe
        except Exception as _e:
            logger.debug(f"MAX_LOSS_PER_TRADE check failed (fail-open): {_e}")

        # V10 #8 — Loss-streak size scaling: after N consecutive losers, halve
        # the next M trades. Smarter than blunt flood-exit which locks losses
        # at market in a panic move.
        halved_note = ""
        if getattr(self, "_size_halved_remaining", 0) > 0:
            pre_halve = final_lots
            final_lots = max(cfg.min_lots, final_lots // 2)
            self._size_halved_remaining -= 1
            halved_note = (
                f" [LOSS-STREAK HALVED: {pre_halve}→{final_lots}, "
                f"{self._size_halved_remaining} more recovery trades]"
            )

        logger.info(
            f"Position sizing: equity={equity:.0f} risk={risk_amount:.0f} "
            f"loss/lot={loss_per_lot:.1f} raw={raw_lots:.1f} "
            f"kelly_edge={kelly_edge:.2f} kelly_lots={kelly_lots:.1f} → {final_lots} lots "
            f"({final_lots * cfg.lot_size} qty){halved_note}"
        )
        return int(final_lots)

    def _compute_futures_position_size(
        self, sl_pts: float, ml_confidence: float, spot: float,
    ) -> int:
        """
        Futures counterpart to _compute_position_size(). Only used when
        cfg.execution_mode == "futures" (default "options" — this method
        is unreachable in today's live behavior).

        Same risk-based Kelly core as the options sizer, with two real
        differences:
          - delta is fixed at 1.0 (a futures point move is a 1:1 P&L point
            move — no option Greeks discount), so loss_per_lot only needs
            sl_pts * lot_size, not sl_pts * delta * lot_size.
          - an ADDITIONAL margin-capital constraint options never needed:
            options only cost the premium (small, already implicitly
            bounded by the risk sizing above); futures require SPAN+
            exposure MARGIN per lot, which for NIFTY is roughly 10-15% of
            full notional (spot * lot_size) — order-of-magnitude more
            capital per lot than an option's premium. Uses a conservative
            offline ESTIMATE (cfg.futures_margin_pct_estimate) since this
            repo has no confirmed live margin-calculator integration yet
            (see get_funds() below — its response format isn't parsed
            anywhere in this codebase, so it's logged for confirmation
            rather than trusted as a hard gate until verified against a
            real account).
        """
        cfg = self.config
        equity = cfg.account_equity
        risk_amount = equity * cfg.max_risk_pct

        loss_per_lot = sl_pts * cfg.lot_size  # delta=1.0 for futures
        if loss_per_lot <= 0:
            logger.warning("Futures position sizing: loss_per_lot <= 0, defaulting to 1 lot")
            return cfg.min_lots

        raw_lots = risk_amount / loss_per_lot
        kelly_edge = max(0.0, min(1.0, (ml_confidence - 0.50) * 2))
        try:
            actual_kelly = self._compute_actual_kelly()
            if actual_kelly is not None:
                kelly_edge = max(0.0, min(1.0, 0.7 * actual_kelly + 0.3 * kelly_edge))
        except Exception as _e:
            logger.debug(f"Futures sizing: True Kelly calc failed (fallback): {_e}")

        kelly_lots = raw_lots * cfg.kelly_fraction * (1 + kelly_edge)
        final_lots = max(cfg.min_lots, min(cfg.max_lots, round(kelly_lots)))

        # MAX_LOSS_PER_TRADE hard enforcement — same rule as options sizing.
        try:
            _max_loss = float(getattr(self.trader.risk, "max_loss_per_trade", 2000.0))
            if _max_loss > 0 and loss_per_lot > 0:
                _max_lots_safe = int(_max_loss / loss_per_lot)
                if _max_lots_safe < cfg.min_lots:
                    logger.warning(
                        f"Futures position sizing: MAX_LOSS_PER_TRADE=₹{_max_loss:.0f} "
                        f"BLOCK — 1 lot would lose ₹{loss_per_lot:.0f} "
                        f"(sl={sl_pts:.0f}pts × lot_size={cfg.lot_size}) > limit. "
                        f"Returning 0 → trade will be SKIPPED."
                    )
                    return 0
                elif _max_lots_safe < final_lots:
                    final_lots = _max_lots_safe
        except Exception as _e:
            logger.debug(f"Futures sizing: MAX_LOSS_PER_TRADE check failed (fail-open): {_e}")

        # ── Margin-capital constraint (offline estimate) ────────────────
        # No live per-order margin calculator confirmed available — this
        # is a deliberately conservative % of notional, not a broker quote.
        notional_per_lot = spot * cfg.lot_size
        margin_per_lot_estimate = notional_per_lot * cfg.futures_margin_pct_estimate
        if margin_per_lot_estimate > 0:
            max_lots_by_margin = int(equity / margin_per_lot_estimate)
            if max_lots_by_margin < cfg.min_lots:
                logger.warning(
                    f"Futures position sizing: MARGIN BLOCK — 1 lot needs an "
                    f"estimated ₹{margin_per_lot_estimate:.0f} margin "
                    f"({cfg.futures_margin_pct_estimate:.0%} of ₹{notional_per_lot:.0f} "
                    f"notional) > equity ₹{equity:.0f}. Returning 0 → trade SKIPPED."
                )
                return 0
            elif max_lots_by_margin < final_lots:
                logger.warning(
                    f"Futures position sizing: MARGIN — reducing {final_lots} lots → "
                    f"{max_lots_by_margin} lots (est. margin/lot=₹{margin_per_lot_estimate:.0f})"
                )
                final_lots = max_lots_by_margin

        # Best-effort live funds check — logged only, NOT enforced yet.
        # Confirm the real field names against a live account before this
        # can safely become a hard gate; wrong field name silently
        # reading 0/None would be worse than not checking at all.
        try:
            funds = self.trader.get_funds_available()
            logger.info(f"Futures position sizing: live get_funds() response (unparsed): {funds}")
        except Exception as _e:
            logger.debug(f"Futures sizing: live funds check unavailable: {_e}")

        if getattr(self, "_size_halved_remaining", 0) > 0:
            final_lots = max(cfg.min_lots, final_lots // 2)
            self._size_halved_remaining -= 1

        logger.info(
            f"Futures position sizing: equity={equity:.0f} risk={risk_amount:.0f} "
            f"loss/lot={loss_per_lot:.1f} margin_est/lot={margin_per_lot_estimate:.0f} "
            f"→ {final_lots} lots ({final_lots * cfg.lot_size} qty)"
        )
        return int(final_lots)

    # ------------------------------------------------------------------
    # Position monitor — runs in separate thread, checks SL/TP/trail
    # ------------------------------------------------------------------

    _MONITOR_INTERVAL = 1      # seconds — tightest safe poll rate for REST fallback
    _PREMIUM_POLL_INTERVAL = 3 # seconds — separate thread polls premium every 3s

    def _premium_poller_loop(self):
        """
        Background daemon: polls option LTP every _PREMIUM_POLL_INTERVAL seconds
        and writes to _premium_cache. Completely isolated from the SL/TP monitor.

        Phase-1 blocking-I/O fix: this decouples the REST call for premium from
        the 1-second SL/TP evaluation loop so a Kotak API stall can never
        delay stop-loss execution.

        Uses call_with_timeout to guarantee the REST call never blocks longer
        than _PREMIUM_POLL_TIMEOUT_SEC. On timeout the cache is not updated;
        the monitor loop detects the stale timestamp and disables premium-stop
        gracefully while spot-stop continues to fire normally.
        """
        _TIMEOUT = 2.0   # REST budget: Kotak quota allows 8 req/s; 2s is generous

        while self._running:
            try:
                with self._lock:
                    pos = self._live_position

                if pos is None or not self._is_market_hours():
                    time.sleep(self._PREMIUM_POLL_INTERVAL)
                    continue
                if pos.state != PositionState.OPEN:
                    time.sleep(self._PREMIUM_POLL_INTERVAL)
                    continue
                if not pos.entry_premium or not pos.symbol or pos.symbol == "PAPER":
                    time.sleep(self._PREMIUM_POLL_INTERVAL)
                    continue

                symbol = pos.symbol

                # ── Timed REST call — never blocks the caller ────────────────
                from core.watchdog_helper import call_with_timeout as _cwt
                quote = _cwt(
                    fn=lambda: self.trader.client.get_quote(symbol, self.trader.exchange),
                    timeout_sec=_TIMEOUT,
                    default=None,
                    name="premium_poll",
                )

                if quote is not None:
                    ltp = 0.0
                    if isinstance(quote.get("data"), dict):
                        ltp = float(quote["data"].get("ltp", 0) or 0)
                    elif isinstance(quote.get("ltp"), (int, float)):
                        ltp = float(quote["ltp"])

                    if ltp > 0:
                        with self._premium_cache_lock:
                            self._premium_cache["premium"] = ltp
                            self._premium_cache["ts"]      = time.monotonic()
                            self._premium_cache["symbol"]  = symbol

            except Exception as _e:
                logger.debug(f"PremiumPoller: unhandled error ({_e}) — continuing")

            time.sleep(self._PREMIUM_POLL_INTERVAL)

    def _position_monitor_loop(self):
        """
        High-frequency position monitor: polls at 1-second intervals.

        Spot price strategy (fastest wins):
          1. WebSocket tick  — if _ws_spot was updated within 2s, use it (sub-second fresh)
          2. REST API        — fallback when WS is not connected / NATIVE_OHLCV=false

        Premium polling is still REST (no WS stream for option LTP in this version).
        """
        while self._running:
            try:
                with self._lock:
                    pos = self._live_position
                if pos is None:
                    time.sleep(self._MONITOR_INTERVAL)
                    continue
                # Operational-safety fix #4 (2026-07-21): do NOT gate
                # monitoring on _is_market_hours() when a position is open.
                # Previously, once market hours expired (e.g. a square-off
                # retry pushed past 15:15:00), this check would go False on
                # the very next iteration and the loop would skip straight
                # past all the exit/retry logic below, abandoning an
                # in-progress close retry with the position left open and
                # unmonitored. An open position must keep being retried for
                # exit/square-off regardless of market hours -- market-hours
                # is enforced for NEW entries elsewhere, not here.

                # Skip if position is being closed (CLOSING state)
                if pos.state != PositionState.OPEN:
                    time.sleep(self._MONITOR_INTERVAL)
                    continue

                # ── Spot price (Phase-1: timeout-guarded REST fallback) ───────────
                # Primary:  WebSocket tick if <2s old (zero I/O, always fast)
                # Fallback: REST call with hard 1-second cap via call_with_timeout
                _cycle_start = time.monotonic()
                try:
                    _ws_age = time.monotonic() - self._ws_spot_ts
                    if self._ws_spot > 0 and _ws_age < 2.0:
                        spot = self._ws_spot                        # 0ms — WS path
                    else:
                        from core.watchdog_helper import call_with_timeout as _cwt
                        spot = _cwt(
                            fn=self.trader.get_nifty_spot,
                            timeout_sec=1.0,                        # hard cap
                            default=0.0,
                            name="monitor_get_spot",
                        )
                        if not spot:
                            time.sleep(self._MONITOR_INTERVAL)
                            continue
                except Exception:
                    time.sleep(self._MONITOR_INTERVAL)
                    continue

                # ── Premium — read from cache (Phase-1: never blocks SL/TP) ──────
                # _premium_poller_loop updates _premium_cache every 3s in background.
                # If cache is stale (>30s) or for wrong symbol, premium-stop is
                # disabled for this cycle. Spot-based SL/TP fires normally regardless.
                premium_stop_hit = False
                _cache_premium = 0.0
                _premium_feed_valid = False
                if pos.entry_premium > 0 and pos.symbol and pos.symbol != "PAPER":
                    with self._premium_cache_lock:
                        _cached = dict(self._premium_cache)
                    _cache_age = time.monotonic() - _cached["ts"]
                    if (
                        _cached["premium"] > 0
                        and _cached["symbol"] == pos.symbol
                        and _cache_age <= self._PREMIUM_CACHE_MAX_AGE_SEC
                    ):
                        _cache_premium = _cached["premium"]
                        _premium_feed_valid = True
                        pos.current_premium = _cache_premium   # update for trail/BE logic
                        if _cache_premium > pos.peak_premium:
                            pos.peak_premium = _cache_premium

                    if _premium_feed_valid and _cache_premium > 0:
                        premium_drop_pct = (pos.entry_premium - _cache_premium) / pos.entry_premium
                        # V10: Premium hard stop now REQUIRES directional confirmation.
                        # Old logic locked losses on IV crush noise (gap-down, news → premium
                        # drops 30% in 5 min then recovers 80%). Now we only fire if the
                        # underlying has ALSO moved against us by >= 0.5 × ATR.
                        _spot_unreal = (spot - pos.entry_price) if pos.direction == "CALL" \
                                                                   else (pos.entry_price - spot)
                        _atr = pos.atr_at_entry or 0.0
                        _dir_against = (_atr > 0 and _spot_unreal <= -0.5 * _atr)
                        if (premium_drop_pct >= pos.premium_hard_stop_pct) and _dir_against:
                            premium_stop_hit = True
                            logger.warning(
                                f"PREMIUM HARD STOP: entry={pos.entry_premium:.1f} "
                                f"now={_cache_premium:.1f} "
                                f"drop={premium_drop_pct:.0%} >= {pos.premium_hard_stop_pct:.0%} "
                                f"AND spot moved {_spot_unreal:+.0f}pts against (<=-0.5×ATR)"
                            )
                        elif premium_drop_pct >= 0.65:
                            # 2026-05-15 #3: Catastrophic premium decay = exit no matter what.
                            # Even if "IV-crush noise" it's too painful to wait.
                            premium_stop_hit = True
                            logger.warning(
                                f"PREMIUM CATASTROPHIC: drop={premium_drop_pct:.0%}>=65% "
                                f"regardless of spot move {_spot_unreal:+.0f}pts. Exit now."
                            )
                        elif premium_drop_pct >= pos.premium_hard_stop_pct:
                            logger.info(
                                f"PREMIUM drop={premium_drop_pct:.0%} but spot only "
                                f"{_spot_unreal:+.0f}pts (need <=-{0.5*_atr:.0f}) "
                                f"— ignoring as IV-crush noise"
                            )
                    elif not _premium_feed_valid and pos.entry_premium > 0:
                        logger.debug(
                            f"Premium cache stale/missing (age={_cache_age:.0f}s) — "
                            f"premium-stop disabled this cycle; spot-stop active"
                        )

                # --- Premium trailing stop (additive, opt-in) ---
                # Distinct from the hard drop-from-entry stop above: exits if
                # premium retraces from its OWN peak, not from entry. Skipped
                # if the hard stop already fired this cycle (avoid double-count).
                premium_trail_hit = False
                if (
                    self.config.use_premium_trailing_stop
                    and _premium_feed_valid
                    and not premium_stop_hit
                ):
                    premium_trail_hit = _premium_trail_stop_hit(
                        pos.peak_premium, pos.current_premium, self.config.premium_trail_giveback_pct
                    )
                    if premium_trail_hit:
                        logger.warning(
                            f"PREMIUM TRAIL STOP: peak={pos.peak_premium:.1f} "
                            f"now={pos.current_premium:.1f} "
                            f"giveback>={self.config.premium_trail_giveback_pct:.0%}"
                        )

                unrealized = (spot - pos.entry_price) if pos.direction == "CALL" \
                    else (pos.entry_price - spot)
                sl_dist = abs(pos.entry_price - pos.initial_sl)

                # ══════════════════════════════════════════════════════════
                # 2026-05-15 #2: PARTIAL PROFIT-TAKING at +1R
                # Exit 50% of position at +1R profit, let runner ride with BE SL.
                # ══════════════════════════════════════════════════════════
                if not pos.partial_exited and sl_dist > 0 and unrealized >= sl_dist:
                    try:
                        partial_qty = pos.original_qty // 2
                        if partial_qty > 0:
                            logger.warning(
                                f"📈 PARTIAL EXIT +1R: "
                                f"selling {partial_qty}/{pos.original_qty} qty at "
                                f"+{unrealized:.0f}pts profit. Runner continues with BE SL."
                            )
                            try:
                                if pos.symbol:
                                    # Try partial close — fall back gracefully
                                    if hasattr(self.trader, "close_position_partial"):
                                        self.trader.close_position_partial(
                                            pos.symbol, qty=partial_qty)
                            except Exception as _e:
                                logger.warning(f"Partial close call failed: {_e}")
                            pos.partial_exited = True
                            # Move SL to breakeven on the runner
                            pos.sl_price = pos.entry_price
                            pos.breakeven_activated = True
                            logger.info(
                                f"→ SL moved to breakeven {pos.entry_price:.0f} on runner"
                            )
                            try:
                                if self.notifier:
                                    self.notifier.notify_trade(
                                        action="PARTIAL_EXIT_1R",
                                        symbol=pos.symbol or f"NIFTY {pos.direction}",
                                        side="SELL",
                                        qty=partial_qty,
                                        price=spot,
                                        order_id="",
                                        status="executed",
                                        details=f"Locked +{unrealized:.0f}pts on 50%, runner @ BE",
                                    )
                            except Exception:
                                pass
                    except Exception as _e:
                        logger.warning(f"Partial-exit logic failed: {_e}")

                # ══════════════════════════════════════════════════════════
                # 2026-05-15 #6: GREEKS-BASED EXIT (delta degradation)
                # If delta dropped >40% from entry, option-pricing thesis is broken.
                # ══════════════════════════════════════════════════════════
                try:
                    if pos.entry_delta > 0 and unrealized < 0:
                        # Only check on losing positions (winners handled by TP/trail)
                        # Recompute delta on every Nth cycle (expensive call)
                        if int(time.monotonic()) % 60 == 0:   # ~1 check/min
                            from core.options_chain import get_atm_greeks
                            from core.expiry_utils import get_dte
                            from core.watchdog_helper import call_with_timeout
                            opt_type = "CE" if pos.direction == "CALL" else "PE"
                            g_now = call_with_timeout(
                                fn=lambda: get_atm_greeks(
                                    kotak_client=self.trader.client,
                                    spot=spot,
                                    dte_days=max(1, get_dte()),
                                    option_type=opt_type,
                                ),
                                timeout_sec=4.0,
                                default=None,
                                name="get_atm_greeks_monitor",
                            )
                            if g_now and g_now.get("delta") is not None:
                                delta_now = abs(g_now["delta"])
                                delta_entry = abs(pos.entry_delta)
                                if delta_entry > 0 and delta_now / delta_entry < 0.60:
                                    logger.warning(
                                        f"⚠️ DELTA DEGRADED: entry={delta_entry:.2f} "
                                        f"now={delta_now:.2f} (-{(1-delta_now/delta_entry)*100:.0f}%) "
                                        f"→ tightening SL to current spot"
                                    )
                                    # Tighten SL: cut distance in half
                                    if pos.direction == "CALL":
                                        pos.sl_price = max(pos.sl_price,
                                                           spot - sl_dist * 0.3)
                                    else:
                                        pos.sl_price = min(pos.sl_price,
                                                           spot + sl_dist * 0.3)
                except Exception as _e:
                    logger.debug(f"Greeks-exit check failed: {_e}")

                # Track peak profit & max drawdown for journal
                if unrealized > self._pos_peak_profit_pts:
                    self._pos_peak_profit_pts = unrealized
                drawdown_from_peak = self._pos_peak_profit_pts - unrealized
                if drawdown_from_peak > self._pos_max_drawdown_pts:
                    self._pos_max_drawdown_pts = drawdown_from_peak

                # --- Compute Premium-ATR (rolling range of last 12 prints ≈ 2 min) ---
                if pos.current_premium > 0:
                    pos.premium_history.append(pos.current_premium)
                    if len(pos.premium_history) > 12:
                        pos.premium_history = pos.premium_history[-12:]
                premium_atr = 0.0
                if len(pos.premium_history) >= 4:
                    premium_atr = max(pos.premium_history) - min(pos.premium_history)
                # Floor at 5% of entry premium so a quiet print sequence still gives us a usable trigger
                if pos.entry_premium > 0:
                    premium_atr = max(premium_atr, pos.entry_premium * 0.05)

                # Premium profit (in option points). Long-only — both CALL & PUT are bought.
                premium_profit = (pos.current_premium - pos.entry_premium) if pos.entry_premium > 0 else 0.0

                # --- Breakeven stop: V10 — trigger at 0.5R (half stop distance) ---
                # Old logic used a fixed 15pts trigger which fired on noise (50pt ATR
                # day → BE on a 30% pullback before move develops). 0.5R is the proven
                # asymmetric breakeven point — gives the move room to breathe.
                be_trigger_r = getattr(self.config, "breakeven_trigger_r", 0.5)
                be_trigger_spot = max(
                    self.config.breakeven_trigger_pts,   # absolute floor (legacy)
                    be_trigger_r * sl_dist,              # 0.5 × initial SL distance
                )
                # Premium primary trigger: profit >= 1.5 × premium-ATR (was 1.0 — too jumpy)
                be_trigger_prem = 1.5 * premium_atr
                # premium_feed_alive: True when we have a valid, fresh cached premium
                premium_feed_alive = _premium_feed_valid and pos.current_premium > 0 and premium_atr > 0
                be_ready = (
                    (premium_feed_alive and premium_profit >= be_trigger_prem
                                        and unrealized >= be_trigger_spot * 0.6)
                    or (not premium_feed_alive and unrealized >= be_trigger_spot)
                )
                if (self.config.use_breakeven_stop and not pos.breakeven_activated
                        and not pos.trail_activated and be_ready):
                        pos.breakeven_activated = True
                        lock = self.config.breakeven_lock_pts
                        if pos.direction == "CALL":
                            new_sl = pos.entry_price + lock
                            if new_sl > pos.sl_price:
                                old_sl = pos.sl_price
                                pos.sl_price = new_sl
                                logger.info(
                                    f"BREAKEVEN ACTIVATED: SL {old_sl:.0f} -> {new_sl:.0f} "
                                    f"(locking +{lock:.0f}pts, zero-risk)"
                                )
                        else:
                            new_sl = pos.entry_price - lock
                            if new_sl < pos.sl_price:
                                old_sl = pos.sl_price
                                pos.sl_price = new_sl
                                logger.info(
                                    f"BREAKEVEN ACTIVATED: SL {old_sl:.0f} -> {new_sl:.0f} "
                                    f"(locking +{lock:.0f}pts, zero-risk)"
                                )

                # --- Trailing stop: premium-ATR trigger w/ spot fallback ---
                # Primary:  premium_profit >= 1.5 × Premium_ATR
                # Fallback: spot unrealized >= 0.8 × SL_distance (original logic)
                trail_trigger_prem = 1.5 * premium_atr
                trail_ready = (
                    (premium_feed_alive and premium_profit >= trail_trigger_prem)
                    or (not premium_feed_alive and unrealized >= sl_dist * self.config.trail_activation_r)
                )
                if (self.config.use_trailing_stop and not pos.trail_activated and trail_ready):
                    pos.trail_activated = True
                    if premium_feed_alive:
                        logger.info(
                            f"TRAIL ACTIVATED (premium): profit=+{premium_profit:.1f} "
                            f"premium_ATR={premium_atr:.1f} trigger=1.5×ATR={trail_trigger_prem:.1f}"
                        )
                    else:
                        logger.info(
                            f"TRAIL ACTIVATED (spot fallback — premium feed dead): "
                            f"unrealized={unrealized:+.0f}pts threshold={sl_dist*self.config.trail_activation_r:.0f}pts"
                        )

                if pos.trail_activated:
                    # Move SL to lock in trail_step_pct of unrealized profit
                    lock_amount = unrealized * self.config.trail_step_pct
                    if pos.direction == "CALL":
                        new_sl = pos.entry_price + lock_amount
                        if new_sl > pos.sl_price:
                            old_sl = pos.sl_price
                            pos.sl_price = new_sl
                            logger.info(
                                f"TRAIL SL moved: {old_sl:.0f} -> {new_sl:.0f} "
                                f"(locking {lock_amount:.0f}pts profit)"
                            )
                    else:
                        new_sl = pos.entry_price - lock_amount
                        if new_sl < pos.sl_price:
                            old_sl = pos.sl_price
                            pos.sl_price = new_sl
                            logger.info(
                                f"TRAIL SL moved: {old_sl:.0f} -> {new_sl:.0f} "
                                f"(locking {lock_amount:.0f}pts profit)"
                            )

                # --- ATR-based trailing stop (additive, opt-in) ---
                # Takes whichever of {profit-lock trail above, ATR-multiple
                # trail} is MORE protective — never loosens the SL.
                if pos.trail_activated and self.config.use_atr_trailing_stop and self._current_atr > 0:
                    atr_candidate = _atr_trail_sl_candidate(
                        pos.direction, spot, self._current_atr, self.config.atr_trail_multiplier
                    )
                    if pos.direction == "CALL":
                        if atr_candidate > pos.sl_price:
                            old_sl = pos.sl_price
                            pos.sl_price = atr_candidate
                            logger.info(
                                f"ATR TRAIL SL moved: {old_sl:.0f} -> {atr_candidate:.0f} "
                                f"({self.config.atr_trail_multiplier}x ATR={self._current_atr:.1f} behind spot)"
                            )
                    else:
                        if atr_candidate < pos.sl_price:
                            old_sl = pos.sl_price
                            pos.sl_price = atr_candidate
                            logger.info(
                                f"ATR TRAIL SL moved: {old_sl:.0f} -> {atr_candidate:.0f} "
                                f"({self.config.atr_trail_multiplier}x ATR={self._current_atr:.1f} behind spot)"
                            )

                # --- Trailing TP logic ---
                # When price reaches 75% of TP, extend TP if momentum continues
                if self.config.use_trailing_tp and unrealized > 0:
                    # Track peak unrealized
                    if unrealized > pos.peak_unrealized:
                        pos.peak_unrealized = unrealized

                    # Original TP distance
                    if pos.original_tp == 0.0:
                        pos.original_tp = pos.tp_price
                    orig_tp_dist = abs(pos.original_tp - pos.entry_price)

                    # Activate trailing TP when price reaches 75% of original TP
                    activation_threshold = orig_tp_dist * self.config.trail_tp_activation_pct
                    if unrealized >= activation_threshold:
                        # Extend TP: new_tp = entry + peak_unrealized + 50% of remaining momentum
                        extend = pos.peak_unrealized * self.config.trail_tp_extend_pct
                        if pos.direction == "CALL":
                            new_tp = pos.entry_price + pos.peak_unrealized + extend
                            if new_tp > pos.tp_price:
                                old_tp = pos.tp_price
                                pos.tp_price = new_tp
                                if not pos.tp_trailing:
                                    pos.tp_trailing = True
                                    logger.info(
                                        f"TRAIL TP ACTIVATED: TP {old_tp:.0f} -> {new_tp:.0f} "
                                        f"(momentum extending, peak={pos.peak_unrealized:.0f}pts)"
                                    )
                                else:
                                    logger.debug(
                                        f"TRAIL TP moved: {old_tp:.0f} -> {new_tp:.0f}"
                                    )
                        else:
                            new_tp = pos.entry_price - pos.peak_unrealized - extend
                            if new_tp < pos.tp_price:
                                old_tp = pos.tp_price
                                pos.tp_price = new_tp
                                if not pos.tp_trailing:
                                    pos.tp_trailing = True
                                    logger.info(
                                        f"TRAIL TP ACTIVATED: TP {old_tp:.0f} -> {new_tp:.0f} "
                                        f"(momentum extending, peak={pos.peak_unrealized:.0f}pts)"
                                    )
                                else:
                                    logger.debug(
                                        f"TRAIL TP moved: {old_tp:.0f} -> {new_tp:.0f}"
                                    )

                # --- Check SL hit ---
                sl_hit = (pos.direction == "CALL" and spot <= pos.sl_price) or \
                         (pos.direction == "PUT" and spot >= pos.sl_price)

                # --- Check TP hit ---
                tp_hit = (pos.direction == "CALL" and spot >= pos.tp_price) or \
                         (pos.direction == "PUT" and spot <= pos.tp_price)

                # --- Check time-based exit (15:14:30 — beat broker auto-square-off) ---
                now = datetime.now()
                time_exit = now.hour == 15 and (now.minute > 14 or (now.minute == 14 and now.second >= 30))
                # V10: On expiry day, force-close ALL positions at 14:30 — gamma & theta
                # past this point destroy P&L faster than any stop can react.
                try:
                    from core.expiry_utils import is_expiry_day
                    if is_expiry_day() and (now.hour > 14 or (now.hour == 14 and now.minute >= 30)):
                        if not time_exit:
                            logger.warning("EXPIRY DAY 14:30 — force-closing position (gamma/theta zone)")
                        time_exit = True
                except Exception:
                    pass

                # Independent-verification fix #6 (2026-07-21): make the
                # square-off trigger STICKY via the pure _sticky_time_exit
                # combinator (unit-tested in isolation) -- a close retry
                # running past an hour rollover (e.g. 15:59->16:00, where
                # `now.hour == 15` stops matching) must never silently
                # un-trigger a forced square-off mid-retry.
                pos.square_off_latched = _sticky_time_exit(time_exit, pos.square_off_latched)
                time_exit = pos.square_off_latched

                # --- Check max-hold-duration exit (additive, opt-in) ---
                max_hold_hit = _max_hold_exceeded(pos.entry_time, self.config.max_hold_minutes)
                if max_hold_hit:
                    logger.info(
                        f"MAX HOLD TIME exceeded ({self.config.max_hold_minutes}min) — forcing exit"
                    )

                if sl_hit or tp_hit or time_exit or premium_stop_hit or premium_trail_hit or max_hold_hit:
                    if premium_stop_hit:
                        reason = "PREMIUM_STOP"
                    elif premium_trail_hit:
                        reason = "PREMIUM_TRAIL_STOP"
                    elif sl_hit:
                        if pos.trail_activated:
                            reason = "TRAIL_SL"
                        elif pos.breakeven_activated:
                            reason = "BE_SL"
                        else:
                            reason = "SL"
                    elif tp_hit:
                        reason = "TP"
                    elif max_hold_hit:
                        reason = "MAX_HOLD_TIME"
                    else:
                        reason = "TIME_EXIT"

                    # Round-3 independent-verification CRITICAL fix
                    # (2026-07-21): acquire exclusive close ownership FIRST,
                    # before any other side effect (day-halt flag,
                    # anti-whipsaw recording, P&L, broker call). If this
                    # cycle loses the race -- manual_close(), emergency_stop(),
                    # or a concurrent close attempt already owns this exact
                    # position -- do absolutely nothing else: no broker call,
                    # no RiskManager update, no journal write, no state
                    # mutation. The owner is fully responsible for the entire
                    # close+accounting sequence.
                    if not self._try_acquire_close_ownership(pos, "monitor"):
                        logger.info(
                            f"Monitor: lost close ownership for {reason} on "
                            f"{pos.direction} — another close is already in "
                            f"flight, skipping this cycle"
                        )
                        time.sleep(self._MONITOR_INTERVAL)
                        continue

                    pnl_pts = unrealized
                    logger.info(
                        f"POSITION EXIT: {reason} | {pos.direction} "
                        f"entry={pos.entry_price:.0f} exit={spot:.0f} "
                        f"P&L={pnl_pts:+.0f}pts"
                    )

                    # Round-3 canonicalization fix (2026-07-21): anti-whipsaw
                    # / day-halt-after-loss / loss-streak / calibrator / P&L /
                    # journal / cleanup / notification all moved into
                    # _finish_successful_exit() -- see below, only invoked
                    # AFTER the close is broker-confirmed (previously these
                    # two flags fired eagerly here, before confirmation; now
                    # correctly gated on a real close, and no longer
                    # duplicated across manual_close()/emergency_stop()).

                    # Close position via broker (paper positions skip this).
                    # Extracted to _attempt_protected_close() (operational-
                    # safety fix #2, 2026-07-21) — identical statements/order,
                    # moved into its own method purely so it's directly unit-
                    # testable without mocking the whole monitor loop.
                    _close_succeeded, _close_exc = self._attempt_protected_close(pos, reason)

                    if not _close_succeeded:
                        # Broker did not confirm the close. Do NOT record a
                        # false exit (P&L/journal/CLOSED below assume a real
                        # close happened). Revert to OPEN so this exact exit
                        # condition (SL/TP/time-exit) re-evaluates and retries
                        # the close on the next 1s monitor cycle — no new
                        # retry logic needed, this loop already runs every 1s.
                        # Fix #2: the exchange-resident SL was NEVER cancelled
                        # in this path, so the position remains genuinely
                        # protected while the retry continues.
                        logger.critical(
                            "CLOSE_FAILED_POSITION_PROTECTED",
                            extra={
                                "event": "close_failed_position_protected",
                                "reason": reason,
                                "direction": pos.direction,
                                "symbol": pos.symbol or "",
                                "sl_order_id": pos.sl_order_id or "",
                                "error": str(_close_exc) if _close_exc else "",
                            },
                        )
                        logger.critical(
                            f"🚨 CLOSE FAILED — {reason} on {pos.direction} "
                            f"{pos.symbol or '(no symbol)'} — broker did not "
                            f"confirm the close. Position remains OPEN; the "
                            f"exchange-resident SL was NEVER cancelled (fix #2) "
                            f"and remains active, so this position IS STILL "
                            f"PROTECTED. Retrying close every cycle."
                        )
                        try:
                            self.notifier.notify_trade(
                                action="CLOSE_FAILED",
                                symbol=pos.symbol or f"NIFTY {pos.direction}",
                                side="SELL",
                                qty=self.trader.default_qty,
                                price=spot,
                                order_id="",
                                status="failed",
                                details=(
                                    f"Broker did not confirm close on {reason} — "
                                    f"position still open, but the exchange SL "
                                    f"is still ACTIVE (not cancelled until "
                                    f"close confirms). Retrying every cycle."
                                ),
                            )
                        except Exception as _e:
                            logger.debug(f"Close-failure notification failed: {_e}")
                        # Release ownership (CLOSING -> OPEN) instead of an
                        # unconditional revert, so a stale/losing caller can
                        # never accidentally reopen a position someone else
                        # has since taken ownership of.
                        self._release_close_ownership(pos, "monitor")
                        time.sleep(self._MONITOR_INTERVAL)
                        continue

                    # Round-3 canonicalization fix (2026-07-21): every
                    # post-close side effect (RiskManager accounting,
                    # analytics, journal, cleanup, notification) now lives
                    # in exactly one place — see _finish_successful_exit()
                    # — instead of being duplicated (and able to drift)
                    # across the normal exit path, manual_close(), and
                    # emergency_stop().
                    self._finish_successful_exit(pos, reason, spot, pnl_pts)

            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            # ── Latency watchdog metric ─────────────────────────────────────
            # Records cycle duration so operators can verify the <100ms target.
            # Log a warning if cycle exceeded 200ms (data feed issue, not I/O).
            try:
                _cycle_ms = (time.monotonic() - _cycle_start) * 1000
                self._monitor_last_cycle_ms = _cycle_ms
                if _cycle_ms > 200:
                    logger.warning(
                        f"Monitor cycle slow: {_cycle_ms:.0f}ms > 200ms target "
                        f"(spot_source={'WS' if self._ws_spot > 0 and (time.monotonic()-self._ws_spot_ts)<2 else 'REST'})"
                    )
            except Exception:
                pass

            time.sleep(self._MONITOR_INTERVAL)

    # ------------------------------------------------------------------
    # ML prediction helper
    # ------------------------------------------------------------------

    def _run_ml_prediction(self, spot):
        """
        Build a full multi-timeframe bar buffer and run V8 ML prediction.
        Fetches 5-min, 15-min, 30-min, 60-min and daily bars from TradingView
        and merges them exactly as done during V8 training.
        """
        import pandas as pd
        from core.ml_engine import predict, predict_precomputed
        from core.tv_fetcher import get_tv_fetcher
        from core.ml_engine import _fcols

        try:
            tv = get_tv_fetcher()

            # ── Fetch all timeframes ──────────────────────────────
            # TV returns ~120 live bars = 10 hours.  Some features need a
            # 252-bar rolling window (e.g. vol_percentile = rv20.rolling(252)).
            # 120 bars < 252 → those features are always NaN for the first ~11h.
            # Fix: prepend historical bars from nifty_5min.csv so the rolling
            # window has enough warmup data regardless of session time.
            df5 = tv.get_nifty_5min(n_bars=120)
            try:
                import pandas as _pd
                from pathlib import Path as _Path
                _csv5 = _Path("data/nifty_5min.csv")
                if _csv5.exists() and not df5.empty:
                    _WARMUP = 300   # need at least 252+20 bars for vol_percentile
                    _hist = _pd.read_csv(_csv5, index_col=0, parse_dates=True)
                    _hist.columns = [c.lower() for c in _hist.columns]
                    # Only use bars BEFORE the first live bar to avoid duplicates
                    _cutoff = df5.index[0]
                    _hist = _hist[_hist.index < _cutoff].tail(_WARMUP)
                    if not _hist.empty:
                        df5 = _pd.concat([_hist, df5]).sort_index()
                        df5 = df5[~df5.index.duplicated(keep="last")]
                        logger.debug(
                            f"5m warmup: prepended {len(_hist)} historical bars "
                            f"({_hist.index[0].date()} → {_hist.index[-1].date()}) "
                            f"→ total {len(df5)} bars for rolling features"
                        )
            except Exception as _we:
                logger.debug(f"5m warmup prepend failed (non-critical): {_we}")

            df15 = tv.get_nifty_15min(n_bars=50)
            df30 = tv.get_nifty_30min(n_bars=30)
            df60 = tv.get_nifty_60min(n_bars=20)

            # Daily — fetch via TradingView
            df_day = tv.get_ohlcv("NIFTY", "NSE", interval_minutes="D", n_bars=30)

            # India VIX — daily
            df_vix = pd.DataFrame()
            try:
                # Need 35 daily bars: vix_rank_30d requires 30 days,
                # vix_pct20/vix_zscore_20d require 20 days. 10 was too few.
                raw_vix = tv.get_ohlcv("INDIAVIX", "NSE", interval_minutes="D", n_bars=35)
                if raw_vix is None or raw_vix.empty:
                    raw_vix = tv.get_ohlcv("INDIA VIX", "NSE", interval_minutes="D", n_bars=35)
                if raw_vix is not None and not raw_vix.empty:
                    df_vix = raw_vix.rename(columns={"close": "vix"})[["vix"]]
                    logger.debug(f"VIX live: {df_vix['vix'].iloc[-1]:.2f}")
            except Exception as e:
                logger.debug(f"VIX live fetch failed: {e}")

            # ── BAR QUALITY GATE (Issue #5) ───────────────────────────────
            # Check raw OHLCV bars before computing any features.
            # A stale, gap-ridden, or frozen feed produces wrong indicator
            # values that silently corrupt the ML input vector.
            try:
                from core.data_quality import check_bar_data
                _dq_bars = check_bar_data(df5, df15, df30, df60)
                if _dq_bars.severity == "block":
                    logger.warning(
                        f"DATA QUALITY BLOCK (bars): {_dq_bars.summary()} "
                        f"| metrics={_dq_bars.metrics}"
                    )
                    return 2, [0.0, 0.0, 1.0], 0.0, {
                        "dq_block": True, "dq_issues": _dq_bars.issues
                    }
                if _dq_bars.issues:
                    logger.warning(f"DATA QUALITY WARN (bars): {_dq_bars.summary()}")
            except Exception as _dq_e:
                logger.debug(f"Bar quality check skipped (fail-open): {_dq_e}")

            if df5.empty or len(df5) < 55:
                logger.debug("Not enough 5-min bars for V8 prediction")
                return 2, [0.0, 0.0, 1.0], 0.0, {}

            # ── Import feature engineering matching the loaded model ──
            # CRITICAL: must match the script that produced the .pkl in
            # ml_engine. V9 model trained on V9 features ≠ V8 features —
            # mixing them silently zero-fills missing columns and the
            # model collapses to ~33% probabilities (the SKIP basin).
            import sys
            from pathlib import Path
            root = Path(__file__).parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

            # V9 feature engineering is used for both V9 and V10 models.
            # V10 = V9 base features (175) + 13 option chain features (188 total).
            # V8 is no longer trained or deployed — always use V9 engineering.
            from scripts.train_model_v9 import (
                features_5min, features_15min, features_30min,
                features_60min, features_daily, features_vix,
                merge_htf, merge_daily_onto_5min,
                merge_vix_onto_5min, add_intraday_context
            )

            # ── Compute features ──────────────────────────────────
            df_feat = features_5min(df5)

            if not df15.empty:
                feat15 = features_15min(df15)
                df_feat = merge_htf(df_feat, feat15, "15m")

            if not df30.empty:
                feat30 = features_30min(df30)
                df_feat = merge_htf(df_feat, feat30, "30m")

            if not df60.empty:
                feat60 = features_60min(df60)
                df_feat = merge_htf(df_feat, feat60, "60m")

            if not df_day.empty:
                df_day_clean = df_day.copy()
                df_day_clean = df_day_clean[~df_day_clean.index.duplicated(keep="last")]
                feat_day = features_daily(df_day_clean)
                df_feat  = merge_daily_onto_5min(df_feat, feat_day)

            # Merge VIX if available
            if not df_vix.empty:
                try:
                    feat_vix = features_vix(df_vix)
                    df_feat  = merge_vix_onto_5min(df_feat, feat_vix)
                except Exception as e:
                    logger.debug(f"VIX merge failed: {e}")

            df_feat = add_intraday_context(df_feat)
            # Forward-fill NaN from rolling warmup (RSI, ATR, ADX, etc.)
            # Don't dropna — it kills ALL rows when warmup NaN overlaps across timeframes
            df_feat.ffill(inplace=True)

            # ── FEATURE QUALITY GATE (Issue #5) ──────────────────────────
            # Check AFTER ffill but BEFORE fillna(0) so genuine NaN from
            # data gaps are still visible. fillna(0) hides them — a 0 RSI
            # or 0 ATR silently poisons the model input.
            try:
                from core.data_quality import check_feature_matrix
                _dq_feat = check_feature_matrix(df_feat)
                if _dq_feat.severity == "block":
                    logger.warning(
                        f"DATA QUALITY BLOCK (features): {_dq_feat.summary()} "
                        f"| metrics={_dq_feat.metrics}"
                    )
                    return 2, [0.0, 0.0, 1.0], 0.0, {
                        "dq_block": True, "dq_issues": _dq_feat.issues
                    }
                if _dq_feat.issues:
                    logger.warning(f"DATA QUALITY WARN (features): {_dq_feat.summary()}")
                    # Log the exact NaN feature names so we can diagnose on Hostinger
                    try:
                        _nan_names = [
                            c for c in df_feat.columns
                            if df_feat[c].iloc[-1] != df_feat[c].iloc[-1]  # isnan
                        ]
                        if _nan_names:
                            logger.warning(
                                f"DATA QUALITY NaN features: {_nan_names}"
                            )
                    except Exception:
                        pass
            except Exception as _dq_e:
                logger.debug(f"Feature quality check skipped (fail-open): {_dq_e}")

            # Zero-fill remaining NaN (rolling warmup tails, not gaps)
            df_feat.fillna(0, inplace=True)

            if len(df_feat) == 0:
                return 2, [0.0, 0.0, 1.0], 0.0, {}

            # ── Log how many features matched ─────────────────────
            if _fcols:
                matched  = [c for c in _fcols if c in df_feat.columns]
                missing  = [c for c in _fcols if c not in df_feat.columns]
                if missing:
                    logger.debug(f"ML: {len(missing)} features missing — filling with 0")
                    for col in missing:
                        df_feat[col] = 0.0
                logger.debug(f"ML: {len(matched)}/{len(_fcols)} features matched")

            signal, proba, conf, indicators = predict_precomputed(df_feat.tail(60))

            # ── 2026-06-12 STAGE-1 SHADOW DEPLOYMENT ──────────────────────
            # Candidate model (sandbox_tb) predicts on the same features and
            # is logged side-by-side. Passive: no decision impact, no-op if
            # artifacts absent, exceptions swallowed inside observe().
            try:
                from core.shadow_model import observe as _shadow_observe
                _shadow_observe(getattr(self, "_cycle_count", 0),
                                df_feat, signal, proba, spot)
            except Exception:
                pass

            # ── 2026-06-13 PRODUCTION SHADOW FRAMEWORK ────────────────────
            # Second shadow comparator: runs the HTF36 candidate on the same
            # features and records direction/confidence/trade-decision
            # agreement to data/shadow/. Passive: no decision impact, no-op
            # if disabled or artifacts absent, exceptions swallowed inside
            # observe(). See core/shadow_framework.py.
            try:
                from core.shadow_framework import observe as _shadow_fw_observe
                _shadow_fw_observe(getattr(self, "_cycle_count", 0),
                                   df_feat, signal, proba, conf, spot)
            except Exception:
                pass

            # ── Inject 60m trend features into indicators ─────────────────
            # Makes tf60_* available to the confidence-gate logic in the
            # main analysis loop for trend-continuation threshold adjustment
            # (used after retraining includes these features in the model).
            try:
                _last = df_feat.iloc[-1]
                for _tf60_col in [
                    "tf60_consec_bear", "tf60_consec_bull",
                    "tf60_trend_slope_3h",
                    "tf60_dist_from_3h_high", "tf60_dist_from_3h_low",
                    "tf60_ema21_slope",
                ]:
                    if _tf60_col in df_feat.columns:
                        indicators[_tf60_col] = round(float(_last[_tf60_col] or 0), 4)
            except Exception:
                pass

            return signal, proba, conf, indicators

        except Exception as e:
            logger.warning(f"Multi-TF prediction failed: {e} — falling back to 5-min only")

        # ── Fallback: 5-min only (V6 style) ──────────────────────
        try:
            df = get_tv_fetcher().get_nifty_5min(n_bars=60)
            if not df.empty and len(df) >= 55:
                return predict(df[["open", "high", "low", "close"]].tail(60))
        except Exception as e:
            logger.debug(f"5-min fallback also failed: {e}")

        # ── Last resort: synthetic bars ───────────────────────────
        if len(self.analyzer._spot_history) >= 55:
            prices = [p for _, p in self.analyzer._spot_history]
            records = [{"open": p, "high": p*1.0002, "low": p*0.9998, "close": p}
                       for p in prices]
            df = pd.DataFrame(records)
            df.index = pd.date_range(end=datetime.now(), periods=len(records), freq="5min")
            return predict(df)

        return 2, [0.0, 0.0, 1.0], 0.0, {}
