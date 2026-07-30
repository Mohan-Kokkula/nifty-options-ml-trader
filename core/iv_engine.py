"""
iv_engine.py — Live Implied Volatility Engine for Nifty Options
=================================================================
Pulls per-strike IV from the NSE option chain (which computes IV
server-side and returns it in every response) and derives the metrics
the pilot needs before committing to a trade.

Why this matters:
  India VIX is a 30-day weighted average across all strikes. On any
  given morning the ATM IV of the front-week expiry can be 3–5 pts
  higher or lower than VIX — that spread IS the mispricing you either
  exploit or lose to. Buying options when ATM IV > 1.3 × VIX means
  you need the spot to move extra hard just to break even.

Metrics produced every 60 s (cached):
  atm_iv          — IV of ATM strike  (CE+PE average), in %
  ce_iv / pe_iv   — split IV at ATM
  iv_skew         — pe_iv − ce_iv  (positive = fear / put premium)
  put_call_iv_ratio — mean(PE IV) / mean(CE IV) across ±5 strikes
  iv_vs_vix       — atm_iv − vix_level  (how rich options are vs index)
  iv_change_pct   — % change from session-open IV (IV expansion / crush)
  iv_rank         — percentile of atm_iv vs rolling 60-day history (0–100)
  live_delta      — actual BS delta at each nearby strike (replaces hardcoded map)

Data sources (priority order):
  1. NSE option chain API  — server-computed IV, no auth needed
  2. Kotak Neo option chain + Black-Scholes solver  — broker-fresh prices
  3. Cached snapshot       — if all fetches fail, last known values

IV history is persisted to data/iv_history.jsonl (one line per day)
so iv_rank improves over time as the bot collects data.

Usage:
    engine = IVEngine(kotak_client=trader.client)
    snap   = engine.get_snapshot(spot=24350.0, vix_level=14.5)
    if snap.available and snap.atm_iv > snap.vix_level * 1.3:
        ...  # expensive day — skip or tighten TP
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
_CACHE_TTL          = 60.0          # seconds between chain refetches
_HISTORY_FILE       = Path("data/iv_history.jsonl")
_HISTORY_DAYS       = 60            # rolling window for iv_rank
_RISK_FREE_RATE     = 0.065         # India 10-yr G-sec approximate (annualized)
_NIFTY_STRIKE_STEP  = 50
_CHAIN_STRIKES_SIDE = 10            # fetch 10 strikes each side of ATM

# Newton-Raphson IV solver limits
_IV_MAX_ITER = 50
_IV_TOL      = 1e-6
_IV_MIN      = 0.001                # 0.1%
_IV_MAX      = 5.0                  # 500%


# ──────────────────────────────────────────────────────────────────
# Black-Scholes helpers  (European options, continuous dividend = 0)
# ──────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    """Standard normal CDF using math.erf (accurate to 1e-15)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(S: float, K: float, T: float, r: float,
               sigma: float, flag: str) -> float:
    """
    Black-Scholes price for European option.

    Args:
        S     : spot price
        K     : strike price
        T     : time to expiry in years (must be > 0)
        r     : risk-free rate (fraction, e.g. 0.065)
        sigma : IV (fraction, e.g. 0.15)
        flag  : 'c' = call, 'p' = put

    Returns:
        Theoretical option price (in same units as S and K).
    """
    if T <= 0 or sigma <= 0:
        # At-expiry intrinsic value
        intrinsic = max(0.0, (S - K) if flag == 'c' else (K - S))
        return intrinsic

    sqrt_T  = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = math.exp(-r * T)            # discount factor

    if flag == 'c':
        return S * _ncdf(d1) - K * df * _ncdf(d2)
    else:
        return K * df * _ncdf(-d2) - S * _ncdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes vega (∂price/∂sigma).
    Identical for calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1     = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return S * _npdf(d1) * sqrt_T


def _bs_delta(S: float, K: float, T: float, r: float,
               sigma: float, flag: str) -> float:
    """Black-Scholes delta (∂price/∂S)."""
    if T <= 0 or sigma <= 0:
        if flag == 'c':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    sqrt_T = math.sqrt(T)
    d1     = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    if flag == 'c':
        return _ncdf(d1)
    else:
        return _ncdf(d1) - 1.0


def black_scholes_iv(market_price: float, S: float, K: float,
                     T_years: float, r: float = _RISK_FREE_RATE,
                     flag: str = 'c') -> float:
    """
    Compute implied volatility from market price using Newton-Raphson.

    Args:
        market_price : observed option LTP
        S            : spot (underlying) price
        K            : strike price
        T_years      : time to expiry in years
        r            : risk-free rate (fraction)
        flag         : 'c' call, 'p' put

    Returns:
        IV as a percentage (e.g. 15.3 for 15.3%).
        Returns 0.0 on failure.
    """
    if market_price <= 0 or S <= 0 or K <= 0 or T_years <= 0:
        return 0.0

    # Intrinsic check — market price can't be below intrinsic
    intrinsic = max(0.0, (S - K) if flag == 'c' else (K - S))
    if market_price < intrinsic:
        return 0.0

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T_years) * (market_price / S)
    sigma = max(_IV_MIN, min(_IV_MAX, sigma))

    for _ in range(_IV_MAX_ITER):
        price = _bs_price(S, K, T_years, r, sigma, flag)
        vega  = _bs_vega(S, K, T_years, r, sigma)
        diff  = price - market_price

        if abs(diff) < _IV_TOL * S:
            break

        if abs(vega) < 1e-10:
            break

        sigma -= diff / vega
        sigma  = max(_IV_MIN, min(_IV_MAX, sigma))

    return round(sigma * 100.0, 4)   # convert fraction → percentage


# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class IVSnapshot:
    """
    Complete IV picture for the current cycle.

    All IV values are in percentage terms (e.g. 15.3 means 15.3%).
    iv_rank is 0–100 (percentile vs 60-day history).
    live_delta values are signed (call delta positive, put negative).
    """
    # Core IV metrics
    atm_iv:            float = 0.0      # (ce_iv + pe_iv) / 2 at ATM
    ce_iv:             float = 0.0      # call IV at ATM strike
    pe_iv:             float = 0.0      # put IV at ATM strike
    iv_skew:           float = 0.0      # pe_iv - ce_iv  (positive = fear)
    put_call_iv_ratio: float = 1.0      # mean(PE IV) / mean(CE IV) ±5 strikes
    iv_vs_vix:         float = 0.0      # atm_iv - vix_level
    iv_change_pct:     float = 0.0      # % change from session-open IV
    iv_rank:           float = 50.0     # percentile vs history (0–100)

    # Live greeks for chosen strikes
    live_delta_atm:    float = 0.50     # BS delta at ATM strike
    live_delta:        dict  = field(default_factory=dict)  # {offset: delta}

    # Context
    spot:              float = 0.0
    atm_strike:        float = 0.0
    vix_level:         float = 0.0
    dte:               float = 0.0      # days to expiry used
    source:            str   = "NONE"   # "NSE" | "KOTAK_BS" | "CACHED"
    timestamp:         datetime = field(default_factory=datetime.now)
    available:         bool  = False

    # ── Risk classification ───────────────────────────────────────
    @property
    def is_elevated(self) -> bool:
        """IV meaningfully above VIX — options are expensive."""
        return self.available and self.iv_vs_vix > 3.0

    @property
    def is_rich(self) -> bool:
        """IV dangerously expensive — >30% above VIX and high percentile."""
        return self.available and self.iv_vs_vix > 5.0 and self.iv_rank > 70

    @property
    def is_crush_risk(self) -> bool:
        """IV rising intraday — potential for IV crush on next event/move."""
        return self.available and self.iv_change_pct > 10.0

    @property
    def summary(self) -> str:
        if not self.available:
            return "IV: N/A"
        return (
            f"ATM_IV={self.atm_iv:.1f}% skew={self.iv_skew:+.1f} "
            f"vs_VIX={self.iv_vs_vix:+.1f} rank={self.iv_rank:.0f}th "
            f"chg={self.iv_change_pct:+.1f}% [{self.source}]"
        )


# ──────────────────────────────────────────────────────────────────
# IVEngine
# ──────────────────────────────────────────────────────────────────

class IVEngine:
    """
    Fetches and caches live IV metrics from NSE option chain.

    Thread-safe — multiple threads can call get_snapshot() concurrently;
    only one will refetch while others wait on the lock.
    """

    def __init__(self, kotak_client=None):
        """
        Args:
            kotak_client : KotakNeoClient instance (optional).
                           Used as fallback IV source when NSE is unavailable.
        """
        self._client    = kotak_client
        self._lock      = threading.Lock()
        self._cache:    Optional[IVSnapshot] = None
        self._cache_ts: float = 0.0

        # Session-open IV: set on first fetch of each day
        self._session_open_iv:   float = 0.0
        self._session_open_date: Optional[date] = None

        # 60-day rolling IV history for rank computation
        self._iv_history: list[dict] = []   # [{date, atm_iv}]
        self._history_loaded = False

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def get_snapshot(self, spot: float, vix_level: float) -> IVSnapshot:
        """
        Return current IV snapshot. Refreshes at most every _CACHE_TTL seconds.
        Thread-safe — safe to call from multiple threads / from pilot loop.

        Args:
            spot      : current Nifty spot price
            vix_level : current India VIX level (%)

        Returns:
            IVSnapshot — always returns (available=False if all sources fail).
        """
        with self._lock:
            if not self._history_loaded:
                self._load_history()

            # Return cached if fresh enough
            if self._cache and (time.monotonic() - self._cache_ts) < _CACHE_TTL:
                return self._cache

            snap = self._fetch(spot, vix_level)
            self._cache    = snap
            self._cache_ts = time.monotonic()
            return snap

    def invalidate(self):
        """Force next call to refetch (call after session reopen)."""
        with self._lock:
            self._cache    = None
            self._cache_ts = 0.0

    # ──────────────────────────────────────────────────────────────
    # Internal fetch logic
    # ──────────────────────────────────────────────────────────────

    def _fetch(self, spot: float, vix_level: float) -> IVSnapshot:
        """Try NSE → Kotak+BS → cached. Returns best available snapshot."""
        # Try NSE first (has server-computed IV — most reliable)
        snap = self._fetch_nse(spot, vix_level)
        if snap.available:
            self._post_process(snap, vix_level)
            self._update_session_open(snap)
            self._append_history(snap)
            logger.info(f"IVEngine: {snap.summary}")
            return snap

        # Try Kotak Neo chain + Black-Scholes
        if self._client:
            snap = self._fetch_kotak_bs(spot, vix_level)
            if snap.available:
                self._post_process(snap, vix_level)
                self._update_session_open(snap)
                self._append_history(snap)
                logger.info(f"IVEngine: {snap.summary}")
                return snap

        # Return stale cache if available, else empty
        if self._cache and self._cache.available:
            stale = IVSnapshot(**self._cache.__dict__)
            stale.source    = "CACHED"
            stale.timestamp = datetime.now()
            logger.warning("IVEngine: both sources failed — using stale cache")
            return stale

        logger.warning("IVEngine: IV unavailable (no cache, no live source)")
        return IVSnapshot(vix_level=vix_level, spot=spot)

    # ──────────────────────────────────────────────────────────────
    # Source 1: NSE option chain (server-computed IV)
    # ──────────────────────────────────────────────────────────────

    def _fetch_nse(self, spot: float, vix_level: float) -> IVSnapshot:
        """
        Pull NSE option chain and extract IV.

        NSE's JSON schema (records.data[]):
          strikePrice, CE.impliedVolatility, PE.impliedVolatility
          CE.delta (sometimes present), CE.openInterest, CE.lastPrice, etc.

        IV is in percentage terms already (16.5 = 16.5%).
        """
        try:
            from core.market_intel import _nse_get   # reuse authenticated session
        except ImportError:
            return IVSnapshot()

        data = _nse_get(
            "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
            timeout=8,
        )
        if not data:
            return IVSnapshot()

        records = data.get("records", {})
        oc_data = records.get("data", [])
        nse_spot = float(records.get("underlyingValue", spot) or spot)
        if nse_spot > 0:
            spot = nse_spot

        if not oc_data:
            return IVSnapshot()

        # Get DTE from expiry_utils
        dte = self._get_dte()

        # Build strike → {ce_iv, pe_iv, ce_ltp, pe_ltp} map
        chain: dict[float, dict] = {}
        for item in oc_data:
            strike = float(item.get("strikePrice", 0))
            if strike <= 0:
                continue
            ce = item.get("CE") or {}
            pe = item.get("PE") or {}
            chain[strike] = {
                "ce_iv":  float(ce.get("impliedVolatility", 0) or 0),
                "pe_iv":  float(pe.get("impliedVolatility", 0) or 0),
                "ce_ltp": float(ce.get("lastPrice", 0) or 0),
                "pe_ltp": float(pe.get("lastPrice", 0) or 0),
                "ce_oi":  float(ce.get("openInterest", 0) or 0),
                "pe_oi":  float(pe.get("openInterest", 0) or 0),
            }

        if not chain:
            return IVSnapshot()

        atm_strike = _nearest_strike(spot, _NIFTY_STRIKE_STEP)
        snap       = self._compute_metrics(chain, atm_strike, spot, dte, vix_level)
        snap.source = "NSE"
        return snap

    # ──────────────────────────────────────────────────────────────
    # Source 2: Kotak Neo chain + Black-Scholes IV solver
    # ──────────────────────────────────────────────────────────────

    def _fetch_kotak_bs(self, spot: float, vix_level: float) -> IVSnapshot:
        """
        Fetch Kotak Neo option chain, compute IV via Black-Scholes from LTP.
        Used when NSE is unavailable (network issue, throttle, etc.).
        """
        try:
            from core.expiry_utils import get_expiry_date
            expiry = get_expiry_date()
            if not expiry:
                return IVSnapshot()

            expiry_str = expiry.strftime("%d%b%y").upper()
            dte        = self._get_dte()
            T_years    = max(dte / 365.0, 1.0 / 365.0)  # floor at 1 day

            # Kotak get_option_chain returns LTP but no IV
            chain_data = None
            for exc in ("NFO", "NSE_INDEX"):
                try:
                    chain_data = self._client.get_option_chain(
                        underlying="NIFTY",
                        exchange=exc,
                        expiry_date=expiry_str,
                        strike_count=_CHAIN_STRIKES_SIDE,
                    )
                    if chain_data and chain_data.get("chain"):
                        break
                except Exception:
                    continue

            if not chain_data or not chain_data.get("chain"):
                logger.warning("IVEngine Kotak: get_option_chain returned empty — no strikes or session issue")
                return IVSnapshot()

            n_entries = len(chain_data["chain"])
            logger.info(f"IVEngine Kotak: chain has {n_entries} strikes for {expiry_str}")
            kotak_spot = float(chain_data.get("underlying_ltp", spot) or spot)
            if kotak_spot > 0:
                spot = kotak_spot

            atm_strike = _nearest_strike(spot, _NIFTY_STRIKE_STEP)

            # Compute IV via Black-Scholes for each strike
            # Handle BOTH Kotak flat keys (ce_ltp, pe_ltp) and
            # nested format (ce: {ltp: X}) — Kotak uses flat keys.
            chain: dict[float, dict] = {}
            for entry in chain_data["chain"]:
                strike  = float(entry.get("strike", 0))
                if strike <= 0:
                    continue

                # Nested format (legacy)
                ce_info = entry.get("ce", {}) or {}
                pe_info = entry.get("pe", {}) or {}
                ce_ltp  = float(ce_info.get("ltp", 0) or 0)
                pe_ltp  = float(pe_info.get("ltp", 0) or 0)
                ce_oi   = float(ce_info.get("oi",  0) or 0)
                pe_oi   = float(pe_info.get("oi",  0) or 0)

                # Flat key fallback (Kotak Neo direct output)
                if ce_ltp == 0:
                    ce_ltp = float(entry.get("ce_ltp", 0) or entry.get("ce_premium", 0) or 0)
                if pe_ltp == 0:
                    pe_ltp = float(entry.get("pe_ltp", 0) or entry.get("pe_premium", 0) or 0)
                if ce_oi == 0:
                    ce_oi  = float(entry.get("ce_oi", 0) or 0)
                if pe_oi == 0:
                    pe_oi  = float(entry.get("pe_oi", 0) or 0)

                ce_iv = (black_scholes_iv(ce_ltp, spot, strike, T_years, flag='c')
                         if ce_ltp > 0 else 0.0)
                pe_iv = (black_scholes_iv(pe_ltp, spot, strike, T_years, flag='p')
                         if pe_ltp > 0 else 0.0)

                chain[strike] = {
                    "ce_iv":  ce_iv,
                    "pe_iv":  pe_iv,
                    "ce_ltp": ce_ltp,
                    "pe_ltp": pe_ltp,
                    "ce_oi":  ce_oi,
                    "pe_oi":  pe_oi,
                }

            # Diagnose if all LTPs are zero (quotes API returned empty)
            valid_ltp = sum(1 for v in chain.values() if v["ce_ltp"] > 0 or v["pe_ltp"] > 0)
            if valid_ltp == 0:
                logger.warning(
                    f"IVEngine Kotak: {len(chain)} strikes found but ALL ltp=0 "
                    f"— quotes API may have failed or market is closed"
                )
                return IVSnapshot()

            logger.info(f"IVEngine Kotak: {valid_ltp}/{len(chain)} strikes have valid LTP")
            snap        = self._compute_metrics(chain, atm_strike, spot, dte, vix_level)
            snap.source = "KOTAK_BS"
            return snap

        except Exception as e:
            logger.warning(f"IVEngine Kotak fetch failed: {e}")
            return IVSnapshot()

    # ──────────────────────────────────────────────────────────────
    # Metrics computation (shared by both sources)
    # ──────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        chain: dict,
        atm_strike: float,
        spot: float,
        dte: float,
        vix_level: float,
    ) -> IVSnapshot:
        """
        Given a chain dict {strike: {ce_iv, pe_iv, ce_ltp, pe_ltp}},
        compute all IVSnapshot metrics.
        """
        snap            = IVSnapshot()
        snap.spot       = spot
        snap.atm_strike = atm_strike
        snap.vix_level  = vix_level
        snap.dte        = dte

        # ATM IV
        atm = chain.get(atm_strike, {})
        ce_iv_atm = float(atm.get("ce_iv", 0))
        pe_iv_atm = float(atm.get("pe_iv", 0))

        # If ATM is missing or zero, find nearest with valid IV
        if ce_iv_atm == 0 and pe_iv_atm == 0:
            strikes_sorted = sorted(chain.keys(), key=lambda s: abs(s - atm_strike))
            for s in strikes_sorted[:3]:
                ce_iv_atm = chain[s].get("ce_iv", 0)
                pe_iv_atm = chain[s].get("pe_iv", 0)
                if ce_iv_atm > 0 or pe_iv_atm > 0:
                    atm_strike = s
                    break

        # Use whichever side has valid IV
        if ce_iv_atm > 0 and pe_iv_atm > 0:
            snap.atm_iv = round((ce_iv_atm + pe_iv_atm) / 2.0, 2)
        elif ce_iv_atm > 0:
            snap.atm_iv = ce_iv_atm
        elif pe_iv_atm > 0:
            snap.atm_iv = pe_iv_atm
        else:
            return snap   # no usable IV — return empty

        snap.ce_iv      = round(ce_iv_atm, 2)
        snap.pe_iv      = round(pe_iv_atm, 2)
        snap.iv_skew    = round(pe_iv_atm - ce_iv_atm, 2)
        snap.iv_vs_vix  = round(snap.atm_iv - vix_level, 2)

        # Put/Call IV ratio across ±5 strikes
        ce_ivs, pe_ivs = [], []
        for k in sorted(chain.keys()):
            d = chain[k]
            if d.get("ce_iv", 0) > 0:
                ce_ivs.append(d["ce_iv"])
            if d.get("pe_iv", 0) > 0:
                pe_ivs.append(d["pe_iv"])

        if ce_ivs and pe_ivs:
            snap.put_call_iv_ratio = round(
                (sum(pe_ivs) / len(pe_ivs)) / (sum(ce_ivs) / len(ce_ivs)), 3
            )

        # Live delta from Black-Scholes for key strikes
        if dte > 0 and snap.atm_iv > 0:
            T_years = max(dte / 365.0, 1.0 / 365.0)
            sigma   = snap.atm_iv / 100.0   # fraction
            snap.live_delta_atm = round(
                _bs_delta(spot, atm_strike, T_years, _RISK_FREE_RATE, sigma, 'c'), 3
            )
            offset_map = {"ATM": 0, "OTM1": 1, "OTM2": 2, "OTM3": 3,
                          "ITM1": -1, "ITM2": -2}
            for offset, steps in offset_map.items():
                strike_c = atm_strike + steps * _NIFTY_STRIKE_STEP
                strike_p = atm_strike - steps * _NIFTY_STRIKE_STEP
                # Use IV of that strike if available, else ATM IV
                ce_sigma = (chain.get(strike_c, {}).get("ce_iv", snap.atm_iv) or snap.atm_iv) / 100.0
                pe_sigma = (chain.get(strike_p, {}).get("pe_iv", snap.atm_iv) or snap.atm_iv) / 100.0
                snap.live_delta[f"CE_{offset}"] = round(
                    _bs_delta(spot, strike_c, T_years, _RISK_FREE_RATE, ce_sigma, 'c'), 3
                )
                snap.live_delta[f"PE_{offset}"] = round(
                    abs(_bs_delta(spot, strike_p, T_years, _RISK_FREE_RATE, pe_sigma, 'p')), 3
                )

        snap.available = snap.atm_iv > 0
        return snap

    def _post_process(self, snap: IVSnapshot, vix_level: float):
        """Compute iv_change_pct and iv_rank in-place after successful fetch."""
        # iv_change_pct vs session open
        if self._session_open_iv > 0:
            snap.iv_change_pct = round(
                (snap.atm_iv - self._session_open_iv) / self._session_open_iv * 100.0, 2
            )

        # iv_rank from history
        snap.iv_rank = self._compute_iv_rank(snap.atm_iv)

    def _update_session_open(self, snap: IVSnapshot):
        """Record session-open IV (first fetch of the day)."""
        today = date.today()
        if self._session_open_date != today:
            self._session_open_iv   = snap.atm_iv
            self._session_open_date = today
            logger.info(f"IVEngine: session open IV = {snap.atm_iv:.1f}%")

    # ──────────────────────────────────────────────────────────────
    # IV history (for iv_rank)
    # ──────────────────────────────────────────────────────────────

    def _load_history(self):
        """Load rolling IV history from disk."""
        self._history_loaded = True
        if not _HISTORY_FILE.exists():
            return
        try:
            entries = []
            with open(_HISTORY_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
            # Keep last _HISTORY_DAYS entries
            self._iv_history = entries[-_HISTORY_DAYS:]
            logger.debug(f"IVEngine: loaded {len(self._iv_history)} historical IV entries")
        except Exception as e:
            logger.debug(f"IVEngine: history load failed: {e}")

    def _append_history(self, snap: IVSnapshot):
        """
        Append today's IV to history file (one entry per day).
        Uses the first reading of the day so it's comparable across days.
        """
        today_str = date.today().isoformat()
        # Check if today already recorded
        if self._iv_history and self._iv_history[-1].get("date") == today_str:
            return

        entry = {"date": today_str, "atm_iv": round(snap.atm_iv, 2)}
        self._iv_history.append(entry)
        # Trim to window
        self._iv_history = self._iv_history[-_HISTORY_DAYS:]

        # Persist
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_HISTORY_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"IVEngine: history write failed: {e}")

    def _compute_iv_rank(self, atm_iv: float) -> float:
        """
        Compute IV rank = (atm_iv − 60d_low) / (60d_high − 60d_low) × 100.
        Returns 50.0 if insufficient history (neutral / unknown).
        """
        if len(self._iv_history) < 5:
            return 50.0   # not enough history yet

        values = [e["atm_iv"] for e in self._iv_history if e.get("atm_iv", 0) > 0]
        if not values:
            return 50.0

        lo, hi = min(values), max(values)
        if hi <= lo:
            return 50.0

        rank = (atm_iv - lo) / (hi - lo) * 100.0
        return round(max(0.0, min(100.0, rank)), 1)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_dte() -> float:
        """Days to expiry, floored at 0.5 (half a trading day)."""
        try:
            from core.expiry_utils import get_dte
            dte = get_dte()
            return max(0.5, float(dte))
        except Exception:
            return 3.0   # safe default


# ──────────────────────────────────────────────────────────────────
# Module-level singleton (shared by claude_pilot + anything else)
# ──────────────────────────────────────────────────────────────────

_engine: Optional[IVEngine] = None


def init_iv_engine(kotak_client=None) -> IVEngine:
    """
    Create (or replace) the module-level IVEngine singleton.
    Call once at startup from main.py after the broker client is ready.
    """
    global _engine
    _engine = IVEngine(kotak_client=kotak_client)
    logger.info("IVEngine: initialised")
    return _engine


def get_iv_engine() -> Optional[IVEngine]:
    """Return the singleton, or None if not yet initialised."""
    return _engine


# ──────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────

def _nearest_strike(spot: float, step: int = _NIFTY_STRIKE_STEP) -> float:
    """Round spot to nearest valid strike."""
    return float(int(round(spot / step)) * step)
