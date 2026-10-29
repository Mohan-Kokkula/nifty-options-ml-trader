"""
gex_options_strategy.py -- S18 GEX Options Entry Strategy
==========================================================
Uses real options chain data to generate trade signals:

  GEX Regime   -> Direction + Strategy Style
  OI Change    -> Flow confirmation
  IV Skew      -> Strike selection
  Max Pain     -> Expiry gravity
  Theta/DTE    -> Strategy type + urgency

Regime Logic:
  POSITIVE GEX (MMs dampen) -> fade at walls, sell premium
  NEGATIVE GEX (MMs amplify) -> trend-follow, buy premium
  NEAR FLIP (edge zone)      -> wait or straddle

Entry Types:
  1. Directional debit   (buy CE/PE)
  2. Credit spread       (sell at wall, buy protection)
  3. Straddle/strangle   (near gamma flip, direction unclear)
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, date


NIFTY_LOT = 75


def compute_max_pain(chain_df):
    strikes = chain_df["strike"].values
    best_strike = strikes[0]
    min_loss = float("inf")
    for s in strikes:
        call_itm_loss = chain_df.apply(
            lambda r: max(0, s - r["strike"]) * r["call_oi"], axis=1
        ).sum()
        put_itm_loss = chain_df.apply(
            lambda r: max(0, r["strike"] - s) * r["put_oi"], axis=1
        ).sum()
        total = call_itm_loss + put_itm_loss
        if total < min_loss:
            min_loss = total
            best_strike = s
    return float(best_strike)


def compute_pcr(chain_df, spot, width=500):
    near = chain_df[
        (chain_df["strike"] >= spot - width) & (chain_df["strike"] <= spot + width)
    ]
    total_put = near["put_oi"].sum()
    total_call = near["call_oi"].sum()
    return total_put / max(total_call, 1)


def compute_chg_oi_pcr(chain_df, spot, width=500):
    near = chain_df[
        (chain_df["strike"] >= spot - width) & (chain_df["strike"] <= spot + width)
    ]
    put_build = near[near["put_chg_oi"] > 0]["put_chg_oi"].sum()
    call_build = near[near["call_chg_oi"] > 0]["call_chg_oi"].sum()
    return put_build / max(call_build, 1)


def iv_skew_atm(chain_df, spot):
    diffs = (chain_df["strike"] - spot).abs()
    atm_idx = diffs.idxmin()
    row = chain_df.loc[atm_idx]
    call_iv = float(row["call_iv"])
    put_iv = float(row["put_iv"])
    return call_iv, put_iv, call_iv - put_iv


def find_highest_volume_strike(chain_df, spot, option_type="call", direction="otm"):
    if option_type == "call":
        if direction == "otm":
            subset = chain_df[chain_df["strike"] >= spot]
        else:
            subset = chain_df[chain_df["strike"] <= spot]
        return float(subset.loc[subset["call_vol"].idxmax(), "strike"])
    else:
        if direction == "otm":
            subset = chain_df[chain_df["strike"] <= spot]
        else:
            subset = chain_df[chain_df["strike"] >= spot]
        return float(subset.loc[subset["put_vol"].idxmax(), "strike"])


def theta_per_day_pct(dte):
    if dte <= 0:
        return 100.0
    if dte == 1:
        return 15.0
    if dte <= 3:
        return 6.0
    if dte <= 5:
        return 3.0
    return 1.5


class GEXOptionsStrategy:
    """
    Generates options trade signals from GEX analysis.

    Usage:
        strat = GEXOptionsStrategy()
        signal = strat.analyze(chain_df, spot, gex_result, dte,
                               today_open, today_high, today_low, hhmm)
    """

    # Thresholds
    WALL_PROXIMITY = 50       # pts to consider "near" a wall
    FLIP_PROXIMITY = 30       # pts to consider "at" gamma flip
    MIN_RR = 1.5              # minimum reward:risk for directional
    MIN_VOLUME = 50000        # minimum option volume for liquidity
    MAX_PREMIUM_PCT = 0.8     # max % of ATR to pay for premium
    NO_ENTRY_AFTER = 1500     # no new trades after 3:00 PM
    NO_ENTRY_BEFORE = 945     # no trades in first 15 min

    def analyze(self, chain_df, spot, gex_result, dte,
                today_open, today_high, today_low, hhmm,
                prev_close=None):
        """
        Main analysis: returns a dict with trade signal or NO_TRADE.

        Returns dict:
            action:      "BUY_CE" | "BUY_PE" | "SELL_CE_SPREAD" | "SELL_PE_SPREAD"
                         | "BUY_STRADDLE" | "NO_TRADE"
            strike:      recommended strike
            reason:      human-readable explanation
            confidence:  0-100
            sl_pct:      stop loss as % of premium
            tp_pct:      take profit as % of premium
            regime:      "positive" | "negative" | "neutral"
            levels:      dict of key levels
            metrics:     dict of supporting metrics
        """
        regime = gex_result["regime"]
        call_wall = gex_result["call_wall"]
        put_wall = gex_result["put_wall"]
        gamma_flip = gex_result["gamma_flip"]
        total_gex = gex_result["total_gex"]

        # Core metrics
        pcr = compute_pcr(chain_df, spot)
        chg_pcr = compute_chg_oi_pcr(chain_df, spot)
        max_pain = compute_max_pain(chain_df)
        call_iv, put_iv, iv_skew = iv_skew_atm(chain_df, spot)
        theta_daily = theta_per_day_pct(dte)

        # Distances
        dist_call_wall = call_wall - spot
        dist_put_wall = spot - put_wall
        dist_flip = spot - gamma_flip
        dist_max_pain = spot - max_pain

        # Intraday move already made
        intraday_move = spot - today_open if today_open > 0 else 0
        intraday_range = today_high - today_low if today_high > today_low else 1
        move_pct_of_range = abs(intraday_move) / intraday_range

        # OI flow direction
        # chg_pcr > 1.5 = strong put building = bullish flow
        # chg_pcr < 0.7 = strong call building = bearish flow
        if chg_pcr > 1.5:
            flow = "bullish"
        elif chg_pcr < 0.7:
            flow = "bearish"
        else:
            flow = "neutral"

        levels = {
            "call_wall": call_wall,
            "put_wall": put_wall,
            "gamma_flip": gamma_flip,
            "max_pain": max_pain,
        }

        metrics = {
            "pcr": round(pcr, 3),
            "chg_oi_pcr": round(chg_pcr, 3),
            "flow": flow,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "iv_skew": round(iv_skew, 2),
            "theta_daily_pct": theta_daily,
            "dte": dte,
            "intraday_move": round(intraday_move, 1),
            "move_pct_of_range": round(move_pct_of_range, 2),
            "dist_call_wall": round(dist_call_wall, 0),
            "dist_put_wall": round(dist_put_wall, 0),
            "dist_gamma_flip": round(dist_flip, 0),
            "dist_max_pain": round(dist_max_pain, 0),
        }

        base = {
            "regime": regime,
            "levels": levels,
            "metrics": metrics,
        }

        # Time filter
        if hhmm >= self.NO_ENTRY_AFTER:
            return {**base, "action": "NO_TRADE", "strike": 0,
                    "reason": "Too late in session (after 3:00 PM)",
                    "confidence": 0, "sl_pct": 0, "tp_pct": 0}

        if hhmm < self.NO_ENTRY_BEFORE:
            return {**base, "action": "NO_TRADE", "strike": 0,
                    "reason": "Auction period (before 9:45 AM)",
                    "confidence": 0, "sl_pct": 0, "tp_pct": 0}

        # Edge zone: regime flips within a few pts of gamma flip.
        # If spot is within FLIP_PROXIMITY of flip, regime is unstable.
        at_flip_edge = abs(dist_flip) <= self.FLIP_PROXIMITY
        if at_flip_edge:
            metrics["regime_note"] = "EDGE ZONE -- regime unstable near gamma flip"

        # === POSITIVE GEX: Mean Reversion at Walls ===
        if regime == "positive" and not at_flip_edge:
            return self._positive_gex_signal(
                chain_df, spot, dte, call_wall, put_wall, gamma_flip,
                max_pain, pcr, chg_pcr, flow, call_iv, put_iv,
                iv_skew, theta_daily, dist_call_wall, dist_put_wall,
                dist_flip, dist_max_pain, intraday_move, move_pct_of_range,
                base
            )

        # === NEGATIVE GEX: Trend Follow ===
        elif regime == "negative" and not at_flip_edge:
            return self._negative_gex_signal(
                chain_df, spot, dte, call_wall, put_wall, gamma_flip,
                max_pain, pcr, chg_pcr, flow, call_iv, put_iv,
                iv_skew, theta_daily, dist_call_wall, dist_put_wall,
                dist_flip, dist_max_pain, intraday_move, move_pct_of_range,
                base
            )

        # === AT GAMMA FLIP EDGE or NEUTRAL ===
        else:
            return self._flip_edge_signal(
                chain_df, spot, dte, call_wall, put_wall, gamma_flip,
                max_pain, pcr, chg_pcr, flow, call_iv, put_iv,
                iv_skew, theta_daily, dist_call_wall, dist_put_wall,
                dist_flip, dist_max_pain, intraday_move, move_pct_of_range,
                base
            )

    def _positive_gex_signal(self, chain_df, spot, dte,
                              call_wall, put_wall, gamma_flip, max_pain,
                              pcr, chg_pcr, flow, call_iv, put_iv,
                              iv_skew, theta_daily,
                              dist_cw, dist_pw, dist_flip, dist_mp,
                              intraday_move, move_pct, base):
        """
        Positive GEX: MMs dampen moves. Fade at walls, sell premium.

        Best trades:
        - Near call wall + bearish flow -> sell CE spread or buy PE
        - Near put wall + bullish flow -> sell PE spread or buy CE
        - Between walls -> sell iron condor (if DTE allows)
        - Near max pain close to expiry -> theta burn favors sellers
        """
        confidence = 0
        reasons = []

        # --- NEAR CALL WALL: SHORT BIAS ---
        if dist_cw <= self.WALL_PROXIMITY:
            confidence += 30
            reasons.append(f"Near call wall ({call_wall:.0f}, {dist_cw:.0f} pts away)")

            if flow == "bearish" or flow == "neutral":
                confidence += 15
                reasons.append(f"Flow not bullish ({flow}, chgPCR={chg_pcr:.2f})")

            if dist_mp > 0:
                confidence += 10
                reasons.append(f"Above max pain ({max_pain:.0f}) -- gravity pulls down")

            if move_pct > 0.7 and intraday_move > 0:
                confidence += 10
                reasons.append("Already rallied most of day's range")

            if dte <= 3:
                # Near expiry at resistance: sell premium
                strike = call_wall
                sl_pct = 50
                tp_pct = 80
                action = "SELL_CE_SPREAD"
                reasons.append(f"Short DTE ({dte}) favors selling at resistance")
                confidence += 10
            else:
                # Buy put for reversal
                strike = self._select_strike(chain_df, spot, "put", "atm")
                sl_pct = 30
                tp_pct = 60
                action = "BUY_PE"

            if confidence >= 50:
                return {**base, "action": action, "strike": strike,
                        "reason": " | ".join(reasons),
                        "confidence": min(confidence, 95),
                        "sl_pct": sl_pct, "tp_pct": tp_pct}

        # --- NEAR PUT WALL: LONG BIAS ---
        if dist_pw <= self.WALL_PROXIMITY:
            confidence = 30
            reasons = [f"Near put wall ({put_wall:.0f}, {dist_pw:.0f} pts away)"]

            if flow == "bullish":
                confidence += 20
                reasons.append(f"Bullish flow (chgPCR={chg_pcr:.2f}, puts being written)")

            if dist_mp < 0:
                confidence += 10
                reasons.append(f"Below max pain ({max_pain:.0f}) -- gravity pulls up")

            if move_pct > 0.7 and intraday_move < 0:
                confidence += 10
                reasons.append("Already sold off most of day's range")

            if dte <= 3:
                strike = put_wall
                sl_pct = 50
                tp_pct = 80
                action = "SELL_PE_SPREAD"
                reasons.append(f"Short DTE ({dte}) favors selling at support")
                confidence += 10
            else:
                strike = self._select_strike(chain_df, spot, "call", "atm")
                sl_pct = 30
                tp_pct = 60
                action = "BUY_CE"

            if confidence >= 50:
                return {**base, "action": action, "strike": strike,
                        "reason": " | ".join(reasons),
                        "confidence": min(confidence, 95),
                        "sl_pct": sl_pct, "tp_pct": tp_pct}

        # --- BETWEEN WALLS: RANGE PLAY ---
        if dist_cw > self.WALL_PROXIMITY and dist_pw > self.WALL_PROXIMITY:
            if dte <= 3 and dist_cw > 100 and dist_pw > 100:
                confidence = 40
                reasons = ["Between walls, short DTE -- theta favors sellers"]
                if pcr > 1.0 and pcr < 2.0:
                    confidence += 15
                    reasons.append(f"Balanced PCR ({pcr:.2f})")
                return {**base, "action": "NO_TRADE", "strike": 0,
                        "reason": "Between walls -- wait for price to approach a wall | " + " | ".join(reasons),
                        "confidence": 0, "sl_pct": 0, "tp_pct": 0}

        return {**base, "action": "NO_TRADE", "strike": 0,
                "reason": "Positive GEX but no clear wall setup",
                "confidence": 0, "sl_pct": 0, "tp_pct": 0}

    def _negative_gex_signal(self, chain_df, spot, dte,
                              call_wall, put_wall, gamma_flip, max_pain,
                              pcr, chg_pcr, flow, call_iv, put_iv,
                              iv_skew, theta_daily,
                              dist_cw, dist_pw, dist_flip, dist_mp,
                              intraday_move, move_pct, base):
        """
        Negative GEX: MMs amplify moves. Trend-follow with momentum.

        Best trades:
        - Above gamma flip + bullish flow -> buy CE
        - Below gamma flip + bearish flow -> buy PE
        - AT gamma flip -> straddle (direction unclear, vol expanding)
        - Breaking through wall -> add to direction (gamma squeeze/cascade)

        Key: buy directional premium, DON'T sell premium in neg gamma.
        """
        confidence = 0
        reasons = []

        # --- AT GAMMA FLIP: VOLATILITY PLAY ---
        if abs(dist_flip) <= self.FLIP_PROXIMITY:
            confidence = 35
            reasons.append(f"At gamma flip ({gamma_flip:.0f}, {dist_flip:+.0f} pts)")
            reasons.append("Negative GEX at flip = big move brewing, direction unclear")

            if dte >= 3:
                strike = self._select_strike(chain_df, spot, "call", "atm")
                confidence += 15
                reasons.append(f"DTE={dte} gives time for the move")

                if flow == "bullish":
                    confidence += 15
                    reasons.append(f"Flow leans bullish (chgPCR={chg_pcr:.2f})")
                    action = "BUY_CE"
                elif flow == "bearish":
                    confidence += 15
                    reasons.append(f"Flow leans bearish (chgPCR={chg_pcr:.2f})")
                    action = "BUY_PE"
                    strike = self._select_strike(chain_df, spot, "put", "atm")
                else:
                    action = "BUY_STRADDLE"
                    reasons.append("Neutral flow -- straddle to capture the move")

                # Chase filter: don't enter after big intraday move
                if move_pct > 0.8:
                    confidence -= 20
                    reasons.append(f"WARNING: {move_pct:.0%} of range used -- late entry")

                if confidence >= 50:
                    return {**base, "action": action, "strike": strike,
                            "reason": " | ".join(reasons),
                            "confidence": min(confidence, 95),
                            "sl_pct": 30, "tp_pct": 50}
            else:
                reasons.append(f"DTE={dte} too short for vol play at flip")

        # --- ABOVE GAMMA FLIP: LONG BIAS ---
        if dist_flip > self.FLIP_PROXIMITY:
            confidence = 25
            reasons = [f"Above gamma flip ({gamma_flip:.0f}) by {dist_flip:.0f} pts"]
            reasons.append("Negative GEX above flip = MMs amplify upside")

            if flow == "bullish":
                confidence += 25
                reasons.append(f"Bullish flow confirms (chgPCR={chg_pcr:.2f})")
            elif flow == "neutral":
                confidence += 10

            # Room to call wall
            if dist_cw > 80:
                confidence += 10
                reasons.append(f"Room to call wall ({call_wall:.0f}, {dist_cw:.0f} pts)")
            elif dist_cw <= 50:
                confidence -= 10
                reasons.append(f"Near call wall -- limited upside ({dist_cw:.0f} pts)")

            # Chase filter
            if move_pct > 0.8 and intraday_move > 0:
                confidence -= 25
                reasons.append(f"CHASING: {move_pct:.0%} of range used, rallied {intraday_move:.0f} pts")
            elif move_pct < 0.4:
                confidence += 10
                reasons.append("Early in the move -- good entry timing")

            # Pullback entry: price dipped then recovered above flip
            if intraday_move > 0 and (spot - today_open) < intraday_move * 0.6:
                confidence += 5

            # DTE check
            if dte <= 1:
                confidence -= 10
                reasons.append("Very short DTE -- theta risk")

            strike = self._select_strike(chain_df, spot, "call", "atm")
            sl_pct = 25
            tp_pct = 50

            # Breaking call wall = gamma squeeze
            if dist_cw <= 0:
                confidence += 20
                reasons.append("BREAKING CALL WALL -- potential gamma squeeze")
                tp_pct = 80

            if confidence >= 50:
                return {**base, "action": "BUY_CE", "strike": strike,
                        "reason": " | ".join(reasons),
                        "confidence": min(confidence, 95),
                        "sl_pct": sl_pct, "tp_pct": tp_pct}

        # --- BELOW GAMMA FLIP: SHORT BIAS ---
        if dist_flip < -self.FLIP_PROXIMITY:
            confidence = 25
            reasons = [f"Below gamma flip ({gamma_flip:.0f}) by {abs(dist_flip):.0f} pts"]
            reasons.append("Negative GEX below flip = MMs amplify downside")

            if flow == "bearish":
                confidence += 25
                reasons.append(f"Bearish flow confirms (chgPCR={chg_pcr:.2f})")
            elif flow == "neutral":
                confidence += 10

            if dist_pw > 80:
                confidence += 10
                reasons.append(f"Room to put wall ({put_wall:.0f}, {dist_pw:.0f} pts)")
            elif dist_pw <= 50:
                confidence -= 10
                reasons.append(f"Near put wall -- limited downside ({dist_pw:.0f} pts)")

            if move_pct > 0.8 and intraday_move < 0:
                confidence -= 25
                reasons.append(f"CHASING: {move_pct:.0%} of range used, sold {abs(intraday_move):.0f} pts")
            elif move_pct < 0.4:
                confidence += 10
                reasons.append("Early in the move -- good entry timing")

            if dte <= 1:
                confidence -= 10
                reasons.append("Very short DTE -- theta risk")

            strike = self._select_strike(chain_df, spot, "put", "atm")
            sl_pct = 25
            tp_pct = 50

            if dist_pw <= 0:
                confidence += 20
                reasons.append("BREAKING PUT WALL -- potential gamma cascade")
                tp_pct = 80

            if confidence >= 50:
                return {**base, "action": "BUY_PE", "strike": strike,
                        "reason": " | ".join(reasons),
                        "confidence": min(confidence, 95),
                        "sl_pct": sl_pct, "tp_pct": tp_pct}

        # --- WALL BOUNCE IN NEGATIVE GEX ---
        # Near put wall with bullish flow = wall likely holds, bounce LONG
        if dist_pw <= self.WALL_PROXIMITY and flow == "bullish":
            confidence = 30
            reasons = [f"Near put wall ({put_wall:.0f}, {dist_pw:.0f} pts) in negative GEX"]
            reasons.append(f"Strong bullish flow (chgPCR={chg_pcr:.2f}) suggests wall holds")
            reasons.append("Bounce from put wall = reversal LONG")

            if chg_pcr > 2.0:
                confidence += 20
                reasons.append("Very strong put writing -- support being built")
            else:
                confidence += 10

            if move_pct < 0.5:
                confidence += 10
                reasons.append("Early in session -- time for the bounce")
            if move_pct > 0.8 and intraday_move < 0:
                confidence -= 10
                reasons.append("Late selloff -- bounce may be weak")

            strike = self._select_strike(chain_df, spot, "call", "atm")
            if confidence >= 50:
                return {**base, "action": "BUY_CE", "strike": strike,
                        "reason": " | ".join(reasons),
                        "confidence": min(confidence, 85),
                        "sl_pct": 25, "tp_pct": 50}

        # Near call wall with bearish flow = wall likely holds, reversal SHORT
        if dist_cw <= self.WALL_PROXIMITY and flow == "bearish":
            confidence = 30
            reasons = [f"Near call wall ({call_wall:.0f}, {dist_cw:.0f} pts) in negative GEX"]
            reasons.append(f"Bearish flow (chgPCR={chg_pcr:.2f}) suggests wall holds")
            reasons.append("Rejection from call wall = reversal SHORT")

            if chg_pcr < 0.5:
                confidence += 20
                reasons.append("Heavy call writing -- resistance strong")
            else:
                confidence += 10

            if move_pct < 0.5:
                confidence += 10
            if move_pct > 0.8 and intraday_move > 0:
                confidence -= 10

            strike = self._select_strike(chain_df, spot, "put", "atm")
            if confidence >= 50:
                return {**base, "action": "BUY_PE", "strike": strike,
                        "reason": " | ".join(reasons),
                        "confidence": min(confidence, 85),
                        "sl_pct": 25, "tp_pct": 50}

        # --- FLOW CONFLICT: price moving one way, flow says other ---
        if dist_flip < -self.FLIP_PROXIMITY and flow == "bullish":
            return {**base, "action": "NO_TRADE", "strike": 0,
                    "reason": "FLOW CONFLICT: price below flip (bearish) but flow is bullish (chgPCR="
                              f"{chg_pcr:.2f}) -- wait for direction to resolve",
                    "confidence": 0, "sl_pct": 0, "tp_pct": 0}
        if dist_flip > self.FLIP_PROXIMITY and flow == "bearish":
            return {**base, "action": "NO_TRADE", "strike": 0,
                    "reason": "FLOW CONFLICT: price above flip (bullish) but flow is bearish (chgPCR="
                              f"{chg_pcr:.2f}) -- wait for direction to resolve",
                    "confidence": 0, "sl_pct": 0, "tp_pct": 0}

        return {**base, "action": "NO_TRADE", "strike": 0,
                "reason": "Negative GEX but filters not met: " + " | ".join(reasons),
                "confidence": 0, "sl_pct": 0, "tp_pct": 0}

    def _flip_edge_signal(self, chain_df, spot, dte,
                          call_wall, put_wall, gamma_flip, max_pain,
                          pcr, chg_pcr, flow, call_iv, put_iv,
                          iv_skew, theta_daily,
                          dist_cw, dist_pw, dist_flip, dist_mp,
                          intraday_move, move_pct, base):
        """
        At gamma flip edge: regime is unstable, big move likely.
        Use flow + max pain + OI change to pick direction.
        If direction unclear, straddle or wait.
        """
        confidence = 30
        reasons = [f"At gamma flip edge ({gamma_flip:.0f}, {dist_flip:+.0f} pts)"]
        reasons.append("Regime unstable here -- big move possible either way")

        direction_score = 0

        # Flow direction
        if flow == "bullish":
            direction_score += 2
            reasons.append(f"Bullish flow (chgPCR={chg_pcr:.2f})")
        elif flow == "bearish":
            direction_score -= 2
            reasons.append(f"Bearish flow (chgPCR={chg_pcr:.2f})")

        # Max pain pull
        if dist_mp > 30:
            direction_score -= 1
            reasons.append(f"Above max pain ({max_pain:.0f}) -- gravity pulls down")
        elif dist_mp < -30:
            direction_score += 1
            reasons.append(f"Below max pain ({max_pain:.0f}) -- gravity pulls up")

        # Room to walls (favor direction with more room)
        if dist_cw > dist_pw + 50:
            direction_score += 1
            reasons.append(f"More room upside (CW {dist_cw:.0f} vs PW {dist_pw:.0f})")
        elif dist_pw > dist_cw + 50:
            direction_score -= 1
            reasons.append(f"More room downside (PW {dist_pw:.0f} vs CW {dist_cw:.0f})")

        # PCR bias
        if pcr > 1.3:
            direction_score += 1
            reasons.append(f"High PCR ({pcr:.2f}) = put support, bullish lean")
        elif pcr < 0.7:
            direction_score -= 1
            reasons.append(f"Low PCR ({pcr:.2f}) = call heavy, bearish lean")

        # Chase filter
        if move_pct > 0.75:
            confidence -= 15
            reasons.append(f"Late entry -- {move_pct:.0%} of range used")

        if dte <= 1:
            confidence -= 10
            reasons.append(f"Short DTE ({dte}) -- theta risk at flip")

        if direction_score >= 2:
            confidence += 20
            action = "BUY_CE"
            strike = self._select_strike(chain_df, spot, "call", "atm")
            reasons.append(f"Direction score {direction_score:+d} -> BULLISH")
        elif direction_score <= -2:
            confidence += 20
            action = "BUY_PE"
            strike = self._select_strike(chain_df, spot, "put", "atm")
            reasons.append(f"Direction score {direction_score:+d} -> BEARISH")
        elif dte >= 3:
            confidence += 10
            action = "BUY_STRADDLE"
            strike = self._select_strike(chain_df, spot, "call", "atm")
            reasons.append(f"Direction unclear (score={direction_score:+d}) + DTE={dte} -> straddle")
        else:
            return {**base, "action": "NO_TRADE", "strike": 0,
                    "reason": "At flip edge, direction unclear, short DTE -- wait | " + " | ".join(reasons),
                    "confidence": 0, "sl_pct": 0, "tp_pct": 0}

        if confidence >= 45:
            return {**base, "action": action, "strike": strike,
                    "reason": " | ".join(reasons),
                    "confidence": min(confidence, 90),
                    "sl_pct": 25, "tp_pct": 50}

        return {**base, "action": "NO_TRADE", "strike": 0,
                "reason": "At flip edge, confidence too low | " + " | ".join(reasons),
                "confidence": 0, "sl_pct": 0, "tp_pct": 0}

    def _select_strike(self, chain_df, spot, option_type, style="atm"):
        """
        Select the best strike based on liquidity and proximity.
        ATM = nearest to spot with decent volume.
        """
        if option_type == "call":
            candidates = chain_df[chain_df["strike"] >= spot - 50].copy()
            candidates["dist"] = (candidates["strike"] - spot).abs()
            candidates = candidates[candidates["call_vol"] >= self.MIN_VOLUME]
            if candidates.empty:
                candidates = chain_df[chain_df["strike"] >= spot - 50].copy()
                candidates["dist"] = (candidates["strike"] - spot).abs()
            return float(candidates.loc[candidates["dist"].idxmin(), "strike"])
        else:
            candidates = chain_df[chain_df["strike"] <= spot + 50].copy()
            candidates["dist"] = (candidates["strike"] - spot).abs()
            candidates = candidates[candidates["put_vol"] >= self.MIN_VOLUME]
            if candidates.empty:
                candidates = chain_df[chain_df["strike"] <= spot + 50].copy()
                candidates["dist"] = (candidates["strike"] - spot).abs()
            return float(candidates.loc[candidates["dist"].idxmin(), "strike"])


def format_signal(signal):
    """Pretty-print a trade signal."""
    lines = []
    lines.append("=" * 65)
    lines.append("  S18 GEX OPTIONS STRATEGY -- SIGNAL")
    lines.append("=" * 65)

    action = signal["action"]
    if action == "NO_TRADE":
        lines.append(f"  ACTION:     NO TRADE")
        lines.append(f"  Reason:     {signal['reason']}")
    else:
        lines.append(f"  ACTION:     {action}")
        lines.append(f"  STRIKE:     {signal['strike']:.0f}")
        lines.append(f"  CONFIDENCE: {signal['confidence']}%")
        lines.append(f"  SL:         {signal['sl_pct']}% of premium")
        lines.append(f"  TP:         {signal['tp_pct']}% of premium")
        lines.append(f"  Reason:     {signal['reason']}")

    lines.append("")
    lines.append(f"  Regime:     {signal['regime'].upper()}")
    lvl = signal["levels"]
    lines.append(f"  Call Wall:  {lvl['call_wall']:.0f}")
    lines.append(f"  Put Wall:   {lvl['put_wall']:.0f}")
    lines.append(f"  Gamma Flip: {lvl['gamma_flip']:.0f}")
    lines.append(f"  Max Pain:   {lvl['max_pain']:.0f}")

    m = signal["metrics"]
    lines.append("")
    lines.append(f"  PCR:        {m['pcr']:.3f}")
    lines.append(f"  ChgOI PCR:  {m['chg_oi_pcr']:.3f}  ({m['flow']})")
    lines.append(f"  IV:         CE={m['call_iv']:.2f}  PE={m['put_iv']:.2f}  Skew={m['iv_skew']:+.2f}")
    lines.append(f"  DTE:        {m['dte']}  (theta ~{m['theta_daily_pct']:.1f}%/day)")
    lines.append(f"  Intraday:   {m['intraday_move']:+.0f} pts  ({m['move_pct_of_range']:.0%} of range)")
    lines.append(f"  Dist:       CW={m['dist_call_wall']:+.0f}  PW={m['dist_put_wall']:+.0f}  GF={m['dist_gamma_flip']:+.0f}  MP={m['dist_max_pain']:+.0f}")

    lines.append("=" * 65)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from core.gex_engine import compute_gex_from_chain

    chain = pd.read_csv("data/nifty_options_chain_20261029.csv")
    DTE = 5

    strat = GEXOptionsStrategy()

    scenarios = [
        ("CURRENT: Spot=22865, rallied 175 pts, 1:36 PM", 22865.0, 22690.65, 22876.55, 22656.75, 1336),
        ("PULLBACK TO FLIP: Spot=22845, dipped to flip", 22845.0, 22690.65, 22876.55, 22656.75, 1100),
        ("MORNING DIP TO PUT WALL: Spot=22710, early AM", 22710.0, 22690.65, 22720.0, 22660.0, 1000),
        ("BREAK ABOVE 22900: Spot=22920, morning breakout", 22920.0, 22690.65, 22930.0, 22680.0, 1030),
        ("SELLOFF BELOW FLIP: Spot=22780, dropped through", 22780.0, 22850.0, 22860.0, 22770.0, 1130),
        ("NEAR CALL WALL: Spot=22970, approaching wall", 22970.0, 22700.0, 22980.0, 22690.0, 1200),
    ]

    for label, spot, day_open, day_high, day_low, hhmm in scenarios:
        gex = compute_gex_from_chain(chain, spot, days_to_expiry=DTE)
        signal = strat.analyze(chain, spot, gex, DTE, day_open, day_high, day_low, hhmm)
        print(f"\n  SCENARIO: {label}")
        print(format_signal(signal))
