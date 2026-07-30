"""
Claude Market Analyzer — Enhanced with SMC + Technical Analysis + ML Integration
================================================================================
Dual-brain architecture:
  1. ML model (V6 ensemble) gives fast signal: CALL/PUT/SKIP
  2. Claude receives ML signal + indicators + spot trend -> confirms or vetoes

Claude is only called when ML says CALL or PUT (saves ~95% of API costs).
"""

import json
import logging
import time
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Nifty options trading agent combining Smart Money Concepts (SMC),
technical analysis, ML model signals, and multi-timeframe awareness.
Your job is to CONFIRM or VETO the ML model's directional signal.

━━━ MARKET STATE DETECTION (check FIRST before anything else) ━━━
Read the 15m_ADX value from MARKET CONTEXT:

SIDEWAYS  (ADX < 20):
- VETO all directional trades → action = WAIT
- Reason: no trend = whipsaw = SL hits
- Do NOT trade CE or PE regardless of ML signal
- Exception: BB squeeze detected → breakout imminent → allow with OTM_2 only

WEAK TREND (ADX 20-25):
- Allow trade only if 4+ indicators align (higher bar than normal)
- Reduce confidence by 10 points
- Use OTM_1 instead of ATM (smaller risk)

STRONG TREND (ADX > 25):
- Trade normally
- DI+ > DI- = bullish trend → only confirm CALL signals
- DI- > DI+ = bearish trend → only confirm PUT signals
- Counter-trend signals → VETO even if ML says otherwise

━━━ TREND ALIGNMENT RULE ━━━
- 15m trend = UP + ML = CALL → CONFIRM (aligned)
- 15m trend = DOWN + ML = PUT → CONFIRM (aligned)
- 15m trend = UP + ML = PUT → VETO (counter-trend)
- 15m trend = DOWN + ML = CALL → VETO (counter-trend)
- 15m trend = SIDEWAYS → VETO always

━━━ SMC FRAMEWORK ━━━
- Higher highs + higher lows = BULLISH → favor CE
- Lower highs + lower lows = BEARISH → favor PE
- Break of Structure (BOS): continuation signal
- Change of Character (CHoCH): reversal signal
- Order blocks = institutional zones (strong entry areas)
- Fair Value Gaps = imbalance zones price fills
- Liquidity sweeps (stop hunts) before real moves

━━━ TECHNICAL CONFLUENCE ━━━
- RSI >70 overbought (PE bias), <30 oversold (CE bias)
- MACD histogram positive+rising = bullish, negative+falling = bearish
- EMA 9>21 on 5-min = short term uptrend
- Daily EMA 9>20 = bullish for weekly options (most relevant)
- Daily EMA 20>50 = bullish for monthly options
- EMA 200 IGNORED — irrelevant for options under 50 days expiry
- BB squeeze = breakout imminent (allow OTM_2 trade)
- Stochastic crossover in extremes = reversal
- ATR ratio >1.2 = vol expanding (buy options)
- ATR ratio <0.8 = vol contracting (avoid buying)

━━━ INDIA VIX RULES (DYNAMIC REGIME) ━━━
- VIX > 30 (EXTREME): WAIT unless VIX is FALLING (recovery trade only, OTM_2, max 1 trade)
- VIX 25-30 (VERY_HIGH): trade WITH 15m trend ONLY, OTM_2, wider SL (1.8x), reduce confidence by 20
- VIX 20-25 (HIGH): tradeable with caution — OTM_1, wider SL (1.5x), bigger TP (1.6x), reduce conf by 15
- VIX 16-20 (ELEVATED): slightly wider SL (1.15x), reduce confidence by 5
- VIX 12-16 (NORMAL): trade normally
- VIX < 12 (CALM): tighter SL/TP (0.85x), increase confidence by 5
NOTE: High VIX = wider ranges = bigger opportunity with adjusted risk. VIX mean-reverts.

━━━ SESSION RULES (IST) ━━━
- Opening 09:15-09:30: gap trades only — trade WITH gap direction immediately
- Morning 09:30-10:30: high vol, strong directional moves — best time
- Mid 10:30-13:30: often range-bound — require ADX > 22 to trade
- Afternoon 13:30-15:00: trend continuation only — no reversals
- After 15:00: NO NEW ENTRIES — only manage existing positions
- After 15:10: WAIT always — theta decay and illiquid
- EXPIRY DAY (from NIFTY_EXPIRY date): trade only 09:15-12:00, use OTM_1 not ATM, reduce confidence by 10
- GAP > 150pts: trade immediately at open in gap direction — highest probability setup

━━━ DECISION LOGIC ━━━
Step 1: Is ADX < 20? → WAIT (sideways, no exceptions unless BB squeeze)
Step 2: Does ML signal align with 15m trend direction? → if no → WAIT
Step 3: Is VIX > 30 (and RISING)? → WAIT. VIX 20-30 = tradeable with VIX regime adjustments
Step 4: Do 3+ indicators confirm? → if no → WAIT
Step 5: All checks passed → CONFIRM with appropriate confidence

RESPONSE FORMAT (strict JSON only, no markdown):
{
  "action": "BUY" | "WAIT",
  "option_type": "CE" | "PE" | null,
  "strike_mode": "ATM" | "OTM_1" | "OTM_2",
  "confidence": 0-100,
  "analysis": "2-3 sentences explaining decision",
  "market_bias": "BULLISH" | "BEARISH" | "NEUTRAL" | "SIDEWAYS",
  "market_state": "TRENDING" | "SIDEWAYS" | "WEAK_TREND",
  "stop_loss_points": 45,
  "target_points": 70,
  "indicators_aligned": 0-6,
  "smc_context": "BOS_UP|BOS_DOWN|CHOCH|ORDER_BLOCK|FVG|NONE",
  "veto_reason": "null or reason if WAIT",
  "exit_by": "15:15"
}

━━━ DYNAMIC SL/TP GUIDANCE (ATR + VIX SCALED — tuned for 90%+ win rate) ━━━
System uses ATR-based SL/TP: SL=2.2x ATR, TP=2.2x ATR.
Wide SL absorbs noise (fewer SL hits), quick TP captures frequent wins.
Bounds: SL 30-80pts, TP 30-130pts (wider in high-VIX regimes).

VIX-scaled adjustments:
- CALM VIX (<12): SL x0.85, TP x0.85 (low vol = tight range)
- NORMAL VIX (12-16): SL/TP as-is
- ELEVATED VIX (16-20): SL x1.15, TP x1.2
- HIGH VIX (20-25): SL x1.5, TP x1.6 (wider range, bigger opportunity)
- VERY HIGH VIX (25-30): SL x1.8, TP x2.0 (trend-only, OTM_2)
- EXTREME VIX (>30): WAIT unless VIX falling (recovery trade)

TRADE MANAGEMENT (critical for 90%+ win rate):
- Breakeven stop: once +15pts profit, move SL to entry+3pts (zero-risk trade)
- Trailing stop: activates at 0.8x SL distance profit, locks 60% of unrealized
- Combined effect: most trades become zero-risk early, then lock profit as it grows

RULES:
- Confidence below 55 = WAIT always
- Sideways market (ADX<20) = WAIT always
- Counter-trend trade = WAIT always (EMA9 must align with trade direction)
- VIX > 30 + RISING = WAIT. VIX 22-30 = tradeable with regime adjustments
- No new entries after 15:00 — only manage existing positions
- Conservative > overtrading — when in doubt, WAIT
- First 20 minutes (09:15-09:35): higher confidence bar (0.60), 3-bar confirmation needed"""


class ClaudeAnalyzer:
    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_OPENAI_MODEL = "gpt-4o"

    def __init__(self, api_key="", model="", max_tokens=1024,
                 auto_execute=False, openai_api_key="", openai_model=""):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.auto_execute = auto_execute
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model or self.DEFAULT_OPENAI_MODEL

        if not self.api_key and not self.openai_api_key:
            logger.warning("No LLM API key set - analyzer disabled")

        providers = []
        if self.api_key:
            providers.append(f"Anthropic ({self.model})")
        if self.openai_api_key:
            providers.append(f"OpenAI ({self.openai_model})")
        if providers:
            logger.info(f"LLM: {' -> '.join(providers)}")

        self._spot_history: list[tuple[str, float]] = []
        self._max_history = 30

    @property
    def enabled(self):
        return bool(self.api_key) or bool(self.openai_api_key)

    def record_spot(self, spot: float):
        ts = datetime.now().strftime("%H:%M")
        self._spot_history.append((ts, spot))
        if len(self._spot_history) > self._max_history:
            self._spot_history.pop(0)

    def analyze(self, spot, option_chain=None, ml_signal=2,
                ml_proba=None, ml_indicators=None, extra_context=""):
        if not self.enabled:
            return {"action": "WAIT", "analysis": "LLM disabled", "confidence": 0}

        if ml_signal == 2:
            return {"action": "WAIT", "analysis": "ML says SKIP",
                    "confidence": 0, "market_bias": "NEUTRAL"}

        ml_dir = "CALL" if ml_signal == 0 else "PUT"
        prompt = self._build_prompt(
            spot, option_chain or [], ml_dir,
            ml_proba or [0.33, 0.33, 0.34],
            ml_indicators or {}, extra_context)

        try:
            resp = self._call_llm(prompt)
            rec = self._parse_response(resp)
            logger.info(
                f"Claude: {rec.get('action')} {rec.get('option_type','')} "
                f"conf={rec.get('confidence')}% aligned={rec.get('indicators_aligned','?')}/6 "
                f"smc={rec.get('smc_context','?')}")
            return rec
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            # Safety: do NOT blindly trust ML when confirmation layer fails.
            # Return WAIT so the pilot skips this cycle rather than taking
            # an unconfirmed trade.
            return {"action": "WAIT", "option_type": None, "strike_mode": "ATM",
                    "confidence": 0,
                    "analysis": f"LLM unavailable ({e}) — skipping for safety",
                    "market_bias": "NEUTRAL"}

    def analyze_with_trader(self, trader, ml_signal=2, ml_proba=None,
                            ml_indicators=None, extra_context=""):
        if not self.enabled:
            return {"action": "WAIT", "analysis": "LLM disabled", "confidence": 0}

        logger.info("Claude analyzing...")
        chain_data = {}
        chain, spot, atm = [], 0, 0
        try:
            chain_data = trader.get_option_chain(width=5)
            spot = chain_data.get("spot", 0)
            chain = chain_data.get("chain", [])
            atm = chain_data.get("atm", 0)
        except Exception:
            pass
        if spot <= 0:
            try:
                spot = trader.get_nifty_spot()
                atm = trader.get_atm_strike()
            except Exception as e:
                return {"action": "WAIT", "analysis": f"No spot: {e}", "confidence": 0}

        self.record_spot(spot)

        # ── Base context ──────────────────────────────────────────
        extra = ""
        if extra_context:
            extra += extra_context.strip() + " "
        if chain_data.get("pcr"):
            extra += f"PCR={chain_data['pcr']:.2f} "
        if chain_data.get("max_ce_oi_strike"):
            extra += f"Resistance(MaxCE_OI)={chain_data['max_ce_oi_strike']} "
        if chain_data.get("max_pe_oi_strike"):
            extra += f"Support(MaxPE_OI)={chain_data['max_pe_oi_strike']} "
        if chain_data.get("source"):
            extra += f"[chain={chain_data['source']}] "

        # ── India VIX ─────────────────────────────────────────────
        try:
            # Try different VIX symbol names — varies by broker
            for vix_sym in ["INDIAVIX", "INDIA VIX", "INDIA_VIX"]:
                try:
                    vix_quote = trader.client.get_quote(vix_sym, "NSE_INDEX")
                    vix = float(vix_quote.get("data", {}).get("ltp", 0) or 0)
                    if vix > 0:
                        if vix > 30: vix_state = "EXTREME"
                        elif vix > 25: vix_state = "VERY_HIGH"
                        elif vix > 20: vix_state = "HIGH"
                        elif vix > 16: vix_state = "ELEVATED"
                        elif vix > 12: vix_state = "NORMAL"
                        else: vix_state = "CALM"
                        extra += f"IndiVIX={vix:.2f}({vix_state}) "
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # ── GIFT Nifty / SGX pre-market ───────────────────────────
        try:
            from core.tv_fetcher import get_tv_fetcher
            # GIFT Nifty only available with authenticated TV session
            gift_df = get_tv_fetcher().get_ohlcv(
                "NIFTY1!", "NSE", interval_minutes=5, n_bars=3
            )
            if not gift_df.empty:
                gift_close = float(gift_df["close"].iloc[-1])
                gift_gap   = gift_close - spot
                extra += f"GIFT_Nifty={gift_close:.0f}(gap={gift_gap:+.0f}) "
        except Exception:
            pass

        # ── Previous day High / Low / Close ───────────────────────
        try:
            from core.tv_fetcher import get_tv_fetcher
            day_df = get_tv_fetcher().get_ohlcv(
                "NIFTY", "NSE", interval_minutes="D", n_bars=3
            )
            if not day_df.empty and len(day_df) >= 2:
                prev = day_df.iloc[-2]
                pd_h = float(prev["high"]);  pd_l = float(prev["low"])
                pd_c = float(prev["close"])
                day_gap_pct = (spot - pd_c) / pd_c * 100
                extra += (f"PrevDay H={pd_h:.0f} L={pd_l:.0f} C={pd_c:.0f} "
                          f"Gap={day_gap_pct:+.2f}% ")
        except Exception:
            pass

        # ── 15-min ADX trend state ────────────────────────────────
        try:
            from core.tv_fetcher import get_tv_fetcher
            df15 = get_tv_fetcher().get_nifty_15min(n_bars=30)
            if not df15.empty and len(df15) >= 15:
                h15, l15, c15 = df15["high"], df15["low"], df15["close"]
                tr15 = pd.concat([
                    h15 - l15,
                    (h15 - c15.shift()).abs(),
                    (l15 - c15.shift()).abs()
                ], axis=1).max(axis=1)
                up15   = h15 - h15.shift(1)
                dn15   = l15.shift(1) - l15
                pdm15  = up15.where((up15 > dn15) & (up15 > 0), 0.0)
                ndm15  = dn15.where((dn15 > up15) & (dn15 > 0), 0.0)
                atr15  = tr15.ewm(span=14, adjust=False).mean()
                pdi15  = 100 * pdm15.ewm(span=14, adjust=False).mean() / (atr15 + 1e-9)
                ndi15  = 100 * ndm15.ewm(span=14, adjust=False).mean() / (atr15 + 1e-9)
                dx15   = 100 * (pdi15 - ndi15).abs() / (pdi15 + ndi15 + 1e-9)
                adx15  = float(dx15.ewm(span=14, adjust=False).mean().iloc[-1])
                dp15   = float(pdi15.iloc[-1]); dm15 = float(ndi15.iloc[-1])
                if adx15 > 25:
                    trend_state = "TREND_UP" if dp15 > dm15 else "TREND_DOWN"
                elif adx15 < 20:
                    trend_state = "SIDEWAYS"
                else:
                    trend_state = "WEAK_TREND"
                extra += f"15m_ADX={adx15:.1f}({trend_state}) DI+={dp15:.1f} DI-={dm15:.1f} "
        except Exception:
            pass

        # ── Days to expiry (from actual NIFTY_EXPIRY date, not hardcoded weekday) ──
        try:
            from core.expiry_utils import get_dte, is_expiry_day, is_pre_expiry
            dte = get_dte()
            expiry_flag = "EXPIRY_DAY" if is_expiry_day() else (
                "PRE_EXPIRY" if is_pre_expiry() else f"DTE={dte}")
            extra += f"{expiry_flag} "
        except Exception:
            pass

        rec = self.analyze(spot, chain, ml_signal, ml_proba, ml_indicators, extra)
        rec["_market_data"] = {
            "spot": spot, "atm": atm,
            "chain_strikes": len(chain),
            "pcr": chain_data.get("pcr", 0),
            "source": chain_data.get("source", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }
        return rec

    def _build_prompt(self, spot, chain, ml_dir, ml_proba, ind, extra):
        now = datetime.now()
        hm = now.hour * 100 + now.minute
        if hm < 915:  sess = "PRE-MARKET"
        elif hm < 930: sess = "OPENING"
        elif hm < 1030: sess = "MORNING"
        elif hm < 1330: sess = "MID-SESSION"
        elif hm < 1510: sess = "AFTERNOON"
        else: sess = "CLOSED"

        p = [
            f"TIME: {now.strftime('%H:%M')} | SESSION: {sess} | SPOT: {spot:.2f}",
            f"ML SIGNAL: {ml_dir} (CALL={ml_proba[0]:.3f} PUT={ml_proba[1]:.3f} SKIP={ml_proba[2]:.3f})",
            "",
        ]

        # ── Market context (new in V8) ────────────────────────────
        if extra:
            p.append("MARKET CONTEXT:")
            # Parse key fields from extra string for readable display
            for chunk in extra.strip().split(" "):
                if chunk:
                    p.append(f"  {chunk}")
            p.append("")

        # ── Technical indicators ──────────────────────────────────
        if ind:
            p.append("INDICATORS (5-min):")
            p.append(f"  RSI14={ind.get('rsi14','?')} RSI7={ind.get('rsi7','?')} MACD_H={ind.get('macd_hist','?')}")
            p.append(f"  EMA_aligned={ind.get('ema_aligned','?')} slope={ind.get('ema_slope','?')}")
            p.append(f"  Stoch5={ind.get('stoch_5','?')} Stoch14={ind.get('stoch_14','?')}")
            p.append(f"  BB_pos={ind.get('bb_position','?')} BB_squeeze={ind.get('bb_squeeze','?')}")
            p.append(f"  CCI={ind.get('cci20','?')} WillR={ind.get('williams_r','?')}")
            p.append(f"  ATR_ratio={ind.get('atr_ratio','?')} Vol_regime={ind.get('vol_regime','?')}")
            p.append(f"  Candle: body={ind.get('body_ratio','?')} doji={ind.get('is_doji','?')} bull_eng={ind.get('bull_engulfing','?')} bear_eng={ind.get('bear_engulfing','?')}")
            p.append(f"  Streak: bull={ind.get('consec_bull',0)} bear={ind.get('consec_bear',0)}")
            p.append("")

        # ── Spot trend ───────────────────────────────────────────
        if len(self._spot_history) >= 2:
            recent = self._spot_history[-8:]
            p.append("SPOT TREND: " + " -> ".join(f"{pr:.0f}" for _, pr in recent))
            chg = recent[-1][1] - recent[0][1]
            all_p = [x[1] for x in self._spot_history]
            p.append(f"Move: {chg:+.0f}pts | High: {max(all_p):.0f} Low: {min(all_p):.0f}")
            p.append("")

        # ── Option chain ─────────────────────────────────────────
        has_data = any((s.get("ce_premium") or 0) > 0 for s in chain)
        if has_data:
            p.append("OPTION CHAIN:")
            total_ce_oi = 0
            total_pe_oi = 0
            for s in chain:
                tag = " *ATM" if s.get("is_atm") else ""
                ce_p  = s.get("ce_premium", 0) or 0
                pe_p  = s.get("pe_premium", 0) or 0
                ce_oi = s.get("ce_oi", 0) or 0
                pe_oi = s.get("pe_oi", 0) or 0
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi
                line = f"  {s['strike']} | CE {ce_p:.1f}"
                if s.get("ce_iv"):
                    line += f" IV={s['ce_iv']:.1f}%"
                line += f" OI={ce_oi:,}"
                if s.get("ce_oi_change"):
                    line += f"({s['ce_oi_change']:+,})"
                line += f" | PE {pe_p:.1f}"
                if s.get("pe_iv"):
                    line += f" IV={s['pe_iv']:.1f}%"
                line += f" OI={pe_oi:,}"
                if s.get("pe_oi_change"):
                    line += f"({s['pe_oi_change']:+,})"
                line += tag
                p.append(line)
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            p.append(f"  Total CE_OI={total_ce_oi:,} PE_OI={total_pe_oi:,} PCR={pcr:.2f}")
        else:
            p.append("CHAIN: N/A. Use indicators + trend.")

        from core.strike_selector import round_to_strike
        atm = round_to_strike(spot)
        p.append(f"\nATM={atm}")
        p.append("\nConfirm or veto. JSON only.")
        return "\n".join(p)

    def _call_llm(self, prompt):
        if self.api_key:
            try:
                return self._call_anthropic(prompt)
            except Exception as e:
                if self.openai_api_key:
                    logger.warning(f"Anthropic failed ({e}), trying OpenAI...")
                else:
                    raise
        if self.openai_api_key:
            return self._call_openai(prompt)
        raise RuntimeError("No LLM available")

    def _call_anthropic(self, prompt):
        r = requests.post(self.ANTHROPIC_API_URL,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.model, "max_tokens": self.max_tokens,
                  "system": SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        if r.status_code != 200:
            logger.error(f"Anthropic {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        txt = ""
        for b in r.json().get("content", []):
            if b.get("type") == "text":
                txt += b.get("text", "")
        return txt

    def _call_openai(self, prompt):
        r = requests.post(self.OPENAI_API_URL,
            headers={"Authorization": f"Bearer {self.openai_api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.openai_model, "max_tokens": self.max_tokens,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": prompt}]},
            timeout=30)
        if r.status_code != 200:
            logger.error(f"OpenAI {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def _parse_response(self, text):
        c = text.strip()
        if c.startswith("```"):
            lines = [l for l in c.split("\n") if not l.strip().startswith("```")]
            c = "\n".join(lines).strip()
        try:
            r = json.loads(c)
        except json.JSONDecodeError:
            return {"action": "WAIT", "analysis": f"Parse fail: {text[:100]}", "confidence": 0}
        r.setdefault("action", "WAIT")
        r.setdefault("option_type", None)
        r.setdefault("strike_mode", "ATM")
        r.setdefault("confidence", 0)
        r.setdefault("analysis", "")
        r.setdefault("market_bias", "NEUTRAL")
        r.setdefault("stop_loss_points", 50)
        r.setdefault("target_points", 100)
        if r["confidence"] < 55:
            r["action"] = "WAIT"
        return r
