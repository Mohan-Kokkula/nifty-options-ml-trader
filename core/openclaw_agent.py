"""
openclaw_agent.py -- Autonomous Trading Agent (Agentic AI)
==========================================================
Replaces manual morning briefs and Claude Desktop.

This agent:
  1. Searches the web for market context (VIX, FII/DII, GIFT Nifty, US markets)
  2. Builds a comprehensive market view
  3. Decides whether to trade, trade cautiously, or sit out
  4. Sends context to /analyze and controls /pilot
  5. Monitors positions intraday and handles square-off

Runs on schedule:
  - 09:00  Morning routine (search + analyze + start pilot)
  - Every 30 min  Intraday check (positions + risk)
  - 13:00  Mid-session review
  - 15:15  Evening square-off

Uses Anthropic API with tool_use (web search + bot control).
Falls back to OpenAI if Anthropic unavailable.
"""

import json
import logging
import os
import threading
import time
import requests
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)

# Bot base URL
BOT_URL = "http://localhost:8080"

# ======================================================================
# TOOLS -- what the agent can do
# ======================================================================

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current market information. "
            "Use for: Nifty outlook, India VIX, GIFT Nifty, FII DII data, "
            "US market close, economic events, RBI decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'Nifty outlook today 24 March 2026'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "bot_analyze",
        "description": (
            "Send market context to the trading bot's /analyze endpoint. "
            "Claude inside the bot will use this context for better trade decisions. "
            "Pass your full market summary as context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Market context string, e.g. 'VIX=14.2(CALM) FII=+2100cr GIFT=+80pts BIAS=BULLISH'"
                },
                "execute": {
                    "type": "boolean",
                    "description": "If true, auto-execute the trade if confidence is high enough",
                    "default": False
                }
            },
            "required": ["context"]
        }
    },
    {
        "name": "bot_pilot",
        "description": (
            "Control the auto-pilot. Actions: 'start', 'stop', 'status'. "
            "When starting, set min_confidence higher on risky days."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "status"]
                },
                "min_confidence": {
                    "type": "integer",
                    "description": "Minimum confidence % to execute trades (60-80)",
                    "default": 65
                },
                "max_trades": {
                    "type": "integer",
                    "description": "Max trades per day (2-5)",
                    "default": 5
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "bot_positions",
        "description": "Check current open positions and P&L from the broker.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "bot_close_all",
        "description": (
            "Emergency: stop pilot + cancel all orders. "
            "Use when market crashes, VIX spikes, or daily loss limit approached."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
]

# ======================================================================
# TOOL EXECUTION
# ======================================================================

def _web_search(query: str) -> str:
    """Search using DuckDuckGo Instant Answer API (no key needed)."""
    try:
        # DuckDuckGo HTML search
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code == 200:
            # Extract text snippets from results
            from html.parser import HTMLParser
            class SnippetParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.snippets = []
                    self._in_snippet = False
                    self._current = ""
                def handle_starttag(self, tag, attrs):
                    cls = dict(attrs).get("class", "")
                    if "result__snippet" in cls:
                        self._in_snippet = True
                        self._current = ""
                def handle_endtag(self, tag):
                    if self._in_snippet:
                        self._in_snippet = False
                        if self._current.strip():
                            self.snippets.append(self._current.strip())
                def handle_data(self, data):
                    if self._in_snippet:
                        self._current += data

            parser = SnippetParser()
            parser.feed(resp.text)
            if parser.snippets:
                return "\n".join(parser.snippets[:5])

        return f"Search completed but no clear results for: {query}"
    except Exception as e:
        return f"Search failed: {e}"


def _bot_call(endpoint: str, body: dict = None, method="POST") -> dict:
    """Call bot HTTP endpoint."""
    url = f"{BOT_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=body or {}, timeout=30)
        return resp.json()
    except requests.ConnectionError:
        return {"error": "Bot not running"}
    except Exception as e:
        return {"error": str(e)}


def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool and return result as string."""
    try:
        if name == "web_search":
            result = _web_search(input_data["query"])
        elif name == "bot_analyze":
            result = _bot_call("/analyze", {
                "context": input_data.get("context", ""),
                "execute": input_data.get("execute", False),
            })
        elif name == "bot_pilot":
            body = {"action": input_data["action"]}
            if "min_confidence" in input_data:
                body["min_confidence"] = input_data["min_confidence"]
            if "max_trades" in input_data:
                body["max_trades"] = input_data["max_trades"]
            result = _bot_call("/pilot", body)
        elif name == "bot_positions":
            result = _bot_call("/positions", {})
        elif name == "bot_close_all":
            # Operational-safety fix #3 (2026-07-21): flattens any open
            # position (close-then-confirm) before stopping the pilot,
            # instead of only stopping the loop + cancelling pending orders.
            result = _bot_call("/emergency_stop", {"reason": "agent_bot_close_all"})
        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ======================================================================
# SYSTEM PROMPT
# ======================================================================

AGENT_SYSTEM_PROMPT = """You are OpenClaw, an autonomous Nifty options trading agent.

TODAY: {today}  |  DAY: {day_of_week}  |  TIME: {time_now}

You have tools to:
1. web_search - Search for market news, VIX, FII/DII data, GIFT Nifty
2. bot_analyze - Send market context to the trading bot
3. bot_pilot - Start/stop/check the auto-pilot
4. bot_positions - Check open positions
5. bot_close_all - Emergency stop

=== YOUR ROLE ===
You are the intelligence layer ABOVE the trading bot. The bot handles price
mechanics (ML model, option chain, order execution, risk management). YOU handle
world context -- news, sentiment, events, macro conditions.

The bot CANNOT search the web. Without your context, it trades blind. Your job
is to ensure every decision has the full picture.

=== ACCURACY CHECKLIST (run before every bot_analyze call) ===
Before calling bot_analyze, you MUST have answers for ALL of these:
- Today's VIX level (search if unknown)
- FII/DII data (net buyer or seller?)
- GIFT Nifty / SGX direction (gap up or down?)
- US markets overnight (Dow, S&P, Nasdaq)
- Any major event today (RBI, FOMC, Budget, expiry?)
- Is today the weekly expiry day? (check NIFTY_EXPIRY date)
- Is market sideways or trending?

If ANY item is unknown -- search the web first, then call bot_analyze.

=== MORNING ROUTINE (09:00) ===
Search for ALL of these (make separate searches):
- "Nifty outlook today {today_str}"
- "India VIX today"
- "GIFT Nifty today pre market"
- "FII DII data today NSE"
- "US markets overnight Dow S&P Nasdaq"

Then decide ONE of:

A) TRADE NORMALLY: VIX < 16, clear FII flow, no events, GIFT gap < 100pts
   -> bot_pilot start with min_confidence=65, max_trades=5
   -> bot_analyze with full context

B) TRADE WITH CAUTION: VIX 16-20, or mixed signals, or minor event
   -> bot_pilot start with min_confidence=75, max_trades=2
   -> bot_analyze with context + warnings

C) DO NOT TRADE TODAY:
   - VIX > 22 (extreme fear)
   - Binary event with unknown outcome (RBI decision day BEFORE announcement)
   - US markets crashed > 1.5% overnight AND VIX elevated
   - Budget day (too unpredictable)
   - Global circuit breaker / panic day
   -> Do NOT start pilot. Explain why.

D) SIDEWAYS -- WAIT FOR BREAKOUT:
   - No clear FII buy or sell (near zero)
   - VIX flat, no events, global markets flat
   - Previous 2-3 days ranging without breakout
   - Nifty day range < 150 pts
   -> Do NOT start pilot yet
   -> Monitor every 30 min for breakout
   -> If Nifty breaks above prev day high or below prev day low -> start pilot

ALWAYS call bot_analyze with your full context string like:
"VIX=14.2(CALM) FII=+2100cr(BUY) GIFT=+80pts US=DOW+0.3% EVENTS=NONE BIAS=BULLISH TREND=UP SESSION=MORNING"

=== MARKET INTELLIGENCE RULES ===

BULLISH signals (prefer CE trades):
- FII buying > 1000 crore
- GIFT Nifty gap up > 50 pts
- US markets up overnight
- VIX falling or below 13
- Nifty above previous day high
- RBI kept rates unchanged (positive surprise)

BEARISH signals (prefer PE trades):
- FII selling > 1000 crore
- GIFT Nifty gap down > 50 pts
- US markets fell overnight
- VIX rising or above 16
- Nifty below previous day low
- Rate hike surprise, bad macro data

STAY OUT signals:
- VIX > 22
- Binary event (RBI meeting BEFORE announcement)
- Budget day
- Nifty in tight 100pt range (sideways)
- Global circuit breaker / panic

=== EXPIRY DAY RULES (based on NIFTY_EXPIRY date — usually Tuesday, shifts on holidays) ===
- Options decay fast -- avoid buying options after 13:00
- If trading, only trade between 09:30-12:30
- Set higher confidence: min_confidence=75, max_trades=2
- Reduce risk -- expiry day volatility is unpredictable

=== SIDEWAYS MARKET HANDLING ===
The bot has ADX < 20 sideways gate, but YOU must also detect sideways from context.

How to detect sideways from news:
- "Nifty consolidating" or "Nifty range bound" in search results
- No catalyst, global markets flat
- VIX below 12 = extreme calm = no movement expected
- Previous 2-3 days ranging

What to do:
- Do NOT start pilot
- Monitor for breakout every 30 min (search "Nifty live")
- If breakout confirmed (Nifty breaks prev day high/low with volume):
  -> bot_pilot start with min_confidence=70
  -> bot_analyze with breakout context

=== INTRADAY CHECK (every 30 min, 09:30-15:00) ===
- bot_positions to check P&L
- bot_pilot status to check trades taken
- If 3+ trades and net losing -> stop pilot
- If market fell > 300pts from open -> bot_close_all
- If VIX spiked suddenly -> stop pilot
- Search "Nifty live" to check current direction

=== MID-SESSION REVIEW (13:00) ===
- Search "Nifty afternoon outlook"
- Check positions and P&L
- Decide: continue, raise confidence, or stop for the day
- Call bot_analyze with updated context if continuing
- On expiry days: STOP all new trades at 13:00

=== EVENING ROUTINE (15:15) ===
- bot_pilot stop
- bot_positions for final P&L
- Close any remaining positions if needed (bot_close_all)
- Report the day's performance

=== CONTEXT FORMAT ===
Build one-line summary for bot_analyze:
VIX=14.2(CALM) | FII=+2100cr(BUY) | GIFT=+80pts | US=DOW+0.3% |
EVENTS=NONE | BIAS=BULLISH | TREND=UP | SESSION=MORNING

The more context you pass, the better Claude inside the bot performs.

Be concise. Act decisively. Search first, decide second."""


# ======================================================================
# AGENT LOOP -- calls LLM with tools in a loop
# ======================================================================

class OpenClawAgent:
    """Autonomous trading agent using Claude/OpenAI with tool use."""

    def __init__(self, anthropic_key: str = "", openai_key: str = "",
                 model: str = "claude-sonnet-4-6", openai_model: str = "gpt-4o"):
        self.anthropic_key = anthropic_key
        self.openai_key = openai_key
        self.model = model
        self.openai_model = openai_model
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_morning: Optional[date] = None
        self._last_intraday: float = 0
        self._last_result: str = ""

        if not anthropic_key and not openai_key:
            logger.warning("OpenClaw agent: no API keys -- agent disabled")

    @property
    def enabled(self):
        return bool(self.anthropic_key) or bool(self.openai_key)

    def run_task(self, task: str, max_turns: int = 10) -> str:
        """Run agent with a task. Returns final text response."""
        if not self.enabled:
            return "Agent disabled (no API keys)"

        now = datetime.now()
        system = AGENT_SYSTEM_PROMPT.format(
            today=now.strftime("%Y-%m-%d"),
            today_str=now.strftime("%d %B %Y"),
            day_of_week=now.strftime("%A"),
            time_now=now.strftime("%H:%M"),
        )

        messages = [{"role": "user", "content": task}]

        for turn in range(max_turns):
            try:
                response = self._call_llm(system, messages)
            except Exception as e:
                logger.error(f"Agent LLM call failed: {e}")
                return f"Agent error: {e}"

            # Check if there are tool calls
            tool_calls = [b for b in response.get("content", [])
                         if b.get("type") == "tool_use"]

            if not tool_calls:
                # No more tools -- extract final text
                text_parts = [b.get("text", "") for b in response.get("content", [])
                             if b.get("type") == "text"]
                final = "\n".join(text_parts).strip()
                self._last_result = final
                logger.info(f"Agent completed in {turn + 1} turns")
                return final

            # Execute tools and continue
            # Add assistant message with tool calls
            messages.append({"role": "assistant", "content": response["content"]})

            # Build tool results
            tool_results = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc.get("input", {})
                logger.info(f"Agent tool: {tool_name}({json.dumps(tool_input)[:100]})")

                result_str = execute_tool(tool_name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result_str[:3000],  # truncate large results
                })

            messages.append({"role": "user", "content": tool_results})

        return "Agent reached max turns without completing"

    def _call_llm(self, system: str, messages: list) -> dict:
        """Call Anthropic or OpenAI with tools."""
        if self.anthropic_key:
            try:
                return self._call_anthropic(system, messages)
            except Exception as e:
                logger.warning(f"Anthropic failed: {e}")
                if self.openai_key:
                    return self._call_openai(system, messages)
                raise

        if self.openai_key:
            return self._call_openai(system, messages)

        raise RuntimeError("No API keys available")

    def _call_anthropic(self, system: str, messages: list) -> dict:
        """Call Anthropic Claude API with tools."""
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 2048,
                "system": system,
                "tools": TOOLS,
                "messages": messages,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def _call_openai(self, system: str, messages: list) -> dict:
        """Call OpenAI API with tools, map to Anthropic format."""
        # Convert Anthropic tool format to OpenAI
        oai_tools = []
        for t in TOOLS:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
            })

        # Convert Anthropic messages to OpenAI format
        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    oai_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Tool results
                    for item in content:
                        if item.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"],
                            })
            elif msg["role"] == "assistant":
                content = msg["content"]
                if isinstance(content, list):
                    text = ""
                    tool_calls = []
                    for item in content:
                        if item.get("type") == "text":
                            text += item.get("text", "")
                        elif item.get("type") == "tool_use":
                            tool_calls.append({
                                "id": item["id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item.get("input", {})),
                                }
                            })
                    oai_msg = {"role": "assistant", "content": text or None}
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    oai_messages.append(oai_msg)

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.openai_model,
                "max_tokens": 2048,
                "tools": oai_tools,
                "messages": oai_messages,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]

        # Convert OpenAI response back to Anthropic format
        content = []
        if choice.get("content"):
            content.append({"type": "text", "text": choice["content"]})
        for tc in choice.get("tool_calls", []):
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })
        return {"content": content}

    # ==================================================================
    # SCHEDULED TASKS
    # ==================================================================

    def morning_routine(self):
        """Run full morning routine -- search, analyze, start pilot."""
        today = date.today()
        if self._last_morning == today:
            logger.info("Agent: morning routine already done today")
            return self._last_result

        logger.info("=== OpenClaw Agent: MORNING ROUTINE ===")
        result = self.run_task(
            "Run your morning routine. Search for all market context, "
            "build your view, call bot_analyze with context, and start "
            "the pilot with appropriate settings. Be thorough with searches."
        )
        self._last_morning = today
        logger.info(f"Agent morning result: {result[:200]}")
        return result

    def intraday_check(self):
        """Quick intraday check -- positions, risk, status."""
        logger.info("=== OpenClaw Agent: INTRADAY CHECK ===")
        result = self.run_task(
            "Quick intraday check. Check bot_positions for P&L, "
            "bot_pilot status for trades taken. If losing badly or "
            "3+ losing trades, stop the pilot. Search 'Nifty live' "
            "for current market direction. Keep it brief."
        )
        self._last_intraday = time.monotonic()
        logger.info(f"Agent intraday result: {result[:200]}")
        return result

    def mid_session_review(self):
        """13:00 review -- decide whether to continue."""
        logger.info("=== OpenClaw Agent: MID-SESSION REVIEW ===")
        return self.run_task(
            "Mid-session review at 13:00. Check positions, search for "
            "'Nifty afternoon outlook'. Decide: continue trading, raise "
            "confidence threshold, or stop for the day. Call bot_analyze "
            "with updated context if continuing."
        )

    def evening_squareoff(self):
        """15:15 routine -- stop everything, report P&L."""
        logger.info("=== OpenClaw Agent: EVENING SQUARE-OFF ===")
        return self.run_task(
            "Evening square-off. Stop the pilot, check final positions, "
            "report the day's P&L. Close any remaining positions if needed."
        )

    # ==================================================================
    # BACKGROUND SCHEDULER
    # ==================================================================

    def start(self):
        """Start the agent scheduler in background."""
        if self._running:
            return
        if not self.enabled:
            logger.warning("Agent not started (no API keys)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop,
                                        daemon=True, name="openclaw-agent")
        self._thread.start()
        logger.info("OpenClaw Agent STARTED (autonomous mode)")

    def stop(self):
        self._running = False
        logger.info("OpenClaw Agent STOPPED")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "last_morning": str(self._last_morning) if self._last_morning else None,
            "last_result": self._last_result[:300] if self._last_result else None,
        }

    def _scheduler_loop(self):
        """Main scheduler -- runs tasks at the right times."""
        while self._running:
            try:
                now = datetime.now()
                hm = now.hour * 100 + now.minute
                weekday = now.weekday()  # 0=Mon ... 6=Sun

                # Skip weekends
                if weekday >= 5:
                    time.sleep(60)
                    continue

                # Morning routine: 09:00 - 09:10
                if 900 <= hm <= 910 and self._last_morning != now.date():
                    self.morning_routine()

                # Intraday checks: every 30 min from 09:30 to 15:00
                elif (930 <= hm <= 1500
                      and time.monotonic() - self._last_intraday > 1800):
                    if hm >= 1255 and hm <= 1310:
                        self.mid_session_review()
                    else:
                        self.intraday_check()

                # Evening square-off: 15:15 - 15:20
                elif 1515 <= hm <= 1520:
                    self.evening_squareoff()
                    # Sleep until next day
                    time.sleep(3600)
                    continue

            except Exception as e:
                logger.error(f"Agent scheduler error: {e}", exc_info=True)

            time.sleep(30)  # check every 30 seconds


# ======================================================================
# SINGLETON
# ======================================================================

_agent: Optional[OpenClawAgent] = None


def get_agent() -> Optional[OpenClawAgent]:
    return _agent


def init_agent(anthropic_key: str = "", openai_key: str = "",
               model: str = "claude-sonnet-4-6",
               openai_model: str = "gpt-4o") -> OpenClawAgent:
    global _agent
    _agent = OpenClawAgent(anthropic_key, openai_key, model, openai_model)
    return _agent
