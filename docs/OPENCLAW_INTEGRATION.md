# Connecting OpenClaw to the Nifty Trader

## Quick Start

Once the Nifty Trader server is running (`python main.py`),
tell OpenClaw to install the skill from the `skills/` directory.

### Option A: Paste the skill directly

1. Open OpenClaw
2. Say: "I want to add a custom skill for Nifty options trading"
3. Paste the contents of `skills/SKILL.md`
4. OpenClaw will now understand the trading commands

### Option B: Point to the local server

Tell OpenClaw:
```
I have a Nifty options trading server running at http://localhost:8080.
Here are the endpoints:

- POST /trade — Place a trade. Body: {"action": "BUY|SELL", "option_type": "CE|PE", "strike": 24500}
- POST /quote — Get a quote. Body: {"option_type": "CE|PE", "strike": 24500}
- POST /positions — Get positions. Body: {}
- POST /funds — Get funds. Body: {}
- POST /cancel_all — Cancel all orders. Body: {}
- POST /close — Close a position. Body: {"symbol": "NIFTY..."}
- GET /health — Health check

Use this server for all my Nifty options trades.
```

### Option C: Use as an OpenAlgo Skill

OpenAlgo also has its own skill system. You can install the skill in OpenAlgo
and let OpenClaw call OpenAlgo directly:

1. Copy `skills/SKILL.md` to OpenAlgo's skills directory
2. Configure OpenClaw with OpenAlgo's URL and API key
3. OpenClaw will use OpenAlgo's unified API directly

## Example Conversations with OpenClaw

Once the skill is installed, you can say things like:

```
"Buy NIFTY ATM call option"
→ Buys 1 lot of NIFTY CE at the current ATM strike

"Sell 24500 PE expiry next week"
→ Sells NIFTY 24500 PE with next Thursday's expiry

"What's the price of 24600 CE?"
→ Fetches live quote for NIFTY 24600 CE

"Show my positions"
→ Lists all open Nifty option positions

"Close all positions"
→ Squares off everything

"How much margin do I have?"
→ Shows available funds in your Kotak Neo account
```

## Safety Notes

- OpenClaw will route all commands through the security layer
- Every trade triggers email + WhatsApp alerts to your phone
- The withdrawal guard blocks any attempt to move funds out
- Risk limits (daily loss cap, max positions) are enforced automatically
- All trades are logged to the audit CSV

## Monitoring

While OpenClaw trades, you can monitor via:

1. **Terminal logs** — the server prints every action
2. **Email** — HTML-formatted trade alerts
3. **WhatsApp** — instant notifications on your phone
4. **Audit CSV** — full history at `/tmp/trade_audit.csv`
5. **OpenAlgo dashboard** — real-time P&L at http://127.0.0.1:5000
