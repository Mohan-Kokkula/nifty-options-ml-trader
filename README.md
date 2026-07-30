# OpenClaw Nifty Options Trader — V9

AI-powered Nifty options trading system.
Three brains working together: **ML Model** → **Claude AI** → **OpenAlgo** → **Kotak Neo**

---

## System Architecture

```
Claude Desktop (OpenClaw Agent)
    ↓ researches VIX, FII, news, gap direction
    ↓ calls http://localhost:8080
Python Bot (port 8080)
    ↓ fetches TradingView 5TF bars
    ↓ runs V9 ML model (156 features + VIX)
    ↓ calls Claude API for confirmation
    ↓ places order if confidence ≥ 60%
OpenAlgo (port 5000)
    ↓ broker bridge
Kotak Neo → NSE (Nifty Options)
```

---

## One-Time Setup (do this only once)

### 1. Install dependencies
```powershell
pip install uv
uv sync
pip install tradingview-datafeed xgboost lightgbm
```

### 2. Configure settings
```powershell
copy config\settings.example.env config\settings.env
# Edit config\settings.env with your credentials
```

Key settings to fill in:
```env
OPENALGO_API_KEY=your_key
KOTAK_CONSUMER_KEY=your_key
KOTAK_ACCESS_TOKEN=your_token
TV_USERNAME=your_tradingview_email
TV_PASSWORD=your_tradingview_password
CLAUDE_API_KEY=sk-ant-...
NIFTY_EXPIRY=24MAR26          # Update every week on Friday
DEFAULT_QTY=65                 # Current Nifty lot size
MAX_DAILY_LOSS=3000
```

### 3. Download historical data
```powershell
# Download India VIX history (one time, ~2 min)
uv run scripts\download_vix.py

# Verify all CSVs are present
.\update_data.ps1 --verify
```

### 4. Train the ML model
```powershell
# Takes 15-20 minutes
uv run scripts\train_model_v8.py
```

Expected output:
```
Labels: CALL=XX,XXX  PUT=XX,XXX  SKIP=XXX,XXX
ML model loaded: ['xgb', 'lgb'] | 156 features
V8 Training complete!
```

### 5. Setup Claude Desktop MCP
```powershell
# Create Claude config folder
New-Item -ItemType Directory -Force -Path "$env:APPDATA\Claude"

# Create config file
'{"mcpServers":{"fetch":{"command":"uvx","args":["mcp-server-fetch"]}}}' | Out-File -FilePath "$env:APPDATA\Claude\claude_desktop_config.json" -Encoding UTF8

# Install fetch server
uv tool install mcp-server-fetch
```
Then restart Claude Desktop. Test: type `Fetch http://localhost:8080/health` — should return `{"status":"ok"}`.

### 6. Schedule automation (run once as Administrator)
```powershell
# Morning: start bot + morning brief at 09:00 weekdays
$a1 = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File E:\algo-bot\openclaw-v9\morning_brief.ps1" -WorkingDirectory "E:\algo-bot\openclaw-v9"
$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "09:00"
Register-ScheduledTask -TaskName "OpenClaw Morning" -Action $a1 -Trigger $t1 -RunLevel Highest

# Evening: square off at 15:15 weekdays
$a2 = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File E:\algo-bot\openclaw-v9\evening_squareoff.ps1" -WorkingDirectory "E:\algo-bot\openclaw-v9"
$t2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "15:15"
Register-ScheduledTask -TaskName "OpenClaw Evening" -Action $a2 -Trigger $t2 -RunLevel Highest

# Data update at 15:35 weekdays
$a3 = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-Command `"cd E:\algo-bot\openclaw-v9; uv run scripts\update_data.py`"" -WorkingDirectory "E:\algo-bot\openclaw-v9"
$t3 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "15:35"
Register-ScheduledTask -TaskName "OpenClaw DataUpdate" -Action $a3 -Trigger $t3 -RunLevel Highest
```

### 7. Disable laptop sleep when plugged in
```powershell
# Run as Administrator
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
# Also: Settings → Power & Sleep → "When plugged in, closing lid does" → Do Nothing
```

---

## Daily Workflow (every trading day)

### Step 1 — Before market (09:00)
```powershell
# Terminal 1: Start OpenAlgo (if not already running)
cd C:\openalgo
python app.py

# Terminal 2: Start the trading bot
cd E:\algo-bot\openclaw-v9
.\run.ps1
```

Verify bot started:
```
ML model loaded: ['xgb', 'lgb'] | 156 features   ← V9 loaded
PILOT_AUTO_START — dual-brain mode (ML + Claude)  ← all 3 brains active
```

### Step 2 — Start Claude Desktop agent (09:00)
1. Open **Claude Desktop** app
2. Start a new conversation
3. Paste the contents of `skills\SKILL.md`
4. Say: *"Research today's market conditions and start the morning routine"*

Claude will:
- Search web for VIX, FII data, GIFT Nifty, US markets
- Call `POST http://localhost:8080/analyze` with context
- Start monitoring every 30 minutes

### Step 3 — Run morning brief (09:00, auto if scheduled)
```powershell
.\morning_brief.ps1
```
This sends basic context (date, expiry day flag) to the bot and confirms pilot is running.

### Step 4 — Watch the logs
```
Cycle #1: 15m ADX=28.5 (TRENDING) → passing to ML
ML V9: PUT | CALL=0.12 PUT=0.68 SKIP=0.20 | RSI=38.2 VIX=14.1
Claude: BUY PE ATM conf=72%
EXECUTING BUY PE 23250...
```

### Step 5 — End of day (15:15, auto if scheduled)
```powershell
.\evening_squareoff.ps1
```
Stops pilot, cancels open orders, closes any positions.

### Step 6 — Update data (15:35, auto if scheduled)
```powershell
.\update_data.ps1
```
Appends today's bars + VIX to all CSVs.

---

## Weekly Tasks (Friday evening)

### Update expiry date
```env
# config\settings.env
NIFTY_EXPIRY=30MAR26    # change to next Monday's date
```

### Check if retraining needed
Retrain monthly or after 3+ consecutive losing days:
```powershell
uv run scripts\train_model_v8.py
```

---

## Backtest

```powershell
# Full 11-year backtest
uv run backtest.py

# Specific year
uv run backtest.py 2026

# Check model probability distribution
uv run backtest.py debug
```

**V9 backtest results (11 years, 2015-2026):**
- Total trades: 4,444
- Win rate: 86.6%
- Total P&L: ₹38,61,369
- Avg per trade: ₹869

---

## Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_QTY` | 65 | Nifty lot size (current = 65) |
| `MAX_DAILY_LOSS` | 3000 | Stop trading if day loss hits ₹3,000 |
| `MAX_OPEN_POSITIONS` | 3 | Max simultaneous open trades |
| `NIFTY_EXPIRY` | 24MAR26 | Current week expiry — update every Friday |
| `MIN_CONFIDENCE` | 60 | Claude confidence % needed to execute |
| `PILOT_AUTO_START` | true | Start auto-pilot on server startup |
| `TV_USERNAME` | — | TradingView email (for data feed) |
| `TV_PASSWORD` | — | TradingView password |
| `CLAUDE_API_KEY` | — | Anthropic API key |

---

## File Structure

```
openclaw-v9\
├── main.py                    ← HTTP server (port 8080)
├── backtest.py                ← ML backtest
├── run.ps1                    ← Start the bot
├── morning_brief.ps1          ← Morning startup (09:00)
├── evening_squareoff.ps1      ← End of day (15:15)
├── update_data.ps1            ← Update data CSVs
├── config\
│   ├── settings.env           ← Your credentials (never commit)
│   └── settings.example.env   ← Template
├── core\
│   ├── claude_pilot.py        ← Auto-pilot loop
│   ├── claude_analyzer.py     ← Claude AI analysis
│   ├── ml_engine.py           ← ML model inference
│   ├── tv_fetcher.py          ← TradingView data
│   ├── strike_selector.py     ← Option strike selection
│   ├── trader.py              ← Order execution
│   └── openalgo_client.py     ← OpenAlgo REST client
├── scripts\
│   ├── train_model_v8.py      ← V9 model training
│   ├── download_vix.py        ← Download India VIX history
│   └── update_data.py         ← Daily data updater
├── data\
│   ├── nifty_5min.csv         ← 5-min OHLCV (2015-today)
│   ├── nifty_15min.csv        ← 15-min OHLCV
│   ├── nifty_30min.csv        ← 30-min OHLCV
│   ├── nifty_60min.csv        ← 60-min OHLCV
│   ├── nifty_day.csv          ← Daily OHLCV
│   └── india_vix.csv          ← India VIX daily
├── models\
│   ├── nifty_v8_models.pkl    ← Trained XGB + LGB models
│   ├── nifty_v8_scaler.pkl    ← Feature scaler
│   └── feature_cols_v8.pkl    ← Feature list (156 cols)
├── skills\
│   └── SKILL.md               ← Paste into Claude Desktop
├── logs\
│   └── YYYY-MM-DD.log         ← Daily log files
└── docs\
    ├── KOTAK_NEO_SETUP.md
    ├── NOTIFICATION_SETUP.md
    └── FALLBACK_WITHOUT_CLAUDE.md
```

---

## Endpoints (port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/trade` | POST | Place a trade manually |
| `/analyze` | POST | Run Claude analysis |
| `/pilot` | POST | Start/stop/status auto-pilot |
| `/positions` | POST | Get open positions |
| `/funds` | POST | Get available margin |
| `/cancel_all` | POST | Cancel all orders |
| `/close` | POST | Close a specific position |
| `/strikes` | POST | Get option chain / strike info |
| `/quote` | POST | Get option quote |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No ML model found` | Run `uv run scripts\train_model_v8.py` |
| `ML=SKIP always` | Run `uv run backtest.py debug` to check probabilities |
| `india_vix.csv not found` | Run `uv run scripts\download_vix.py` |
| `error while signin` (TradingView) | Normal warning — data flows fine. Add TV credentials to settings.env |
| `OpenAlgo 400 error on /expiry` | Check `NIFTY_EXPIRY` in settings.env is correct format (e.g. 24MAR26) |
| `NSE 403 error` | NSE blocks direct requests — use OpenAlgo chain instead |
| Claude Desktop no MCP | Restart Claude Desktop after editing config file |
| Bot not finding signals | Market may be sideways (ADX < 20) — normal, wait for trending day |

---

## Fallback (if Claude Desktop unavailable)

The bot trades on its own without Claude Desktop. ML + Claude API still work.
Only missing: morning web research (VIX, FII, news).

```powershell
# Minimum daily routine without Claude Desktop:
.\morning_brief.ps1       # 09:00 — start pilot
.\evening_squareoff.ps1   # 15:15 — square off
.\update_data.ps1         # 15:35 — update data
```

See `docs\FALLBACK_WITHOUT_CLAUDE.md` for full details.

---

## Disclaimer

This software is for educational purposes only. Trading in derivatives involves
substantial risk of loss. AI-generated signals are not financial advice.
Always paper-trade first before using real money. The authors are not
responsible for any financial losses incurred using this software.
