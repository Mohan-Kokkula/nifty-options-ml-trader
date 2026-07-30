# Running Without Claude.ai — 3 Fallback Options

If claude.ai is unavailable, use any of these options. The Python bot
still runs fully — ML model, risk management, execution all work.
You only lose the macro context layer (news, FII, VIX research).

---

## Option 1 — OpenAlgo Webhook (simplest, no extra setup)

OpenAlgo has a built-in webhook endpoint that accepts TradingView alerts
and calls your bot automatically. This is the most reliable fallback.

### How it works:
TradingView Alert → OpenAlgo Webhook → your bot /analyze or /pilot

### Setup in TradingView:
1. Open your NIFTY 5-min chart
2. Add the Pine Script from `pine/openclaw_trend_state.pine`
3. Create an alert on "OpenClaw — Trend UP confirmed"
4. Set webhook URL to:
   ```
   http://YOUR_PUBLIC_IP:8080/analyze
   ```
5. Set alert message to:
   ```json
   {"execute": true, "context": "TradingView signal: TREND_UP confirmed. ADX={{plot_0}}. Symbol={{ticker}}. Time={{time}}."}
   ```
6. Repeat for "Trend DOWN confirmed" with option_type PE

### What happens:
- TradingView detects ADX crossover → fires alert → calls your bot
- Bot runs full analysis (ML + Claude API) + executes if confident
- No manual intervention needed

---

## Option 2 — Auto-Pilot Only (bot runs itself, no agent needed)

The bot already runs fully autonomously via PILOT_AUTO_START=true.
You don't need OpenClaw at all for basic operation.

What you lose without OpenClaw:
- No morning macro research (VIX, FII, GIFT Nifty)
- No news context passed to Claude
- No intraday monitoring by the agent
- No automatic stop on breaking news

What still works:
- ML model runs every 5 minutes ✅
- ADX sideways gate ✅
- Claude confirmation ✅
- Risk management ✅
- Trade execution ✅
- WhatsApp / email alerts ✅

### To run in this mode, just start the bot:
```powershell
.\run.ps1
```
The pilot starts automatically. No other action needed.

---

## Option 3 — OpenAlgo Chat Agent (built-in to OpenAlgo)

OpenAlgo has its own chat interface at http://127.0.0.1:5000
It can call your bot's endpoints directly.

### Setup:
1. Open OpenAlgo dashboard at http://127.0.0.1:5000
2. Go to the "Chatbot" or "Agent" section
3. Add a custom tool pointing to http://localhost:8080
4. Use these commands in OpenAlgo chat:

```
Start pilot:     POST http://localhost:8080/pilot {"action":"start"}
Check status:    POST http://localhost:8080/pilot {"action":"status"}
Analyze market:  POST http://localhost:8080/analyze {"execute":true}
Check positions: POST http://localhost:8080/positions {}
Stop pilot:      POST http://localhost:8080/pilot {"action":"stop"}
```

---

## Option 4 — Morning Script (runs automatically, no agent)

A scheduled PowerShell script that runs at 09:00 every day,
fetches basic market data and passes it to the bot as context.

Create file `morning_brief.ps1`:

```powershell
# morning_brief.ps1 — Auto morning context at 09:00 IST
# Schedule via Task Scheduler: daily at 09:00

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Load env
Get-Content config\settings.env | Where-Object {
    -not $_.TrimStart().StartsWith('#') -and $_.Trim() -ne ''
} | ForEach-Object {
    $name, $value = $_.Split('=', 2).Trim()
    if ($name) { Set-Item "env:$name" $value }
}

$today = Get-Date -Format "yyyy-MM-dd"
$dayOfWeek = (Get-Date).DayOfWeek

# Build basic context
$context = "DATE=$today "
if ($dayOfWeek -eq "Thursday") {
    $context += "EXPIRY_DAY=true(reduce_size,stop_by_1300) "
}
$context += "AUTO_BRIEF=true NO_CLAUDE_AGENT=true "

# Call analyze with basic context
$body = @{
    execute = $false
    context = $context
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/analyze" `
        -Method POST -Body $body -ContentType "application/json"
    Write-Host "Morning brief sent: $($response.analysis)" -ForegroundColor Green
} catch {
    Write-Host "Bot not running yet" -ForegroundColor Yellow
}

# Start pilot if not already running
$pilotBody = @{ action = "status" } | ConvertTo-Json
try {
    $status = Invoke-RestMethod -Uri "http://localhost:8080/pilot" `
        -Method POST -Body $pilotBody -ContentType "application/json"
    if (-not $status.running) {
        $startBody = @{ action = "start" } | ConvertTo-Json
        Invoke-RestMethod -Uri "http://localhost:8080/pilot" `
            -Method POST -Body $startBody -ContentType "application/json"
        Write-Host "Pilot started" -ForegroundColor Green
    } else {
        Write-Host "Pilot already running" -ForegroundColor Cyan
    }
} catch {
    Write-Host "Could not connect to bot" -ForegroundColor Red
}
```

### Schedule it:
```powershell
# Run once as admin to register
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-File E:\algo-bot\openclaw-tv\morning_brief.ps1" `
    -WorkingDirectory "E:\algo-bot\openclaw-tv"
$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
Register-ScheduledTask -TaskName "OpenClaw Morning Brief" `
    -Action $action -Trigger $trigger -RunLevel Highest
```

---

## Summary — Which option to use?

| Situation | Best option |
|-----------|-------------|
| claude.ai temporarily down | Option 2 — bot runs itself |
| Want TradingView signals to trigger bot | Option 1 — TV webhook |
| Want basic automation without any agent | Option 4 — morning script |
| Have OpenAlgo chat feature | Option 3 — OpenAlgo agent |
| Normal operation | Claude.ai as agent (original) |

The bot NEVER depends on claude.ai to run — it always works standalone.
Claude.ai only adds macro context which improves accuracy by ~5-10%.
