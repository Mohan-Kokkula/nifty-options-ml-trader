# morning_brief.ps1 - Auto morning context at 09:00 IST
# Schedule via Windows Task Scheduler: daily at 09:00.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host ""
Write-Host "=== OpenClaw Morning Brief ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Gray

# Load env (PS 5.1 safe)
$envFile = "config\settings.env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object {
        -not $_.TrimStart().StartsWith('#') -and $_.Trim() -ne ''
    } | ForEach-Object {
        $parts = $_.Split('=', 2)
        if ($parts.Count -ge 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($name) { Set-Item "env:$name" $value }
        }
    }
}

$today      = Get-Date -Format "yyyy-MM-dd"
$dayOfWeek  = (Get-Date).DayOfWeek
$hour       = (Get-Date).Hour

# Weekend guard
if ($dayOfWeek -eq "Saturday" -or $dayOfWeek -eq "Sunday") {
    Write-Host "Weekend - market closed, skipping" -ForegroundColor Gray
    exit 0
}

# Market hours guard
if ($hour -lt 8 -or $hour -ge 16) {
    Write-Host "Outside market hours (hour=$hour) - skipping" -ForegroundColor Gray
    exit 0
}

# Build context string
$context = "DATE=$today SESSION=MORNING "

if ($dayOfWeek -eq "Thursday") {
    $context += "EXPIRY_DAY=true "
    Write-Host "!! EXPIRY DAY - reduced size, stop by 13:00" -ForegroundColor Yellow
}
if ($dayOfWeek -eq "Monday") {
    $context += "MONDAY=true "
    Write-Host "Monday - gap risk, wait before first trade" -ForegroundColor Yellow
}
if ($dayOfWeek -eq "Friday") {
    $context += "FRIDAY=true "
    Write-Host "Friday - close early, no late trades" -ForegroundColor Yellow
}

$context += "SOURCE=morning_brief_script "
Write-Host "Context: $context" -ForegroundColor Gray

# Check bot is running
$botUrl = "http://localhost:8080"
try {
    $health = Invoke-RestMethod -Uri "$botUrl/health" -Method GET -TimeoutSec 5
    if ($health.status -eq "ok") {
        Write-Host "[OK] Bot is running" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Bot responded but status=$($health.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[FAIL] Bot not running at $botUrl - start with: .\run.ps1" -ForegroundColor Red
    exit 1
}

# Send morning context to analyze
$analyzeBody = @{
    execute = $false
    context = $context
} | ConvertTo-Json

try {
    $rec = Invoke-RestMethod -Uri "$botUrl/analyze" `
        -Method POST -Body $analyzeBody -ContentType "application/json" `
        -TimeoutSec 30

    if ($rec.error) {
        Write-Host "[WARN] Analyze: $($rec.error)" -ForegroundColor Yellow
        Write-Host "  Pilot will run in ML-only mode" -ForegroundColor Gray
    } else {
        $bias = if ($rec.market_bias) { $rec.market_bias } else { "UNKNOWN" }
        $conf = if ($rec.confidence) { $rec.confidence } else { 0 }
        Write-Host "[OK] Morning analysis: $bias conf=$conf%" -ForegroundColor Green
        if ($rec.analysis) {
            $maxLen = [Math]::Min(150, $rec.analysis.Length)
            $snippet = $rec.analysis.Substring(0, $maxLen)
            Write-Host "  $snippet" -ForegroundColor Gray
        }
    }
} catch {
    $statusCode = 0
    $errMsg = $_.Exception.Message
    try {
        $statusCode = [int]$_.Exception.Response.StatusCode
        $reader = [System.IO.StreamReader]::new($_.Exception.Response.GetResponseStream())
        $errBody = $reader.ReadToEnd() | ConvertFrom-Json
        $errMsg = $errBody.error
    } catch {}
    if ($statusCode -eq 400) {
        Write-Host "[WARN] Claude not enabled - $errMsg" -ForegroundColor Yellow
    } else {
        Write-Host "[WARN] Analyze failed (HTTP $statusCode): $errMsg" -ForegroundColor Yellow
    }
}

# Start pilot if not running
$statusBody = @{ action = "status" } | ConvertTo-Json
try {
    $status = Invoke-RestMethod -Uri "$botUrl/pilot" `
        -Method POST -Body $statusBody -ContentType "application/json" -TimeoutSec 5

    if ($status.running -eq $true) {
        Write-Host "[OK] Pilot already running (cycle=$($status.cycle) trades=$($status.trades))" -ForegroundColor Cyan
    } else {
        $minConf = if ($dayOfWeek -eq "Thursday") { 70 } else { 60 }
        $maxTrades = if ($dayOfWeek -eq "Thursday") { 2 } else { 5 }

        $startBody = @{
            action         = "start"
            min_confidence = $minConf
            max_trades     = $maxTrades
        } | ConvertTo-Json

        $startResult = Invoke-RestMethod -Uri "$botUrl/pilot" `
            -Method POST -Body $startBody -ContentType "application/json" -TimeoutSec 5

        if ($startResult.status -eq "started") {
            Write-Host "[OK] Pilot started (conf>=$minConf% max_trades=$maxTrades)" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Pilot start: $($startResult | ConvertTo-Json -Compress)" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "[WARN] Pilot check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Auto square-off: 15:15 IST (run evening_squareoff.ps1)" -ForegroundColor Gray
Write-Host ""
