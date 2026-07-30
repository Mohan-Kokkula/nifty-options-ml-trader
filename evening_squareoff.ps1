# evening_squareoff.ps1 - Auto square-off at 15:15 IST
# Schedule via Task Scheduler: daily at 15:15.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host ""
Write-Host "=== OpenClaw Evening Square-Off ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Gray

# Load env
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

try {
    $health = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method GET -TimeoutSec 3
} catch {
    Write-Host "Bot not running - nothing to do" -ForegroundColor Gray
    exit 0
}

# Stop pilot
$stopBody = @{ action = "stop" } | ConvertTo-Json
try {
    Invoke-RestMethod -Uri "http://localhost:8080/pilot" `
        -Method POST -Body $stopBody -ContentType "application/json" | Out-Null
    Write-Host "[OK] Pilot stopped" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not stop pilot" -ForegroundColor Yellow
}

# Cancel all open orders
try {
    Invoke-RestMethod -Uri "http://localhost:8080/cancel_all" `
        -Method POST -Body "{}" -ContentType "application/json" | Out-Null
    Write-Host "[OK] All orders cancelled" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Cancel all failed" -ForegroundColor Yellow
}

# Check remaining positions
try {
    $pos = Invoke-RestMethod -Uri "http://localhost:8080/positions" `
        -Method POST -Body "{}" -ContentType "application/json"
    $open = $pos.positions.data | Where-Object { [int]($_.netqty ?? $_.quantity ?? 0) -ne 0 }
    if ($open -and $open.Count -gt 0) {
        Write-Host "[WARN] $($open.Count) position(s) still open - closing..." -ForegroundColor Yellow
        foreach ($p in $open) {
            $closeBody = @{ symbol = $p.symbol } | ConvertTo-Json
            try {
                Invoke-RestMethod -Uri "http://localhost:8080/close" `
                    -Method POST -Body $closeBody -ContentType "application/json" | Out-Null
                Write-Host "   Closed: $($p.symbol)" -ForegroundColor Green
            } catch {
                Write-Host "   Failed to close: $($p.symbol)" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "[OK] No open positions" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARN] Could not check positions" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[OK] Square-off complete. Good trading day!" -ForegroundColor Green
Write-Host ""
