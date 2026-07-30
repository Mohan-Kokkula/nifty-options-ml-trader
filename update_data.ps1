# update_data.ps1 — Fetch today's Nifty bars and append to CSVs
# Run after 15:30 IST every trading day.
# Scheduled via Windows Task Scheduler at 15:35.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "`n=== OpenClaw Data Update ===" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Gray

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
    Write-Host "Loaded env from $envFile" -ForegroundColor Green
}

python scripts/update_data.py $args

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Data update complete`n" -ForegroundColor Green
} else {
    Write-Host "`n❌ Data update failed`n" -ForegroundColor Red
}
