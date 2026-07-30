# run.ps1 - Load env + run app

Write-Host "Loading environment variables from config/settings.env..." -ForegroundColor Cyan

Get-Content config\settings.env | Where-Object { 
    -not $_.TrimStart().StartsWith('#') -and $_.Trim() -ne '' 
} | ForEach-Object {
    $parts = $_.Split('=', 2)
    if ($parts.Count -ge 2) {
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ($name) {
            Set-Item "env:$name" $value
            Write-Host "   Loaded: $name" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "Starting app with uv..." -ForegroundColor Cyan
uv run main.py
