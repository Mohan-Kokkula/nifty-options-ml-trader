#!/usr/bin/env bash
# cron_refresh_data.sh — Append fresh bars to all CSVs (5/15/30/60/day + VIX)
# Scheduled daily after market close (16:00 IST Mon–Fri).
set -euo pipefail

# Resolve project root (scripts/ -> ../)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/cron"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/refresh_${STAMP}.log"

export TZ="Asia/Kolkata"

# Prefer uv if available (matches project's uv workflow); else fall back to python3
if command -v uv >/dev/null 2>&1; then
    RUNNER="uv run"
else
    RUNNER="python3"
fi

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') : data refresh starting ==="
    $RUNNER scripts/update_data.py --days 1
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') : data refresh done ==="
} >> "$LOG_FILE" 2>&1
