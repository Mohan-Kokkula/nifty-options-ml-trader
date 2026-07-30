#!/usr/bin/env bash
# cron_train_model.sh — Retrain the V9 LightGBM model on refreshed CSVs.
# Scheduled weekly (Saturday early morning) so it never fights the live bot.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/cron"
MODEL_DIR="$PROJECT_ROOT/models"
BACKUP_DIR="$MODEL_DIR/backups"
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

STAMP="$(date +%Y-%m-%d_%H%M)"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

export TZ="Asia/Kolkata"

if command -v uv >/dev/null 2>&1; then
    RUNNER="uv run"
else
    RUNNER="python3"
fi

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') : training starting ==="

    # Back up current model before overwrite (best-effort; ignore if absent)
    if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR"/*.pkl 2>/dev/null || true)" ]; then
        tar -czf "$BACKUP_DIR/model_${STAMP}.tgz" -C "$MODEL_DIR" . \
            --exclude='backups' || true
        echo "  backed up current model → $BACKUP_DIR/model_${STAMP}.tgz"
    fi

    # Step 1: make sure data is current (idempotent — no-ops if already updated)
    $RUNNER scripts/update_data.py --days 1 || echo "  (data refresh non-fatal failure)"

    # Step 2: retrain. Futures CSV is optional — include only if it exists.
    FUT_ARG=""
    if [ -f "data/nifty_fut_5min.csv" ]; then
        FUT_ARG="--csvfut data/nifty_fut_5min.csv"
    fi

    $RUNNER scripts/train_model_v9.py \
        --csv5   data/nifty_5min.csv  \
        --csv15  data/nifty_15min.csv \
        --csv30  data/nifty_30min.csv \
        --csv60  data/nifty_60min.csv \
        --csvday data/nifty_day.csv   \
        --csvvix data/india_vix.csv   \
        $FUT_ARG \
        --prune-shap

    # Retain only last 8 weekly backups
    ls -1t "$BACKUP_DIR"/model_*.tgz 2>/dev/null | tail -n +9 | xargs -r rm -f

    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') : training done ==="
} >> "$LOG_FILE" 2>&1
