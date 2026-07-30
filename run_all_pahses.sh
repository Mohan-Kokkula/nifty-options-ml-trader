#!/bin/bash
# run_all_phases.sh — chained Phase 2 → 3 → 4 with logging, resume, and
# market-hours pausing so the live bot always has the CPU during trading.

set -u

cd "$(dirname "$0")"
source .venv/bin/activate

STATUS_LOG="logs/pipeline_status.txt"
mkdir -p logs/phase2 logs/phase3 logs/phase4

log() {
    local msg="$1"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $msg" | tee -a "$STATUS_LOG"
}

wait_out_market_hours() {
    while true; do
        local dow=$(TZ='Asia/Kolkata' date +%u)         # 1=Mon..7=Sun
        local hhmm=$(TZ='Asia/Kolkata' date +%H%M)
        # dow 6,7 = weekend, always OK to run
        # dow 1-5 (weekday), pause if 0915 <= HHMM <= 1530
        if [ $dow -ge 6 ] || [ $hhmm -lt 0915 ] || [ $hhmm -gt 1530 ]; then
            return
        fi
        log "PAUSE  waiting for market close (currently $hhmm IST, dow=$dow)"
        sleep 600      # re-check every 10 minutes
    done
}

run_phase() {
    local name="$1"
    local cmd="$2"
    local out="$3"
    wait_out_market_hours          # ← pause if market is open
    log "START  $name  →  log: $out"
    local t_start=$(date +%s)
    if bash -c "$cmd" > "$out" 2>&1; then
        local elapsed=$(( $(date +%s) - t_start ))
        log "OK     $name  (${elapsed}s)"
        return 0
    else
        local rc=$?
        local elapsed=$(( $(date +%s) - t_start ))
        log "FAIL   $name  (${elapsed}s, exit=$rc)  ← see $out"
        return $rc
    fi
}

log "═══════════════ Pipeline started ═══════════════"

# ── Phase 3 first (Phase 4 consumes its outputs) ──
run_phase "Phase 3 (arch diversity)" \
    "python phase3_arch_diversity.py --brains xgb,lgb,cat,mlp" \
    "logs/phase3/run_full.out"
PH3_OK=$?

# ── Phase 4 only if Phase 3 succeeded ──
if [ $PH3_OK -eq 0 ]; then
    run_phase "Phase 4 (calibration)" \
        "python phase4_calibration.py" \
        "logs/phase4/run_full.out"
else
    log "SKIP   Phase 4  (Phase 3 failed; fix and re-run)"
fi

# ── Phase 2 last: independent, ~30-50 hrs, biggest job ──
run_phase "Phase 2 (XGB HPO, 30 trials × 3 inner × 8 folds)" \
    "python phase2_launcher.py" \
    "logs/phase2/run_full.out"

log "═══════════════ Pipeline finished ═══════════════"