"""
retrain_weekly.py — Daily ML Model Retraining Pipeline
=======================================================
Runs every weekday at 17:00 IST (after market close, well before next open).

Pipeline:
  0. Pre-flight: tests/test_leakage_safety.py — abort if the feature
     pipeline has regressed a look-ahead leak (no training/promotion)
  1. Train V9  — fast (~3 min), just needs nifty_5min.csv (updated daily by EOD saver)
  2. Train V10 — medium (~8 min), uses v10_training_features.csv
                 Feature rebuild (build_training_dataset.py) only runs on Sundays
                 or when features are >6 days old (needs NSE bhavcopy, slow)
  3. If a model was promoted, refresh models/calibration/calibrator.pkl
     (scripts/calibration.py --live) so it matches the newly-promoted
     probability distribution
  4. Send Telegram notification with training summary

This script only trains/promotes/calibrates — it does NOT restart the bot
service. If the nightly stop/start cron cycle is running (stop ~23:45,
start ~04:00), a promoted model loads automatically at the next start —
no manual action needed. Otherwise restart manually:
  docker compose -f /home/trader/openclaw-v9-docker/docker-compose.yml restart

Cron setup on Hostinger (as trader user), Docker deployment:
  crontab -e
  # — NSE bhavcopy from previous trading day is ready (published ~7:30 PM)
  # — Training finishes well before the bot's 04:00 restart
  # — Bot loads the fresh model automatically on that startup
  # `docker compose run` reuses the live image's own Python env — no
  # separate venv to keep in sync with what's actually scoring live.
  30 2 * * 2-6  cd /home/trader/openclaw-v9-docker && docker compose run --rm -T --name nifty-trader-retrain --entrypoint python nifty-trader scripts/retrain_weekly.py >> logs/retrain.log 2>&1
  # Note: 2-6 = Tue–Sat (trains Mon–Fri session data the next morning)

Usage:
  python scripts/retrain_weekly.py              # auto-decide what to rebuild
  python scripts/retrain_weekly.py --skip-build # skip dataset rebuild (use existing)
  python scripts/retrain_weekly.py --force-build # force dataset rebuild today
  python scripts/retrain_weekly.py --v9-only    # only retrain V9 (fastest)
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"
LOGS_DIR   = ROOT / "logs"
VENV_PY    = ROOT / "venv" / "bin" / "python"
SCRIPTS    = ROOT / "scripts"

# Use venv python if available, else current interpreter
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("retrain")


def _run(cmd: list, label: str, timeout: int = 1800) -> bool:
    """Run a subprocess. Returns True on success."""
    log.info(f"▶ {label}")
    log.info(f"  Command: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=False,
            timeout=timeout,
        )
        if result.returncode == 0:
            log.info(f"  ✅ {label} — done (exit 0)")
            return True
        else:
            log.error(f"  ❌ {label} — FAILED (exit {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"  ❌ {label} — TIMED OUT after {timeout}s")
        return False
    except Exception as e:
        log.error(f"  ❌ {label} — ERROR: {e}")
        return False


def _model_summary() -> str:
    """Return a short summary of current model file sizes and dates."""
    lines = []
    for name in ["nifty_v9_models.pkl", "nifty_v10_models.pkl"]:
        p = MODELS_DIR / name
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size  = p.stat().st_size // 1024
            lines.append(f"  {name}: {size}KB (modified {mtime})")
        else:
            lines.append(f"  {name}: NOT FOUND")
    return "\n".join(lines)


def _telegram_notify(msg: str):
    """Send Telegram notification via the bot's notifier."""
    try:
        sys.path.insert(0, str(ROOT))
        from dotenv import load_dotenv
        load_dotenv(ROOT / "config" / "settings.env")
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        import urllib.request, json
        payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        log.info("📨 Telegram notification sent")
    except Exception as e:
        log.warning(f"Telegram notify failed: {e}")


def _promote(version: str) -> tuple[bool, str]:
    """Call the registry promotion gate (Champion/Challenger mode) for the latest candidate."""
    try:
        sys.path.insert(0, str(ROOT))
        from core.model_registry import promote_if_passes_gate
        return promote_if_passes_gate(version, cc_mode=True)
    except Exception as e:
        return False, f"Promotion error: {e}"


def _cc_report(version: str) -> str:
    """Return a short formatted C/C comparison string for the Telegram summary."""
    try:
        sys.path.insert(0, str(ROOT))
        from core.model_registry import get_cc_report
        r = get_cc_report(version)
        verdict  = r.get("cc_verdict", "?")
        champ_a  = r.get("champion_test_dir_acc")
        chal_a   = r.get("challenger_test_dir_acc")
        gap      = r.get("gap")
        live_wr  = r.get("live_win_rate")

        parts = [f"verdict={verdict}"]
        if champ_a is not None:
            parts.append(f"champion={champ_a:.3f}")
        if chal_a is not None:
            parts.append(f"challenger={chal_a:.3f}")
        if gap is not None:
            parts.append(f"gap={gap:+.3f}")
        if live_wr is not None:
            parts.append(f"live_wr={live_wr:.0%}")
        return "  " + "  |  ".join(parts)
    except Exception as e:
        return f"  cc_report error: {e}"


def _do_rollback(spec: str) -> bool:
    """
    Roll back to a specific registry candidate.
    spec format: "v10:20260604_091500_def456"
    """
    try:
        sys.path.insert(0, str(ROOT))
        from core.model_registry import rollback, list_versions

        if ":" not in spec:
            log.error(f"Rollback spec must be 'version:candidate_id', got: {spec!r}")
            return False

        version, candidate_id = spec.split(":", 1)
        log.info(f"Rolling back {version} to candidate {candidate_id}")
        ok = rollback(version, candidate_id)
        if ok:
            log.info(f"Rollback successful: {version}/{candidate_id}")
        else:
            log.error(f"Rollback failed: {version}/{candidate_id}")
        return ok
    except Exception as e:
        log.error(f"Rollback error: {e}")
        return False


def _list_registry(version: str) -> None:
    """Print the last 5 registry candidates for a version."""
    try:
        sys.path.insert(0, str(ROOT))
        from core.model_registry import list_versions
        entries = list_versions(version, n=5)
        log.info(f"Registry entries for {version} (newest first):")
        for e in entries:
            promoted = "ACTIVE" if e.get("promoted") else "      "
            acc = e.get("test_dir_acc", "?")
            sig = e.get("test_signals", "?")
            cid = e.get("candidate_id", "?")
            log.info(f"  [{promoted}] {cid}  test_dir_acc={acc}  signals={sig}")
    except Exception as e:
        log.warning(f"Could not list registry: {e}")


def main():
    parser = argparse.ArgumentParser(description="Daily ML retraining pipeline")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build_training_dataset.py (use existing features)")
    parser.add_argument("--force-build", action="store_true",
                        help="Force dataset rebuild even if features are fresh")
    parser.add_argument("--v9-only", action="store_true",
                        help="Only retrain V9 (no option chain features needed)")
    parser.add_argument("--rollback", metavar="VERSION:CANDIDATE_ID",
                        help="Roll back to a specific registry entry, e.g. "
                             "v10:20260604_091500_def456. "
                             "Use --list-registry to see available candidates.")
    parser.add_argument("--list-registry", metavar="VERSION",
                        help="List recent registry entries for a version (e.g. v10)")
    args = parser.parse_args()

    # ── Utility modes (no training) ───────────────────────────────────────────
    if args.list_registry:
        _list_registry(args.list_registry)
        sys.exit(0)

    if args.rollback:
        ok = _do_rollback(args.rollback)
        if ok:
            log.info("Rollback complete — loads automatically at the next "
                      "04:00 bot restart, or restart manually: "
                      "docker compose -f /home/trader/openclaw-v9-docker/docker-compose.yml restart")
        sys.exit(0 if ok else 1)

    start = time.time()
    log.info("=" * 60)
    log.info(f"OpenClaw ML Retraining — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # ── Step 0: Pre-flight leakage-safety gate ─────────────────────────────────
    # tests/test_leakage_safety.py locks in the 2026-06-11 Phase-0 fixes
    # (same-day daily merge, partial-HTF-bar merge, unshifted VIX/option
    # context, ...). If the feature pipeline has regressed, abort BEFORE
    # spending ~10min training (and possibly promoting) a leaky model.
    if not _run([PYTHON, str(ROOT / "tests" / "test_leakage_safety.py")],
                 label="Pre-flight: feature-pipeline leakage safety tests",
                 timeout=120):
        log.error("Leakage-safety tests FAILED — aborting retrain "
                   "(feature pipeline regression). Training/promotion skipped.")
        _telegram_notify(
            "<b>OpenClaw Retraining ABORTED</b>\n"
            "tests/test_leakage_safety.py FAILED — feature pipeline has a "
            "look-ahead leak regression. Training skipped to avoid promoting "
            "a leaky model. Check logs/retrain.log."
        )
        sys.exit(1)

    trained_v10 = False
    trained_v9  = False

    # ── Step 1: Update V10 training features ──────────────────────────────────
    # Auto-decide: rebuild features only on Sundays or if >6 days stale.
    # build_training_dataset.py fetches NSE bhavcopy data (slow, ~5 min).
    # On weekdays we skip it and train with yesterday's features + today's 5min bars.
    if not args.skip_build and not args.v9_only:
        features_csv = DATA_DIR / "v10_training_features.csv"
        age_days = 0.0
        if features_csv.exists():
            age_days = (time.time() - features_csv.stat().st_mtime) / 86400
            log.info(f"v10_training_features.csv age: {age_days:.1f} days")

        # At 3:30 AM, download YESTERDAY's bhavcopy (published ~7:30 PM last evening).
        # Skip only if --skip-build flag is passed explicitly.
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        log.info(f"Downloading yesterday's bhavcopy ({yesterday})...")
        ok = _run(
            [PYTHON, str(SCRIPTS / "download_nse_bhavcopy.py"),
             "--start", yesterday,
             "--end",   yesterday],
            label=f"Download NSE F&O bhavcopy for {yesterday}",
            timeout=120,
        )
        if ok:
            _run(
                [PYTHON, str(SCRIPTS / "build_training_dataset.py")],
                label="Build V10 training features",
                timeout=600,
            )
        else:
            log.warning(
                f"Bhavcopy download failed for {yesterday} — "
                "V10 training with existing features. "
                "Normal if yesterday was a market holiday."
            )

    # ── Step 2: Train V10 ─────────────────────────────────────────────────────
    if not args.v9_only:
        trained_v10 = _run(
            [PYTHON, str(SCRIPTS / "train_model_v10.py")],
            label="Train V10 model (XGB + LGB + option chain features)",
            timeout=1800,
        )
        if trained_v10:
            log.info("✅ V10 model updated successfully")
        else:
            log.warning("V10 training failed — falling back to V9")

    # ── Step 3: Train V9 (always — as a baseline / fallback) ──────────────────
    trained_v9 = _run(
        [PYTHON, str(SCRIPTS / "train_model_v9.py")],
        label="Train V9 model (XGB + LGB, no option features)",
        timeout=1200,
    )
    if trained_v9:
        log.info("V9 training subprocess complete")

    # ── Step 4: Promotion gate ─────────────────────────────────────────────────
    # Each training subprocess saved a CANDIDATE to the registry.
    # We now run the gate: if the candidate passes quality checks, it is
    # atomically copied to the live model paths. If it fails, the previous
    # live model is untouched.
    promoted_v10, promoted_v9 = False, False
    promote_reason_v10 = promote_reason_v9 = "skipped (training failed)"

    if trained_v10:
        promoted_v10, promote_reason_v10 = _promote("v10")
        if promoted_v10:
            log.info(f"V10 promoted: {promote_reason_v10}")
        else:
            log.warning(f"V10 NOT promoted: {promote_reason_v10}")

    if trained_v9:
        promoted_v9, promote_reason_v9 = _promote("v9")
        if promoted_v9:
            log.info(f"V9 promoted: {promote_reason_v9}")
        else:
            log.warning(f"V9 NOT promoted: {promote_reason_v9}")

    # Prune old registry entries (keep last 14 per version = 2 weeks)
    try:
        sys.path.insert(0, str(ROOT))
        from core.model_registry import prune_old_candidates
        prune_old_candidates("v10", keep=14)
        prune_old_candidates("v9",  keep=14)
    except Exception as e:
        log.debug(f"Registry prune skipped: {e}")

    # ── Step 5: Recalibrate — only if at least one model was promoted ─────────
    # scripts/calibration.py --live fits Platt/Isotonic on the VAL segment
    # against whichever model core/ml_engine.load_model() will load as the
    # production champion (V10 preferred, V9 fallback), so calibrated
    # probabilities stay matched to the model that's about to go live.
    any_promoted = promoted_v10 or promoted_v9
    recalibrated = False
    if any_promoted:
        recalibrated = _run(
            [PYTHON, str(SCRIPTS / "calibration.py"), "--live"],
            label="Refresh probability calibrator for promoted model",
            timeout=600,
        )
        if recalibrated:
            log.info("✅ Calibrator refreshed for newly promoted model")
        else:
            log.warning("Calibration refresh failed — previous calibrator.pkl stays in place")

    # ── Step 6: Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - start

    def _pline(label, trained, promoted, reason, version=""):
        if not trained:
            return f"<b>{label}:</b> Training FAILED"
        if promoted:
            cc = _cc_report(version) if version else ""
            return f"<b>{label}:</b> Trained + PROMOTED\n{cc}"
        return f"<b>{label}:</b> Trained but NOT promoted: {reason}"

    summary_lines = [
        "<b>OpenClaw Daily Retraining</b>",
        f"<b>Date:</b> {datetime.now().strftime('%d %b %Y %H:%M')}",
        "",
        _pline("V10", trained_v10, promoted_v10, promote_reason_v10, "v10"),
        _pline("V9",  trained_v9,  promoted_v9,  promote_reason_v9,  "v9"),
        f"<b>Calibrator:</b> {'Refreshed' if recalibrated else ('Refresh FAILED' if any_promoted else 'Not needed')}",
        f"<b>Restart:</b> {'Auto-loads at next 04:00 bot restart' if any_promoted else 'No'}",
        f"<b>Duration:</b> {elapsed/60:.1f} min",
        "",
        "<b>Model files:</b>",
        _model_summary(),
    ]
    if not any_promoted:
        summary_lines.insert(2, "<b>*** GATE FAILED — previous model still live ***</b>")

    summary = "\n".join(summary_lines)
    log.info("\n" + summary)
    _telegram_notify(summary)

    if trained_v10 or trained_v9:
        log.info("Retraining pipeline complete")
        sys.exit(0)
    else:
        log.error("All training steps failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
