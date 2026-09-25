"""PSAR Gate Harness — simulates the full claude_pilot gate sequence.

Run this BEFORE deploying to catch gate conflicts. It replays historical
bars through every gate in claude_pilot's execution path and reports
which gates would block each PSAR signal.

Usage:
    python scripts/gate_harness.py [--days N]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from core.psar_engine import PSAREngine
from core.tv_fetcher import get_tv_fetcher
from core.trap_detector import TrapDetector

def run_harness(n_days: int = 3):
    tv = get_tv_fetcher()
    print("Fetching data...")
    df5 = tv.get_nifty_5min(n_bars=n_days * 75 + 200)
    df15 = tv.get_nifty_15min(n_bars=n_days * 25 + 100)
    df30 = tv.get_nifty_30min(n_bars=n_days * 13 + 60)

    if df5.empty:
        print("ERROR: No 5m data")
        return

    vix_val = 15.0
    try:
        raw_vix = tv.get_ohlcv("INDIAVIX", "NSE", interval_minutes="D", n_bars=10)
        if raw_vix is None or raw_vix.empty:
            raw_vix = tv.get_ohlcv("INDIA VIX", "NSE", interval_minutes="D", n_bars=10)
        if raw_vix is not None and not raw_vix.empty:
            vix_val = float(raw_vix["close"].iloc[-1])
    except Exception:
        pass

    MAX_LOSS = 2000
    LOT_SIZE = 65

    trading_days = sorted(set(df5.index.date))
    target_days = trading_days[-n_days:]

    print(f"VIX: {vix_val:.1f} | Days: {target_days}")
    print(f"Alignment mode: {'STRICT 3/3' if vix_val >= 22 else 'FAST 2/3'}")

    gate_stats = {}
    all_signals = []

    for day in target_days:
        df5_day = df5[df5.index.date == day].copy()
        if df5_day.empty:
            continue

        engine = PSAREngine()
        trap = TrapDetector()
        in_position = False

        print(f"\n{'='*80}")
        print(f"  {day} — Gate-by-Gate Analysis")
        print(f"{'='*80}")

        for i in range(len(df5_day)):
            bar = df5_day.iloc[i]
            bar_time = df5_day.index[i]
            hm = bar_time.hour * 100 + bar_time.minute
            spot = float(bar["close"])

            if hm < 915 or hm > 1520:
                continue

            trap.update_tick(spot, bar_time)

            if in_position:
                continue

            cutoff = bar_time
            hist_5m = df5[df5.index <= cutoff].tail(60)
            hist_15m = df15[df15.index <= cutoff].tail(30)
            hist_30m = df30[df30.index <= cutoff].tail(20)

            if len(hist_5m) < 10 or len(hist_15m) < 5 or len(hist_30m) < 5:
                continue

            signal, proba, conf, indicators = engine.predict(
                hist_5m, hist_15m, hist_30m, vix=vix_val, current_hm=hm
            )

            if signal == 2:
                continue

            direction = "CALL" if signal == 0 else "PUT"
            option_type = "CE" if signal == 0 else "PE"
            sl_pts, tp_pts = engine.get_sl_tp(
                vix=vix_val, max_loss_budget=MAX_LOSS, lot_size=LOT_SIZE
            )
            aligned = indicators.get("aligned_count", 0)
            partial = indicators.get("partial_alignment", False)
            session_minutes = (bar_time.hour - 9) * 60 + bar_time.minute - 15

            gates_hit = []
            passed_all = True

            # ── Gate 1: OPEN_SETTLE ──
            if hm < engine.OPEN_SETTLE:
                gates_hit.append("OPEN_SETTLE")
                passed_all = False

            # ── Gate 2: Morning hard block (15 min for PSAR) ──
            if session_minutes < 15:
                gates_hit.append("MORNING_HARD_BLOCK")
                passed_all = False

            # ── Gate 3: Lunch chop ──
            if engine.LUNCH_START <= hm < engine.LUNCH_END:
                gates_hit.append("LUNCH_CHOP")
                passed_all = False

            # ── Gate 4: Confidence vs effective_min_conf ──
            # For PSAR, regime floor is capped at 50%
            effective_min = 50
            if conf * 100 < effective_min:
                gates_hit.append(f"LOW_CONF({conf*100:.0f}%<{effective_min}%)")
                passed_all = False

            # ── Gate 5: Trap detector ──
            verdict = trap.is_trap(
                option_type=option_type, spot=spot,
                ml_indicators=indicators, now=bar_time
            )
            if verdict.is_trap:
                gates_hit.append(f"TRAP:{verdict.reason}")
                passed_all = False

            # ── Gate 6: Multi-TF disagree (BYPASSED for PSAR) ──
            # PSAR has its own multi-TF alignment

            # ── Gate 7: MAX_LOSS_PER_TRADE ──
            risk = sl_pts * LOT_SIZE
            if risk > MAX_LOSS:
                gates_hit.append(f"MAX_LOSS(risk={risk:.0f}>budget={MAX_LOSS})")
                passed_all = False

            # ── Gate 8: trading_enabled (BYPASSED for PSAR) ──
            # ── Gate 9: Day halt (BYPASSED for PSAR) ──
            # ── Gate 10: CALL<85% (BYPASSED for PSAR) ──

            status = "PASS" if passed_all else "BLOCKED"
            tag = f"{'[P]' if partial else '[F]'}"

            if gates_hit:
                print(f"  {bar_time.strftime('%H:%M')} {direction:4s} conf={conf:.2f} "
                      f"align={aligned}/3{tag} SL={sl_pts} TP={tp_pts} "
                      f"→ BLOCKED by: {', '.join(gates_hit)}")
            else:
                print(f"  {bar_time.strftime('%H:%M')} {direction:4s} conf={conf:.2f} "
                      f"align={aligned}/3{tag} SL={sl_pts} TP={tp_pts} "
                      f"→ PASS ✓")
                engine.record_trade()
                in_position = True

            for g in gates_hit:
                gate_name = g.split("(")[0].split(":")[0]
                gate_stats[gate_name] = gate_stats.get(gate_name, 0) + 1

            all_signals.append({
                "date": str(day),
                "time": bar_time.strftime('%H:%M'),
                "direction": direction,
                "conf": round(conf, 2),
                "aligned": aligned,
                "partial": partial,
                "status": status,
                "gates_hit": gates_hit,
            })

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  GATE AUDIT SUMMARY — {len(all_signals)} signals across {len(target_days)} days")
    print(f"{'='*80}")

    passed = [s for s in all_signals if s["status"] == "PASS"]
    blocked = [s for s in all_signals if s["status"] == "BLOCKED"]
    print(f"  Passed: {len(passed)} | Blocked: {len(blocked)} | Total: {len(all_signals)}")

    if gate_stats:
        print(f"\n  BLOCK FREQUENCY:")
        for gate, count in sorted(gate_stats.items(), key=lambda x: -x[1]):
            print(f"    {gate:<30s} {count:>3d} blocks")

    if passed:
        print(f"\n  PASSED SIGNALS:")
        for s in passed:
            tag = "[P]" if s["partial"] else "[F]"
            print(f"    {s['date']} {s['time']} {s['direction']:4s} "
                  f"conf={s['conf']:.2f} align={s['aligned']}/3{tag}")

    if blocked:
        print(f"\n  BLOCKED SIGNALS (review these):")
        for s in blocked:
            tag = "[P]" if s["partial"] else "[F]"
            print(f"    {s['date']} {s['time']} {s['direction']:4s} "
                  f"conf={s['conf']:.2f} align={s['aligned']}/3{tag} "
                  f"→ {', '.join(s['gates_hit'])}")

    print(f"\n  PSAR BYPASSES ACTIVE:")
    print(f"    - Regime floor capped at 50% (was 65-85% in CHOP)")
    print(f"    - Multi-TF EMA disagree skipped (PSAR has its own alignment)")
    print(f"    - trading_enabled kill switch bypassed")
    print(f"    - Day halt after loss bypassed")
    print(f"    - CALL<85% filter bypassed")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    run_harness(args.days)
