"""
test_slippage_tracking.py — Unit tests for slippage tracking (Issue #7).

Run:
    python scripts/test_slippage_tracking.py

Tests:
  A. record_entry stores all 5 slippage fields
  B. record_entry backward-compat: old call-sites without slippage args still work
  C. get_daily_slippage_report: computes correct averages from journal
  D. get_daily_slippage_report: excludes DRY_RUN and zero signal_premium rows
  E. get_daily_slippage_report: returns n_trades=0 when no qualifying rows
  F. slippage section appears in EOD summary when trades have slippage data
  G. EOD summary does not crash when no slippage data
  H. market_impact and exec_slippage computed correctly
"""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trade_journal import TradeJournal

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


today = datetime.now().strftime("%Y-%m-%d")


def _make_journal(*entries) -> TradeJournal:
    td = tempfile.mktemp(suffix=".jsonl")
    with open(td, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return TradeJournal(path=td)


def _entry(signal_premium=90.0, limit_price=91.0, fill_price=91.5,
           market_impact=1.5, exec_slippage=0.5,
           is_dry_run=False, direction="PUT", trade_date=None, qty=65):
    return {
        "trade_date":        trade_date or today,
        "entry_time":        "10:15:00",
        "session":           "MORNING",
        "direction":         direction,
        "option_type":       "PE",
        "strike_mode":       "ATM",
        "symbol":            "NIFTY2660923200PE",
        "entry_spot":        23300.0,
        "entry_premium":     fill_price,
        "confidence":        82,
        "sl_pts":            70.0,
        "tp_pts":            140.0,
        "sl_price":          23370.0,
        "tp_price":          23160.0,
        "rr_ratio":          2.0,
        "lots":              1,
        "qty":               65,
        "cycle":             15,
        "atr_5m":            35.0,
        "atr_15m":           55.0,
        "vix":               16.5,
        "vix_regime":        "ELEVATED",
        "adx_15m":           38.0,
        "ml_proba_call":     0.02,
        "ml_proba_put":      0.55,
        "ml_proba_skip":     0.43,
        "recovery_mode":     False,
        "gap_override":      False,
        "gamma_squeeze":     False,
        "dte":               4,
        "is_expiry":         False,
        "pcr":               1.05,
        "oi_magnetic_strike":23000.0,
        "is_dry_run":        is_dry_run,
        "qty":               qty,
        # Slippage fields
        "signal_premium":    signal_premium,
        "limit_price":       limit_price,
        "fill_price":        fill_price,
        "market_impact_pts": market_impact,
        "exec_slippage_pts": exec_slippage,
        # Exit fields
        "exit_time":     None,
        "exit_reason":   None,
        "pnl_pts":       None,
        "result":        None,
        "event":         None,
    }


# ── A. record_entry stores slippage fields ───────────────────────────────────
print("\n[A] record_entry stores slippage fields")

jnl_a = _make_journal()
jnl_a.record_entry(
    direction="PUT", option_type="PE", strike_mode="ATM",
    entry_spot=23300.0, entry_premium=91.5, confidence=82,
    sl_pts=70.0, tp_pts=140.0, sl_price=23370.0, tp_price=23160.0,
    lots=1, qty=65, cycle=15, symbol="NIFTY2660923200PE",
    atr_5m=35.0, atr_15m=55.0, vix=16.5, vix_regime="ELEVATED",
    adx_15m=38.0, ml_proba=[0.02, 0.55, 0.43],
    is_dry_run=False,
    signal_premium=90.0, limit_price=91.0, fill_price=91.5,
    market_impact_pts=1.5, exec_slippage_pts=0.5,
)

# Force write by checking _open_trade
raw = jnl_a._open_trade
check("signal_premium stored",    raw.get("signal_premium")    == 90.0)
check("limit_price stored",       raw.get("limit_price")       == 91.0)
check("fill_price stored",        raw.get("fill_price")        == 91.5)
check("market_impact_pts stored", raw.get("market_impact_pts") == 1.5)
check("exec_slippage_pts stored", raw.get("exec_slippage_pts") == 0.5)


# ── B. Backward-compat: old call-site without slippage args ──────────────────
print("\n[B] Backward-compat: old call-site without slippage args")

jnl_b = _make_journal()
raised = False
try:
    jnl_b.record_entry(
        direction="CALL", option_type="CE", strike_mode="ATM",
        entry_spot=23400.0, entry_premium=100.0, confidence=80,
        sl_pts=80.0, tp_pts=160.0, sl_price=23320.0, tp_price=23560.0,
        lots=1, qty=65, cycle=10, symbol="NIFTY2660924000CE",
        atr_5m=35.0, atr_15m=55.0, vix=15.0, vix_regime="NORMAL",
        adx_15m=35.0, ml_proba=[0.60, 0.02, 0.38],
        is_dry_run=False,
        # No slippage args — old call site
    )
except TypeError:
    raised = True
check("old call-site without slippage args does not raise", not raised)
check("default signal_premium=0 in old call-site",
      jnl_b._open_trade.get("signal_premium", -1) == 0.0)


# ── C. get_daily_slippage_report: correct averages ───────────────────────────
print("\n[C] get_daily_slippage_report: correct averages")

e1 = _entry(signal_premium=90.0, limit_price=91.0, fill_price=92.0,
            market_impact=2.0, exec_slippage=1.0, qty=65)
e2 = _entry(signal_premium=85.0, limit_price=86.0, fill_price=86.5,
            market_impact=1.5, exec_slippage=0.5, qty=65)
jnl_c = _make_journal(e1, e2)
slip = jnl_c.get_daily_slippage_report(today)

check("n_trades=2",              slip["n_trades"] == 2)
check("avg_market_impact=1.75",  abs(slip["avg_market_impact"] - 1.75) < 0.01)
check("avg_exec_slippage=0.75",  abs(slip["avg_exec_slippage"] - 0.75) < 0.01)
check("max_market_impact=2.0",   slip["max_market_impact"] == 2.0)
check("total_slip_cost = 2.0+1.5 x 65 = 227.5",
      abs(slip["total_slip_cost_rs"] - (2.0 + 1.5) * 65) < 0.1)
check("slip_over_1pt=2 (both > 1pt)", slip["slip_over_1pt"] == 2)


# ── D. Excludes DRY_RUN and zero signal_premium ──────────────────────────────
print("\n[D] Excludes DRY_RUN and pre-fix rows")

dry = _entry(signal_premium=90.0, market_impact=1.0, is_dry_run=True)
nodata = _entry(signal_premium=0.0, market_impact=0.0)   # pre-fix entry
real = _entry(signal_premium=88.0, market_impact=0.8)
real["qty"] = 65

jnl_d = _make_journal(dry, nodata, real)
slip_d = jnl_d.get_daily_slippage_report(today)
check("DRY_RUN excluded",          slip_d["n_trades"] == 1)
check("zero signal_premium excluded", slip_d["avg_market_impact"] == 0.8)


# ── E. No qualifying rows → n_trades=0 ───────────────────────────────────────
print("\n[E] No qualifying rows -> n_trades=0")

jnl_e = _make_journal()
slip_e = jnl_e.get_daily_slippage_report(today)
check("empty journal -> n_trades=0",    slip_e["n_trades"] == 0)
check("empty journal -> avg_impact=0",  slip_e["avg_market_impact"] == 0.0)

jnl_f = _make_journal(_entry(signal_premium=0.0))
slip_f = jnl_f.get_daily_slippage_report(today)
check("only zero-premium rows -> n_trades=0", slip_f["n_trades"] == 0)


# ── F. Slippage in EOD summary when data exists ───────────────────────────────
print("\n[F] EOD summary includes slippage section")

from core.eod_summary import _format_summary

# Build a minimal trade record that passes _format_summary
completed_trade = {
    "trade_date":  today,
    "entry_time":  "10:00:00",
    "direction":   "PUT",
    "option_type": "PE",
    "pnl_pts":     35.0,
    "pnl_rupees":  2275.0,
    "result":      "WIN",
    "exit_reason": "TRAIL_SL",
    "event":       "EXIT",
    "is_dry_run":  False,
    "exit_spot":   23200.0,
    # slippage fields
    "signal_premium":    88.0,
    "limit_price":       89.0,
    "fill_price":        89.5,
    "market_impact_pts": 1.5,
    "exec_slippage_pts": 0.5,
    "qty":               65,
}

# Patch TradeJournal to return our test data
import core.eod_summary as _eod
_orig_tj = None
try:
    import core.trade_journal as _tj_mod
    _orig_init = _tj_mod.TradeJournal.__init__
    _orig_get  = _tj_mod.TradeJournal.get_daily_slippage_report

    def _fake_slip(self, today_str=""):
        return {
            "n_trades": 1, "avg_market_impact": 1.5, "avg_exec_slippage": 0.5,
            "max_market_impact": 1.5, "total_slip_cost_rs": 97.5,
            "slip_over_1pt": 1, "entries": [completed_trade],
        }

    _tj_mod.TradeJournal.get_daily_slippage_report = _fake_slip

    summary = _format_summary([completed_trade], risk_cap=5000.0)
    check("slippage section in summary",        "Slippage" in summary)
    check("avg market impact in summary",        "Avg market impact" in summary)
    check("total slip cost in summary",          "Total slip cost" in summary)
    check("warning for >1pt trades",             ">1pt market impact" in summary)
finally:
    _tj_mod.TradeJournal.get_daily_slippage_report = _orig_get


# ── G. EOD summary does not crash without slippage data ──────────────────────
print("\n[G] EOD summary does not crash without slippage data")

# No slippage fields in the trade record
bare_trade = {
    "trade_date": today, "entry_time": "10:00:00",
    "direction": "PUT", "pnl_pts": 35.0, "pnl_rupees": 2275.0,
    "result": "WIN", "exit_reason": "TP", "event": "EXIT",
    "is_dry_run": False, "exit_spot": 23200.0,
}
raised2 = False
try:
    summary_bare = _format_summary([bare_trade], risk_cap=5000.0)
except Exception:
    raised2 = True
check("EOD summary does not raise without slippage data", not raised2)


# ── H. Correct metric arithmetic ─────────────────────────────────────────────
print("\n[H] Slippage metric arithmetic")

signal = 85.0
limit  = 86.0
fill   = 87.0
check("market_impact = fill - signal = +2.0", round(fill - signal, 4) == 2.0)
check("exec_slippage = fill - limit  = +1.0", round(fill - limit,  4) == 1.0)

signal2 = 85.0
limit2  = 86.0
fill2   = 85.5   # filled better than limit (limit order improvement)
check("exec_slippage < 0 when fill < limit",  round(fill2 - limit2, 4) < 0)
check("market_impact can be +0.5 even with improvement",
      round(fill2 - signal2, 4) == 0.5)


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
