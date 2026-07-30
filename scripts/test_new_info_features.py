"""
test_new_info_features.py — unit tests for the new-information feature code.

Tests the PURE, deterministic logic only (formulas, parsers) on tiny
hand-constructed inputs — NOT market-data validation. Covers:
  P2 futures_archiver: normalize_depth, compute_depth_features
  P3 oi_archiver:      parse_chain_rows
  P4 oi_features:      atm_ce/pe_oi_chg_pct, oi_buildup_signed, net_oi_flow
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS  {name}")
    else:    FAIL += 1; print(f"  FAIL  {name}")


# ── P2: futures_archiver depth helpers ──────────────────────────────────────
print("\n[P2] futures_archiver depth helpers")
from scripts.futures_archiver import normalize_depth, compute_depth_features

# list-style payload
raw_list = {"data": {"bids": [{"price": 100.0, "quantity": 30},
                              {"price": 99.5, "quantity": 20}],
                     "asks": [{"price": 100.5, "quantity": 10},
                              {"price": 101.0, "quantity": 10}]}}
nd = normalize_depth(raw_list)
check("normalize list-style bids", nd["bids"][0] == (100.0, 30))
check("normalize list-style asks", nd["asks"][0] == (100.5, 10))

# Kotak flat bp/bq payload
raw_flat = {"data": {"bp1": 200, "bq1": 50, "sp1": 200.5, "sq1": 25}}
nd2 = normalize_depth(raw_flat)
check("normalize flat bp/bq", nd2["bids"][0] == (200.0, 50.0) and nd2["asks"][0] == (200.5, 25.0))

f = compute_depth_features(nd)
# bid_qty=50, ask_qty=20 -> obi=(50-20)/70
check("obi sign/value", abs(f["obi"] - (30/70)) < 1e-5)   # obi rounded to 6 dp
check("best bid/ask", f["best_bid"] == 100.0 and f["best_ask"] == 100.5)
# microprice between bid and ask
check("microprice bounded", 100.0 <= f["microprice"] <= 100.5)
check("empty depth safe", compute_depth_features({"bids": [], "asks": []})["obi"] == 0.0)


# ── P3: oi_archiver full-chain parser ───────────────────────────────────────
print("\n[P3] oi_archiver parse_chain_rows")
from scripts.oi_archiver import parse_chain_rows

chain = {"atm_strike": 23400, "underlying_ltp": 23412.5, "chain": [
    {"strike": 23300, "ce": {"oi": 100, "iv": 14.1}, "pe": {"oi": 200, "iv": 15.2}},
    {"strike": 23400, "ce": {"oi": 500, "iv": 13.0}, "pe": {"oi": 600, "iv": 13.5}},
    {"strike": 23500, "ce": {"oi": 300, "iv": 12.8}, "pe": {"oi": 150, "iv": 16.0}},
]}
rows = parse_chain_rows(chain, n_strikes=11)
check("parses all strikes", len(rows) == 3)
atm_row = next(r for r in rows if r["strike"] == 23400)
check("ATM ce_oi captured", atm_row["ce_oi"] == 500)
check("ATM pe_iv captured (real IV)", abs(atm_row["pe_iv"] - 13.5) < 1e-9)
check("rows sorted by strike", [r["strike"] for r in rows] == [23300, 23400, 23500])
check("empty chain safe", parse_chain_rows({"chain": []}) == [])

# Kotak FLAT schema ({spot, ce_oi, ce_iv, ...}) must also parse with real IV
chain_flat = {"atm_strike": 23400, "spot": 23410.0, "chain": [
    {"strike": 23400, "ce_oi": 700, "ce_iv": 12.4, "pe_oi": 800, "pe_iv": 12.9},
    {"strike": 23450, "ce_oi": 250, "ce_iv": 11.8, "pe_oi": 120, "pe_iv": 14.1},
]}
rf = parse_chain_rows(chain_flat, n_strikes=11)
check("flat schema parsed", len(rf) == 2)
atm_f = next(r for r in rf if r["strike"] == 23400)
check("flat ce_oi captured", atm_f["ce_oi"] == 700)
check("flat ce_iv captured (real IV, non-zero)", abs(atm_f["ce_iv"] - 12.4) < 1e-9)
check("flat spot captured", atm_f["spot"] == 23410.0)
# n_strikes window keeps closest-to-ATM
narrow = parse_chain_rows(chain, n_strikes=0)   # ATM +/-0 -> 1 strike
check("n_strikes window limits", len(narrow) == 1 and narrow[0]["strike"] == 23400)


# ── P4: oi_features new features ────────────────────────────────────────────
print("\n[P4] oi_features Priority-4 features")
from scripts.oi_features import compute_features

# constructed 1-min day: constant ATM strike, monotone spot up,
# call_oi falling, put_oi rising (classic bullish put-writing regime)
n = 40
ts = pd.date_range("2026-06-16 09:15", periods=n, freq="1min")
df = pd.DataFrame({
    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
    "spot": np.linspace(23400, 23440, n),
    "atm_strike": 23400,
    "call_oi": np.linspace(500000, 460000, n),    # falling
    "put_oi":  np.linspace(500000, 560000, n),    # rising
    "call_iv": 13.0, "put_iv": 13.5,
    "pcr": np.linspace(1.0, 1.12, n),
})
out = compute_features(df)
for col in ["atm_ce_oi_chg_pct", "atm_pe_oi_chg_pct", "oi_buildup_signed", "net_oi_flow"]:
    check(f"{col} present", col in out.columns)
last = out.iloc[-1]
check("atm_ce_oi_chg_pct negative (calls unwinding)", last["atm_ce_oi_chg_pct"] < 0)
check("atm_pe_oi_chg_pct positive (puts writing)", last["atm_pe_oi_chg_pct"] > 0)
# spot up + total OI up (puts rise faster) -> buildup_signed = +1
check("oi_buildup_signed in {-1,0,1}", set(np.unique(out["oi_buildup_signed"].dropna())) <= {-1.0, 0.0, 1.0})
check("net_oi_flow in [-1,1]", out["net_oi_flow"].dropna().abs().max() <= 1.0 + 1e-9)
check("net_oi_flow positive (put-dominant flow)", last["net_oi_flow"] > 0)
# fixed-strike masking: an ATM re-reference should NaN the delta at the splice
df2 = df.copy(); df2.loc[20:, "atm_strike"] = 23450
out2 = compute_features(df2)
check("strike-roll masks chg at splice",
      np.isnan(out2["atm_ce_oi_chg_pct"].iloc[21]))


# ── P2b: futures_archiver deployment infra (restart safety, gaps, storage) ──
print("\n[P2b] futures_archiver deployment infrastructure")
import tempfile
from datetime import datetime as _dt
import scripts.futures_archiver as fa

fa.OUTDIR = Path(tempfile.mkdtemp())          # isolate from real archive dir
check("_validate_storage passes on writable dir", fa._validate_storage() is True)

row = {c: 0 for c in fa.COLS}
row.update({"timestamp": "2026-06-17 10:00:00", "fut_symbol": "NIFTYFUT",
            "fut_ltp": 23400.0, "bid_qty_1": 50})
fa.append(row)
data_path, gaps_path = fa._paths(_dt.now())
check("append created daily data file", data_path.exists())
check("_resume_ts returns last timestamp (restart safety)",
      fa._resume_ts() == "2026-06-17 10:00:00")

fa.log_gap("2026-06-17 10:00:00", "2026-06-17 10:06:00", 360, "test")
check("log_gap wrote gap file", gaps_path.exists())

if data_path.suffix == ".csv":
    data_path.write_text("wrong,header\n1,2\n", encoding="utf-8")   # corrupt header
    fa.append(row)
    rot = data_path.with_name(data_path.stem + "_oldschema.csv")
    check("schema rotation moved mismatched file aside", rot.exists())
    check("append rewrote a clean current file", data_path.exists())
else:
    check("schema rotation (parquet path — n/a)", True)

# backward-compat: snapshot row schema unchanged (COLS still starts with timestamp)
check("COLS backward-compatible (timestamp first)", fa.COLS[0] == "timestamp")


# ── P2c: deployment-reliability fixes (disk, atomic, quarantine, lockfile) ──
print("\n[P2c] deployment-reliability primitives")
import importlib, tempfile as _tf
import pandas as _pd

# fresh import after OUTDIR was patched above
importlib.reload(fa)
fa.OUTDIR = Path(_tf.mkdtemp())
fa._LOCKFILE = fa.OUTDIR / ".archiver.lock"

# 1. disk-low check toggles correctly via threshold (pass tempdir explicitly:
# the production default arg was bound at module import, before our test patch)
import shutil
real_free = shutil.disk_usage(fa.OUTDIR).free
fa._FREE_MIN_BYTES = real_free * 2          # force "low"
check("_disk_low True when threshold above free", fa._disk_low(fa.OUTDIR) is True)
fa._FREE_MIN_BYTES = 1                      # restore
check("_disk_low False when threshold tiny", fa._disk_low(fa.OUTDIR) is False)

# 2. atomic parquet write: no .tmp left behind, target file is valid
try:
    import pyarrow  # noqa: F401
    has_parquet = True
except Exception:
    has_parquet = False
if has_parquet:
    p = fa.OUTDIR / "atom.parquet"
    fa._atomic_parquet_write(_pd.DataFrame({"a": [1, 2]}), p)
    check("atomic write target exists", p.exists())
    check("no .tmp left behind", not p.with_suffix(".parquet.tmp").exists())
    df = _pd.read_parquet(p); check("atomic write payload OK", list(df["a"]) == [1, 2])
else:
    check("atomic parquet write (skipped — no pyarrow)", True)

# 3. corrupt parquet quarantine
bad = fa.OUTDIR / "bad.parquet"
bad.write_bytes(b"NOT A PARQUET FILE")
out = fa._read_parquet_or_quarantine(bad)
check("corrupt parquet returns None", out is None)
check("corrupt parquet renamed away", not bad.exists())
quarantined = list(fa.OUTDIR.glob("bad_corrupt_*.parquet"))
check("corrupt file quarantined with marker", len(quarantined) == 1)

# 4. lockfile claim + stale reclaim + same-pid idempotent + foreign-PID refusal
fa._LOCKFILE.unlink(missing_ok=True) if hasattr(Path, "unlink") else None
check("first claim succeeds", fa._claim_lock() is True)
check("lockfile exists after claim", fa._LOCKFILE.exists())
check("lockfile contains own pid", fa._LOCKFILE.read_text().strip() == str(os.getpid()))
# stale lock (dead PID) should be reclaimed
fa._LOCKFILE.write_text("99999999")         # almost certainly dead
check("stale-pid lockfile reclaimed", fa._claim_lock() is True)
check("after reclaim, lockfile is ours", fa._LOCKFILE.read_text().strip() == str(os.getpid()))
# foreign live PID (pid 1 = init on Linux, always alive; on Windows os.kill(1,0)
# raises -> treated as stale, so this assertion is platform-specific). Skip
# strict foreign-pid test on Windows to keep it portable.
if sys.platform != "win32":
    fa._LOCKFILE.write_text("1")
    check("live foreign PID refused", fa._claim_lock() is False)
    # cleanup foreign-PID stub so the release test below operates on OUR lock
    fa._LOCKFILE.unlink(missing_ok=True)
    fa._claim_lock()
else:
    check("foreign-PID test (skipped on Windows)", True)
# release lock
fa._release_lock()
check("lockfile removed on release", not fa._LOCKFILE.exists())

# also confirm oi_archiver exposes the same primitives (parity)
import scripts.oi_archiver as oa
for name in ("_disk_low", "_atomic_parquet_write",
             "_read_parquet_or_quarantine", "_claim_lock", "_release_lock"):
    check(f"oi_archiver.{name} present", hasattr(oa, name))


print(f"\n{'='*50}\n  RESULT: {PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
