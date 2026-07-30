"""
test_purged_split.py — Unit tests for purged_train_val_test_split (Issue #2).

Run:
    python scripts/test_purged_split.py

Verifies the split has NO label leakage:
  * sets are disjoint and chronological (train < val < test)
  * a purge+embargo GAP separates consecutive sets
  * the key no-leak property: max(prev) + FWD_BARS < min(next)
  * scaler-fit independence is structural (train indices come first)
  * tiny-n guard raises rather than returning an empty set
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model_v9 import (
    purged_train_val_test_split, FWD_BARS, EMBARGO_BARS,
    TRAIN_FRAC, VAL_FRAC,
)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}")


GAP = FWD_BARS + EMBARGO_BARS
print(f"\nConstants: FWD_BARS={FWD_BARS} EMBARGO_BARS={EMBARGO_BARS} "
      f"GAP={GAP} TRAIN_FRAC={TRAIN_FRAC} VAL_FRAC={VAL_FRAC}")

# ── Core split on a realistic size ──────────────────────────────────────────
n = 200_000
tr, va, te = purged_train_val_test_split(n)
print(f"\n[A] n={n:,} -> TRAIN={len(tr):,} VAL={len(va):,} TEST={len(te):,}")

# A1: non-empty
check("all sets non-empty", len(tr) and len(va) and len(te))

# A2: disjoint
s_tr, s_va, s_te = set(tr), set(va), set(te)
check("train/val disjoint", s_tr.isdisjoint(s_va))
check("val/test disjoint",  s_va.isdisjoint(s_te))
check("train/test disjoint", s_tr.isdisjoint(s_te))

# A3: chronological ordering
check("train before val", tr.max() < va.min())
check("val before test",  va.max() < te.min())

# A4: PURGE — the critical no-leak property.
# A train bar at position p has a label using close[p+FWD_BARS]. For no leak,
# the last used train position + FWD_BARS must be strictly before val starts.
check("no train->val label leak (gap >= FWD_BARS)",
      tr.max() + FWD_BARS < va.min())
check("no val->test label leak (gap >= FWD_BARS)",
      va.max() + FWD_BARS < te.min())

# A5: full purge+embargo gap honoured
check("train->val gap == GAP", va.min() - tr.max() - 1 == GAP)
check("val->test gap == GAP",  te.min() - va.max() - 1 == GAP)

# A6: test set is the most recent data (frozen holdout = newest)
check("test set ends at n-1", te.max() == n - 1)

# A7: fractions approximately honoured
check("train ~85%", abs(len(tr) / n - TRAIN_FRAC) < 0.01)
check("val ~7.5%",  abs(len(va) / n - VAL_FRAC) < 0.01)

# ── Tiny-n guard ────────────────────────────────────────────────────────────
print("\n[B] tiny-n guard")
raised = False
try:
    purged_train_val_test_split(10)   # smaller than the gap -> empty set
except ValueError:
    raised = True
check("tiny n raises ValueError (no silent empty set)", raised)

# ── Determinism ─────────────────────────────────────────────────────────────
print("\n[C] determinism")
tr2, va2, te2 = purged_train_val_test_split(n)
check("split is deterministic",
      np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2))

print(f"\n{'='*50}")
print(f"  RESULT: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(1 if FAIL else 0)
