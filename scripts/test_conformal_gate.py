"""
test_conformal_gate.py — one calibrated abstention rule vs twelve hand-tuned gates.

WHY
    claude_pilot.py puts 12 gates between a model signal and an order. Five
    cannot be validated at all (1 IV record, 29 days of OI, 18 brief files,
    no 3-minute bars). scripts/replay_gate_stack.py measured the other seven
    and found the stack removes 87% of trades while barely improving
    selection: PF 0.638 gated vs 0.660 ungated over the last 91 sessions.

    Every one of those gates answers the same question by assertion --
    "when should I trust this model?" Conformal prediction answers it with a
    finite-sample coverage guarantee and ONE knob instead of forty.

METHOD (split conformal, Mondrian / class-conditional)
    1. TRAIN is split into proper-train and a calibration block, separated
       by the same purge gap used everywhere else in this project.
    2. Fit the V11 ensemble on proper-train only.
    3. On calibration, nonconformity s_i = 1 - p(true class of i).
    4. Per class c, q_c = the ceil((n+1)(1-alpha))/n quantile of that
       class's own scores. Class-conditional because CALL/PUT/SKIP are
       badly imbalanced and a pooled quantile would let SKIP dominate.
    5. Prediction SET = { c : 1 - p(c) <= q_c }.
    6. TRADE only when the set is the singleton {CALL} or {PUT}. Empty,
       containing SKIP, or containing both directions all mean the model is
       not confident enough to act -- abstain.

    Step 6 is the point: abstention is DERIVED from calibration data, not
    hand-tuned like the twelve gates it would replace.

PRE-REGISTERED, stated before the run
    selection metric = VAL profit factor.
    Adopt only if conformal beats the live decision rule on VAL PF *and*
    holds on TEST. Coverage is reported as a correctness check on the
    method itself, never as evidence of profitability.

USAGE
    python scripts/test_conformal_gate.py
    python scripts/test_conformal_gate.py --alphas 0.05,0.10,0.20,0.30
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_model_v9 as V9  # noqa: E402
import scripts.train_model_v11 as V11  # noqa: E402

FRICTION = 5.9
ATR_STOP = 4.0
EOD, ENTRY_START, ENTRY_END = "15:15", "09:20", "15:00"
CALL, PUT, SKIP = 0, 1, 2


def conformal_quantiles(p_cal: np.ndarray, y_cal: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Class-conditional split-conformal thresholds."""
    q = np.ones(3)
    for c in range(3):
        m = y_cal == c
        n = int(m.sum())
        if n < 50:
            q[c] = 1.0
            continue
        s = 1.0 - p_cal[m, c]                      # nonconformity
        lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q[c] = float(np.quantile(s, lvl, method="higher"))
    return q


def predict_sets(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Boolean (n,3) membership matrix of the conformal prediction sets."""
    return (1.0 - p) <= q[None, :]


def conformal_signal(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Act only on an unambiguous singleton {CALL} or {PUT}."""
    S = predict_sets(p, q)
    sig = np.full(len(p), SKIP, dtype=np.int8)
    lone = S.sum(axis=1) == 1
    sig[lone & S[:, CALL]] = CALL
    sig[lone & S[:, PUT]] = PUT
    return sig


def simulate(df: pd.DataFrame, rows: np.ndarray, sig: np.ndarray) -> dict:
    """4xATR stop, ride to the close. Identical for every config compared."""
    sub = df.iloc[rows]
    H, L, C = sub["high"].values, sub["low"].values, sub["close"].values
    atr = (sub["a14"] * sub["close"]).values
    hhmm = sub.index.strftime("%H:%M").values
    days = sub.index.normalize().values
    trades, pos, cur = [], None, None
    for i in range(len(sub)):
        if days[i] != cur:
            cur, pos = days[i], None
        if pos is not None:
            d, e, sl = pos
            hit = None
            if d == CALL and L[i] <= sl:
                hit = sl - e
            elif d == PUT and H[i] >= sl:
                hit = e - sl
            if hit is None and hhmm[i] >= EOD:
                hit = (C[i] - e) if d == CALL else (e - C[i])
            if hit is not None:
                trades.append(hit - FRICTION)
                pos = None
            continue
        if sig[i] == SKIP or not (ENTRY_START <= hhmm[i] < ENTRY_END):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        stop = a * ATR_STOP
        e = C[i]
        pos = (int(sig[i]), e, e - stop if sig[i] == CALL else e + stop)
    arr = np.array(trades, dtype=float)
    if len(arr) == 0:
        return {"pf": 0.0, "n": 0, "avg": 0.0, "total": 0.0}
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return {"pf": float(w / l) if l > 0 else float("inf"), "n": len(arr),
            "avg": float(arr.mean()), "total": float(arr.sum())}


def dir_acc(sig: np.ndarray, y: np.ndarray) -> float:
    m = (sig != SKIP) & (y != SKIP)
    return float((sig[m] == y[m]).mean()) if m.any() else float("nan")


def row(tag: str, s: dict, acc: float, cov: float = float("nan")) -> str:
    pf = "  inf " if s["pf"] == float("inf") else f"{s['pf']:6.3f}"
    cs = "     -" if not np.isfinite(cov) else f"{cov:6.1%}"
    return (f"{tag:<26}{pf}{s['n']:>7}{s['avg']:>+9.2f}{s['total']:>+9.0f}"
            f"{acc:>9.1%}{cs:>9}")


HDR = (f"{'config':<26}{'PF':>6}{'n':>7}{'avg':>9}{'total':>9}"
       f"{'dir_acc':>9}{'cover':>9}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.30,0.40")
    a = ap.parse_args()
    t0 = time.time()

    print("PRE-REGISTERED: selection metric = VAL profit factor. Adopt only")
    print("if conformal beats the live rule on VAL PF and holds on TEST.")
    print("Coverage validates the METHOD, never the profitability.\n")

    df = V11.build_dataset(V9.CSV_5M, V9.CSV_15M, V9.CSV_30M, V9.CSV_60M,
                           V9.CSV_DAY, V9.CSV_VIX, V9.CSV_FUT)
    fcols = list(V11.V11_FEATURES)
    X = df[fcols].replace([np.inf, -np.inf], 0.0).fillna(0.0).values
    y = df["label"].values.astype(int)
    tr, va, te = V9.purged_train_val_test_split(len(df))

    gap = V9.FWD_BARS + V9.EMBARGO_BARS
    cut = int(len(tr) * 0.85)
    ptr, cal = tr[:cut - gap], tr[cut:]
    print(f"proper-train {len(ptr):,} | calibration {len(cal):,} "
          f"(purge {gap}) | VAL {len(va):,} | TEST {len(te):,}\n")

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[ptr])
    models = V9._fit_ensemble(Xtr, y[ptr], sc.transform(X[va]), y[va],
                              V11.build_sample_weights(df, ptr, y[ptr]))

    def proba(idx):
        return np.mean([m.predict_proba(sc.transform(X[idx]))
                        for m in models.values()], axis=0)

    p_cal, p_va, p_te = proba(cal), proba(va), proba(te)

    print("=" * 75)
    print("BASELINE — the live decision rule (what runs today)")
    print("=" * 75)
    print(HDR)
    base = {}
    for name, p, idx in (("VAL", p_va, va), ("TEST", p_te, te)):
        s = V9.live_rule_signals(p)
        st = simulate(df, idx, s)
        base[name] = st
        print(row(f"live rule  {name}", st, dir_acc(s, y[idx])))

    print(f"\n{'='*75}\nCONFORMAL — one knob, calibrated abstention\n{'='*75}")
    print(HDR)
    best = None
    for al in [float(x) for x in a.alphas.split(",")]:
        q = conformal_quantiles(p_cal, y[cal], al)
        for name, p, idx in (("VAL", p_va, va), ("TEST", p_te, te)):
            sg = conformal_signal(p, q)
            S = predict_sets(p, q)
            cov = float(S[np.arange(len(p)), y[idx]].mean())
            st = simulate(df, idx, sg)
            print(row(f"alpha={al:.2f}  {name}", st, dir_acc(sg, y[idx]), cov))
            if name == "VAL" and st["n"] >= 30:
                if best is None or st["pf"] > best[0]["pf"]:
                    best = (st, al)
        print("-" * 75)

    if best is None:
        print("\nNo alpha produced enough VAL trades to judge.")
        return
    bs, bal = best
    q = conformal_quantiles(p_cal, y[cal], bal)
    ts = simulate(df, te, conformal_signal(p_te, q))
    dv = bs["pf"] - base["VAL"]["pf"]
    dt = ts["pf"] - base["TEST"]["pf"]
    print(f"\nVAL picked alpha={bal:.2f}")
    print(f"  VAL  conformal {bs['pf']:.3f} vs live {base['VAL']['pf']:.3f}"
          f"   delta {dv:+.3f}")
    print(f"  TEST conformal {ts['pf']:.3f} vs live {base['TEST']['pf']:.3f}"
          f"   delta {dt:+.3f}")

    print("\nVERDICT")
    if dv <= 0:
        print(f"  REJECT — conformal does not beat the live rule on VAL ({dv:+.3f}).")
    elif dt <= 0:
        print(f"  REJECT — VAL improved ({dv:+.3f}) but TEST did not ({dt:+.3f}).")
        print("  Same VAL-only pattern that has failed every time this session.")
    else:
        print(f"  ADOPT-CANDIDATE — VAL {dv:+.3f} and TEST {dt:+.3f} both positive.")
        print("  Needs walk-forward confirmation before it replaces any gate.")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
