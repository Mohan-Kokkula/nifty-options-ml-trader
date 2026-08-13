"""
test_profile_as_features.py — does price density help the MODEL?

Today's scripts/test_profile_setups.py tested Market Profile as trading
RULES and they lost: 27 variants, best PF 0.899 over 3,754 trades. That was
the narrow test. A feature can carry real information while a naive rule
built on it still loses money -- dist_vwap is exactly that in this project:
a losing rule on its own, yet it survived V9's cut into V11's 26 features.

So this asks the other question. Same V11 model, same purged split, same
live decision rule -- only the feature list differs.

    baseline   V11_FEATURES                      (26)
    augmented  V11_FEATURES + 7 profile features (33)

THE SEVEN, all strictly causal (developing profile sees only bars 0..i of
its own session; prior-day profile is yesterday's completed one):

    poc_dist_atr    close vs the DEVELOPING POC, in ATR
    ppoc_dist_atr   close vs yesterday's POC
    pvah_dist_atr   close vs yesterday's value-area high
    pval_dist_atr   close vs yesterday's value-area low
    in_value        1 when price sits inside yesterday's value area
    va_width_atr    yesterday's value-area width, in ATR
    poc_shift_atr   today's developing POC vs yesterday's -- value migration

PRE-REGISTERED, stated before the run so it cannot be chosen afterwards:
    selection metric = VAL profit factor.
    Adopt only if augmented beats baseline on VAL PF *and* holds on TEST.
    Earlier in this project a post-hoc metric (VAL acc x recall) picked a
    variant that then failed TEST; that mistake is not repeated here.

    dir_acc is reported but is NOT the criterion. Accuracy has repeatedly
    moved without PF following it -- 65% direction against a 52% path AUC
    is the whole reason this project exists.

NOTE ON VOLUME: NIFTY spot volume is 0 for 2015-2025, so these are TPO
(time-at-price) profiles, not volume profiles. build_profile() accepts a
weights array, so this becomes a real volume profile once the futures
archive is deep enough. It has ~2 months.

USAGE
    python scripts/test_profile_as_features.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_model_v9 as V9  # noqa: E402
import scripts.train_model_v11 as V11  # noqa: E402
from core.profile_features import developing_profiles, session_profiles  # noqa: E402

PROFILE_CACHE = ROOT / "data" / ".profile_cache.pkl"
FRICTION = 5.9
LOT_ATR_STOP = 4.0          # the exit that validated: 4 x ATR, ride to close
EOD, ENTRY_START, ENTRY_END = "15:15", "09:20", "15:00"

PROFILE_FEATURES = ["poc_dist_atr", "ppoc_dist_atr", "pvah_dist_atr",
                    "pval_dist_atr", "in_value", "va_width_atr",
                    "poc_shift_atr"]


def attach_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Join causal profile levels and derive the seven features."""
    if PROFILE_CACHE.exists():
        print(f"  loading profile levels from {PROFILE_CACHE.name}")
        cached = joblib.load(PROFILE_CACHE)
        lv = cached[["p_poc", "p_vah", "p_val", "d_poc"]]
    else:
        print("  building profiles (causal, ~2min)")
        raw = pd.read_csv(V9.CSV_5M, parse_dates=["date"]).set_index("date")
        raw = raw[~raw.index.duplicated(keep="first")].sort_index()
        sp = session_profiles(raw).shift(1).add_prefix("p_")
        day = raw.index.normalize()
        for c in sp.columns:
            raw[c] = sp[c].reindex(day).values
        lv = raw.join(developing_profiles(raw))[
            ["p_poc", "p_vah", "p_val", "d_poc"]]

    df = df.join(lv, how="left")

    # a14 is ATR/close (train_model_v9.py:432) -- recover points
    atr = (df["a14"] * df["close"]).replace([np.inf, -np.inf], np.nan)
    atr = atr.where(atr > 0, np.nan)
    c = df["close"]

    df["poc_dist_atr"] = (c - df["d_poc"]) / atr
    df["ppoc_dist_atr"] = (c - df["p_poc"]) / atr
    df["pvah_dist_atr"] = (c - df["p_vah"]) / atr
    df["pval_dist_atr"] = (c - df["p_val"]) / atr
    df["in_value"] = ((c >= df["p_val"]) & (c <= df["p_vah"])).astype(float)
    df["va_width_atr"] = (df["p_vah"] - df["p_val"]) / atr
    df["poc_shift_atr"] = (df["d_poc"] - df["p_poc"]) / atr

    for f in PROFILE_FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def fit_and_score(df: pd.DataFrame, fcols: list, idx: tuple) -> dict:
    """Train the V11 ensemble on `fcols`; return VAL/TEST metrics."""
    from sklearn.preprocessing import StandardScaler

    tr, va, te = idx
    X = df[fcols].replace([np.inf, -np.inf], 0.0).fillna(0.0).values
    y = df["label"].values

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr])          # fit on TRAIN only
    Xvl, Xte = sc.transform(X[va]), sc.transform(X[te])
    w = V11.build_sample_weights(df, tr, y[tr])
    models = V9._fit_ensemble(Xtr, y[tr], Xvl, y[va], w)

    out = {}
    for name, Xe, rows in (("val", Xvl, va), ("test", Xte, te)):
        p = np.mean([m.predict_proba(Xe) for m in models.values()], axis=0)
        sig = V9.live_rule_signals(p)
        m = V9.live_rule_metrics(p, y[rows])
        out[name] = {"dir_acc": m["dir_acc"], "recall": m["recall"],
                     "n_fire": int((sig != 2).sum())}
        out[name].update(simulate(df, rows, sig))
    return out


def simulate(df: pd.DataFrame, rows: np.ndarray, sig: np.ndarray) -> dict:
    """Common exit for both models: 4xATR stop, ride to the close."""
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
            if d == 0 and L[i] <= sl:
                hit = sl - e
            elif d == 1 and H[i] >= sl:
                hit = e - sl
            if hit is None and hhmm[i] >= EOD:
                hit = (C[i] - e) if d == 0 else (e - C[i])
            if hit is not None:
                trades.append(hit - FRICTION)
                pos = None
            continue
        if sig[i] == 2 or not (ENTRY_START <= hhmm[i] < ENTRY_END):
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        stop = a * LOT_ATR_STOP
        e = C[i]
        pos = (int(sig[i]), e, e - stop if sig[i] == 0 else e + stop)

    arr = np.array(trades, dtype=float)
    if len(arr) == 0:
        return {"pf": 0.0, "n": 0, "avg": 0.0, "total": 0.0}
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return {"pf": float(w / l) if l > 0 else float("inf"), "n": len(arr),
            "avg": float(arr.mean()), "total": float(arr.sum())}


def line(tag: str, m: dict) -> str:
    pf = "  inf " if m["pf"] == float("inf") else f"{m['pf']:6.3f}"
    return (f"{tag:<26} {m['dir_acc']:>7.1%} {m['recall']:>8.1%} "
            f"{pf} {m['n']:>6} {m['avg']:>+8.2f} {m['total']:>+9.0f}")


HDR = (f"{'model / split':<26} {'dir_acc':>7} {'recall':>8} {'PF':>6} "
       f"{'n':>6} {'avg':>8} {'total':>9}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()

    print("PRE-REGISTERED: selection metric is VAL profit factor. Adopt only")
    print("if augmented beats baseline on VAL PF and holds on TEST.\n")

    print("building V11 dataset...")
    df = V11.build_dataset(V9.CSV_5M, V9.CSV_15M, V9.CSV_30M, V9.CSV_60M,
                           V9.CSV_DAY, V9.CSV_VIX, V9.CSV_FUT)
    print(f"  rows={len(df):,}  ({time.time()-t0:.0f}s)")

    print("attaching profile features...")
    df = attach_profile(df)
    cov = {f: float(df[f].ne(0).mean()) for f in PROFILE_FEATURES}
    print("  non-zero coverage: " + ", ".join(f"{k}={v:.0%}" for k, v in cov.items()))

    base = list(V11.V11_FEATURES)
    missing = [c for c in base if c not in df.columns]
    if missing:
        raise SystemExit(f"V11 features missing: {missing}")
    aug = base + PROFILE_FEATURES
    idx = V9.purged_train_val_test_split(len(df))
    print(f"\nsplit: TRAIN {len(idx[0]):,} | VAL {len(idx[1]):,} | TEST {len(idx[2]):,}")

    print(f"\ntraining baseline ({len(base)} features)...")
    b = fit_and_score(df, base, idx)
    print(f"training augmented ({len(aug)} features)...")
    a = fit_and_score(df, aug, idx)

    print(f"\n{'='*88}\nRESULTS — same split, same live rule, same 4xATR/EOD exit")
    print(f"{'='*88}\n{HDR}")
    print(line(f"baseline  ({len(base)}f)  VAL", b["val"]))
    print(line(f"augmented ({len(aug)}f)  VAL", a["val"]))
    print("-" * 88)
    print(line(f"baseline  ({len(base)}f)  TEST", b["test"]))
    print(line(f"augmented ({len(aug)}f)  TEST", a["test"]))

    dv = a["val"]["pf"] - b["val"]["pf"]
    dt = a["test"]["pf"] - b["test"]["pf"]
    print(f"\nVAL  PF delta {dv:+.3f}   (the pre-registered criterion)")
    print(f"TEST PF delta {dt:+.3f}")
    print(f"VAL  dir_acc delta {a['val']['dir_acc']-b['val']['dir_acc']:+.1%}")
    print(f"TEST dir_acc delta {a['test']['dir_acc']-b['test']['dir_acc']:+.1%}")

    print("\nVERDICT")
    if dv <= 0:
        print(f"  REJECT — profile features do not improve VAL PF ({dv:+.3f}).")
        print("  Price density carries nothing the model did not already have.")
    elif dt <= 0:
        print(f"  REJECT — VAL improved ({dv:+.3f}) but TEST did not ({dt:+.3f}).")
        print("  That is the VAL-only pattern that has failed all session.")
    else:
        print(f"  ADOPT-CANDIDATE — VAL {dv:+.3f} and TEST {dt:+.3f} both positive.")
        print("  Worth a walk-forward confirmation before it goes anywhere near live.")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
