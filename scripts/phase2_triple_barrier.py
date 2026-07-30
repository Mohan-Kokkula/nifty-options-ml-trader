"""
phase2_triple_barrier.py — PHASE 2: leakage-safe triple-barrier label aligned
to actual production trading outcomes.

LABEL SPECIFICATION (production settings, zero tuning):
  For each 5-min bar t in the live trade window (10:00–15:00 entries,
  matching the morning hard-block and late cutoff):
    entry      = open of bar t+1                  (same as sim/live)
    SL         = clamp(2.2 × ATR15_completed(t), 30, 80) spot pts (production)
    TP         = 2 × SL                           (production)
    time exit  = end of day                       (production flat rule)
  Simulate LONG and SHORT hypothetical trades, first-touch wins, same-bar
  SL+TP ambiguity resolves to SL (conservative, same as validation sim).
  label = 0 (CALL) if the LONG trade hits TP first (before SHORT's TP)
        = 1 (PUT)  if the SHORT trade hits TP first
        = 2 (SKIP) otherwise (both stop out / EOD drift / out of window)
  VIX>35 bars forced SKIP (production risk posture, as in the old labeler).
  NO RSI/ADX conditioning — removes the label-feature coupling entirely.

  ATR15 uses COMPLETED 15-min bars only (consistent with the fixed pipeline).

UNIQUENESS WEIGHTS (López de Prado): event i spans [t+1, exit_i];
  concurrency c_s = #active events at bar s; u_i = mean(1/c_s) over the span.
  Final training weight = class_balance × uniqueness × expiry-era boost
  (the existing production weighting formula, with uniqueness multiplied in).

TRAINING: identical _fit_ensemble hyperparams, identical purged 85/7.5/7.5
split, scaler fit on train only, SAME 170 feature columns as the clean
baseline (sandbox_clean) — the label is the ONLY changed variable.

VALIDATION: both models (TB vs clean-baseline) evaluated on the SAME frozen
test rows: bar-level Spearman IC vs forward returns, gate-sim (NEW config)
PF/DD/trades/expectancy, bootstrap CI on the IC difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model_v9 import (                          # noqa: E402
    purged_train_val_test_split, _fit_ensemble, FWD_BARS,
)
from scripts.phase0_retrain_clean import build_features       # noqa: E402
from scripts.phase1_edge_discovery import build_aux           # noqa: E402
from scripts.validate_fixes_sim import run_variant            # noqa: E402

RES = ROOT / "data/validation_results"
OUT = ROOT / "models/sandbox_tb"
OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.22
NEW_VARIANT = dict(thr_call=0.32, thr_put=0.25, skip_ceil=0.65,
                   g2_call=72, g2_put=58, new_regime_floors=True,
                   pe_chop_cap=True, f_rsi=True, f_slt=True, f_rev=True,
                   f_mtf=True, cr_pen=5, cr_floor=65)


def completed_atr15(df):
    df15 = df[["open", "high", "low", "close"]].resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr = pd.concat([df15.high - df15.low,
                    (df15.high - df15.close.shift()).abs(),
                    (df15.low - df15.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr.index = atr.index + pd.Timedelta(minutes=10)   # completed-bar labeling
    return atr.reindex(df.index, method="ffill").fillna(30.0)


def barrier_labels(df):
    """Triple-barrier labels + exit indices. Returns (labels, exit_idx)."""
    o = df["open"].values; h = df["high"].values; l = df["low"].values
    atr = completed_atr15(df).values
    minutes = df.index.hour * 60 + df.index.minute
    dates = df.index.date
    n = len(df)
    lab = np.full(n, 2, dtype=np.int8)
    exit_i = np.arange(n)

    i = 0
    while i < n - 1:
        if not (600 <= minutes[i] <= 900) or dates[i + 1] != dates[i]:
            i += 1
            continue
        sl = min(80.0, max(30.0, 2.2 * atr[i]))
        tp = 2.0 * sl
        entry = o[i + 1]
        long_done = short_done = 0      # 0=open 1=TP 2=SL
        long_j = short_j = -1
        j = i + 1
        while j < n and dates[j] == dates[i]:
            if long_done == 0:
                if l[j] <= entry - sl:
                    long_done, long_j = 2, j
                elif h[j] >= entry + tp:
                    long_done, long_j = 1, j
            if short_done == 0:
                if h[j] >= entry + sl:
                    short_done, short_j = 2, j
                elif l[j] <= entry - tp:
                    short_done, short_j = 1, j
            if long_done and short_done:
                break
            j += 1
        eod = j - 1 if j <= n - 1 else n - 1
        if long_done == 1 and (short_done != 1 or long_j <= short_j):
            lab[i] = 0
            exit_i[i] = long_j
        elif short_done == 1:
            lab[i] = 1
            exit_i[i] = short_j
        else:
            lab[i] = 2
            exit_i[i] = max(long_j, short_j, eod)
        i += 1
    return lab, exit_i


def uniqueness_weights(lab, exit_i):
    n = len(lab)
    conc = np.zeros(n + 1)
    starts = np.arange(n) + 1
    for i in range(n):
        if lab[i] != 2:
            conc[starts[i]] += 1
            e = min(exit_i[i] + 1, n)
            conc[e] -= 1 if e < n + 1 else 0
    conc = np.cumsum(conc[:n])
    inv = 1.0 / np.maximum(conc, 1.0)
    cs = np.concatenate([[0.0], np.cumsum(inv)])
    u = np.ones(n)
    for i in range(n):
        if lab[i] != 2:
            s, e = min(starts[i], n - 1), min(exit_i[i], n - 1)
            u[i] = (cs[e + 1] - cs[s]) / max(e - s + 1, 1)
    return u


def sim_metrics(tr):
    if tr.empty:
        return {"trades": 0, "pf": None, "dd": 0, "net": 0, "exp": None, "wr": None}
    pnl = tr["pnl"]
    daily = tr.groupby(pd.to_datetime(tr["ts"]).dt.date)["pnl"].sum().cumsum()
    return {"trades": len(tr), "wr": round(float((pnl > 0).mean()), 3),
            "pf": round(float(pnl[pnl > 0].sum() / max(abs(pnl[pnl <= 0].sum()), 1e-9)), 2),
            "dd": round(float((daily - daily.cummax()).min()), 0),
            "net": round(float(pnl.sum()), 0),
            "exp": round(float(pnl.mean()), 0)}


def main():
    print("[1/6] Clean feature matrix...")
    dfm = build_features()
    fcols = joblib.load(ROOT / "models/sandbox_clean/feature_cols.pkl")
    dfm = dfm.dropna(subset=fcols)
    dfm["fwd_ret"] = dfm["close"].shift(-FWD_BARS) / dfm["close"] - 1

    print("[2/6] Barrier engine...")
    lab, exit_i = barrier_labels(dfm)
    if "vix_level" in dfm.columns:
        lab[(dfm["vix_level"].values > 35.0)] = 2
    dfm["label"] = lab

    bc = pd.Series(lab).value_counts().sort_index()
    base = {"CALL": int(bc.get(0, 0)), "PUT": int(bc.get(1, 0)),
            "SKIP": int(bc.get(2, 0))}
    print(f"   base rates: {base} | directional "
          f"{(base['CALL']+base['PUT'])/len(lab):.1%}")
    yearly_rates = pd.DataFrame({
        "year": dfm.index.year, "lab": lab}).groupby("year")["lab"].apply(
        lambda s: round(float((s != 2).mean()), 4)).to_dict()

    print("[3/6] Uniqueness weights...")
    u = uniqueness_weights(lab, exit_i)
    print(f"   uniqueness: mean={u[lab != 2].mean():.3f} "
          f"min={u[lab != 2].min():.3f}")

    print("[4/6] Train (same hyperparams/split/features as clean baseline)...")
    X, y = dfm[fcols].values, dfm["label"].values
    tr_i, va_i, te_i = purged_train_val_test_split(len(X))
    sc = StandardScaler()
    Xtr, Xvl, Xte = (sc.fit_transform(X[tr_i]), sc.transform(X[va_i]),
                     sc.transform(X[te_i]))
    ytr = y[tr_i]
    call_n, put_n = (ytr == 0).sum(), (ytr == 1).sum()
    dir_n = call_n + put_n
    skip_pct = (ytr == 2).mean()
    trade_w = skip_pct / (1 - skip_pct) if skip_pct < 1 else 2.0
    cb = dir_n / (2.0 * call_n) if call_n else 1.0
    pb = dir_n / (2.0 * put_n) if put_n else 1.0
    w = np.where(ytr == 0, trade_w * cb, np.where(ytr == 1, trade_w * pb, 1.0))
    if "expiry_is_tue" in dfm.columns:
        w = w * np.where(dfm["expiry_is_tue"].values[tr_i].astype(bool), 6.0, 1.0)
    # uniqueness: normalize to mean 1.0 across directional train events so it
    # REDISTRIBUTES weight among overlapping events without shrinking the
    # directional class aggregate (raw mean ≈0.085 collapsed the model to
    # all-SKIP on the first run)
    u_tr = u[tr_i].copy()
    dir_mask = ytr != 2
    if dir_mask.any() and u_tr[dir_mask].mean() > 0:
        u_tr[dir_mask] = u_tr[dir_mask] / u_tr[dir_mask].mean()
    w = w * u_tr
    models = _fit_ensemble(Xtr, ytr, Xvl, y[va_i], w)
    joblib.dump(models, OUT / "models.pkl")
    joblib.dump(sc, OUT / "scaler.pkl")
    joblib.dump(fcols, OUT / "feature_cols.pkl")

    print("[5/6] Evaluate TB model and clean baseline on SAME test rows...")
    report = {"label_base_rates": base, "directional_rate_by_year": yearly_rates,
              "uniqueness_mean": round(float(u[lab != 2].mean()), 4)}
    yte = y[te_i]
    te_rows = dfm.iloc[te_i]
    fwd = te_rows["fwd_ret"].values
    rng = np.random.default_rng(13)

    def eval_model(tag, mdl, scl, fc):
        Xe = scl.transform(te_rows[fc].values)
        p = np.mean([m.predict_proba(Xe) for m in mdl.values()], axis=0)
        s = np.where(p.max(axis=1) >= CONF, p.argmax(axis=1), 2)
        dm = (s != 2) & (yte != 2)
        da = float((yte[dm] == s[dm]).mean()) if dm.any() else None
        edge = p[:, 0] - p[:, 1]
        m = ~np.isnan(fwd)
        ic, icp = spearmanr(edge[m], fwd[m])
        dft = te_rows[["open", "high", "low", "close"]].copy()
        dft["call_p"], dft["put_p"], dft["skip_p"] = p[:, 0], p[:, 1], p[:, 2]
        dft["rsi14"] = te_rows["rsi14"] if "rsi14" in te_rows.columns else 50.0
        sim = sim_metrics(run_variant(dft, build_aux(dft), NEW_VARIANT))
        report[tag] = {"dir_acc_vs_TB_labels": round(da, 4) if da else None,
                       "IC_vs_fwd_ret": round(float(ic), 4),
                       "IC_p": round(float(icp), 6), "sim": sim}
        print(f"   {tag:14s} acc(TB-labels)={da if da else 0:.4f} "
              f"IC={ic:+.4f}(p={icp:.4f}) | sim n={sim['trades']} "
              f"pf={sim['pf']} dd={sim['dd']} exp={sim['exp']}")
        return edge

    e_tb = eval_model("TB_model", models, sc, fcols)
    base_m = joblib.load(ROOT / "models/sandbox_clean/models.pkl")
    base_s = joblib.load(ROOT / "models/sandbox_clean/scaler.pkl")
    e_bl = eval_model("baseline_model", base_m, base_s, fcols)

    print("[6/6] Significance: bootstrap CI on dIC (TB - baseline)...")
    msk = ~np.isnan(fwd)
    fwd_c, e_tb_c, e_bl_c = fwd[msk], e_tb[msk], e_bl[msk]
    n = len(fwd_c)
    deltas = np.empty(2000)
    for k in range(2000):
        ix = rng.integers(0, n, n)
        deltas[k] = (spearmanr(e_tb_c[ix], fwd_c[ix])[0]
                     - spearmanr(e_bl_c[ix], fwd_c[ix])[0])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    report["delta_IC_bootstrap"] = {
        "mean": round(float(deltas.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "significant": bool(lo > 0 or hi < 0)}
    print(f"   dIC = {deltas.mean():+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")

    json.dump(report, open(RES / "phase2_tb_report.json", "w"), indent=1,
              default=str)
    print(f"\nReport -> {RES / 'phase2_tb_report.json'}")


if __name__ == "__main__":
    main()
