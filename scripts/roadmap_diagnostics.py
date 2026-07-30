"""
roadmap_diagnostics.py -- read-only diagnostics for the engineering-roadmap
review. Covers exactly the 3 areas of the 7-area request that have NO
prior evidence anywhere in this project and are NOT on the "do not repeat"
list: (5) ensemble diversity between XGB and LGB, (6) error clustering
into named regime buckets, (7) regime-segmentation achievable-PF estimate.

Reuses the unmodified production pipeline (build_frame/train_fold/
simulate_trades from backtest_threshold_sweep/backtest_options,
PRODUCTION_BASELINE thresholds) across the full 8-fold walk-forward --
same models, same thresholds, same data as every other experiment this
arc. NO new model architecture, NO threshold/label changes, nothing
written back to any production file. Only NEW measurement: keeping
XGB and LGB predictions separate (production averages them) and tagging
every traded/skipped bar with regime context for slicing.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import (
    build_frame, train_fold, _proba3, signals_from_probas, EMBARGO_DAYS, add_months,
    metrics as trade_metrics,
)
from backtest_options import build_iv_map, simulate_trades
from threshold_opt import PRODUCTION_BASELINE as PB

CALL_THR, PUT_THR, SKIP_CEIL = PB.call_thr, PB.put_thr, PB.skip_ceil

FOLDS = [
    (date(2024, 7, 1), date(2024, 10, 1)), (date(2024, 10, 1), date(2025, 1, 1)),
    (date(2025, 1, 1), date(2025, 4, 1)), (date(2025, 4, 1), date(2025, 7, 1)),
    (date(2025, 7, 1), date(2025, 10, 1)), (date(2025, 10, 1), date(2026, 1, 1)),
    (date(2026, 1, 1), date(2026, 4, 1)), (date(2026, 4, 1), date(2026, 5, 1)),
]


def main():
    t0 = time.time()
    feat, fcols = build_frame()
    iv, exp = build_iv_map()

    all_rows = []          # per-traded-bar diversity + regime data
    all_fp_rows, all_fn_rows = [], []   # raw context for error clustering
    per_fold_metrics = {}

    for k, (a, b) in enumerate(FOLDS, 1):
        cutoff = a - timedelta(days=EMBARGO_DAYS)
        tr_mask = feat.index.date < cutoff
        te_mask = (feat.index.date >= a) & (feat.index.date < b)
        if tr_mask.sum() < 5000 or te_mask.sum() < 50:
            continue
        test = feat[te_mask]
        models, sc = train_fold(feat, fcols, tr_mask)
        if models is None:
            continue
        Xte = sc.transform(test[fcols].values)
        p_xgb = _proba3(models["xgb"], Xte)
        p_lgb = _proba3(models["lgb"], Xte)
        p_ens = (p_xgb + p_lgb) / 2
        sig = signals_from_probas(p_ens, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_xgb = signals_from_probas(p_xgb, CALL_THR, PUT_THR, SKIP_CEIL)
        sig_lgb = signals_from_probas(p_lgb, CALL_THR, PUT_THR, SKIP_CEIL)

        tdf = simulate_trades(test, sig, p_ens, iv, exp)
        m = trade_metrics(tdf)
        per_fold_metrics[k] = dict(test_range=[str(a), str(b)],
                                    n_trades=(m or {}).get("trades", 0),
                                    pf=(m or {}).get("pf"))
        print(f"[Fold {k}/8] test {a}..{b}  trades={(m or {}).get('trades',0)}  "
              f"pf={(m or {}).get('pf')}")

        pnl_map = dict(zip(tdf["entry_time"], tdf["net_option"])) if len(tdf) and "entry_time" in tdf.columns else {}

        for i, ts in enumerate(test.index):
            row = test.iloc[i]
            traded = sig[i] != 2
            agree = int(sig_xgb[i] == sig_lgb[i])
            rec = dict(fold=k, ts=str(ts), agree=agree,
                       p_xgb_call=float(p_xgb[i, 0]), p_xgb_put=float(p_xgb[i, 1]),
                       p_lgb_call=float(p_lgb[i, 0]), p_lgb_put=float(p_lgb[i, 1]),
                       vix_regime=row.get("vix_regime"), is_expiry=row.get("is_expiry"),
                       tf15_adx=row.get("tf15_adx"), is_morning=row.get("is_morning"),
                       is_midday=row.get("is_midday"), is_afternoon=row.get("is_afternoon"),
                       tf15_bb_squeeze=row.get("tf15_bb_squeeze"), traded=int(traded))
            if traded:
                pnl = pnl_map.get(ts)
                rec["pnl"] = float(pnl) if pnl is not None else None
                if pnl is not None and pnl <= 0:
                    fp_row = row.copy(); fp_row["fold"] = k; fp_row["pnl"] = pnl
                    all_fp_rows.append(fp_row)
            all_rows.append(rec)

        # false negatives: SKIP bars where the frozen ATR-barrier oracle would
        # have fired -- reuse the exact labeler already validated in the
        # label-redesign study (not recomputed logic, imported).
        from scripts.label_redesign_study import label_atr_barrier
        closes = test["close"].values.astype(np.float64)
        highs = test["high"].values.astype(np.float64)
        lows = test["low"].values.astype(np.float64)
        prev_close = np.roll(closes, 1); prev_close[0] = closes[0]
        tr_ = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
        atr10 = pd.Series(tr_, index=test.index).rolling(10).mean().values
        oracle = label_atr_barrier(test, closes, highs, lows, atr10)["labels"]
        for i, ts in enumerate(test.index):
            if sig[i] == 2 and oracle[i] != 2:
                fn_row = test.iloc[i].copy(); fn_row["fold"] = k
                all_fn_rows.append(fn_row)

    diversity_df = pd.DataFrame(all_rows)
    fp_df = pd.DataFrame(all_fp_rows)
    fn_df = pd.DataFrame(all_fn_rows)
    print(f"\nPooled: traded bars={diversity_df['traded'].sum()}  FP rows={len(fp_df)}  FN rows={len(fn_df)}")

    # =========================================================================
    # AREA 5: Ensemble diversity (XGB vs LGB)
    # =========================================================================
    traded_df = diversity_df[diversity_df["traded"] == 1].copy()
    corr_call = np.corrcoef(diversity_df["p_xgb_call"], diversity_df["p_lgb_call"])[0, 1]
    corr_put = np.corrcoef(diversity_df["p_xgb_put"], diversity_df["p_lgb_put"])[0, 1]
    agree_rate = diversity_df["agree"].mean()
    disagree_traded = traded_df[traded_df["agree"] == 0]
    disagree_pnl = disagree_traded["pnl"].dropna()
    agree_traded = traded_df[traded_df["agree"] == 1]
    agree_pnl = agree_traded["pnl"].dropna()
    diversity_result = dict(
        corr_p_call=float(corr_call), corr_p_put=float(corr_put),
        signal_agreement_rate=float(agree_rate),
        n_traded_agree=len(agree_pnl), pf_when_agree=float(agree_pnl[agree_pnl > 0].sum() / -agree_pnl[agree_pnl <= 0].sum()) if (agree_pnl <= 0).any() and agree_pnl[agree_pnl <= 0].sum() < 0 else None,
        n_traded_disagree=len(disagree_pnl), pf_when_disagree=float(disagree_pnl[disagree_pnl > 0].sum() / -disagree_pnl[disagree_pnl <= 0].sum()) if (disagree_pnl <= 0).any() and disagree_pnl[disagree_pnl <= 0].sum() < 0 else None,
    )
    print("\n=== AREA 5: Ensemble diversity ===")
    print(json.dumps(diversity_result, indent=2, default=str))

    # =========================================================================
    # AREA 6: Error clustering (rule-based, named buckets, on REAL per-row data)
    # =========================================================================
    def bucket_row(row):
        tags = []
        if row.get("is_expiry"): tags.append("expiry_day")
        if row.get("is_midday"): tags.append("lunchtime_midday")
        if row.get("vix_regime") is not None and row.get("vix_regime") == 0: tags.append("low_vol_regime")
        if row.get("vix_spike"): tags.append("vix_spike")
        if row.get("tf15_bb_squeeze"): tags.append("squeeze_context")
        if row.get("tf15_adx") is not None and row.get("tf15_adx") < 20: tags.append("range_low_adx")
        if row.get("tf15_adx") is not None and row.get("tf15_adx") > 30: tags.append("trend_high_adx")
        return tags or ["unclassified"]

    def cluster_summary(df_, name):
        if len(df_) == 0:
            return {}
        counts = {}
        for _, row in df_.iterrows():
            for t in bucket_row(row):
                counts[t] = counts.get(t, 0) + 1
        n = len(df_)
        return {t: dict(n=c, pct=round(100 * c / n, 1)) for t, c in sorted(counts.items(), key=lambda kv: -kv[1])}

    fp_clusters = cluster_summary(fp_df, "FP")
    fn_clusters = cluster_summary(fn_df.sample(min(len(fn_df), 3000), random_state=42) if len(fn_df) else fn_df, "FN")
    print("\n=== AREA 6: Error clustering ===")
    print("False positives (losing trades) by bucket:")
    print(json.dumps(fp_clusters, indent=2))
    print("False negatives (missed oracle-directional SKIPs) by bucket (3000-row sample if larger):")
    print(json.dumps(fn_clusters, indent=2))

    # =========================================================================
    # AREA 7: Regime segmentation -- achievable PF if separated
    # =========================================================================
    def regime_of(row):
        adx = row.get("tf15_adx")
        trend = "trend" if (adx is not None and adx > 25) else "range"
        vix = "high_vol" if row.get("vix_spike") else ("low_vol" if row.get("vix_regime") == 0 else "normal_vol")
        return f"{trend}_{vix}"

    traded_df["regime"] = traded_df.apply(regime_of, axis=1)
    regime_pf = {}
    for r, g in traded_df.groupby("regime"):
        pnl = g["pnl"].dropna()
        if len(pnl) < 10:
            regime_pf[r] = dict(n=len(pnl), pf=None, note="too few trades (<10) for a reliable PF")
            continue
        gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
        pf = float(gp / gl) if gl > 0 else float("inf")
        regime_pf[r] = dict(n=len(pnl), pf=pf, net=float(pnl.sum()), win_rate=float((pnl > 0).mean()))
    print("\n=== AREA 7: Regime-segmented PF (from real pooled trade data) ===")
    print(json.dumps(regime_pf, indent=2, default=str))

    overall_pnl = traded_df["pnl"].dropna()
    overall_pf = float(overall_pnl[overall_pnl > 0].sum() / -overall_pnl[overall_pnl <= 0].sum()) \
        if (overall_pnl <= 0).sum() and overall_pnl[overall_pnl <= 0].sum() < 0 else None
    print(f"\nOverall pooled PF (all folds, all regimes): {overall_pf}")

    result = dict(wall_time_s=round(time.time() - t0, 1), per_fold_metrics=per_fold_metrics,
                  ensemble_diversity=diversity_result, fp_clusters=fp_clusters, fn_clusters=fn_clusters,
                  regime_segmented_pf=regime_pf, overall_pooled_pf=overall_pf,
                  n_fp=len(fp_df), n_fn_sampled=len(fn_df) if len(fn_df) <= 3000 else 3000, n_fn_total=len(fn_df))
    with open(ROOT / "logs/roadmap_diagnostics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nDone in {result['wall_time_s']}s. Wrote logs/roadmap_diagnostics.json")


if __name__ == "__main__":
    main()
