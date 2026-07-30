"""
check_top_features_direction.py — empirical: do the top-20 +IC features
actually predict UP, and the top-20 −IC features actually predict DOWN?

Method:
  1. Load leak-clean V9 frame (same as feature_audit.py used).
  2. Read feature_audit.csv to get the top-20 +IC and top-20 −IC feature names.
  3. Z-score each, build composite bullish_score (avg of top-20 +IC z-scores)
     and bearish_score (avg of top-20 −IC z-scores).
  4. Bucket each into quintiles and measure:
        * % positive forward 15-min return
        * mean forward return
        * sample size
  5. Confusion-matrix-style result for combined bullish_score - bearish_score.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest_threshold_sweep import build_frame
FWD_BARS = 3      # 15-min forward (matches feature_audit.py)

print("Loading audit + frame...")
audit = pd.read_csv(ROOT / "logs/feature_audit.csv")
feat, _ = build_frame()
px = feat["close"]
feat["fwd_15m"] = px.shift(-FWD_BARS) / px - 1
# same-day only: shift the date column by N rows; if not equal, drop
feat["date"] = pd.to_datetime(feat.index).normalize()
date_arr = feat["date"].values
date_fwd = np.roll(date_arr, -FWD_BARS)
date_fwd[-FWD_BARS:] = np.datetime64("NaT")
feat.loc[date_arr != date_fwd, "fwd_15m"] = np.nan

# top-20 positive and top-20 negative by signed IC
top_pos = audit.sort_values("ic", ascending=False).head(20)["feature"].tolist()
top_neg = audit.sort_values("ic", ascending=True).head(20)["feature"].tolist()

# z-score each, build composite scores
def z(s):
    return (s - s.mean()) / s.std()

pos_z = pd.concat([z(feat[c]) for c in top_pos if c in feat.columns], axis=1).mean(axis=1)
neg_z = pd.concat([z(feat[c]) for c in top_neg if c in feat.columns], axis=1).mean(axis=1)

# directional score: high when bullish-features high AND bearish-features low
score = pos_z - neg_z

# slice by quintile and measure forward returns
d = pd.DataFrame({"score": score, "pos_score": pos_z, "neg_score": neg_z,
                  "fwd": feat["fwd_15m"]}).dropna()
print(f"  usable bars: {len(d):,}")
print()

def bucket_report(s, fwd, label):
    qs = pd.qcut(s, 5, labels=["Q1-low", "Q2", "Q3", "Q4", "Q5-high"], duplicates="drop")
    g = d.groupby(qs, observed=True)["fwd"].agg(["count", "mean", lambda x: (x>0).mean()*100])
    g.columns = ["n", "mean_fwd", "pct_positive"]
    print(f"=== {label} (5 quintiles) ===")
    print(f"  {'bucket':<10}{'n':>8}{'mean fwd %':>14}{'% positive':>14}")
    for idx, row in g.iterrows():
        print(f"  {str(idx):<10}{int(row['n']):>8}{row['mean_fwd']*100:>+13.3f}%{row['pct_positive']:>13.1f}%")
    print()
    return g

print("Q1=feature score lowest, Q5=feature score highest")
print()
g_pos = bucket_report(d["pos_score"], d["fwd"], "TOP-20 POSITIVE-IC composite (should predict UP when HIGH)")
g_neg = bucket_report(d["neg_score"], d["fwd"], "TOP-20 NEGATIVE-IC composite (should predict DOWN when HIGH)")
g_net = bucket_report(d["score"],   d["fwd"], "NET score = pos - neg (should predict UP when HIGH)")

# Headline directional accuracy
hi = d[d["score"] >= d["score"].quantile(0.8)]
lo = d[d["score"] <= d["score"].quantile(0.2)]
print("=" * 70)
print("  HEADLINE DIRECTIONAL ACCURACY (top 20% vs bottom 20% of net score)")
print("=" * 70)
print(f"  Top 20% of score (n={len(hi)}):   "
      f"% fwd positive = {(hi['fwd']>0).mean()*100:.1f}%   "
      f"mean fwd = {hi['fwd'].mean()*100:+.3f}%")
print(f"  Bot 20% of score (n={len(lo)}):   "
      f"% fwd positive = {(lo['fwd']>0).mean()*100:.1f}%   "
      f"mean fwd = {lo['fwd'].mean()*100:+.3f}%")
print(f"  Spread in % positive: {(hi['fwd']>0).mean()*100 - (lo['fwd']>0).mean()*100:+.1f} pp")
print(f"  Spread in mean fwd:   {(hi['fwd'].mean() - lo['fwd'].mean())*100:+.4f}%")
print()
# Friction context
print(f"  For reference: 15-min option round-trip friction ≈ 0.12%")
print(f"  Per-bar mean fwd spread captured: {(hi['fwd'].mean() - lo['fwd'].mean())*100:.4f}% — "
      f"vs 0.12% friction → ratio {(hi['fwd'].mean() - lo['fwd'].mean())*100/0.12:.2f}")
