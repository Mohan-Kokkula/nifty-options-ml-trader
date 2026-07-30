"""
validate_confidence_edge.py — no-look-ahead validation of the confidence edge
=============================================================================
Built 2026-06-16. The post-hoc test (PF 1.25 at top-25%) chose the percentile
cutoff using the WHOLE sample. This re-tests it causally:

  * Input: the 618 walk-forward OOS trades (each already from a fold whose
    model saw only prior data) tagged with entry directional confidence.
  * Selection rule (CAUSAL): for each trade, the keep-threshold is the
    EXPANDING quantile of confidence over ONLY the trades that closed before
    it (min 40-trade warm-up). A trade is kept iff its confidence >= that
    trailing threshold. No future information enters the threshold.
  * Reports PF / Sharpe / Net / count / MaxDD per fold (quarter) and aggregate
    for top 75/50/25/10%.
  * Confidence-decile calibration + monotonicity test.
  * Bootstrap significance + Bonferroni correction for testing 4 thresholds.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

MIN_HISTORY = 40          # warm-up trades before selection activates
PCTS = [75, 50, 25, 10]


def pf(x):
    g = x[x > 0].sum(); l = -x[x <= 0].sum()
    return g / l if l > 0 else float("inf")


def maxdd(net_series_time_sorted):
    eq = np.cumsum(net_series_time_sorted)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min()) if len(eq) else 0.0  # most negative


def agg_metrics(t):
    if len(t) == 0:
        return dict(n=0, pf=np.nan, sharpe=np.nan, net=0.0, dd=0.0, wr=np.nan)
    net = t.sort_values("entry_time")["net_option"].values
    n = len(net); std = net.std(ddof=1) if n > 1 else 0.0
    span = max((pd.to_datetime(t["entry_time"]).max()
                - pd.to_datetime(t["entry_time"]).min()).days, 1)
    tpy = n / (span / 365.25)
    sharpe = (net.mean() / std * np.sqrt(tpy)) if std > 0 else 0.0
    return dict(n=n, pf=pf(net), sharpe=sharpe, net=net.sum(),
                dd=maxdd(net), wr=(net > 0).mean() * 100)


def causal_keep(df, pct):
    """Expanding-quantile trailing threshold; keep trade iff conf>=threshold."""
    conf = df["conf"].values
    keep = np.zeros(len(df), bool)
    q = 1 - pct / 100.0
    for i in range(len(df)):
        if i < MIN_HISTORY:
            continue                      # warm-up — not evaluated
        thr = np.quantile(conf[:i], q)    # only prior trades
        keep[i] = conf[i] >= thr
    return keep


def main():
    d = pd.read_csv("logs/exec_baseline_trades.csv").dropna(subset=["conf"])
    d = d.sort_values("entry_time").reset_index(drop=True)
    d["q"] = pd.to_datetime(d["entry_time"]).dt.to_period("Q")
    print(f"OOS trades: {len(d)} | conf {d.conf.min():.3f}-{d.conf.max():.3f} | "
          f"warm-up {MIN_HISTORY}\n")

    # ---------- 1-5: causal selection per threshold ----------
    print("=" * 92)
    print("  NO-LOOK-AHEAD CAUSAL SELECTION (expanding-quantile trailing threshold)")
    print("=" * 92)
    print(f"  {'keep':<10}{'trades':>7}{'PF':>7}{'Sharpe':>8}{'WinRt':>7}"
          f"{'Net':>13}{'MaxDD':>13}  quarters+")
    print("  " + "-" * 88)
    kept_sets = {}
    base = agg_metrics(d)
    print(f"  {'ALL':<10}{base['n']:>7}{base['pf']:>7.2f}{base['sharpe']:>8.2f}"
          f"{base['wr']:>6.0f}%{('Rs.'+format(base['net'],'+,.0f')):>13}"
          f"{('Rs.'+format(base['dd'],',.0f')):>13}")
    for p in PCTS:
        keep = causal_keep(d, p)
        sub = d[keep]
        kept_sets[p] = sub
        m = agg_metrics(sub)
        pq = sub.groupby("q").net_option.sum()
        qpos = f"{(pq>0).sum()}/{len(pq)}"
        print(f"  top {p:<5}%{m['n']:>7}{m['pf']:>7.2f}{m['sharpe']:>8.2f}"
              f"{m['wr']:>6.0f}%{('Rs.'+format(m['net'],'+,.0f')):>13}"
              f"{('Rs.'+format(m['dd'],',.0f')):>13}  {qpos}")

    # per-fold(quarter) for the headline top-25%
    print("\n  Per-quarter (causal top-25%):")
    s25 = kept_sets[25]
    for q, g in s25.groupby("q"):
        m = agg_metrics(g)
        print(f"    {q}: n={m['n']:>2}  PF {m['pf']:>4.2f}  "
              f"net Rs.{m['net']:+,.0f}  dd Rs.{m['dd']:,.0f}")

    # ---------- 6-7: decile calibration + monotonicity ----------
    print("\n" + "=" * 92)
    print("  CONFIDENCE-DECILE CALIBRATION (all 618 OOS trades)")
    print("=" * 92)
    d["dec"] = pd.qcut(d["conf"], 10, labels=False, duplicates="drop")
    print(f"  {'decile':<8}{'conf_lo':>9}{'conf_hi':>9}{'n':>5}{'win%':>7}{'avg_net':>11}{'PF':>7}")
    decavg = []
    for dec, g in d.groupby("dec"):
        m = agg_metrics(g)
        decavg.append((dec, g.net_option.mean()))
        print(f"  {int(dec):<8}{g.conf.min():>9.3f}{g.conf.max():>9.3f}"
              f"{len(g):>5}{(g.net_option>0).mean()*100:>6.0f}%"
              f"{('Rs.'+format(g.net_option.mean(),'+,.0f')):>11}{pf(g.net_option.values):>7.2f}")
    decs = np.array([x[0] for x in decavg]); avgs = np.array([x[1] for x in decavg])
    rho, prho = stats.spearmanr(decs, avgs)
    print(f"\n  Monotonicity (decile rank vs avg net): Spearman={rho:+.2f} p={prho:.3f}")

    # ---------- 8: bootstrap on causal top-25% & top-10% ----------
    print("\n" + "=" * 92)
    print("  BOOTSTRAP SIGNIFICANCE (causal-selected subsets, 5000 resamples)")
    print("=" * 92)
    rng = np.random.default_rng(7)
    results = {}
    for p in PCTS:
        net = kept_sets[p]["net_option"].values
        if len(net) < 10:
            continue
        bs_mean = np.array([rng.choice(net, len(net), replace=True).mean() for _ in range(5000)])
        p_le0 = (bs_mean <= 0).mean()          # one-sided: P(mean<=0)
        lo, hi = np.percentile(bs_mean, [2.5, 97.5])
        results[p] = p_le0
        print(f"  top {p:>2}%: mean Rs.{net.mean():+,.0f}/trade  "
              f"CI[Rs.{lo:+,.0f}, Rs.{hi:+,.0f}]  P(mean<=0)={p_le0:.3f}")

    # ---------- 9: multiple-testing correction ----------
    print("\n" + "=" * 92)
    print("  MULTIPLE-TESTING ADJUSTMENT (4 thresholds tried)")
    print("=" * 92)
    if results:
        best_p = min(results, key=results.get)
        raw = results[best_p]
        bonf = min(1.0, raw * len(results))
        print(f"  Best threshold: top {best_p}%  raw p={raw:.3f}  "
              f"Bonferroni x{len(results)} -> p_adj={bonf:.3f}")
        print(f"  Significant at 0.05 after correction? "
              f"{'YES' if bonf < 0.05 else 'NO'}")


if __name__ == "__main__":
    main()
