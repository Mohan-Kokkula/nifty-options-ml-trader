"""Regime classification for the existing (frozen) Phase-5 fold boundaries.

For each fold, take the NIFTY close on the first and last bar of the test
window, compute the total return and its annualised equivalent. Classify:

  * bull if annualised return >= +8%
  * bear if annualised return <= -8%
  * sideways otherwise

Fold boundaries are NOT changed — this only labels them for reporting.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ._base import BEAR_MAX_ANNUALISED, BULL_MIN_ANNUALISED


def _annualised(total_return: float, days: float) -> float:
    if days <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (365.0 / days) - 1.0)


def classify_fold(test_df: pd.DataFrame) -> dict:
    if "close" not in test_df.columns or len(test_df) < 2:
        return {
            "close_start": float("nan"),
            "close_end": float("nan"),
            "total_return": float("nan"),
            "annualised_return": float("nan"),
            "days": 0,
            "regime": "UNKNOWN",
        }
    c0 = float(test_df["close"].iloc[0])
    c1 = float(test_df["close"].iloc[-1])
    if c0 <= 0:
        return {
            "close_start": c0, "close_end": c1,
            "total_return": float("nan"),
            "annualised_return": float("nan"),
            "days": 0, "regime": "UNKNOWN",
        }
    total = (c1 - c0) / c0
    days = (test_df.index[-1] - test_df.index[0]).days or 1
    ann = _annualised(total, days)
    if ann >= BULL_MIN_ANNUALISED:
        regime = "BULL"
    elif ann <= BEAR_MAX_ANNUALISED:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"
    return {
        "close_start": c0,
        "close_end": c1,
        "total_return": total,
        "annualised_return": ann,
        "days": int(days),
        "regime": regime,
    }


def classify_folds(fold_data: Mapping[int, tuple]) -> dict:
    """Return {fold: classification} for every fold."""
    out = {}
    for fold, (probs, test_df, iv, exp) in fold_data.items():
        out[int(fold)] = classify_fold(test_df)
    # summary
    regimes = [v["regime"] for v in out.values()]
    counts = {r: regimes.count(r) for r in set(regimes)}
    return {"per_fold": out, "counts": counts}
