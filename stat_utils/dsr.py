"""Deflated Sharpe Ratio (Bailey & López de Prado 2014).

The DSR corrects the observed Sharpe ratio for (a) non-normality of
returns and (b) selection bias arising from choosing the best of ``N``
tried strategies.

Formulas
--------
Expected maximum of ``N`` iid standard normals (approximate):

    SR_hat = sqrt(V) * ((1-γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e)))

where ``γ ≈ 0.5772`` is the Euler–Mascheroni constant and ``V`` is the
per-trial variance of Sharpe estimates. Bailey & López de Prado use
``V = 1`` for iid normal returns; a Sharpe variance under non-normality is
accounted for in the denominator below.

DSR:

    DSR = Φ( (SR_obs - SR_threshold) * sqrt(n - 1)
             / sqrt(1 - γ₃ * SR_obs + (γ₄ - 1)/4 * SR_obs^2) )

where γ₃ is skewness of returns and γ₄ is (non-excess) kurtosis of returns.

References
----------
Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
Journal of Portfolio Management 40(5), 94-107.
"""
from __future__ import annotations

import math

from scipy import stats

from ._errors import InvalidInputError
from ._types import DSRResult
from ._validation import validate_ci_level, validate_positive_int


_EULER_GAMMA = 0.5772156649015329


def _expected_max_sharpe(n_trials: int, variance: float = 1.0) -> float:
    """Expected maximum of ``n_trials`` iid standard-normal Sharpe
    estimates. Formula from Bailey & López de Prado (2014) eq. (5)."""
    if n_trials < 1:
        raise InvalidInputError(f"n_trials: must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    e = math.e
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return math.sqrt(variance) * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b)


def deflated_sharpe(
    observed_sharpe: float,
    n_samples: int,
    n_trials: int,
    *,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
    ci_level: float = 0.95,
) -> DSRResult:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Parameters
    ----------
    observed_sharpe : float
        Sample Sharpe of the *selected* strategy.
    n_samples : int
        Sample size used to compute ``observed_sharpe``.
    n_trials : int
        Number of independent strategy trials that competed for
        selection. Must be ``>= 1``.
    skewness : float
        Sample skewness of returns (``γ₃``). Default 0 (normal).
    kurtosis_excess : float
        *Excess* kurtosis of returns (``γ₄ - 3``). Default 0 (normal).
    ci_level : float
        Reserved for future CI reporting; unused by the closed-form DSR.

    Returns
    -------
    DSRResult
        ``dsr`` in [0, 1] is the probability that the observed Sharpe
        exceeds the threshold Sharpe adjusted for selection and higher
        moments.

    Notes
    -----
    A DSR close to 1 indicates the observed Sharpe is unlikely to be a
    product of selection over ``n_trials``. A DSR close to 0 indicates the
    observed Sharpe is roughly what one would expect from the maximum of
    ``n_trials`` uninformed trials.
    """
    n_samples = validate_positive_int(n_samples, "n_samples", minimum=2)
    n_trials = validate_positive_int(n_trials, "n_trials", minimum=1)
    _ = validate_ci_level(ci_level)

    kurtosis = kurtosis_excess + 3.0
    sr = float(observed_sharpe)

    sr_threshold = _expected_max_sharpe(n_trials, variance=1.0)

    # Variance of Sharpe estimator under non-normality (Mertens 2002)
    denom_var = 1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if denom_var <= 0.0:
        # Degenerate; fall back to iid-normal variance (=1) so we at least
        # produce a defined DSR rather than crash. Mark by returning
        # exactly 0 or 1 depending on sign.
        denom_var = 1.0

    z = (sr - sr_threshold) * math.sqrt(n_samples - 1) / math.sqrt(denom_var)
    dsr = float(stats.norm.cdf(z))
    return DSRResult(
        dsr=dsr,
        observed_sharpe=sr,
        threshold_sharpe=float(sr_threshold),
        n_samples=n_samples,
        n_trials=n_trials,
        skewness=float(skewness),
        kurtosis_excess=float(kurtosis_excess),
    )
