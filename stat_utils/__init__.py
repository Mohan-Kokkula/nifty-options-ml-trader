"""stat_utils — Publication-grade statistical utilities for OpenClaw v2.

Public API (stable). Downstream phase scripts import from this module
only; direct submodule imports are discouraged.

Run the test suite:
    pytest stat_utils/tests -v
"""
from __future__ import annotations

from ._errors import (
    InsufficientDataError,
    InvalidInputError,
    StatUtilsError,
)
from ._types import (
    BootstrapCI,
    DMResult,
    DSRResult,
    KSResult,
    KendallResult,
    LeveneResult,
    MHTResult,
    PBOResult,
    PermutationResult,
    SPAResult,
    WhiteRCResult,
)
from .block_length import BlockLengthSelector, cbrt_block_length
from .bootstrap import block_bootstrap_ci, paired_block_bootstrap_ci
from .dm import diebold_mariano
from .dsr import deflated_sharpe
from .hac import newey_west_variance
from .helpers import kendall_tau, ks_2samp, levene, permutation_test
from .metrics import (
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    win_rate,
)
from .mht import benjamini_hochberg, holm_bonferroni
from .pbo import probability_backtest_overfitting
from .spa import hansen_spa
from .white_rc import white_reality_check

__all__ = [
    # Errors
    "StatUtilsError", "InvalidInputError", "InsufficientDataError",
    # Types
    "BootstrapCI", "DMResult", "WhiteRCResult", "SPAResult", "DSRResult",
    "PBOResult", "MHTResult", "KendallResult", "LeveneResult", "KSResult",
    "PermutationResult",
    # Metrics
    "profit_factor", "sharpe", "sortino", "max_drawdown", "win_rate",
    "expectancy",
    # Bootstrap
    "block_bootstrap_ci", "paired_block_bootstrap_ci",
    # Block-length
    "BlockLengthSelector", "cbrt_block_length",
    # Tests
    "diebold_mariano", "white_reality_check", "hansen_spa",
    "deflated_sharpe", "probability_backtest_overfitting",
    # MHT
    "holm_bonferroni", "benjamini_hochberg",
    # Helpers
    "kendall_tau", "levene", "ks_2samp", "permutation_test",
    # Low-level
    "newey_west_variance",
]

__version__ = "1.0.0"
