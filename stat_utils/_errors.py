"""Exception hierarchy for stat_utils.

All exceptions raised by public API derive from :class:`StatUtilsError`
so downstream code can catch a single base class.
"""
from __future__ import annotations


class StatUtilsError(Exception):
    """Base class for all stat_utils exceptions."""


class InvalidInputError(StatUtilsError, ValueError):
    """Raised when a caller supplies structurally invalid input.

    Examples: wrong dtype, wrong shape, mismatched lengths, negative
    counts where positives are required, empty required arrays.
    """


class InsufficientDataError(StatUtilsError, ValueError):
    """Raised when input is structurally valid but too small to produce
    a meaningful statistic (e.g. bootstrap of an empty stream, DM with
    fewer than ``lag + 2`` observations).
    """
