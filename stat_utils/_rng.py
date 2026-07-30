"""Deterministic RNG helpers.

Every stochastic procedure in stat_utils uses this module so that:
  * ``seed=None``      -> non-reproducible (system entropy)
  * ``seed=int``       -> reproducible
  * ``seed=Generator`` -> caller supplies the RNG state

For parallel execution we pre-spawn independent child ``SeedSequence``
objects, one per bootstrap replicate. Chunking the replicate axis across
workers therefore does not affect output — determinism is ``(seed, B)``
invariant regardless of ``n_jobs``.
"""
from __future__ import annotations

from typing import Union

import numpy as np

SeedLike = Union[int, np.random.Generator, np.random.SeedSequence, None]


def as_generator(seed: SeedLike) -> np.random.Generator:
    """Coerce a seed-like argument to a :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed : int, np.random.Generator, np.random.SeedSequence, or None
        Anything acceptable to :func:`numpy.random.default_rng` plus
        ``Generator`` (returned as-is).

    Returns
    -------
    np.random.Generator
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_children(seed: SeedLike, n: int) -> list[np.random.SeedSequence]:
    """Return *n* independent child :class:`SeedSequence` objects.

    The i-th child produces the same stream regardless of how the caller
    later chunks the range ``[0, n)`` across workers. This is the primary
    determinism guarantee for parallel bootstrap.

    Parameters
    ----------
    seed : SeedLike
    n : int
        Number of children to spawn.

    Returns
    -------
    list of np.random.SeedSequence
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if isinstance(seed, np.random.SeedSequence):
        ss = seed
    elif isinstance(seed, np.random.Generator):
        # Extract the underlying SeedSequence if available; otherwise
        # bootstrap a fresh one from a draw. Round-trip through int for
        # broad Generator compatibility.
        ss = np.random.SeedSequence(int(seed.integers(0, 2**63 - 1)))
    else:
        ss = np.random.SeedSequence(seed)
    return list(ss.spawn(n))


def child_generator(child: np.random.SeedSequence) -> np.random.Generator:
    """Materialise a child SeedSequence into a Generator."""
    return np.random.default_rng(child)
