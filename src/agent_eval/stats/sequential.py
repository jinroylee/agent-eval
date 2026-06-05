"""Anytime-valid confidence sequence for online drift monitoring.

A confidence sequence is valid simultaneously at all times, so you can peek after every trace
(sampled live traffic) and trigger drift alerts / auto-rollback without inflating the false-alarm
rate (the "peeking problem"). This is a two-sided sub-Gaussian normal-mixture sequence
(Robbins; Howard, Ramdas, McAuliffe, Sekhon, 2021).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ConfidenceSequence(NamedTuple):
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    radius: np.ndarray


def anytime_valid_sequence(
    stream,
    alpha: float = 0.05,
    sigma2: float = 0.25,
    rho: float | None = None,
) -> ConfidenceSequence:
    """Running mean with a time-uniform ``(1 - alpha)`` band for bounded ``[0, 1]`` observations.

    ``sigma2`` is the per-step sub-Gaussian variance proxy (1/4 for [0, 1] data). ``rho`` tunes
    where the boundary is tightest. The boundary controls two-sided error: the true mean leaves
    the band at *some* time with probability at most ``alpha``.
    """
    x = np.asarray(stream, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("stream must be non-empty")
    t = np.arange(1, n + 1)
    mean = np.cumsum(x) / t
    if rho is None:
        rho = sigma2
    v = t * sigma2
    # Two-sided normal-mixture boundary on the cumulative sum: use alpha/2 per side.
    boundary_s = np.sqrt((v + rho) * np.log((v + rho) / (rho * (alpha / 2.0) ** 2)))
    radius = boundary_s / t
    lo = np.clip(mean - radius, 0.0, 1.0)
    hi = np.clip(mean + radius, 0.0, 1.0)
    return ConfidenceSequence(mean, lo, hi, radius)
