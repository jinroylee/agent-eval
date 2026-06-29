"""Confidence intervals used by the offline runner.

We default to **Wilson** for proportions rather than the normal-approximation/CLT interval, which
under-covers at small n; and a **bootstrap** for quantiles, because latency is heavy-tailed and a
normality assumption would be wrong. Each returns ``(point, lo, hi)``.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.proportion import proportion_confint


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Stays inside ``[0, 1]`` by construction."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return p, float(lo), float(hi)


def percentile_ci(
    samples,
    q: float = 0.95,
    alpha: float = 0.05,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a quantile (e.g. p95/p99 latency). Returns ``(q_hat, lo, hi)``."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    rng = rng if rng is not None else np.random.default_rng(0)
    q_hat = float(np.quantile(arr, q))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_q = np.quantile(arr[idx], q, axis=1)
    lo = float(np.quantile(boot_q, alpha / 2.0))
    hi = float(np.quantile(boot_q, 1.0 - alpha / 2.0))
    return q_hat, lo, hi
