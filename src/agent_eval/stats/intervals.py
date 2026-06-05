"""Binomial proportion confidence intervals.

We default to Wilson (and Clopper-Pearson) rather than the normal-approximation/CLT
interval, which under-covers at small n (Bowyer et al., ICML 2025). CI helpers return
``(point, lo, hi)`` with bounds clamped to the unit interval.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(p_hat, lo, hi)``. Stays inside ``[0, 1]`` by construction.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return p, float(lo), float(hi)


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Clopper-Pearson exact interval (Beta method). Conservative; at least as wide as Wilson."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="beta")
    return p, float(lo), float(hi)


def beta_binomial_ci(
    successes: int, n: int, a: float = 1.0, b: float = 1.0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bayesian Beta-Binomial credible interval.

    Conjugate posterior is ``Beta(a + successes, b + n - successes)``; returns
    ``(posterior_mean, lo, hi)`` as the equal-tailed credible interval. Document priors.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    if a <= 0 or b <= 0:
        raise ValueError("prior parameters a, b must be positive")
    post_a = a + successes
    post_b = b + (n - successes)
    mean = post_a / (post_a + post_b)
    lo = stats.beta.ppf(alpha / 2.0, post_a, post_b)
    hi = stats.beta.ppf(1.0 - alpha / 2.0, post_a, post_b)
    return float(mean), float(lo), float(hi)


def percentile_ci(
    samples,
    q: float,
    alpha: float = 0.05,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a quantile (e.g. p95/p99 latency).

    Latency is heavy-tailed, so we bootstrap the quantile rather than assume normality.
    Returns ``(q_hat, lo, hi)``.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    rng = rng if rng is not None else np.random.default_rng()
    q_hat = float(np.quantile(arr, q))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_q = np.quantile(arr[idx], q, axis=1)
    lo = float(np.quantile(boot_q, alpha / 2.0))
    hi = float(np.quantile(boot_q, 1.0 - alpha / 2.0))
    return q_hat, lo, hi
