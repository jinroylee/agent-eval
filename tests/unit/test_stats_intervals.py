"""Tests for binomial confidence intervals.

The load-bearing property (Bowyer et al., ICML 2025): the normal-approximation/CLT
interval *under-covers* at small n, so we default to Wilson. These tests assert (a) known
values and (b) that Wilson's empirical coverage stays near nominal at small n — the P0 gate.
"""

import numpy as np

from agent_eval.stats.intervals import (
    beta_binomial_ci,
    clopper_pearson,
    percentile_ci,
    wilson_interval,
)


def test_wilson_known_value():
    # 8/10 successes, 95% Wilson interval is approx [0.490, 0.943] (point 0.8).
    p, lo, hi = wilson_interval(8, 10, alpha=0.05)
    assert p == 0.8
    assert lo == 0.0 or lo > 0.0  # within unit interval
    assert 0.47 < lo < 0.52
    assert 0.92 < hi < 0.96
    assert 0.0 <= lo < p < hi <= 1.0


def test_wilson_endpoints_stay_in_unit_interval():
    # All-success and all-fail must not produce bounds outside [0, 1].
    _, lo0, hi0 = wilson_interval(0, 7)
    _, lo1, hi1 = wilson_interval(7, 7)
    assert lo0 == 0.0 or lo0 >= 0.0
    assert 0.0 <= lo0 <= hi0 <= 1.0
    assert 0.0 <= lo1 <= hi1 <= 1.0


def test_wilson_coverage_at_small_n_is_near_nominal():
    """Seeded Monte-Carlo coverage check: the 95% Wilson interval should cover the true p
    at least ~92% of the time even at small n — where the CLT interval notably under-covers."""
    rng = np.random.default_rng(0)
    trials = 4000
    for n in (5, 10, 20):
        # Only n+1 distinct intervals exist; precompute once instead of per-draw.
        bounds = {k: wilson_interval(k, n, alpha=0.05)[1:] for k in range(n + 1)}
        for true_p in (0.2, 0.5, 0.8):
            draws = rng.binomial(n, true_p, size=trials)
            covered = sum(1 for k in draws if bounds[int(k)][0] <= true_p <= bounds[int(k)][1])
            coverage = covered / trials
            assert coverage >= 0.92, f"Wilson under-covered at n={n}, p={true_p}: {coverage:.3f}"


def test_clopper_pearson_known_value_and_conservatism():
    # 8/10, 95% Clopper-Pearson (exact) is approx [0.444, 0.975]; wider than Wilson.
    p, lo, hi = clopper_pearson(8, 10, alpha=0.05)
    assert p == 0.8
    assert 0.42 < lo < 0.47
    assert 0.96 < hi < 0.99
    _, w_lo, w_hi = wilson_interval(8, 10)
    assert (hi - lo) >= (w_hi - w_lo)  # exact interval is at least as wide as Wilson


def test_beta_binomial_uniform_prior_posterior_mean():
    # Uniform Beta(1,1) prior, 8/10 -> posterior Beta(9,3), mean = 9/12 = 0.75.
    mean, lo, hi = beta_binomial_ci(8, 10, a=1.0, b=1.0, alpha=0.05)
    assert abs(mean - 0.75) < 1e-9
    assert 0.0 <= lo < mean < hi <= 1.0


def test_percentile_ci_brackets_true_quantile():
    rng = np.random.default_rng(1)
    samples = np.arange(1, 101, dtype=float)  # 1..100
    qhat, lo, hi = percentile_ci(samples, q=0.95, alpha=0.05, n_boot=2000, rng=rng)
    assert 93.0 <= qhat <= 97.0
    assert lo < qhat < hi
    assert lo >= 1.0 and hi <= 100.0
