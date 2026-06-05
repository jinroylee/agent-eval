"""Tests for PPI (judge-label correction) and the anytime-valid confidence sequence."""

import numpy as np

from agent_eval.stats.ppi import ppi_interval
from agent_eval.stats.sequential import anytime_valid_sequence


def _gold_only_width(y, alpha=0.05):
    y = np.asarray(y, dtype=float)
    se = np.std(y, ddof=1) / np.sqrt(y.size)
    return 2 * 1.959963984540054 * se


def test_ppi_degrades_to_gold_only_under_noise():
    rng = np.random.default_rng(0)
    n, m = 60, 600
    y = rng.binomial(1, 0.6, n).astype(float)
    judge = rng.random(m)  # pure noise, uncorrelated with y
    judge[:n] = rng.random(n)
    gold_idx = np.arange(n)
    theta, lo, hi = ppi_interval(judge, y, gold_idx, alpha=0.05)
    # With an uninformative judge, PPI must fall back to the gold mean + gold-only width.
    assert abs(theta - y.mean()) < 0.05
    assert abs((hi - lo) - _gold_only_width(y)) < 0.02


def test_ppi_informative_judge_narrows_interval():
    rng = np.random.default_rng(2)
    n, m = 80, 800
    y = rng.binomial(1, 0.6, n).astype(float)
    judge = rng.normal(0.6, 0.2, m)
    judge[:n] = y + rng.normal(0, 0.02, n)  # near-perfect judge on the labeled set
    gold_idx = np.arange(n)
    _, lo, hi = ppi_interval(judge, y, gold_idx, alpha=0.05)
    assert (hi - lo) < 0.7 * _gold_only_width(y)  # judge information tightens the interval


def test_confidence_sequence_radius_shrinks():
    rng = np.random.default_rng(0)
    stream = rng.binomial(1, 0.5, 300).astype(float)
    cs = anytime_valid_sequence(stream, alpha=0.05)
    assert cs.radius[9] > cs.radius[-1]  # tightens as evidence accrues
    assert np.all(cs.lo <= cs.hi)


def test_confidence_sequence_type_one_error_under_peeking():
    """Across many null streams, the true mean should leave the band at SOME time
    no more than alpha of the time — the defining anytime-valid guarantee (peeking-safe)."""
    alpha = 0.05
    rng = np.random.default_rng(7)
    for p0 in (0.5, 0.3):
        excluded = 0
        streams = 1500
        for _ in range(streams):
            stream = rng.binomial(1, p0, 150).astype(float)
            cs = anytime_valid_sequence(stream, alpha=alpha)
            if np.any((p0 < cs.lo) | (p0 > cs.hi)):
                excluded += 1
        rate = excluded / streams
        assert rate <= alpha, f"anytime-valid CS exceeded type-I at p0={p0}: {rate:.3f}"
