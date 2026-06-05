"""Tests for significance tests used by the regression gate and ranking."""

import numpy as np

from agent_eval.stats.tests import (
    benjamini_hochberg,
    bradley_terry,
    mcnemar,
    paired_bootstrap,
    wilcoxon,
)


def test_mcnemar_symmetric_not_significant():
    r = mcnemar(10, 10)
    assert r.n_discordant == 20
    assert r.pvalue > 0.5


def test_mcnemar_asymmetric_significant():
    # A large imbalance in discordant pairs => a real change.
    r = mcnemar(20, 3)
    assert r.pvalue < 0.05


def test_benjamini_hochberg_rejects_small_keeps_large():
    pvals = [0.001, 0.008, 0.02, 0.04, 0.9]
    reject, qvals = benjamini_hochberg(pvals, alpha=0.05)
    assert reject[0] and not reject[-1]
    assert qvals.shape == (5,)
    assert np.all(qvals >= 0) and np.all(qvals <= 1)


def test_paired_bootstrap_detects_positive_shift():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.1, 0.03, size=200)
    r = paired_bootstrap(diffs, alpha=0.05, n_boot=2000, rng=rng)
    assert r.mean_diff > 0
    assert r.lo > 0  # CI excludes 0
    assert r.pvalue < 0.05


def test_paired_bootstrap_no_shift_brackets_zero():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.0, 0.05, size=200)
    r = paired_bootstrap(diffs, alpha=0.05, n_boot=2000, rng=rng)
    assert r.lo < 0 < r.hi
    assert r.pvalue > 0.05


def test_wilcoxon_signed_rank():
    a = [2, 3, 4, 5, 6, 7]
    b = [1, 1, 1, 1, 1, 1]
    assert wilcoxon(a, b).pvalue < 0.05
    a2 = [1, 2, 3, 4, 5, 6]
    b2 = [2, 1, 4, 3, 6, 5]
    assert wilcoxon(a2, b2).pvalue > 0.3


def test_bradley_terry_recovers_ordering():
    # A dominates B dominates C.
    wins = np.array([[0, 9, 9], [1, 0, 9], [1, 1, 0]], dtype=float)
    r = bradley_terry(wins)
    s = r.strengths
    assert s[0] > s[1] > s[2]
    assert abs(float(np.sum(s)) - 1.0) < 1e-6  # normalized
