"""Tests for the paired regression gate (block only on a significant, large-enough drop)."""

from agent_eval.offline.regression import bootstrap_regression, mcnemar_regression


def test_mcnemar_detects_significant_drop():
    base = [True] * 20
    cand = [True] * 12 + [False] * 8  # 8 regressions, 0 improvements
    r = mcnemar_regression(base, cand, metric="ex")
    assert r.candidate_rate < r.baseline_rate
    assert r.significant_drop is True


def test_mcnemar_no_drop_when_identical():
    base = [True] * 15 + [False] * 5
    r = mcnemar_regression(base, list(base), metric="ex")
    assert r.significant_drop is False


def test_mcnemar_noise_not_flagged():
    base = [True] * 18 + [False] * 2
    cand = [True] * 17 + [False] * 3  # one discordant pair => not significant
    assert mcnemar_regression(base, cand, metric="ex").significant_drop is False


def test_min_effect_blocks_small_but_significant_drop():
    base = [True] * 20
    cand = [True] * 12 + [False] * 8  # delta -0.4, p significant
    r = mcnemar_regression(base, cand, metric="ex", min_effect=0.5)
    assert r.significant_drop is False  # magnitude below the practical-significance floor


def test_bootstrap_regression_on_continuous_scores():
    import numpy as np

    rng = np.random.default_rng(0)
    base = list(rng.normal(0.8, 0.03, 100))
    cand = list(np.array(base) - 0.1)  # uniform 0.1 drop
    r = bootstrap_regression(base, cand, metric="geval", rng=rng)
    assert r.significant_drop is True
    same = bootstrap_regression(base, list(base), metric="geval", rng=rng)
    assert same.significant_drop is False
