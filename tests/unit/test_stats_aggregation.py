"""Tests for clustered SEs and eval-domain estimators (pass^k, Cost-of-Pass)."""

import numpy as np

from agent_eval.stats.agents import cost_of_pass, pass_hat_k
from agent_eval.stats.clustered import clustered_se


def _naive_se(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def test_clustered_se_reduces_to_iid_for_singleton_clusters():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, size=60)
    clusters = list(range(60))  # each its own cluster
    _, se = clustered_se(values, clusters)
    assert abs(se - _naive_se(values)) < 1e-9


def test_clustered_se_inflates_under_within_cluster_correlation():
    # 4 clusters of 25; within a cluster every value is identical => strong correlation.
    values, clusters = [], []
    for g in range(4):
        values += [float(g)] * 25
        clusters += [g] * 25
    mean, se = clustered_se(values, clusters)
    assert abs(mean - 1.5) < 1e-9
    assert se > 2 * _naive_se(values)  # clustering inflates the SE well beyond the naive one


def test_pass_hat_k_all_success_is_one():
    r = pass_hat_k([5, 5, 5], trials_per_task=5, k=3)
    assert abs(r.estimate - 1.0) < 1e-9


def test_pass_hat_k_partial():
    r = pass_hat_k([5, 5, 0, 5], trials_per_task=5, k=3)
    assert abs(r.estimate - 0.75) < 1e-9
    assert 0.0 <= r.lo <= r.estimate <= r.hi <= 1.0


def test_pass_hat_k_decreases_with_k():
    one = pass_hat_k([3, 4], trials_per_task=5, k=1).estimate
    three = pass_hat_k([3, 4], trials_per_task=5, k=3).estimate
    assert one > three  # all-k-pass is harder than single-pass


def test_cost_of_pass_is_cost_over_success_rate():
    # mean cost 1.0, success rate 0.5 -> $2 per correct answer.
    cop = cost_of_pass([1, 0, 1, 0], [1.0, 1.0, 1.0, 1.0])
    assert abs(cop - 2.0) < 1e-9


def test_cost_of_pass_no_success_is_inf():
    assert cost_of_pass([0, 0], [1.0, 2.0]) == float("inf")
