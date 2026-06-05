"""Tests for inter-rater agreement (Cohen's kappa), used to certify judge-human agreement."""

from agent_eval.stats.agreement import cohen_kappa


def test_perfect_agreement():
    assert cohen_kappa([0, 1, 0, 1, 1], [0, 1, 0, 1, 1]) == 1.0


def test_anticorrelated_is_negative():
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]) < 0


def test_partial_agreement_between_zero_and_one():
    k = cohen_kappa([0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0])
    assert 0.0 < k < 1.0


def test_chance_agreement_near_zero():
    # judge says "1" always; truth is half 1 half 0 -> kappa == 0 (no skill).
    assert abs(cohen_kappa([1, 1, 1, 1], [1, 1, 0, 0])) < 1e-9
