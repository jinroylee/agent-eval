"""Tests for the offline evaluate() runner: per-metric Wilson aggregation + gate verdict."""

from agent_eval.core.contracts import EvalContext, MetricResult, Tier
from agent_eval.core.gate import GatePolicy
from agent_eval.core.metric import BaseMetric
from agent_eval.core.suite import Suite
from agent_eval.metrics.deterministic_ import ExactMatch
from agent_eval.offline.runner import evaluate


class Graded(BaseMetric):
    name = "graded"
    tier = Tier.DETERMINISTIC

    def _compute(self, ctx: EvalContext) -> MetricResult:
        return MetricResult(self.name, float(ctx.metadata["s"]), passed=None)


def _exact_ds(pairs):
    return [EvalContext(input="q", output=o, expected=e) for o, e in pairs]


def test_binary_metric_passes_threshold_with_wilson_ci():
    data = _exact_ds([("a", "a"), ("b", "b"), ("c", "c"), ("x", "y")])  # 3/4 match
    suite = Suite("plain", "response", [ExactMatch()], GatePolicy(thresholds={"exact_match": 0.6}))
    res = evaluate(suite, data)
    agg = res.aggregates[0]
    assert agg.binary and abs(agg.value - 0.75) < 1e-9
    assert agg.ci_low < 0.75 < agg.ci_high  # Wilson interval brackets the point
    assert res.verdict.passed is True
    assert res.verdict.metric_passed["exact_match"] is True


def test_binary_metric_fails_high_threshold_and_require_pass():
    data = _exact_ds([("a", "a"), ("b", "b"), ("c", "c"), ("x", "y")])
    suite = Suite(
        "plain",
        "response",
        [ExactMatch()],
        GatePolicy(thresholds={"exact_match": 0.9}, require_pass=["exact_match"]),
    )
    res = evaluate(suite, data)
    assert res.verdict.passed is False


def test_errors_are_counted_and_excluded_from_n():
    data = [
        EvalContext(input="q", output="a", expected="a"),
        EvalContext(input="q", output="a", expected=None),  # ExactMatch requires expected
    ]
    suite = Suite("plain", "response", [ExactMatch()], GatePolicy(thresholds={"exact_match": 0.5}))
    agg = evaluate(suite, data).aggregates[0]
    assert agg.n_errors == 1 and agg.n == 1 and agg.value == 1.0


def test_continuous_metric_mean_and_ci():
    data = [EvalContext(input="q", output="o", metadata={"s": x}) for x in (0.5, 0.7, 0.9, 0.6)]
    suite = Suite("plain", "response", [Graded()], GatePolicy(thresholds={"graded": 0.5}))
    agg = evaluate(suite, data).aggregates[0]
    assert not agg.binary and abs(agg.value - 0.675) < 1e-9
    assert agg.ci_low <= agg.value <= agg.ci_high


def test_missing_require_pass_metric_fails_gate():
    data = _exact_ds([("a", "a")])
    suite = Suite(
        "plain", "response", [ExactMatch()], GatePolicy(require_pass=["never_ran"])
    )
    assert evaluate(suite, data).verdict.passed is False
