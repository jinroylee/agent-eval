"""Offline runner: per-aggregation rollups (RATE/MEAN/P95), CIs, and the direction-aware gate."""

import pytest

from agent_eval.core.contracts import Aggregation, EvalContext, MetricResult
from agent_eval.core.gate import GatePolicy
from agent_eval.core.metric import BaseMetric
from agent_eval.core.suite import Suite
from agent_eval.offline.runner import evaluate


class Binary(BaseMetric):
    name = "binary"
    aggregation = Aggregation.RATE
    requires = frozenset({"output", "expected"})

    def _compute(self, ctx):
        ok = ctx.output == ctx.expected
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


class Graded(BaseMetric):
    name = "graded"

    def _compute(self, ctx):
        return MetricResult(self.name, float(ctx.metadata["s"]))


class Latency(BaseMetric):
    name = "latency"
    aggregation = Aggregation.P95
    higher_is_better = False
    unit_interval = False

    def _compute(self, ctx):
        return MetricResult(self.name, float(ctx.metadata["ms"]))


def _ds(outs_exps):
    return [EvalContext(input="q", output=o, expected=e) for o, e in outs_exps]


def test_rate_uses_wilson_and_passes_threshold():
    data = _ds([("a", "a"), ("b", "b"), ("c", "c"), ("x", "y")])  # 3/4
    res = evaluate(Suite("p", "c", [Binary()], GatePolicy({"binary": 0.6}, ["binary"])), data)
    agg = res.aggregates[0]
    assert agg.aggregation is Aggregation.RATE and abs(agg.value - 0.75) < 1e-9
    assert agg.ci_low < 0.75 < agg.ci_high
    assert res.verdict.passed


def test_mean_with_clustered_ci():
    data = [EvalContext(input="q", metadata={"s": x}) for x in (0.5, 0.7, 0.9, 0.6)]
    agg = evaluate(Suite("p", "c", [Graded()], GatePolicy({"graded": 0.5})), data).aggregates[0]
    assert abs(agg.value - 0.675) < 1e-9 and agg.ci_low <= agg.value <= agg.ci_high


def test_p95_lower_is_better_gate():
    data = [EvalContext(input="q", metadata={"ms": v}) for v in (100, 120, 130, 5000)]
    # p95 ~ near the 5000 tail -> must be <= 2000 to pass, so this FAILS
    res = evaluate(Suite("p", "c", [Latency()], GatePolicy({"latency": 2000.0}, ["latency"])), data)
    assert res.aggregates[0].aggregation is Aggregation.P95
    assert not res.verdict.passed
    # ... and passes with a generous budget
    ok = evaluate(Suite("p", "c", [Latency()], GatePolicy({"latency": 9000.0})), data)
    assert ok.verdict.passed


def test_errors_excluded_from_denominator():
    data = [EvalContext(input="q", output="a", expected="a"), EvalContext(input="q", output="a")]
    agg = evaluate(Suite("p", "c", [Binary()], GatePolicy({"binary": 0.5})), data).aggregates[0]
    assert agg.n == 1 and agg.n_errors == 1 and agg.value == 1.0


def test_require_pass_missing_metric_fails():
    data = _ds([("a", "a")])
    res = evaluate(Suite("p", "c", [Binary()], GatePolicy(require_pass=["nope"])), data)
    assert not res.verdict.passed


class CriteriaJudge(BaseMetric):
    name = "crit"

    def _compute(self, ctx):
        crit = ctx.metadata["crit"]
        return MetricResult(self.name, sum(crit.values()) / len(crit), detail={"criteria": crit})


def test_mean_aggregate_carries_criteria_breakdown():
    data = [
        EvalContext(input="q", metadata={"crit": {"Clarity": 0.8, "Tone": 0.6}}),
        EvalContext(input="q", metadata={"crit": {"Clarity": 0.4, "Tone": 1.0}}),
    ]
    agg = evaluate(Suite("p", "c", [CriteriaJudge()], GatePolicy()), data).aggregates[0]
    assert agg.breakdown == {"Clarity": pytest.approx(0.6), "Tone": pytest.approx(0.8)}


def test_breakdown_averages_only_items_carrying_each_criterion():
    data = [
        EvalContext(input="q", metadata={"crit": {"Clarity": 1.0}}),
        EvalContext(input="q", metadata={"crit": {"Clarity": 0.0, "Tone": 0.5}}),
    ]
    agg = evaluate(Suite("p", "c", [CriteriaJudge()], GatePolicy()), data).aggregates[0]
    assert agg.breakdown == {"Clarity": pytest.approx(0.5), "Tone": pytest.approx(0.5)}


def test_breakdown_absent_without_criteria_detail():
    data = [EvalContext(input="q", metadata={"s": 0.5})]
    agg = evaluate(Suite("p", "c", [Graded()], GatePolicy()), data).aggregates[0]
    assert agg.breakdown is None
