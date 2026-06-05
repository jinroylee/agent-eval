"""Tests for the core contracts and the BaseMetric behavior all metrics inherit."""

import asyncio

import pytest

from agent_eval.core.contracts import Cost, CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric


class NonEmptyOutput(BaseMetric):
    name = "nonempty_output"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output"})
    cost_class = CostClass.FREE

    def _compute(self, ctx: EvalContext) -> MetricResult:
        s = 1.0 if ctx.output else 0.0
        return MetricResult(self.name, s, passed=bool(s))


class OverflowScore(BaseMetric):
    name = "overflow"
    tier = Tier.DETERMINISTIC

    def _compute(self, ctx: EvalContext) -> MetricResult:
        return MetricResult(self.name, 1.5, None)


class Boom(BaseMetric):
    name = "boom"
    tier = Tier.DETERMINISTIC

    def _compute(self, ctx: EvalContext) -> MetricResult:
        raise RuntimeError("kaboom")


def test_contracts_are_frozen():
    c = Cost(tokens=1)
    with pytest.raises(AttributeError):
        c.tokens = 2  # type: ignore[misc]


def test_evalcontext_defaults():
    ctx = EvalContext(input="q", output="a")
    assert ctx.expected is None
    assert ctx.retrieved_context == ()
    assert ctx.trajectory == ()
    assert ctx.metadata == {}


def test_metric_scores_and_captures_latency():
    r = NonEmptyOutput().score(EvalContext(input="q", output="hello"))
    assert r.score == 1.0 and r.passed is True and r.error is None
    assert r.cost.latency_ms >= 0.0


def test_missing_requirement_returns_error_not_exception():
    r = NonEmptyOutput().score(EvalContext(input="q", output=None))
    assert r.error is not None and "output" in r.error
    assert r.score == 0.0


def test_metric_traps_exceptions():
    r = Boom().score(EvalContext(input="q", output="a"))
    assert r.error is not None and "kaboom" in r.error


def test_score_is_clamped_to_unit_interval():
    assert OverflowScore().score(EvalContext(input="q", output="a")).score == 1.0


def test_async_ascore_bridges_to_sync_by_default():
    r = asyncio.run(NonEmptyOutput().ascore(EvalContext(input="q", output="hi")))
    assert r.score == 1.0
