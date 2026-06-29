"""BaseMetric plumbing: required-field validation, error trapping, normalization."""

from agent_eval.core.contracts import Aggregation, EvalContext, MetricResult
from agent_eval.core.metric import BaseMetric, Metric


class _Graded(BaseMetric):
    name = "graded"
    requires = frozenset({"output", "expected"})

    def _compute(self, ctx):
        return MetricResult(self.name, 2.0)  # deliberately out of [0, 1]


class _Raw(BaseMetric):
    name = "raw"
    unit_interval = False

    def _compute(self, ctx):
        return MetricResult(self.name, 1234.0)


class _Boom(BaseMetric):
    name = "boom"

    def _compute(self, ctx):
        raise RuntimeError("kaboom")


def test_missing_required_field_becomes_error_not_failure():
    r = _Graded().score(EvalContext(input="q", output="a"))  # expected missing
    assert r.error is not None and "expected" in r.error


def test_empty_sequence_counts_as_missing():
    m = _Graded()
    r = m.score(EvalContext(input="q", output="", expected="a"))
    assert r.error is not None  # empty string output is missing


def test_unit_interval_clamps_score():
    r = _Graded().score(EvalContext(input="q", output="a", expected="a"))
    assert r.score == 1.0  # clamped from 2.0


def test_non_unit_interval_keeps_raw_value():
    r = _Raw().score(EvalContext(input="q"))
    assert r.score == 1234.0


def test_exceptions_are_trapped_into_error():
    r = _Boom().score(EvalContext(input="q"))
    assert r.error is not None and "kaboom" in r.error


def test_latency_is_captured():
    r = _Raw().score(EvalContext(input="q"))
    assert r.cost.latency_ms >= 0.0


def test_basemetric_satisfies_metric_protocol():
    assert isinstance(_Graded(), Metric)
    assert _Graded().aggregation is Aggregation.MEAN
