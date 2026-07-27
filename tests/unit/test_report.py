"""Reports: the per-criterion breakdown appears as text sub-rows / a JSON key only when present."""

from agent_eval.core.contracts import Aggregation
from agent_eval.core.gate import GateVerdict, MetricAggregate
from agent_eval.core.suite import SuiteResult
from agent_eval.offline import report


def _result(breakdown=None):
    agg = MetricAggregate("llm_judge", 0.7, 0.6, 0.8, 4, 0, Aggregation.MEAN, True, breakdown)
    return SuiteResult("plain", "response", 4, [agg], GateVerdict(True, {"llm_judge": True}, []))


def test_to_dict_includes_breakdown_only_when_present():
    with_bd = report.to_dict(_result({"Clarity": 0.9}))
    assert with_bd["aggregates"][0]["breakdown"] == {"Clarity": 0.9}
    assert "breakdown" not in report.to_dict(_result())["aggregates"][0]


def test_to_text_renders_criterion_sub_rows():
    text = report.to_text(_result({"Clarity": 0.9, "Tone": 0.5}))
    assert "llm_judge" in text
    assert "· Clarity" in text and "0.900" in text
    assert "· Tone" in text and "0.500" in text


def test_to_junit_is_unaffected_by_breakdown():
    assert report.to_junit([_result({"Clarity": 0.9})]) == report.to_junit([_result()])
