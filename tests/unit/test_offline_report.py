"""Tests for offline reporting: JSON, JUnit XML, and human-readable text."""

import json
import xml.etree.ElementTree as ET

from agent_eval.core.contracts import EvalContext
from agent_eval.core.gate import GatePolicy
from agent_eval.core.suite import Suite
from agent_eval.metrics.deterministic_ import ExactMatch
from agent_eval.offline import report as rep
from agent_eval.offline.runner import evaluate


def _failing_result():
    data = [EvalContext("q", "a", "a"), EvalContext("q", "b", "x")]  # 1/2 match
    suite = Suite(
        "plain",
        "response",
        [ExactMatch()],
        GatePolicy(thresholds={"exact_match": 0.9}, require_pass=["exact_match"]),
    )
    return evaluate(suite, data)


def test_json_report_has_metrics_and_verdict():
    d = json.loads(rep.to_json(_failing_result()))
    assert d["verdict"]["passed"] is False
    assert d["aggregates"][0]["metric"] == "exact_match"
    assert "ci_low" in d["aggregates"][0]


def test_junit_marks_failure():
    xml = rep.to_junit([_failing_result()])
    root = ET.fromstring(xml)
    assert root.tag == "testsuites"
    failures = root.findall(".//failure")
    assert len(failures) >= 1


def test_text_report_mentions_metric_and_verdict():
    txt = rep.to_text(_failing_result())
    assert "exact_match" in txt
    assert "FAIL" in txt
