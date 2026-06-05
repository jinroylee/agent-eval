"""Tests for orchestration trajectory metrics (tool-sequence match + BFCL-style arg validity)."""

from agent_eval.core.contracts import EvalContext
from agent_eval.metrics.trajectory_ import ToolArgValidity, TrajectoryMatch


def _step(tool, **args):
    return {"tool": tool, "args": args}


def _ctx(pred, ref=None, schemas=None):
    md = {"tool_schemas": schemas} if schemas else {}
    return EvalContext(input="q", output=None, expected=ref, trajectory=pred, metadata=md)


def test_strict_match_is_order_sensitive():
    ref = [_step("search", q="x"), _step("answer")]
    assert TrajectoryMatch(mode="strict").score(_ctx(ref, ref)).score == 1.0
    reordered = [_step("answer"), _step("search", q="x")]
    assert TrajectoryMatch(mode="strict").score(_ctx(reordered, ref)).score == 0.0


def test_unordered_allows_valid_alternate_path():
    ref = [_step("a"), _step("b")]
    assert TrajectoryMatch(mode="unordered").score(_ctx([_step("b"), _step("a")], ref)).score == 1.0


def test_subset_is_no_forbidden_tool_gate():
    allowed = [_step("search"), _step("answer"), _step("calc")]
    ok = TrajectoryMatch(mode="subset").score(_ctx([_step("search"), _step("answer")], allowed))
    assert ok.score == 1.0 and ok.passed is True
    forbidden = TrajectoryMatch(mode="subset").score(
        _ctx([_step("search"), _step("delete_db")], allowed)
    )
    assert forbidden.score == 0.0 and forbidden.passed is False


def test_superset_requires_all_reference_tools():
    required = [_step("auth"), _step("fetch")]
    has_all = [_step("auth"), _step("fetch"), _step("log")]
    assert TrajectoryMatch(mode="superset").score(_ctx(has_all, required)).score == 1.0
    assert TrajectoryMatch(mode="superset").score(_ctx([_step("auth")], required)).score == 0.0


def test_tool_arg_validity_bfcl_style():
    schemas = {"search": {"q": "str"}, "calc": {"x": "int", "y": "int"}}
    good = [_step("search", q="hi"), _step("calc", x=1, y=2)]
    assert ToolArgValidity().score(_ctx(good, schemas=schemas)).score == 1.0

    bad = [_step("search"), _step("calc", x="oops", y=2)]  # missing q; wrong type
    r = ToolArgValidity().score(_ctx(bad, schemas=schemas))
    assert r.score < 1.0 and r.passed is False

    unknown = [_step("delete")]  # tool not in the allowed schema
    assert ToolArgValidity().score(_ctx(unknown, schemas=schemas)).score == 0.0
