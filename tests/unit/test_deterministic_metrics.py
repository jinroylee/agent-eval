"""Known-good / known-bad / missing-requirement fixtures for the deterministic metrics."""

from agent_eval.core.contracts import EvalContext, Tier
from agent_eval.metrics.deterministic_ import (
    ExactMatch,
    JsonShape,
    NumericTolerance,
    RegexMatch,
    SetMatch,
)


def _ctx(output, expected=None):
    return EvalContext(input="q", output=output, expected=expected)


def test_exact_match_good_bad_missing():
    m = ExactMatch()
    assert m.tier is Tier.DETERMINISTIC
    assert m.score(_ctx("abc", "abc")).score == 1.0
    assert m.score(_ctx("abc", "xyz")).score == 0.0
    assert m.score(_ctx("abc", None)).error is not None  # expected missing


def test_regex_match():
    m = RegexMatch(pattern=r"\d+")
    assert m.score(_ctx("abc123")).score == 1.0
    assert m.score(_ctx("abc")).score == 0.0


def test_set_match_is_order_insensitive():
    m = SetMatch()
    assert m.score(_ctx([1, 2, 3], [3, 2, 1])).score == 1.0
    assert m.score(_ctx([1, 2], [1, 2, 3])).score == 0.0


def test_numeric_tolerance():
    m = NumericTolerance(tol=0.01)
    assert m.score(_ctx("1.005", "1.0")).score == 1.0
    assert m.score(_ctx("1.5", "1.0")).score == 0.0
    assert m.score(_ctx("not-a-number", "1.0")).error is not None  # trapped, not raised


def test_json_shape_required_keys():
    m = JsonShape(required_keys=["a", "b"])
    assert m.score(_ctx('{"a": 1, "b": 2}')).score == 1.0
    assert m.score(_ctx({"a": 1, "b": 2})).score == 1.0  # dict input
    bad = m.score(_ctx('{"a": 1}'))
    assert bad.score == 0.0 and "b" in bad.detail["missing"]
