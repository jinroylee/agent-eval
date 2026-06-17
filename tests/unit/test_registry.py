"""Tests for the MetricRegistry that builds metrics from config dicts."""

import pytest

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.errors import ConfigError
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import MetricRegistry, default_registry


class _Echo(BaseMetric):
    """A trivial metric for exercising the registry mechanism (output == expected)."""

    name = "echo"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output", "expected"})
    cost_class = CostClass.FREE

    def __init__(self, bonus: float = 0.0) -> None:
        self.bonus = bonus

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = ctx.output == ctx.expected
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


def _reg() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register("echo", lambda p: _Echo(**p))
    return reg


def test_build_by_type_and_overrides_name():
    m = _reg().build({"type": "echo", "name": "em"})
    assert m.name == "em"
    assert m.score(EvalContext(input="q", output="a", expected="a")).score == 1.0


def test_build_with_params():
    m = _reg().build({"type": "echo", "name": "e", "params": {"bonus": 0.5}})
    assert m.bonus == 0.5


def test_build_from_bare_string_uses_type_as_name():
    m = _reg().build("echo")
    assert m.name == "echo"


def test_unknown_type_raises_config_error():
    with pytest.raises(ConfigError):
        _reg().build({"type": "does_not_exist"})


def test_registry_lists_registered_types_sorted():
    reg = _reg()
    reg.register("alpha", lambda p: _Echo(**p))
    assert reg.types() == ["alpha", "echo"]


def test_default_registry_is_empty_base():
    # The deterministic built-ins were removed; the base registry is now empty and metric
    # families are layered on per agent type / by the CLI.
    assert default_registry().types() == []
