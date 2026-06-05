"""Tests for the MetricRegistry that builds metrics from config dicts."""

import pytest

from agent_eval.core.contracts import EvalContext
from agent_eval.core.errors import ConfigError
from agent_eval.core.registry import default_registry


def test_build_by_type_and_overrides_name():
    reg = default_registry()
    m = reg.build({"type": "exact_match", "name": "em"})
    assert m.name == "em"
    assert m.score(EvalContext(input="q", output="a", expected="a")).score == 1.0


def test_build_with_params():
    reg = default_registry()
    m = reg.build({"type": "numeric_tolerance", "name": "nt", "params": {"tol": 0.01}})
    assert m.score(EvalContext(input="q", output="1.005", expected="1.0")).score == 1.0


def test_build_from_bare_string_uses_type_as_name():
    reg = default_registry()
    m = reg.build("exact_match")
    assert m.name == "exact_match"


def test_unknown_type_raises_config_error():
    reg = default_registry()
    with pytest.raises(ConfigError):
        reg.build({"type": "does_not_exist"})


def test_registry_lists_builtin_types():
    types = default_registry().types()
    assert {"exact_match", "regex_match", "set_match", "json_shape", "numeric_tolerance"} <= set(
        types
    )
