"""Metric registry: factory registration, BuildContext passthrough, and the built-in catalog."""

import pytest

from agent_eval.core.contracts import EvalContext, MetricResult
from agent_eval.core.errors import ConfigError
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import BuildContext, MetricRegistry, default_registry

EXPECTED_TYPES = {
    "llm_judge", "bertscore", "p95_latency", "token_usage",  # common
    "recall_at_k", "precision_at_k", "ndcg_at_k", "faithfulness", "consistency",  # rag
    "soft_f1", "component_match", "ast_valid", "t2s_faithfulness", "t2s_consistency",  # t2s
}


class _Echo(BaseMetric):
    name = "echo"

    def __init__(self, value: float = 0.0):
        self.value = value

    def _compute(self, ctx):
        return MetricResult(self.name, self.value)


def test_register_and_build_with_params():
    reg = MetricRegistry()
    reg.register("echo", lambda p, ctx: _Echo(**p))
    m = reg.build({"type": "echo", "name": "my_echo", "params": {"value": 0.7}})
    assert m.name == "my_echo" and m.score(EvalContext(input="q")).score == 0.7


def test_build_passes_build_context():
    reg = MetricRegistry()
    reg.register("ctx_echo", lambda p, ctx: _Echo(ctx.defaults.get("value", 0.0)))
    m = reg.build("ctx_echo", BuildContext(defaults={"value": 0.9}))
    assert m.score(EvalContext(input="q")).score == 0.9


def test_unknown_type_raises():
    with pytest.raises(ConfigError):
        MetricRegistry().build("nope")


def test_default_registry_has_full_catalog():
    assert set(default_registry().types()) == EXPECTED_TYPES


def test_judge_metrics_resolve_system_prompt_from_defaults_and_params():
    reg = default_registry()
    ctx = BuildContext(defaults={"system_prompt": "Global persona."})
    assert reg.build("faithfulness", ctx).system_prompt == "Global persona."
    assert reg.build("llm_judge", ctx).system_prompt == "Global persona."
    override = reg.build(
        {"type": "faithfulness", "name": "f", "params": {"system_prompt": "Metric persona."}}, ctx
    )
    assert override.system_prompt == "Metric persona."
