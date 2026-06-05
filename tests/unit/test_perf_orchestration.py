"""Tests for perf attribution + orchestration per-step critic."""

from agent_eval.agents.orchestration.critic import build_orchestration_critic
from agent_eval.config.schema import ConfigModel, RuntimeCriticModel
from agent_eval.core.contracts import Decision, EvalContext
from agent_eval.metrics.perf_ import (
    LatencyBudget,
    attribute,
    latency_percentiles,
    scenario_pass_hat_k,
)
from agent_eval.runtime.critic import critic_loop
from agent_eval.runtime.fallback import default_fallbacks


def test_attribution_reconciles_to_graph_total():
    spans = [
        {"node": "plan", "tokens": 100, "latency_ms": 50, "usd": 0.01},
        {"node": "tool", "tokens": 40, "latency_ms": 20, "usd": 0.0},
        {"node": "plan", "tokens": 60, "latency_ms": 30, "usd": 0.006},
    ]
    a = attribute(spans)
    assert a["nodes"]["plan"]["tokens"] == 160 and a["nodes"]["tool"]["tokens"] == 40
    assert a["graph"]["tokens"] == 200
    assert a["graph"]["tokens"] == sum(n["tokens"] for n in a["nodes"].values())  # reconciles


def test_latency_percentiles():
    p = latency_percentiles(list(range(1, 101)))
    assert 49 <= p["p50"] <= 51 and 94 <= p["p95"] <= 96 and p["p99"] >= 98


def test_latency_budget_metric():
    m = LatencyBudget(max_ms=1000)
    assert m.score(EvalContext("q", "o", metadata={"latency_ms": 500})).passed is True
    assert m.score(EvalContext("q", "o", metadata={"latency_ms": 1500})).passed is False


def test_scenario_pass_hat_k():
    runs = [[True] * 5, [True, True, True, False, False]]  # 5/5 and 3/5
    r = scenario_pass_hat_k(runs, k=3)
    assert abs(r.estimate - 0.55) < 1e-9  # (1 + C(3,3)/C(5,3)) / 2


def _cfg(max_retries=2, fallback="abstain"):
    return ConfigModel(
        agent_type="orchestration",
        runtime_critic=RuntimeCriticModel(
            tiers=["deterministic"], max_retries=max_retries, fallback=fallback
        ),
    )


def test_orchestration_critic_rejects_invalid_tool_call():
    critic = build_orchestration_critic(_cfg())
    schemas = {"search": {"q": "str"}}
    bad = EvalContext("q", None, trajectory=[{"tool": "search", "args": {}}],
                      metadata={"tool_schemas": schemas})
    d, a = critic.decide(bad, attempt=0)
    assert d is Decision.RETRY and a.hard_fail
    good = EvalContext("q", None, trajectory=[{"tool": "search", "args": {"q": "x"}}],
                       metadata={"tool_schemas": schemas})
    assert critic.decide(good)[0] is Decision.ACCEPT


def test_orchestration_critic_loop_exhausts_to_fallback():
    critic = build_orchestration_critic(_cfg(max_retries=2))
    schemas = {"search": {"q": "str"}}

    def gen(attempt, critique):
        return [{"tool": "search", "args": {}, "n": attempt}]  # always invalid, distinct per attempt

    res = critic_loop(
        gen,
        lambda t: EvalContext("q", None, trajectory=t, metadata={"tool_schemas": schemas}),
        critic,
        fallback=default_fallbacks().get("abstain"),
    )
    assert res.decision is Decision.ABSTAIN and res.attempts == 2
