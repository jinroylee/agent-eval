"""Tests for the LangGraph instrumentation layer against a real tiny StateGraph."""

import asyncio
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent_eval.core.contracts import EvalContext, EvalTarget, Level, MetricResult, Tier
from agent_eval.core.errors import SelectorNotFound
from agent_eval.core.metric import BaseMetric
from agent_eval.instrumentation.active_node import make_critic_node
from agent_eval.instrumentation.context_builder import for_graph, for_node, for_tool
from agent_eval.instrumentation.observe import observe
from agent_eval.instrumentation.topology import introspect, validate_selector
from agent_eval.runtime.critic import Critic
from agent_eval.runtime.policy import CriticPolicy


class _S(TypedDict):
    text: str


def _app():
    def upper(state):
        return {"text": state["text"].upper()}

    def exclaim(state):
        return {"text": state["text"] + "!"}

    g = StateGraph(_S)
    g.add_node("upper", upper)
    g.add_node("exclaim", exclaim)
    g.add_edge(START, "upper")
    g.add_edge("upper", "exclaim")
    g.add_edge("exclaim", END)
    return g.compile()


def test_topology_introspection_and_selector_validation():
    idx = introspect(_app())
    assert {"upper", "exclaim"} <= idx.nodes
    assert "__start__" not in idx.nodes and "__end__" not in idx.nodes
    validate_selector(idx, EvalTarget(level=Level.NODE, selector="upper"))  # no raise
    validate_selector(idx, EvalTarget(level=Level.GRAPH, selector="*"))  # whole graph
    with pytest.raises(SelectorNotFound):
        validate_selector(idx, EvalTarget(level=Level.NODE, selector="does_not_exist"))


def test_observe_captures_node_and_graph_levels():
    targets = [
        EvalTarget(level=Level.NODE, selector="upper"),
        EvalTarget(level=Level.GRAPH, selector="*"),
    ]
    ctxs = asyncio.run(observe(_app(), {"text": "hi"}, targets))
    by_level = {c.metadata["level"]: c for c in ctxs}
    assert by_level["node"].output == {"text": "HI"}  # the 'upper' node's output
    assert by_level["graph"].output == {"text": "HI!"}  # end-to-end final state


def test_context_builder_identical_shape_node_vs_tool():
    node_ctx = for_node("gen", {"in": 1}, {"out": 2})
    tool_ctx = for_tool("search", {"q": "x"}, ["r1", "r2"])
    graph_ctx = for_graph({"text": "hi"}, {"text": "HI!"})
    assert isinstance(node_ctx, EvalContext) and isinstance(tool_ctx, EvalContext)
    assert node_ctx.metadata["level"] == "node" and node_ctx.input == {"in": 1}
    assert tool_ctx.metadata["level"] == "tool" and tool_ctx.input == {"q": "x"}
    assert tool_ctx.output == ["r1", "r2"]  # TOOL: args->input, result->output
    assert graph_ctx.metadata["selector"] == "*"


def test_make_critic_node_routes_on_decision():
    class IsUpper(BaseMetric):
        name = "is_upper"
        tier = Tier.DETERMINISTIC
        requires = frozenset({"output"})

        def _compute(self, ctx: EvalContext) -> MetricResult:
            ok = str(ctx.output).isupper()
            return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)

    critic = Critic(
        {Tier.DETERMINISTIC: [IsUpper()]}, CriticPolicy(tiers=[Tier.DETERMINISTIC], max_retries=2)
    )
    node = make_critic_node(
        critic,
        lambda state: EvalContext("q", state["text"]),
        on_accept="done",
        on_retry="gen",
        on_fallback="human",
    )

    accept = node({"text": "HELLO"})
    assert isinstance(accept, Command) and accept.goto == "done"

    retry = node({"text": "hello"})
    assert retry.goto == "gen" and retry.update["_critic_attempt"] == 1

    exhausted = node({"text": "hello", "_critic_attempt": 2})
    assert exhausted.goto == "human"
