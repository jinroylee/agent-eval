"""LangGraph topology introspection — discover the graph and validate eval-target selectors.

Uses the duck-typed ``compiled.get_graph()`` (no hard langgraph import), so a user can validate
that their config's ``EvalTarget`` selectors name real nodes before any run (fail fast with a list
of valid names) and attach metrics to levels declaratively.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval.core.contracts import EvalTarget, Level
from agent_eval.core.errors import SelectorNotFound

_SPECIAL = {"__start__", "__end__"}


@dataclass
class TopologyIndex:
    nodes: set[str]
    edges: list[tuple[str, str]]


def introspect(compiled) -> TopologyIndex:
    graph = compiled.get_graph()
    nodes = {n for n in graph.nodes if n not in _SPECIAL}
    edges = [(e.source, e.target) for e in graph.edges]
    return TopologyIndex(nodes, edges)


def validate_selector(index: TopologyIndex, target: EvalTarget) -> None:
    """Raise SelectorNotFound if the target names a node that does not exist (whole-graph is ok)."""
    if target.level is Level.GRAPH or target.selector == "*":
        return
    if target.selector not in index.nodes:
        raise SelectorNotFound(
            f"selector {target.selector!r} not found in graph; valid nodes: {sorted(index.nodes)}"
        )
