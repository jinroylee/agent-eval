"""Build the SAME EvalContext at every graph level — only field population differs.

This is the highest-risk correctness point of the multi-level design: TOOL puts call args in
``input`` and the tool result in ``output``; NODE uses node in/out; GRAPH/SUBGRAPH use task input
and boundary state. Centralizing it here keeps the tool-vs-node asymmetry testable in one place.
"""

from __future__ import annotations

from typing import Any

from agent_eval.core.contracts import EvalContext, Level


def for_node(node: str, input_state: Any, output_state: Any, **extra: Any) -> EvalContext:
    return EvalContext(
        input=input_state,
        output=output_state,
        metadata={"level": Level.NODE.value, "selector": node, "langgraph_node": node, **extra},
    )


def for_tool(tool: str, args: Any, result: Any, **extra: Any) -> EvalContext:
    return EvalContext(
        input=args,
        output=result,
        metadata={"level": Level.TOOL.value, "selector": tool, **extra},
    )


def for_subgraph(name: str, boundary_input: Any, boundary_output: Any, **extra: Any) -> EvalContext:
    return EvalContext(
        input=boundary_input,
        output=boundary_output,
        metadata={"level": Level.SUBGRAPH.value, "selector": name, **extra},
    )


def for_graph(graph_input: Any, final_output: Any, **extra: Any) -> EvalContext:
    return EvalContext(
        input=graph_input,
        output=final_output,
        metadata={"level": Level.GRAPH.value, "selector": "*", **extra},
    )
