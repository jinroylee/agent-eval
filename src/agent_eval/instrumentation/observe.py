"""Observational (non-intrusive) instrumentation: capture per-level I/O from astream_events.

Runs the graph unmodified and consumes ``astream_events(version="v2")``, pairing node I/O by
``metadata['langgraph_node']`` and tracking the end-to-end final state. This lets the same metrics
score a production graph at graph/subgraph/node/tool granularity without changing the agent.

Anchored on the stable v2 event API (``on_chain_start``/``on_chain_end`` + ``data.input/output``);
the newer stream-events v3 path is intentionally not depended upon.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_eval.core.contracts import EvalContext, EvalTarget, Level
from agent_eval.instrumentation import context_builder as cb


async def observe(compiled, graph_input, targets: Sequence[EvalTarget]) -> list[EvalContext]:
    node_io: dict[str, dict] = {}
    graph_final = None

    async for event in compiled.astream_events(graph_input, version="v2"):
        if event["event"] != "on_chain_end":
            continue
        node = event.get("metadata", {}).get("langgraph_node")
        data = event.get("data", {})
        if node:
            node_io[node] = {"input": data.get("input"), "output": data.get("output")}
        else:
            graph_final = data.get("output")  # outermost chain completes last -> last wins

    contexts: list[EvalContext] = []
    for target in targets:
        if target.level is Level.GRAPH:
            contexts.append(cb.for_graph(graph_input, graph_final))
        elif target.level in (Level.NODE, Level.SUBGRAPH, Level.TOOL):
            io = node_io.get(target.selector)
            if io is None:
                continue  # node didn't fire on this run
            if target.level is Level.TOOL:
                contexts.append(cb.for_tool(target.selector, io["input"], io["output"]))
            elif target.level is Level.SUBGRAPH:
                contexts.append(cb.for_subgraph(target.selector, io["input"], io["output"]))
            else:
                contexts.append(cb.for_node(target.selector, io["input"], io["output"]))
    return contexts
