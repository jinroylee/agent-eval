"""Internal observability boundary — the framework emits spans here, never to a backend SDK.

A ``Span`` carries OpenTelemetry GenAI-semantic-convention-friendly attributes (``gen_ai.*``).
``TraceBoundary`` fans each emitted span out to every registered backend adapter, so one
instrumentation point feeds multiple swappable backends and nothing in the core depends on a
particular vendor (OTel ``gen_ai.*`` is still experimental, so we isolate ourselves from it here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str  # operation / node / tool name
    kind: str  # graph | chain | llm | tool
    attributes: dict[str, Any] = field(default_factory=dict)  # gen_ai.* + framework keys
    start_ms: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error
    parent: str | None = None


class TraceBoundary:
    def __init__(self) -> None:
        self._adapters: list = []

    def register(self, adapter) -> None:
        self._adapters.append(adapter)

    def emit(self, span: Span) -> None:
        for adapter in self._adapters:
            adapter.export(span)


def span_from_node(node: str, tokens: int = 0, latency_ms: float = 0.0, usd: float = 0.0) -> Span:
    """Build a node span with gen_ai.* attributes from attribution figures."""
    return Span(
        name=node,
        kind="chain",
        attributes={
            "gen_ai.usage.total_tokens": tokens,
            "gen_ai.cost.usd": usd,
            "langgraph.node": node,
        },
        duration_ms=latency_ms,
    )
