"""Core contracts — the universal currency every module shares.

An ``EvalContext`` describes one unit of work to score at *any* graph level (graph / subgraph /
node / tool); a ``Metric`` turns it into a normalized ``MetricResult``. Keeping these tiny and
frozen means the same objects flow through the offline runner and the runtime critic unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Tier(StrEnum):
    """How a metric produces a score; also the runtime escalation order (cheap -> strong)."""

    DETERMINISTIC = "deterministic"
    UNCERTAINTY = "uncertainty"
    JUDGE = "judge"


class Mode(StrEnum):
    OFFLINE = "offline"
    RUNTIME = "runtime"


class Level(StrEnum):
    GRAPH = "graph"
    SUBGRAPH = "subgraph"
    NODE = "node"
    TOOL = "tool"


class CostClass(StrEnum):
    FREE = "free"
    CHEAP = "cheap"
    EXPENSIVE = "expensive"


class Decision(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Cost:
    tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EvalContext:
    """One unit of work under evaluation. Optional fields cover all four agent types."""

    input: Any
    output: Any
    expected: Any = None
    retrieved_context: Sequence[str] = ()
    trajectory: Sequence[Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    metric: str
    score: float
    passed: bool | None = None
    confidence: float | None = None
    cost: Cost = Cost()
    detail: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None  # set => the metric could not RUN (distinct from scoring a failure)


@dataclass(frozen=True)
class EvalTarget:
    """Which LangGraph level a suite evaluates, and how it attaches."""

    level: Level
    selector: str = "*"  # node / tool / subgraph name; "*" == whole graph
    attach: str = "observe"  # "observe" (post-hoc) | "active" (inject critic)
