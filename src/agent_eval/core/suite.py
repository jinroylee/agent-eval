"""A Suite groups the metrics + the gate for one (agent_type, category) evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agent_eval.core.gate import GatePolicy, GateVerdict, MetricAggregate
from agent_eval.core.metric import Metric


@dataclass
class Suite:
    agent_type: str  # plain | rag | t2s (informational — drives nothing in the core)
    category: str  # e.g. retrieval | response | performance
    metrics: Sequence[Metric]
    gate: GatePolicy = field(default_factory=GatePolicy)


@dataclass
class SuiteResult:
    agent_type: str
    category: str
    n_items: int
    aggregates: list[MetricAggregate]
    verdict: GateVerdict
