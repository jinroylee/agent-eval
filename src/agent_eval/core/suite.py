"""A Suite groups metrics + a gate for one (agent_type, category) at one graph level."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agent_eval.core.contracts import EvalTarget
from agent_eval.core.gate import GatePolicy, GateVerdict, MetricAggregate
from agent_eval.core.metric import Metric


@dataclass
class Suite:
    agent_type: str
    category: str  # search | response | latency | scenario
    metrics: Sequence[Metric]
    gate: GatePolicy = field(default_factory=GatePolicy)
    target: EvalTarget | None = None


@dataclass
class SuiteResult:
    agent_type: str
    category: str
    n_items: int
    aggregates: list[MetricAggregate]
    verdict: GateVerdict
