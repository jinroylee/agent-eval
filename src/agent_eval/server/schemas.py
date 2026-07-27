"""Pydantic request/response models — the wire mirror of the core contracts.

Requests carry 1..N :class:`EvalContextIn` items (every field optional; each metric's ``requires``
decides what is enough); responses mirror :class:`MetricResult` per item plus the runner's
:class:`MetricAggregate` as the final value. Field names match the core dataclasses exactly so
payloads read like the framework's own objects.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_eval.core.contracts import EvalContext, MetricResult
from agent_eval.core.gate import MetricAggregate


class EvalContextIn(BaseModel):
    """JSON mirror of :class:`EvalContext` (see MetaKey for the conventional metadata keys)."""

    input: Any = None
    output: Any = None
    expected: Any = None
    retrieved_context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_context(self) -> EvalContext:
        return EvalContext(
            input=self.input,
            output=self.output,
            expected=self.expected,
            retrieved_context=tuple(self.retrieved_context),
            metadata=dict(self.metadata),
        )


class MetricRequest(BaseModel):
    """The uniform request body: 1..N contexts (repeated runs post N) + optional metric params."""

    contexts: list[EvalContextIn] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class CostOut(BaseModel):
    tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0


class MetricResultOut(BaseModel):
    """One item's score. ``error`` set means the metric could not run for this item (it is then
    excluded from the aggregate's denominator) — never conflate it with a low score."""

    metric: str
    score: float
    passed: bool | None = None
    confidence: float | None = None
    cost: CostOut = Field(default_factory=CostOut)
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_result(cls, r: MetricResult) -> MetricResultOut:
        return cls(
            metric=r.metric, score=r.score, passed=r.passed, confidence=r.confidence,
            cost=CostOut(tokens=r.cost.tokens, usd=r.cost.usd, latency_ms=r.cost.latency_ms),
            detail=dict(r.detail), error=r.error,
        )


class AggregateOut(BaseModel):
    """Mirrors :class:`agent_eval.core.gate.MetricAggregate` field-for-field."""

    metric: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    n_errors: int
    aggregation: str
    higher_is_better: bool
    breakdown: dict[str, float] | None = None  # per-criterion means (criteria-based judge metrics)

    @classmethod
    def from_aggregate(cls, a: MetricAggregate) -> AggregateOut:
        return cls(
            metric=a.metric, value=a.value, ci_low=a.ci_low, ci_high=a.ci_high,
            n=a.n, n_errors=a.n_errors, aggregation=str(a.aggregation),
            higher_is_better=a.higher_is_better,
            breakdown=dict(a.breakdown) if a.breakdown else None,
        )


class MetricResponse(BaseModel):
    family: str
    metric: str  # the registry type
    n: int  # contexts posted
    n_errors: int  # items whose metric errored
    results: list[MetricResultOut]  # one per posted context, same order
    aggregate: AggregateOut  # the final value (always present; n=1 is degenerate but well-defined)


class JudgeInfo(BaseModel):
    kind: str  # stub | factory | openai_compatible | injected
    detail: str = ""
    panel_size: int = 1  # judges consulted per item (primary + panel)


class HealthOut(BaseModel):
    status: str
    version: str
    judge: JudgeInfo
    available: dict[str, bool]  # optional capabilities: {"t2s": ..., "bertscore": ...}


class CatalogEntry(BaseModel):
    family: str
    path: str
    type: str
    requires: list[str]
    optional: list[str]
    metadata_keys: list[str]
    metadata_optional: list[str]
    params: list[str]
    judge_based: bool
    cost_class: str | None = None  # None when the metric is unavailable on this server
    aggregation: str | None = None
    available: bool
    unavailable_hint: str | None = None
