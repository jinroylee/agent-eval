"""Core contracts — the small, shared vocabulary every module uses.

An :class:`EvalContext` is one item to score (one user query and the agent's response to it); a
``Metric`` turns it into a normalized :class:`MetricResult`; the offline runner aggregates results
across a dataset into a gated verdict. Keeping these tiny and frozen means datasets, metrics, and
the runner can evolve independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Aggregation(StrEnum):
    """How a metric's per-item scores roll up into a single dataset-level number.

    - ``RATE`` — fraction of items that passed (binary metrics); reported with a Wilson CI.
    - ``MEAN`` — average of graded [0, 1] scores; reported with a cluster-robust CI.
    - ``P95`` — 95th percentile (heavy-tailed observations like latency); bootstrap CI.
    """

    RATE = "rate"
    MEAN = "mean"
    P95 = "p95"


class CostClass(StrEnum):
    """Rough running cost of a metric — lets users see at a glance which metrics need an LLM."""

    FREE = "free"  # pure computation
    CHEAP = "cheap"  # local model / DB execution
    EXPENSIVE = "expensive"  # calls an LLM judge


class MetaKey:
    """The conventional ``EvalContext.metadata`` keys this framework reads.

    Centralized so the "required LangGraph state fields" are one source of truth (docs reference
    these names). A dataset/prediction record populates the ones its metrics need.
    """

    RETRIEVED_IDS = "retrieved_ids"  # RAG: ranked ids the retriever returned (best first)
    RELEVANT_IDS = "relevant_ids"  # RAG: gold relevant ids (GT)
    RELEVANCE = "relevance"  # RAG: optional {id: graded gain} for NDCG (default: binary)
    SQL = "sql"  # T2S: the SQL the agent generated
    GOLD_SQL = "gold_sql"  # T2S: the gold SQL (GT)
    DB_REF = "db_ref"  # T2S: path/URI of a database (used at data-prep/predict time, not at eval time)
    EXECUTION_RESULT = "execution_result"  # T2S: the predicted SQL's result set (rows), from the graph state
    GOLD_EXECUTION_RESULT = "gold_execution_result"  # T2S: the gold SQL's result set (rows), from the dataset (GT)
    LATENCY_MS = "latency_ms"  # perf: wall-clock of the run (auto-filled by the harness)
    TOKENS = "tokens"  # perf: tokens spent on the run (filled by the harness if available)
    CLUSTER_ID = "cluster_id"  # stats: group id for non-iid items (widens the CI)


@dataclass(frozen=True)
class Cost:
    tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EvalContext:
    """One unit of work to score: a user query and the agent's response, plus optional evidence.

    The invariant that keeps metrics uniform across agent types: ``output`` is **always the final
    response shown to the user**, so the common metrics (judge, BERTScore) score it for every agent
    type. Agent-specific artifacts live in dedicated fields / ``metadata`` (see :class:`MetaKey`):
    a RAG agent's ranked ids in ``metadata['retrieved_ids']``, a T2S agent's SQL in
    ``metadata['sql']``, and so on.
    """

    input: Any  # the user message / question
    output: Any = None  # the final response to the user (what the common metrics score)
    expected: Any = None  # gold final response (GT metrics only)
    retrieved_context: Sequence[str] = ()  # RAG: retrieved chunk texts (for faithfulness/consistency)
    metadata: Mapping[str, Any] = field(default_factory=dict)  # see MetaKey for the keys read


@dataclass(frozen=True)
class MetricResult:
    """A normalized score for one item.

    ``error`` (the metric could not run — missing field or exception) is distinct from a low
    ``score`` (it ran and the item scored badly): errored items are excluded from the denominator
    and surfaced separately, never silently counted as failures.
    """

    metric: str
    score: float
    passed: bool | None = None  # set for binary (RATE) metrics; None for graded
    confidence: float | None = None  # judge/panel agreement, if applicable
    cost: Cost = Cost()
    detail: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
