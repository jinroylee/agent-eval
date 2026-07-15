"""Pydantic schema for the agent-eval YAML config (the declarative, config-first surface).

One file describes: how to turn a LangGraph agent's runs into a predictions dataset
(``prediction``), where the data lives (``datasets``), which LLM judge to use (``judge``), and what
to score + gate (``suites``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GateModel(BaseModel):
    thresholds: dict[str, float] = Field(default_factory=dict)  # metric name -> threshold
    require_pass: list[str] = Field(default_factory=list)  # hard ship-blockers


class SuiteModel(BaseModel):
    metrics: list[Any] = Field(default_factory=list)  # bare type-string or {type,name,params}
    gate: GateModel = Field(default_factory=GateModel)
    dataset: str | None = None  # which configured dataset this suite scores


class DatasetModel(BaseModel):
    adapter: str = "jsonl"  # jsonl | tabular (defaults to jsonl; the prediction target is always jsonl)
    path: str
    format: str | None = None
    field_map: dict[str, str] = Field(default_factory=dict)  # target field -> source column
    include_unmapped: bool = True


class PredictionModel(BaseModel):
    """How ``agent-eval predict`` fills a dataset by running a LangGraph agent.

    ``graph`` is ``"module:attr"`` resolving to a compiled LangGraph (or a zero-arg factory that
    returns one). For each record in the ``source`` dataset, the graph is invoked with
    ``{input_key: <the input field>}``; ``state_map`` (target field -> final-state key) projects the
    resulting state onto canonical fields. The merged records are written to the ``target`` dataset.
    With ``n_runs > 1`` the graph runs N times per input and the repeated generations are stored as
    ``metadata['repeated_outputs']`` (and ``['repeated_sql']`` when ``sql`` is mapped) for the
    self-consistency metrics.
    """

    graph: str
    source: str  # dataset name holding the gold inputs + references
    target: str  # dataset name to write predictions into (its `path` is the output file); always jsonl
    input_key: str = "input"  # key under which the question is passed into the graph state
    state_map: dict[str, str] = Field(default_factory=dict)  # target field -> graph state key
    n_runs: int = Field(default=1, ge=1)  # >1 -> repeated runs for the self-consistency metrics
    generate_csv: bool = False  # also write a flattened, human-readable CSV view next to the jsonl


class JudgeModel(BaseModel):
    """Resolve an LLM judge backend. ``factory`` is ``"module:attr"`` returning a JudgeBackend (or a
    list of them = a PoLL panel). Omit this section to use the deterministic offline stub."""

    factory: str | None = None


class ConfigModel(BaseModel):
    version: int = 1
    agent_type: str = "plain"  # plain | rag | t2s (informational)
    defaults: dict[str, Any] = Field(default_factory=dict)  # dialect, k, result_set_policy, ...
    datasets: dict[str, DatasetModel] = Field(default_factory=dict)
    judge: JudgeModel | None = None
    prediction: PredictionModel | None = None
    suites: dict[str, SuiteModel] = Field(default_factory=dict)
