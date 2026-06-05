"""Pydantic schema for the agent-eval YAML config (the declarative, config-first surface)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TargetModel(BaseModel):
    level: str = "graph"
    selector: str = "*"
    attach: str = "observe"


class GateModel(BaseModel):
    thresholds: dict[str, float] = Field(default_factory=dict)
    require_pass: list[str] = Field(default_factory=list)
    min_effect: float = 0.0
    significance_alpha: float = 0.05
    fdr: bool = True


class SuiteModel(BaseModel):
    target: TargetModel = Field(default_factory=TargetModel)
    metrics: list[Any] = Field(default_factory=list)  # bare type-string or {type,name,params}
    gate: GateModel = Field(default_factory=GateModel)
    dataset: str | None = None  # which configured dataset this suite evaluates
    judge: dict[str, Any] | None = None


class DatasetModel(BaseModel):
    adapter: str
    path: str
    format: str | None = None
    field_map: dict[str, str] = Field(default_factory=dict)
    include_unmapped: bool = True
    frozen_slice: str | None = None


class JudgeModel(BaseModel):
    model: str
    prompt_version: str = "v1"
    rubric_id: str = ""
    protocol: str = "pointwise"
    panel: list[str] = Field(default_factory=list)
    confidence_policy: str = "logprob"
    certification_ref: str | None = None


class RuntimeCriticModel(BaseModel):
    tiers: list[str] = Field(default_factory=lambda: ["deterministic", "uncertainty", "judge"])
    max_retries: int = 2
    latency_budget_ms: float = 1500.0
    fallback: str = "abstain"
    tau: dict[str, float] = Field(default_factory=dict)
    self_refine_only_for: list[str] = Field(default_factory=lambda: ["style", "format", "safety"])


class ConfigModel(BaseModel):
    version: int = 1
    agent_type: str
    defaults: dict[str, Any] = Field(default_factory=dict)
    datasets: dict[str, DatasetModel] = Field(default_factory=dict)
    judges: dict[str, JudgeModel] = Field(default_factory=dict)
    suites: dict[str, SuiteModel] = Field(default_factory=dict)
    runtime_critic: RuntimeCriticModel | None = None
    eval_targets: list[TargetModel] = Field(default_factory=list)
