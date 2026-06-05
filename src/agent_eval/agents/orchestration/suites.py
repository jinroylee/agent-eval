"""Orchestration suite wiring: registry with trajectory + perf metrics, and suite builders."""

from __future__ import annotations

from agent_eval.config.loader import build_suite
from agent_eval.config.schema import ConfigModel
from agent_eval.core.registry import MetricRegistry, default_registry
from agent_eval.core.suite import Suite
from agent_eval.metrics.perf_ import register_perf_metrics
from agent_eval.metrics.trajectory_ import register_trajectory_metrics


def orchestration_registry() -> MetricRegistry:
    reg = default_registry()
    register_trajectory_metrics(reg)
    register_perf_metrics(reg)
    return reg


def build_trajectory_suite(cfg: ConfigModel, registry: MetricRegistry | None = None) -> Suite:
    return build_suite(cfg, "search", registry or orchestration_registry())
