"""Plain-agent suite wiring: registry with uncertainty metrics (+ judges configured per deployment)."""

from __future__ import annotations

from agent_eval.config.loader import build_suite
from agent_eval.config.schema import ConfigModel
from agent_eval.core.registry import MetricRegistry, default_registry
from agent_eval.core.suite import Suite
from agent_eval.metrics.uncertainty_ import register_uncertainty_metrics


def plain_registry() -> MetricRegistry:
    reg = default_registry()
    register_uncertainty_metrics(reg)
    return reg


def build_response_suite(cfg: ConfigModel, registry: MetricRegistry | None = None) -> Suite:
    return build_suite(cfg, "response", registry or plain_registry())
