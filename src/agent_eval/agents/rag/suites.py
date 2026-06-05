"""RAG suite wiring: registry with retrieval + uncertainty metrics, and suite builders."""

from __future__ import annotations

from agent_eval.config.loader import build_suite
from agent_eval.config.schema import ConfigModel
from agent_eval.core.registry import MetricRegistry, default_registry
from agent_eval.core.suite import Suite
from agent_eval.metrics.retrieval_ import register_retrieval_metrics
from agent_eval.metrics.uncertainty_ import register_uncertainty_metrics


def rag_registry() -> MetricRegistry:
    reg = default_registry()
    register_retrieval_metrics(reg)
    register_uncertainty_metrics(reg)
    return reg


def build_search_suite(cfg: ConfigModel, registry: MetricRegistry | None = None) -> Suite:
    return build_suite(cfg, "search", registry or rag_registry())


def build_response_suite(cfg: ConfigModel, registry: MetricRegistry | None = None) -> Suite:
    return build_suite(cfg, "response", registry or rag_registry())
