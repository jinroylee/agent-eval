"""T2S suite wiring: a registry preloaded with the T2S metrics, and a search-suite builder."""

from __future__ import annotations

from agent_eval.config.loader import build_suite
from agent_eval.config.schema import ConfigModel
from agent_eval.core.registry import MetricRegistry, default_registry
from agent_eval.core.suite import Suite
from agent_eval.metrics.ast_query_ import register_t2s_metrics


def t2s_registry() -> MetricRegistry:
    """A metric registry with the built-in deterministic metrics + the T2S metrics registered."""
    reg = default_registry()
    register_t2s_metrics(reg)
    return reg


def build_search_suite(cfg: ConfigModel, registry: MetricRegistry | None = None) -> Suite:
    return build_suite(cfg, "search", registry or t2s_registry())
