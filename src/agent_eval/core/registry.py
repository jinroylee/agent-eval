"""Metric registry — builds Metric instances from config (the config-first surface).

A metric spec is either a bare type string (``"exact_match"``) or a dict
``{"type": ..., "name": ..., "params": {...}}``. Future agents and plugins register new types
here; nothing in the core changes. Plugin types are discovered via entry points (group
``agent_eval.metrics``) when ``load_plugins()`` is called.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from agent_eval.core.errors import ConfigError
from agent_eval.core.metric import Metric
from agent_eval.metrics.deterministic_ import (
    ExactMatch,
    JsonShape,
    NumericTolerance,
    RegexMatch,
    SetMatch,
)

MetricFactory = Callable[[dict], Metric]


class MetricRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, MetricFactory] = {}

    def register(self, type_name: str, factory: MetricFactory) -> None:
        self._factories[type_name] = factory

    def build(self, spec: dict | str) -> Metric:
        if isinstance(spec, str):
            spec = {"type": spec, "name": spec}
        type_name = spec.get("type") or spec.get("name")
        if type_name is None:
            raise ConfigError(f"metric spec needs a 'type' or 'name': {spec!r}")
        if type_name not in self._factories:
            raise ConfigError(
                f"unknown metric type {type_name!r}; known: {self.types()}"
            )
        params = spec.get("params") or {}
        metric = self._factories[type_name](params)
        metric.name = spec.get("name") or type_name  # per-instance reporting name
        return metric

    def types(self) -> list[str]:
        return sorted(self._factories)

    def load_plugins(self, group: str = "agent_eval.metrics") -> None:
        """Register metric factories advertised by installed packages via entry points."""
        for ep in entry_points(group=group):
            self.register(ep.name, ep.load())


def default_registry() -> MetricRegistry:
    """A registry pre-loaded with the built-in deterministic metrics."""
    reg = MetricRegistry()
    reg.register("exact_match", lambda p: ExactMatch())
    reg.register("regex_match", lambda p: RegexMatch(**p))
    reg.register("set_match", lambda p: SetMatch())
    reg.register("numeric_tolerance", lambda p: NumericTolerance(**p))
    reg.register("json_shape", lambda p: JsonShape(**p))
    return reg


__all__ = ["MetricRegistry", "MetricFactory", "default_registry"]
