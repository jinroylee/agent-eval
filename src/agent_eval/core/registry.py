"""Metric registry — build ``Metric`` instances from declarative config specs.

A metric spec is either a bare type string (``"recall_at_k"``) or a dict
``{"type": ..., "name": ..., "params": {...}}``. Factories receive the spec ``params`` *and* a
:class:`BuildContext` carrying things that can't live in YAML — the resolved LLM judge backend and
suite-wide defaults (``dialect``, ``k``, ``result_set_policy``). Plugins register new types via
entry points (group ``agent_eval.metrics``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from agent_eval.core.errors import ConfigError
from agent_eval.core.metric import Metric


@dataclass(frozen=True)
class BuildContext:
    """Shared, non-serializable dependencies injected into metric factories at build time."""

    judge: Any = None  # a judges.backend.JudgeBackend (typed Any to avoid an import cycle)
    panel: Sequence[Any] = ()  # extra judge backends => a PoLL panel
    defaults: Mapping[str, Any] = field(default_factory=dict)


MetricFactory = Callable[[dict, BuildContext], Metric]


class MetricRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, MetricFactory] = {}

    def register(self, type_name: str, factory: MetricFactory) -> None:
        self._factories[type_name] = factory

    def build(self, spec: dict | str, ctx: BuildContext | None = None) -> Metric:
        if isinstance(spec, str):
            spec = {"type": spec, "name": spec}
        type_name = spec.get("type") or spec.get("name")
        if type_name is None:
            raise ConfigError(f"metric spec needs a 'type' or 'name': {spec!r}")
        if type_name not in self._factories:
            raise ConfigError(f"unknown metric type {type_name!r}; known: {self.types()}")
        params = spec.get("params") or {}
        metric = self._factories[type_name](params, ctx or BuildContext())
        metric.name = spec.get("name") or type_name  # per-instance reporting name
        return metric

    def types(self) -> list[str]:
        return sorted(self._factories)

    def load_plugins(self, group: str = "agent_eval.metrics") -> None:
        """Register metric factories advertised by installed packages via entry points."""
        for ep in entry_points(group=group):
            self.register(ep.name, ep.load())


def default_registry() -> MetricRegistry:
    """A registry with every built-in metric family registered (common + RAG + T2S).

    T2S needs the optional ``sqlglot`` dependency, so it is registered only if importable. Plugin
    metrics are layered on via :meth:`MetricRegistry.load_plugins`.
    """
    from agent_eval.metrics import common, rag

    reg = MetricRegistry()
    common.register(reg)
    rag.register(reg)
    try:
        from agent_eval.metrics import t2s

        t2s.register(reg)
    except ImportError:
        pass  # T2S metrics unavailable without the [t2s] extra
    reg.load_plugins()
    return reg


__all__ = ["MetricRegistry", "MetricFactory", "BuildContext", "default_registry"]
