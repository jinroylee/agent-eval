"""Backend adapters for the trace boundary — swappable, instrumented once via gen_ai.* spans.

MemoryBackend and JsonlBackend are dependency-free (and the test contract). OTelBackend maps spans
to OpenTelemetry; LangSmith / Phoenix / Langfuse are then reachable through the OTel/OpenInference
exporter without changing the framework — standardize at the instrumentation boundary, swap backends
freely. Prefer the self-hostable OTel-native ones over proprietary backends for governance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol, runtime_checkable

from agent_eval.obs.boundary import Span


@runtime_checkable
class BackendAdapter(Protocol):
    name: str

    def export(self, span: Span) -> None: ...


class MemoryBackend:
    """Collects spans in memory — for tests and assertions."""

    name = "memory"

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


class JsonlBackend:
    """Appends spans as JSON lines — a simple, dependency-free durable backend."""

    name = "jsonl"

    def __init__(self, path: str) -> None:
        self.path = path

    def export(self, span: Span) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(span)) + "\n")


class OTelBackend:
    """Export spans to OpenTelemetry (LangSmith/Phoenix/Langfuse via OTLP). Lazy/optional."""

    name = "otel"

    def __init__(self, tracer=None) -> None:  # pragma: no cover - optional dependency
        if tracer is None:
            try:
                from opentelemetry import trace
            except ImportError as exc:
                raise ImportError("OTelBackend requires the '[obs]' extra") from exc
            tracer = trace.get_tracer("agent_eval")
        self._tracer = tracer

    def export(self, span: Span) -> None:  # pragma: no cover
        with self._tracer.start_as_current_span(span.name) as otel_span:
            for key, value in span.attributes.items():
                otel_span.set_attribute(key, value)
            otel_span.set_attribute("agent_eval.kind", span.kind)
