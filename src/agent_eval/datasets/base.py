"""Bring-your-own-dataset core: normalize arbitrary user records into EvalContext.

Users point a ``field_map`` (target -> source column/key) at their data. Targets that are
EvalContext fields populate the context; any other target lands in ``metadata`` (e.g. ``schema``,
``db_ref``); unmapped columns are preserved in ``metadata`` too so nothing is lost.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_eval.core.contracts import EvalContext
from agent_eval.core.errors import ConfigError

KNOWN_FIELDS = frozenset({"input", "output", "expected", "retrieved_context", "trajectory"})


@dataclass
class DatasetSpec:
    adapter: str  # "jsonl" | "tabular"
    path: str
    format: str | None = None  # csv | excel | parquet | jsonl (inferred from extension if None)
    field_map: Mapping[str, str] = field(default_factory=dict)
    include_unmapped: bool = True


@runtime_checkable
class DatasetAdapter(Protocol):
    def __iter__(self) -> Iterator[EvalContext]: ...


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def record_to_context(
    record: Mapping[str, Any],
    field_map: Mapping[str, str] | None = None,
    include_unmapped: bool = True,
) -> EvalContext:
    """Map one source record to an EvalContext via ``field_map`` (target -> source key)."""
    if not field_map:
        field_map = {k: k for k in KNOWN_FIELDS if k in record}

    kwargs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    used: set[str] = set()
    for target, source in field_map.items():
        used.add(source)
        value = record.get(source)
        if target == "metadata":
            if isinstance(value, Mapping):
                metadata.update(value)
        elif target in KNOWN_FIELDS:
            kwargs[target] = value
        else:
            metadata[target] = value

    if include_unmapped:
        for key, value in record.items():
            if key not in used and key not in metadata:
                metadata[key] = value

    return EvalContext(
        input=kwargs.get("input"),
        output=kwargs.get("output"),
        expected=kwargs.get("expected"),
        retrieved_context=_as_tuple(kwargs.get("retrieved_context")),
        trajectory=_as_tuple(kwargs.get("trajectory")),
        metadata=metadata,
    )


def load_dataset(spec: DatasetSpec) -> Iterable[EvalContext]:
    """Resolve a DatasetSpec to an iterable adapter over EvalContexts."""
    if spec.adapter == "jsonl":
        from agent_eval.datasets.jsonl import JsonlAdapter

        return JsonlAdapter(spec.path, spec.field_map, spec.include_unmapped)
    if spec.adapter == "tabular":
        from agent_eval.datasets.tabular import TabularAdapter

        return TabularAdapter(spec.path, spec.field_map, spec.format, spec.include_unmapped)
    raise ConfigError(f"unknown dataset adapter {spec.adapter!r}")
