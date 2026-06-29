"""Bring-your-own-dataset core: normalize arbitrary records into the canonical eval shape.

A ``field_map`` (target -> source column/key) projects user records onto the canonical fields the
metrics read. Targets that are :class:`EvalContext` fields populate the context; every other target
lands in ``metadata`` (e.g. ``relevant_ids``, ``sql``, ``db_ref``). A nested ``metadata`` dict in the
record is merged in. Unmapped columns are preserved in ``metadata`` so nothing is silently dropped.

The prediction harness writes records already in canonical shape, so they load with no field_map.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_eval.core.contracts import EvalContext
from agent_eval.core.errors import ConfigError

# EvalContext fields that can be a field_map/state_map target; anything else -> metadata.
CONTEXT_FIELDS = frozenset({"input", "output", "expected", "retrieved_context"})


@dataclass
class DatasetSpec:
    adapter: str  # "jsonl" | "tabular"
    path: str
    format: str | None = None  # csv | tsv | excel | parquet (inferred from extension if None)
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


def to_canonical(
    record: Mapping[str, Any],
    field_map: Mapping[str, str] | None = None,
    include_unmapped: bool = True,
) -> dict[str, Any]:
    """Project a source record onto ``{input, output, expected, retrieved_context, metadata}``."""
    if not field_map:
        field_map = {k: k for k in CONTEXT_FIELDS if k in record}

    out: dict[str, Any] = {"metadata": {}}
    metadata: dict[str, Any] = out["metadata"]
    used: set[str] = set()
    for target, source in field_map.items():
        used.add(source)
        value = record.get(source)
        if target == "metadata":
            if isinstance(value, Mapping):
                metadata.update(value)
        elif target in CONTEXT_FIELDS:
            out[target] = value
        else:
            metadata[target] = value

    # A nested metadata dict in the record (canonical records have one) is merged in.
    if isinstance(record.get("metadata"), Mapping) and "metadata" not in field_map:
        metadata.update(record["metadata"])
        used.add("metadata")

    if include_unmapped:
        for key, value in record.items():
            if key not in used and key not in metadata and key not in CONTEXT_FIELDS:
                metadata[key] = value
    return out


def record_to_context(
    record: Mapping[str, Any],
    field_map: Mapping[str, str] | None = None,
    include_unmapped: bool = True,
) -> EvalContext:
    """Map one source record to an EvalContext via ``field_map`` (target -> source key)."""
    c = to_canonical(record, field_map, include_unmapped)
    return EvalContext(
        input=c.get("input"),
        output=c.get("output"),
        expected=c.get("expected"),
        retrieved_context=_as_tuple(c.get("retrieved_context")),
        metadata=c["metadata"],
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


def load_records(spec: DatasetSpec) -> list[dict]:
    """Load the *raw* source records (dicts), before field-mapping — used by the prediction harness."""
    if spec.adapter == "jsonl":
        from agent_eval.datasets.jsonl import read_jsonl_records

        return read_jsonl_records(spec.path)
    if spec.adapter == "tabular":
        from agent_eval.datasets.tabular import read_tabular_records

        return read_tabular_records(spec.path, spec.format)
    raise ConfigError(f"unknown dataset adapter {spec.adapter!r}")
