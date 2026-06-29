"""JSON / JSONL dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from agent_eval.core.contracts import EvalContext
from agent_eval.datasets.base import record_to_context


def read_jsonl_records(path: str) -> list[dict]:
    """Read ``.jsonl`` (one JSON object per line) or a ``.json`` array of objects into raw dicts."""
    text = Path(path).read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl_records(path: str, records: list[dict]) -> None:
    """Write records as JSONL (one object per line)."""
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")


class JsonlAdapter:
    def __init__(
        self, path: str, field_map: Mapping[str, str] | None = None, include_unmapped: bool = True
    ) -> None:
        self.path = path
        self.field_map = field_map or {}
        self.include_unmapped = include_unmapped

    def __iter__(self) -> Iterator[EvalContext]:
        for rec in read_jsonl_records(self.path):
            yield record_to_context(rec, self.field_map, self.include_unmapped)
