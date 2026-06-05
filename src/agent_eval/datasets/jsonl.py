"""JSON / JSONL dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from agent_eval.core.contracts import EvalContext
from agent_eval.datasets.base import record_to_context


class JsonlAdapter:
    """Reads ``.jsonl`` (one JSON object per line) or a ``.json`` array of objects."""

    def __init__(
        self,
        path: str,
        field_map: Mapping[str, str] | None = None,
        include_unmapped: bool = True,
    ) -> None:
        self.path = path
        self.field_map = field_map or {}
        self.include_unmapped = include_unmapped

    def __iter__(self) -> Iterator[EvalContext]:
        text = Path(self.path).read_text().strip()
        if not text:
            return
        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for rec in records:
            yield record_to_context(rec, self.field_map, self.include_unmapped)
