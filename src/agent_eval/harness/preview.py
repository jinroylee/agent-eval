"""Human-readable CSV sidecar for prediction records (write-only, derived from the JSONL).

``agent-eval predict`` always writes the canonical predictions as JSONL — the lossless,
schema-free artifact that ``evaluate`` reads. When ``prediction.generate_csv`` is set, it *also*
writes a flattened CSV next to it purely for eyeballing a run in a spreadsheet. The CSV is a
**derived, write-only view** — never an evaluate input: nested ``metadata`` is flattened to
``metadata.<key>`` columns and bulky result sets are summarized rather than inlined, so it trades
fidelity for readability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Canonical top-level fields, in display order, before the flattened ``metadata.*`` columns.
_CANONICAL_ORDER = ("input", "output", "expected", "retrieved_context")
# Structured payloads too bulky for a spreadsheet cell: summarized as ``<N rows>`` instead of inlined.
_HEAVY_FIELDS = frozenset({"execution_result", "gold_execution_result"})


def csv_preview_path(jsonl_path: str) -> str:
    """Sibling CSV path for a prediction JSONL: ``predictions.jsonl`` -> ``predictions.preview.csv``.

    The ``.preview.`` marker signals a view, not data, so nobody points a suite ``source`` at it.
    """
    return f"{Path(jsonl_path).with_suffix('')}.preview.csv"


def _cell(key: str, value: Any) -> str:
    """Render one value as a spreadsheet-friendly cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    if key in _HEAVY_FIELDS and isinstance(value, (list, tuple)):
        return f"<{len(value)} rows>"
    # lists / dicts / anything else -> compact JSON (ensure_ascii=False keeps non-ASCII legible).
    return json.dumps(value, ensure_ascii=False, default=str)


def _flatten(record: dict) -> dict[str, str]:
    """Flatten one canonical record to ``{column: cell}``, metadata under ``metadata.<key>``."""
    row: dict[str, str] = {}
    for fld in _CANONICAL_ORDER:
        if fld in record:
            row[fld] = _cell(fld, record[fld])
    for key, value in (record.get("metadata") or {}).items():
        row[f"metadata.{key}"] = _cell(key, value)
    return row


def write_csv_preview(jsonl_path: str, records: list[dict]) -> str:
    """Write a UTF-8 CSV view of prediction ``records`` next to the JSONL; return the CSV path.

    Columns are the canonical fields present (in a fixed reading order) followed by the union of
    ``metadata.*`` keys across all records, sorted for a stable header on ragged data.
    """
    rows = [_flatten(r) for r in records]
    ordered = [f for f in _CANONICAL_ORDER if any(f in r for r in rows)]
    meta_cols = sorted({c for r in rows for c in r if c.startswith("metadata.")})
    fieldnames = ordered + meta_cols

    csv_path = csv_preview_path(jsonl_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
