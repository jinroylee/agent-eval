"""Tests for bring-your-own-dataset adapters: record mapping + JSONL/CSV/Parquet loaders."""

import json

import pandas as pd

from agent_eval.datasets.base import DatasetSpec, load_dataset, record_to_context


def test_record_to_context_maps_known_fields_and_metadata():
    rec = {"question": "q1", "gold": "g1", "answer": "a1", "schema": "DDL", "extra": 7}
    ctx = record_to_context(
        rec,
        {"input": "question", "expected": "gold", "output": "answer", "schema": "schema"},
    )
    assert ctx.input == "q1" and ctx.expected == "g1" and ctx.output == "a1"
    assert ctx.metadata["schema"] == "DDL"  # custom target -> metadata
    assert ctx.metadata["extra"] == 7  # unmapped column preserved


def test_record_to_context_identity_when_no_field_map():
    ctx = record_to_context({"input": "q", "output": "a", "expected": "e"})
    assert ctx.input == "q" and ctx.output == "a" and ctx.expected == "e"


def test_jsonl_adapter_roundtrip(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [
        {"question": "q1", "gold": "g1", "answer": "a1"},
        {"question": "q2", "gold": "g2", "answer": "a2"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ds = load_dataset(
        DatasetSpec(
            adapter="jsonl",
            path=str(p),
            field_map={"input": "question", "expected": "gold", "output": "answer"},
        )
    )
    ctxs = list(ds)
    assert len(ctxs) == 2
    assert ctxs[0].input == "q1" and ctxs[1].expected == "g2" and ctxs[1].output == "a2"


def test_tabular_csv_adapter(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame([{"question": "q1", "gold": "g1", "answer": "a1"}]).to_csv(p, index=False)
    ds = load_dataset(
        DatasetSpec(
            adapter="tabular",
            path=str(p),
            format="csv",
            field_map={"input": "question", "expected": "gold", "output": "answer"},
        )
    )
    ctx = list(ds)[0]
    assert ctx.input == "q1" and ctx.output == "a1" and ctx.expected == "g1"


def test_tabular_infers_parquet_from_extension_and_nan_becomes_none(tmp_path):
    p = tmp_path / "d.parquet"
    pd.DataFrame([{"question": "q1", "answer": None}]).to_parquet(p)
    ds = load_dataset(
        DatasetSpec(
            adapter="tabular", path=str(p), field_map={"input": "question", "output": "answer"}
        )
    )
    ctx = list(ds)[0]
    assert ctx.input == "q1"
    assert ctx.output is None  # missing cell normalized to None, not NaN
