"""Dataset normalization: field_map projection, metadata merge, and the jsonl/tabular adapters."""

import json

from agent_eval.datasets.base import DatasetSpec, load_dataset, load_records, to_canonical


def test_field_map_projects_to_canonical_fields():
    rec = {"q": "hi", "ans": "yo", "gold": "yo", "doc": "d1"}
    c = to_canonical(rec, {"input": "q", "output": "ans", "expected": "gold", "retrieved_ids": "doc"})
    assert c["input"] == "hi" and c["output"] == "yo" and c["expected"] == "yo"
    assert c["metadata"]["retrieved_ids"] == "d1"  # non-context target -> metadata


def test_nested_metadata_dict_is_merged():
    rec = {"input": "q", "output": "a", "metadata": {"sql": "SELECT 1", "tokens": 5}}
    c = to_canonical(rec)
    assert c["metadata"]["sql"] == "SELECT 1" and c["metadata"]["tokens"] == 5


def test_unmapped_columns_preserved_in_metadata():
    c = to_canonical({"input": "q", "extra": 42}, {"input": "input"})
    assert c["metadata"]["extra"] == 42


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"q": "1", "rel": ["d1"]}, {"q": "2", "rel": ["d2"]},
    ]))
    spec = DatasetSpec("jsonl", str(p), field_map={"input": "q", "relevant_ids": "rel"})
    ctxs = list(load_dataset(spec))
    assert len(ctxs) == 2 and ctxs[0].input == "1" and ctxs[0].metadata["relevant_ids"] == ["d1"]


def test_load_records_returns_raw_dicts(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"q": "1", "a": "2"}))
    recs = load_records(DatasetSpec("jsonl", str(p)))
    assert recs == [{"q": "1", "a": "2"}]


def test_tabular_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("question,gold\nhi,yo\n")
    spec = DatasetSpec("tabular", str(p), field_map={"input": "question", "expected": "gold"})
    ctxs = list(load_dataset(spec))
    assert ctxs[0].input == "hi" and ctxs[0].expected == "yo"
