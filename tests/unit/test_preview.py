"""CSV preview sidecar: canonical fields flattened, metadata.* columns, heavy fields summarized."""

import csv

from agent_eval.harness.preview import csv_preview_path, write_csv_preview


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_preview_path_is_sibling_csv():
    assert csv_preview_path("a/b/predictions.jsonl") == "a/b/predictions.csv"


def test_flattens_metadata_and_writes_utf8(tmp_path):
    jsonl = tmp_path / "predictions.jsonl"
    records = [
        {"input": "질문", "output": "답변", "expected": "정답",
         "metadata": {"sql": "SELECT 1", "tokens": 5, "latency_ms": 12.5}},
    ]
    csv_path = write_csv_preview(str(jsonl), records)
    assert csv_path == str(tmp_path / "predictions.csv")

    rows = _read(csv_path)
    assert rows[0]["input"] == "질문" and rows[0]["output"] == "답변"
    assert rows[0]["metadata.sql"] == "SELECT 1"
    assert rows[0]["metadata.tokens"] == "5"
    # non-ASCII survives the round trip via utf-8-sig (BOM so Excel renders Korean, not mojibake)
    assert "질문" in (tmp_path / "predictions.csv").read_text(encoding="utf-8-sig")


def test_heavy_result_sets_summarized_not_inlined(tmp_path):
    jsonl = tmp_path / "p.jsonl"
    rows = write_csv_preview(str(jsonl), [
        {"input": "q", "metadata": {"execution_result": [[1], [2], [3]], "sql": "SELECT x"}},
    ]) and _read(str(tmp_path / "p.csv"))
    assert rows[0]["metadata.execution_result"] == "<3 rows>"
    assert rows[0]["metadata.sql"] == "SELECT x"


def test_lists_json_encoded_and_ragged_columns_unioned(tmp_path):
    jsonl = tmp_path / "p.jsonl"
    write_csv_preview(str(jsonl), [
        {"input": "a", "metadata": {"retrieved_ids": ["d1", "d2"]}},
        {"input": "b", "metadata": {"tokens": 7}},          # different metadata keys -> union of columns
    ])
    rows = _read(str(tmp_path / "p.csv"))
    assert rows[0]["metadata.retrieved_ids"] == '["d1", "d2"]'  # short list inlined as JSON
    assert rows[0]["metadata.tokens"] == ""                     # missing cell blank, not dropped
    assert rows[1]["metadata.tokens"] == "7"
