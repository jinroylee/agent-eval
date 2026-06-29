"""CLI acceptance: `evaluate` gates with the right exit code; `predict` fills a dataset."""

import json

import pytest
from typer.testing import CliRunner

from agent_eval.cli.main import app

runner = CliRunner()


def _eval_config(tmp_path, threshold):
    # Predictions already on disk (canonical shape) — no agent run needed for `evaluate`.
    rows = [
        {"input": "q1", "metadata": {"retrieved_ids": ["d1"], "relevant_ids": ["d1"]}},
        {"input": "q2", "metadata": {"retrieved_ids": ["d2"], "relevant_ids": ["d2"]}},
        {"input": "q3", "metadata": {"retrieved_ids": ["x"], "relevant_ids": ["d3"]}},  # miss
    ]
    (tmp_path / "pred.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = f"""
version: 1
agent_type: rag
defaults: {{k: 5}}
datasets:
  predictions: {{adapter: jsonl, path: {tmp_path}/pred.jsonl}}
suites:
  retrieval:
    dataset: predictions
    metrics: [recall_at_k]
    gate: {{thresholds: {{recall_at_k: {threshold}}}, require_pass: [recall_at_k]}}
"""
    p = tmp_path / "c.yaml"
    p.write_text(cfg)
    return p


def test_evaluate_passes(tmp_path):
    result = runner.invoke(app, ["evaluate", "-c", str(_eval_config(tmp_path, 0.5))])  # mean recall 2/3
    assert result.exit_code == 0, result.stdout
    assert "recall_at_k" in result.stdout and "PASS" in result.stdout


def test_evaluate_fails_with_nonzero_exit(tmp_path):
    result = runner.invoke(app, ["evaluate", "-c", str(_eval_config(tmp_path, 0.9))])
    assert result.exit_code == 1 and "FAIL" in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_predict_then_evaluate_cli(tmp_path):
    pytest.importorskip("langgraph")
    (tmp_path / "agent.py").write_text(
        "from typing import TypedDict\n"
        "from langgraph.graph import StateGraph, START, END\n"
        "class S(TypedDict, total=False):\n    input: str\n    output: str\n"
        "def node(s): return {'output': s['input']}\n"
        "_g = StateGraph(S); _g.add_node('n', node); _g.add_edge(START,'n'); _g.add_edge('n',END)\n"
        "graph = _g.compile()\n"
    )
    (tmp_path / "gold.jsonl").write_text(json.dumps({"q": "hi", "ref": "hi"}))
    cfg = f"""
version: 1
agent_type: plain
datasets:
  gold: {{adapter: jsonl, path: {tmp_path}/gold.jsonl, field_map: {{input: q, expected: ref}}}}
  predictions: {{adapter: jsonl, path: {tmp_path}/pred.jsonl}}
prediction:
  graph: {tmp_path}/agent.py:graph
  source: gold
  target: predictions
  input_key: input
  state_map: {{output: output}}
suites:
  response:
    dataset: predictions
    metrics: [llm_judge]
    gate: {{thresholds: {{llm_judge: 0.5}}, require_pass: [llm_judge]}}
"""
    (tmp_path / "c.yaml").write_text(cfg)
    pred = runner.invoke(app, ["predict", "-c", str(tmp_path / "c.yaml")])
    assert pred.exit_code == 0, pred.stdout
    assert (tmp_path / "pred.jsonl").exists()
    ev = runner.invoke(app, ["evaluate", "-c", str(tmp_path / "c.yaml")])
    assert ev.exit_code == 0, ev.stdout
