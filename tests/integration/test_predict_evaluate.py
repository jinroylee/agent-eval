"""End-to-end: run a real (tiny) LangGraph through predict() then evaluate()."""

import json

import pytest

pytest.importorskip("langgraph")

from agent_eval.config.loader import build_dataset_spec, build_suite  # noqa: E402
from agent_eval.config.schema import ConfigModel  # noqa: E402
from agent_eval.core.registry import default_registry  # noqa: E402
from agent_eval.datasets.base import load_dataset, load_records  # noqa: E402
from agent_eval.harness.predict import predict  # noqa: E402
from agent_eval.offline.runner import evaluate  # noqa: E402

_AGENT = '''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict, total=False):
    input: str
    output: str
    tokens: int

def node(state):
    return {"output": state["input"].upper(), "tokens": len(state["input"])}

_g = StateGraph(S)
_g.add_node("n", node)
_g.add_edge(START, "n")
_g.add_edge("n", END)
graph = _g.compile()
'''


def _config(tmp_path):
    (tmp_path / "agent.py").write_text(_AGENT)
    (tmp_path / "gold.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [{"q": "hello", "ref": "HELLO"}, {"q": "world", "ref": "WORLD"}])
    )
    return ConfigModel(
        agent_type="plain",
        datasets={
            "gold": {"adapter": "jsonl", "path": str(tmp_path / "gold.jsonl"),
                     "field_map": {"input": "q", "expected": "ref"}},
            "predictions": {"adapter": "jsonl", "path": str(tmp_path / "pred.jsonl")},
        },
        prediction={
            "graph": f"{tmp_path / 'agent.py'}:graph",
            "source": "gold", "target": "predictions", "input_key": "input",
            "state_map": {"output": "output", "tokens": "tokens"},
        },
        suites={"response": {"metrics": ["llm_judge"], "gate": {"thresholds": {"llm_judge": 0.5},
                                                                "require_pass": ["llm_judge"]},
                             "dataset": "predictions"}},
    )


def test_predict_fills_predictions(tmp_path):
    cfg = _config(tmp_path)
    records = predict(cfg)
    assert len(records) == 2
    # the graph ran: output is the uppercased input, latency was auto-captured
    first = records[0]
    assert first["output"] == "HELLO" and first["metadata"]["tokens"] == 5
    assert "latency_ms" in first["metadata"]
    # and it was written to the target dataset path
    on_disk = load_records(build_dataset_spec(cfg, "predictions"))
    assert on_disk[0]["output"] == "HELLO"


def test_evaluate_predictions_passes_gate(tmp_path):
    cfg = _config(tmp_path)
    predict(cfg)
    suite = build_suite(cfg, "response", default_registry())
    result = evaluate(suite, load_dataset(build_dataset_spec(cfg, "predictions")))
    # output == reference, so the stub judge scores 1.0 -> gate passes
    assert result.verdict.passed and result.aggregates[0].value == 1.0


_MULTI_RUN_AGENT = '''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict, total=False):
    input: str
    output: str
    sql: str

def node(state):
    return {"output": state["input"].upper(), "sql": f"SELECT '{state['input']}'"}

_g = StateGraph(S)
_g.add_node("n", node)
_g.add_edge(START, "n")
_g.add_edge("n", END)
graph = _g.compile()
'''


def test_predict_n_runs_collects_repeated_generations(tmp_path):
    (tmp_path / "agent.py").write_text(_MULTI_RUN_AGENT)
    (tmp_path / "gold.jsonl").write_text(json.dumps({"q": "hello"}))
    cfg = ConfigModel(
        agent_type="t2s",
        datasets={
            "gold": {"adapter": "jsonl", "path": str(tmp_path / "gold.jsonl"),
                     "field_map": {"input": "q"}},
            "predictions": {"adapter": "jsonl", "path": str(tmp_path / "pred.jsonl")},
        },
        prediction={
            "graph": f"{tmp_path / 'agent.py'}:graph",
            "source": "gold", "target": "predictions", "input_key": "input",
            "state_map": {"output": "output", "sql": "sql"},
            "n_runs": 3,
        },
    )
    records = predict(cfg)
    md = records[0]["metadata"]
    # run 1 stays the canonical prediction; all 3 runs are collected as arrays
    assert records[0]["output"] == "HELLO" and md["sql"] == "SELECT 'hello'"
    assert md["repeated_outputs"] == ["HELLO", "HELLO", "HELLO"]
    assert md["repeated_sql"] == ["SELECT 'hello'", "SELECT 'hello'", "SELECT 'hello'"]


def test_predict_single_run_writes_no_repeated_arrays(tmp_path):
    records = predict(_config(tmp_path))  # n_runs defaults to 1
    assert "repeated_outputs" not in records[0]["metadata"]
    assert "repeated_sql" not in records[0]["metadata"]
