"""Tests for the YAML config schema + loader (the config-first surface)."""

from agent_eval.config.loader import (
    build_dataset_spec,
    build_suite,
    load_config,
    write_calibrated_tau,
)
from agent_eval.core.contracts import EvalContext, Level, MetricResult, Tier
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import default_registry
from agent_eval.offline.runner import evaluate


class _ExactMatch(BaseMetric):
    """Local stand-in metric (the deterministic catalog was removed) for the loader tests."""

    name = "exact_match"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output", "expected"})

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = ctx.output == ctx.expected
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)

CONFIG = """
version: 1
agent_type: plain
datasets:
  toy:
    adapter: jsonl
    path: ${DATA_PATH}
    field_map: {input: q, output: a, expected: e}
suites:
  response:
    target: {level: graph, selector: "*"}
    metrics:
      - {type: exact_match, name: exact}
    gate:
      thresholds: {exact: 0.5}
      require_pass: [exact]
runtime_critic:
  tiers: [deterministic]
  tau: {exact: 1.0}
"""


def _write(tmp_path, text=CONFIG):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_load_config_parses_structure(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.agent_type == "plain"
    assert "response" in cfg.suites
    assert cfg.runtime_critic.tau["exact"] == 1.0


def test_env_interpolation_in_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", "/data/toy.jsonl")
    cfg = load_config(_write(tmp_path))
    spec = build_dataset_spec(cfg, "toy")
    assert spec.path == "/data/toy.jsonl"
    assert spec.field_map["input"] == "q"


def test_build_suite_constructs_runnable_suite(tmp_path):
    cfg = load_config(_write(tmp_path))
    reg = default_registry()
    reg.register("exact_match", lambda p: _ExactMatch())  # CONFIG gates on exact_match
    suite = build_suite(cfg, "response", reg)
    assert suite.metrics[0].name == "exact"
    assert suite.gate.thresholds["exact"] == 0.5
    assert suite.target.level is Level.GRAPH
    res = evaluate(suite, [EvalContext(input="q", output="a", expected="a")])
    assert res.verdict.passed is True


def test_write_calibrated_tau_roundtrips(tmp_path):
    p = _write(tmp_path)
    write_calibrated_tau(p, {"exact": 0.9, "schema_linking": 1.0})
    cfg = load_config(p)
    assert cfg.runtime_critic.tau["exact"] == 0.9
    assert cfg.runtime_critic.tau["schema_linking"] == 1.0
