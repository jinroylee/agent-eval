"""P0 acceptance gate: `agent-eval evaluate` runs end-to-end and gates on thresholds."""

import json

from typer.testing import CliRunner

from agent_eval.cli.main import app

runner = CliRunner()


def _setup(tmp_path, threshold):
    rows = [  # mean Recall@k = 2/3 (two perfect retrievals, one miss)
        {"q": "1", "retrieved": ["d1"], "relevant": ["d1"]},
        {"q": "2", "retrieved": ["d2"], "relevant": ["d2"]},
        {"q": "3", "retrieved": ["x"], "relevant": ["d3"]},
    ]
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = f"""
version: 1
agent_type: rag
datasets:
  toy:
    adapter: jsonl
    path: {tmp_path}/d.jsonl
    field_map: {{input: q, output: retrieved, expected: relevant}}
suites:
  search:
    dataset: toy
    metrics: [{{type: recall_at_k, name: recall, params: {{k: 5}}}}]
    gate: {{thresholds: {{recall: {threshold}}}, require_pass: [recall]}}
"""
    p = tmp_path / "c.yaml"
    p.write_text(cfg)
    return p


def test_cli_evaluate_passes_and_shows_ci(tmp_path):
    cfgp = _setup(tmp_path, 0.5)  # 0.667 >= 0.5
    result = runner.invoke(app, ["evaluate", "--config", str(cfgp)])
    assert result.exit_code == 0, result.stdout
    assert "recall" in result.stdout
    assert "PASS" in result.stdout
    assert "[" in result.stdout and "]" in result.stdout  # CI printed


def test_cli_evaluate_fails_gate_with_nonzero_exit(tmp_path):
    cfgp = _setup(tmp_path, 0.9)  # 0.667 < 0.9
    result = runner.invoke(app, ["evaluate", "--config", str(cfgp)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
