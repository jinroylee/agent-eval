"""P0 acceptance gate: `agent-eval evaluate` runs end-to-end and gates on thresholds."""

import json

from typer.testing import CliRunner

from agent_eval.cli.main import app

runner = CliRunner()


def _setup(tmp_path, threshold):
    rows = [  # 2/3 exact match
        {"q": "1", "a": "x", "e": "x"},
        {"q": "2", "a": "y", "e": "y"},
        {"q": "3", "a": "z", "e": "w"},
    ]
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = f"""
version: 1
agent_type: plain
datasets:
  toy:
    adapter: jsonl
    path: {tmp_path}/d.jsonl
    field_map: {{input: q, output: a, expected: e}}
suites:
  response:
    dataset: toy
    metrics: [{{type: exact_match, name: exact}}]
    gate: {{thresholds: {{exact: {threshold}}}, require_pass: [exact]}}
"""
    p = tmp_path / "c.yaml"
    p.write_text(cfg)
    return p


def test_cli_evaluate_passes_and_shows_ci(tmp_path):
    cfgp = _setup(tmp_path, 0.5)  # 0.667 >= 0.5
    result = runner.invoke(app, ["evaluate", "--config", str(cfgp)])
    assert result.exit_code == 0, result.stdout
    assert "exact" in result.stdout
    assert "PASS" in result.stdout
    assert "[" in result.stdout and "]" in result.stdout  # Wilson CI printed


def test_cli_evaluate_fails_gate_with_nonzero_exit(tmp_path):
    cfgp = _setup(tmp_path, 0.9)  # 0.667 < 0.9
    result = runner.invoke(app, ["evaluate", "--config", str(cfgp)])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout
