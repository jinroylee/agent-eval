"""Tests for runtime-threshold calibration (the offline->runtime tau round-trip)."""

import json

from agent_eval.config.loader import load_config
from agent_eval.offline.calibrate import calibrate_and_write, calibrate_tau


def test_calibrate_tau_bounds_false_fail_rate():
    scores = {"conf": [0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.88, 0.82, 0.91, 0.77]}
    tau = calibrate_tau(scores, max_false_fail=0.1)
    assert 0.6 <= tau["conf"] <= 0.78  # ~10th percentile operating point


def test_calibrate_binary_metric_requires_pass():
    tau = calibrate_tau({"ex": [1.0] * 10}, max_false_fail=0.05)
    assert tau["ex"] == 1.0


def test_calibrate_and_write_roundtrips_into_config(tmp_path):
    rows = [{"q": "1", "a": "x", "e": "x"}, {"q": "2", "a": "y", "e": "y"}]  # all match
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg_text = f"""
version: 1
agent_type: plain
datasets:
  d: {{adapter: jsonl, path: {tmp_path}/d.jsonl, field_map: {{input: q, output: a, expected: e}}}}
suites:
  search:
    dataset: d
    metrics: [{{type: exact_match, name: exact_match}}]
    gate: {{thresholds: {{exact_match: 0.5}}}}
runtime_critic: {{tiers: [deterministic], tau: {{}}}}
"""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(cfg_text)

    tau = calibrate_and_write(cfg_path, "search", max_false_fail=0.05)
    assert tau["exact_match"] == 1.0
    reloaded = load_config(cfg_path)
    assert reloaded.runtime_critic.tau["exact_match"] == 1.0  # written back into the same config
