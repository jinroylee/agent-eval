"""Tests for the observability boundary (fan-out) and the anytime-valid drift monitor."""

import json

import numpy as np

from agent_eval.obs.adapters import JsonlBackend, MemoryBackend
from agent_eval.obs.boundary import Span, TraceBoundary
from agent_eval.obs.drift import DriftMonitor


def test_boundary_fans_out_to_multiple_backends():
    boundary = TraceBoundary()
    a, b = MemoryBackend(), MemoryBackend()
    boundary.register(a)
    boundary.register(b)
    boundary.emit(Span(name="generate_sql", kind="llm", attributes={"gen_ai.usage.output_tokens": 12}))
    assert len(a.spans) == 1 and len(b.spans) == 1
    assert a.spans[0].name == "generate_sql"
    assert b.spans[0].attributes["gen_ai.usage.output_tokens"] == 12


def test_jsonl_backend_writes_spans(tmp_path):
    path = tmp_path / "spans.jsonl"
    boundary = TraceBoundary()
    boundary.register(JsonlBackend(str(path)))
    boundary.emit(Span(name="retrieve", kind="tool", attributes={"k": 5}))
    boundary.emit(Span(name="graph", kind="graph", attributes={}))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["name"] == "retrieve"


def test_drift_monitor_fires_on_injected_regression():
    rng = np.random.default_rng(0)
    good = list(rng.binomial(1, 0.95, 40).astype(float))  # healthy
    bad = list(rng.binomial(1, 0.4, 120).astype(float))  # regression
    mon = DriftMonitor(baseline=0.85, alpha=0.05, min_n=20)
    fired = any(mon.observe(x).fired for x in good + bad)
    assert fired


def test_drift_monitor_respects_type_one_under_null():
    alpha = 0.05
    rng = np.random.default_rng(3)
    runs, false_alarms = 200, 0
    for _ in range(runs):
        mon = DriftMonitor(baseline=0.8, alpha=alpha, min_n=20)  # true rate == baseline (null)
        stream = rng.binomial(1, 0.8, 150).astype(float)
        if any(mon.observe(float(x)).fired for x in stream):
            false_alarms += 1
    assert false_alarms / runs <= alpha  # peeking-safe: no inflated false-alarm rate


def test_drift_monitor_deterministic_sampling():
    mon = DriftMonitor(baseline=0.9, sample_rate=0.5, min_n=1)
    for _ in range(100):
        mon.observe(1.0)
    assert 48 <= mon.n_sampled <= 52  # ~50% of traffic recorded, deterministically
