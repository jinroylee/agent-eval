#!/usr/bin/env python
"""Programmatic evaluation of the *plain* example — the import-and-run counterpart to the CLI.

This does exactly what these two commands do::

    uv run agent-eval predict  -c examples/plain/plain.yaml
    uv run agent-eval evaluate -c examples/plain/plain.yaml

…but as an explicit Python script you can read top-to-bottom, edit, and step through in a debugger.
It calls the very same library functions the CLI wraps (``load_config`` → ``predict`` →
``build_suite`` → ``evaluate``); **nothing in ``agent_eval`` is modified**. The YAML stays the single
source of truth for the datasets, ``state_map``, metrics, and thresholds.

Run it (from anywhere — it evaluates the config's repo-root-relative paths like the CLI does)::

    uv run python examples/plain/run_eval.py

Requires the langgraph extra so the harness can drive the agent::

    uv sync --extra langgraph
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_eval.config.loader import build_dataset_spec, build_suite, load_config
from agent_eval.core.contracts import Aggregation
from agent_eval.core.registry import default_registry
from agent_eval.core.suite import SuiteResult
from agent_eval.datasets.base import load_dataset
from agent_eval.harness.predict import predict
from agent_eval.offline.runner import evaluate

# The config uses repo-root-relative paths (e.g. examples/plain/gold.jsonl), exactly like the CLI,
# which resolves them against the current working directory. chdir to the repo root so this script
# works no matter where you launch it from.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "examples/plain/plain.yaml"


def print_report(result: SuiteResult, thresholds, dataset_name: str) -> None:
    """A compact stand-in for the CLI's report table: metric · value · 95% CI · threshold · gate."""
    print(f"\n=== {result.agent_type}/{result.category}  dataset={dataset_name}  n={result.n_items} ===")
    print(f"{'metric':<22} {'value':>9}  {'95% CI':>20}  {'thr':>8}  gate")
    for agg in result.aggregates:
        thr = thresholds.get(agg.metric)
        thr_s = f"{thr:.2f}" if thr is not None else "-"
        gate = "info" if thr is None else ("PASS" if result.verdict.metric_passed.get(agg.metric) else "FAIL")
        # latency/token metrics report raw magnitudes; quality metrics live in [0, 1].
        value = f"{agg.value:>9.1f}" if agg.aggregation is Aggregation.P95 else f"{agg.value:>9.3f}"
        ci = f"[{agg.ci_low:.3f}, {agg.ci_high:.3f}]"
        err = f"  ({agg.n_errors} err)" if agg.n_errors else ""
        print(f"{agg.metric:<22} {value}  {ci:>20}  {thr_s:>8}  {gate}{err}")
    print(f"VERDICT: {'PASS' if result.verdict.passed else 'FAIL'}")
    for reason in result.verdict.reasons:
        print(f"  - {reason}")


def main() -> int:
    os.chdir(REPO_ROOT)
    cfg = load_config(CONFIG)

    # --- 1) predict — run the agent's LangGraph over the gold dataset -> predictions.jsonl --------
    # Same as `agent-eval predict`. (If you already have a predictions file, delete this block and
    # just evaluate it.) `predict` invokes the compiled graph named in cfg.prediction.graph on each
    # gold input, times it, and projects the final graph state onto the canonical eval fields.
    if cfg.prediction is not None:
        records = predict(cfg)
        out_path = cfg.datasets[cfg.prediction.target].path
        print(f"predict: wrote {len(records)} prediction(s) to {out_path}")

    # --- 2) evaluate — score every suite and gate on its thresholds ------------------------------
    # Same as `agent-eval evaluate`. Build one Suite per entry in cfg.suites, run it over the
    # predictions dataset (aggregate per metric with confidence intervals), and apply the gate.
    registry = default_registry()  # common + RAG + T2S metric factories (T2S only if sqlglot present)
    all_passed = True
    for name in cfg.suites:
        suite = build_suite(cfg, name, registry)
        ds_name = cfg.suites[name].dataset or next(iter(cfg.datasets))
        dataset = load_dataset(build_dataset_spec(cfg, ds_name))
        result = evaluate(suite, dataset)
        print_report(result, cfg.suites[name].gate.thresholds, ds_name)
        all_passed = all_passed and result.verdict.passed

    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1  # non-zero exit fails CI, just like the CLI


if __name__ == "__main__":
    raise SystemExit(main())
