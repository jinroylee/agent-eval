"""agent-eval command-line interface.

- ``predict``  — run your LangGraph agent over a gold dataset to fill in predicted answers.
- ``evaluate`` — score the predictions and gate on thresholds (non-zero exit on failure, for CI).
"""

from __future__ import annotations

from pathlib import Path

import typer

from agent_eval.config.loader import build_dataset_spec, build_suite, load_config
from agent_eval.core.contracts import Aggregation
from agent_eval.core.registry import default_registry
from agent_eval.core.suite import SuiteResult
from agent_eval.datasets.base import load_dataset
from agent_eval.offline.runner import evaluate as run_eval

app = typer.Typer(
    add_completion=False,
    help="agent-eval: offline evaluation framework for LangGraph agents",
)


@app.command()
def predict(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to the YAML config"),
) -> None:
    """Run the agent's LangGraph over the gold dataset and write a predictions dataset."""
    from agent_eval.harness.predict import predict as run_predict

    cfg = load_config(config)
    if cfg.prediction is None:
        typer.echo("Config has no 'prediction' section; nothing to do.")
        raise typer.Exit(2)
    records = run_predict(cfg)
    out_path = cfg.datasets[cfg.prediction.target].path
    typer.echo(f"Wrote {len(records)} prediction(s) to {out_path}")
    if cfg.prediction.generate_csv:
        from agent_eval.harness.preview import csv_preview_path

        typer.echo(f"Wrote human-readable CSV view to {csv_preview_path(out_path)}")


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to the YAML config"),
    category: str | None = typer.Option(None, help="Run a single suite (default: all suites)"),
    dataset: str | None = typer.Option(None, help="Override the dataset name to evaluate on"),
) -> None:
    """Run the offline evaluation gate; exit non-zero if any suite fails its thresholds."""
    cfg = load_config(config)
    registry = default_registry()

    categories = [category] if category else list(cfg.suites)
    if not categories:
        typer.echo("No suites defined in config.")
        raise typer.Exit(1)

    all_passed = True
    for cat in categories:
        if cat not in cfg.suites:
            typer.echo(f"Unknown suite: {cat!r}")
            raise typer.Exit(2)
        suite = build_suite(cfg, cat, registry)
        ds_name = dataset or cfg.suites[cat].dataset or (next(iter(cfg.datasets)) if cfg.datasets else None)
        if ds_name is None:
            typer.echo(f"[{cat}] no dataset configured")
            raise typer.Exit(2)
        result = run_eval(suite, load_dataset(build_dataset_spec(cfg, ds_name)))
        _print_report(result, cfg.suites[cat].gate.thresholds, ds_name)
        all_passed = all_passed and result.verdict.passed

    raise typer.Exit(0 if all_passed else 1)


@app.command()
def version() -> None:
    """Print the installed agent-eval version."""
    from importlib.metadata import version as _v

    typer.echo(_v("agent-eval"))


def _fmt_value(value: float, aggregation: Aggregation) -> str:
    # latency/token metrics report raw magnitudes; quality metrics live in [0, 1].
    return f"{value:>9.1f}" if aggregation is Aggregation.P95 else f"{value:>9.3f}"


def _print_report(result: SuiteResult, thresholds, ds_name: str) -> None:
    typer.echo(f"\n=== {result.agent_type}/{result.category}  dataset={ds_name}  n={result.n_items} ===")
    typer.echo(f"{'metric':<22} {'value':>9}  {'95% CI':>20}  {'thr':>8}  gate")
    for agg in result.aggregates:
        thr = thresholds.get(agg.metric)
        thr_s = f"{thr:.2f}" if thr is not None else "-"
        gate = "info" if thr is None else ("PASS" if result.verdict.metric_passed.get(agg.metric) else "FAIL")
        err = f"  ({agg.n_errors} err)" if agg.n_errors else ""
        ci = f"[{agg.ci_low:.3f}, {agg.ci_high:.3f}]"
        value = _fmt_value(agg.value, agg.aggregation)
        typer.echo(f"{agg.metric:<22} {value}  {ci:>20}  {thr_s:>8}  {gate}{err}")
        if agg.breakdown:  # per-criterion means (criteria-based judge metrics)
            for name, mean in agg.breakdown.items():
                typer.echo(f"  · {name:<20} {mean:>9.3f}")

    overall = "PASS" if result.verdict.passed else "FAIL"
    typer.echo(f"VERDICT: {typer.style(overall, bold=True)}")
    for reason in result.verdict.reasons:
        typer.echo(f"  - {reason}")


if __name__ == "__main__":
    app()
