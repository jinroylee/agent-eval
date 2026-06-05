"""agent-eval command-line interface.

``evaluate`` runs the offline CI gate (exit non-zero on failure, JUnit-friendly) and ``calibrate``
derives runtime-critic thresholds from an offline run and writes them into the config. ``introspect``
needs a live compiled graph, so it is exposed through the Python API (``instrumentation.topology``).
"""

from __future__ import annotations

from pathlib import Path

import typer

from agent_eval.config.loader import build_dataset_spec, build_suite, load_config
from agent_eval.core.registry import default_registry
from agent_eval.core.suite import SuiteResult
from agent_eval.datasets.base import load_dataset
from agent_eval.offline.runner import evaluate as run_eval

app = typer.Typer(
    add_completion=False,
    help="agent-eval: evaluation + runtime-critique framework for LangGraph agents",
)


def _build_registry():
    """Registry with built-in deterministic metrics + all pure families (retrieval, uncertainty,
    trajectory, perf) + T2S (if sqlglot installed) + any plugin metrics (entry points)."""
    from agent_eval.metrics.perf_ import register_perf_metrics
    from agent_eval.metrics.retrieval_ import register_retrieval_metrics
    from agent_eval.metrics.trajectory_ import register_trajectory_metrics
    from agent_eval.metrics.uncertainty_ import register_uncertainty_metrics

    registry = default_registry()
    registry.load_plugins()
    register_retrieval_metrics(registry)
    register_uncertainty_metrics(registry)
    register_trajectory_metrics(registry)
    register_perf_metrics(registry)
    try:  # T2S metrics need the optional sqlglot dependency
        from agent_eval.metrics.ast_query_ import register_t2s_metrics

        register_t2s_metrics(registry)
    except ImportError:
        pass
    return registry


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to the YAML config"),
    category: str | None = typer.Option(None, help="Run a single suite/category (default: all)"),
    dataset: str | None = typer.Option(None, help="Override the dataset name to evaluate on"),
) -> None:
    """Run the offline evaluation gate; exit non-zero if any suite fails its thresholds."""
    cfg = load_config(config)
    registry = _build_registry()

    categories = [category] if category else list(cfg.suites)
    if not categories:
        typer.echo("No suites defined in config.")
        raise typer.Exit(1)

    all_passed = True
    for cat in categories:
        if cat not in cfg.suites:
            typer.echo(f"Unknown suite/category: {cat!r}")
            raise typer.Exit(2)
        suite = build_suite(cfg, cat, registry)
        ds_name = dataset or cfg.suites[cat].dataset or (
            next(iter(cfg.datasets)) if cfg.datasets else None
        )
        if ds_name is None:
            typer.echo(f"[{cat}] no dataset configured")
            raise typer.Exit(2)
        result = run_eval(suite, load_dataset(build_dataset_spec(cfg, ds_name)))
        _print_report(result, cfg.suites[cat].gate.thresholds, ds_name)
        all_passed = all_passed and result.verdict.passed

    raise typer.Exit(0 if all_passed else 1)


@app.command()
def calibrate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to the YAML config"),
    category: str = typer.Option("search", help="Suite/category to calibrate thresholds from"),
    max_false_fail: float = typer.Option(
        0.05, help="Bound the false-fail rate on the (good) calibration set"
    ),
) -> None:
    """Derive runtime-critic thresholds (tau) from an offline run and write them into the config."""
    from agent_eval.offline.calibrate import calibrate_and_write

    tau = calibrate_and_write(
        config, category, max_false_fail=max_false_fail, registry=_build_registry()
    )
    typer.echo(f"Wrote calibrated tau into {config} [runtime_critic.tau]:")
    for name, value in tau.items():
        typer.echo(f"  {name}: {value:.4f}")


@app.command()
def version() -> None:
    """Print the installed agent-eval version."""
    from importlib.metadata import version as _v

    typer.echo(_v("agent-eval"))


def _print_report(result: SuiteResult, thresholds, ds_name: str) -> None:
    typer.echo(
        f"\n=== {result.agent_type}/{result.category}  dataset={ds_name}  n={result.n_items} ==="
    )
    typer.echo(f"{'metric':<24} {'value':>7}  {'95% CI':>18}  {'thr':>5}  gate")
    for agg in result.aggregates:
        thr = thresholds.get(agg.metric)
        thr_s = f"{thr:.2f}" if thr is not None else "   -"
        if thr is None:
            gate = "info"
        else:
            gate = "PASS" if result.verdict.metric_passed.get(agg.metric) else "FAIL"
        err = f"  ({agg.n_errors} err)" if agg.n_errors else ""
        ci = f"[{agg.ci_low:.3f}, {agg.ci_high:.3f}]"
        typer.echo(f"{agg.metric:<24} {agg.value:>7.3f}  {ci:>18}  {thr_s:>5}  {gate}{err}")

    overall = "PASS" if result.verdict.passed else "FAIL"
    typer.echo(f"VERDICT: {typer.style(overall, bold=True)}")
    for reason in result.verdict.reasons:
        typer.echo(f"  - {reason}")


if __name__ == "__main__":
    app()
