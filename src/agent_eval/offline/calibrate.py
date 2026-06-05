"""Calibrate runtime-critic thresholds (tau) from an offline run — the offline->runtime link.

For each metric, tau is the operating point that bounds the false-fail rate on the (good)
calibration set: the ``max_false_fail`` empirical quantile of the metric's scores. Writing it back
into ``runtime_critic.tau`` means the in-flight critic accepts/retries using thresholds the offline
study actually justified, so offline and runtime can't drift.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from agent_eval.config.loader import (
    build_dataset_spec,
    build_suite,
    load_config,
    write_calibrated_tau,
)
from agent_eval.core.contracts import EvalContext
from agent_eval.core.registry import MetricRegistry, default_registry
from agent_eval.core.suite import Suite
from agent_eval.datasets.base import load_dataset


def collect_scores(suite: Suite, dataset: Iterable[EvalContext]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {m.name: [] for m in suite.metrics}
    for ctx in dataset:
        for metric in suite.metrics:
            r = metric.score(ctx)
            if r.error is None:
                scores[metric.name].append(float(r.score))
    return scores


def calibrate_tau(
    scores_by_metric: Mapping[str, list[float]], max_false_fail: float = 0.05
) -> dict[str, float]:
    tau: dict[str, float] = {}
    for metric, scores in scores_by_metric.items():
        if scores:
            tau[metric] = float(np.quantile(scores, max_false_fail))
    return tau


def calibrate_and_write(
    config_path: str | Path,
    category: str,
    max_false_fail: float = 0.05,
    registry: MetricRegistry | None = None,
) -> dict[str, float]:
    """Run the suite over its calibration dataset, derive tau, and write it into the config."""
    cfg = load_config(config_path)
    registry = registry or default_registry()
    suite = build_suite(cfg, category, registry)
    ds_name = cfg.suites[category].dataset or next(iter(cfg.datasets))
    data = load_dataset(build_dataset_spec(cfg, ds_name))
    tau = calibrate_tau(collect_scores(suite, data), max_false_fail)
    write_calibrated_tau(config_path, tau)
    return tau
