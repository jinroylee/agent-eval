"""Gate policy and verdict logic.

A gate turns per-metric aggregates into a pass/fail decision: every metric that has a configured
threshold must meet it (direction-aware — ``recall_at_k >= 0.7`` but ``p95_latency <= 2000``), and
``require_pass`` metrics are hard ship-blockers that must have actually run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent_eval.core.contracts import Aggregation


@dataclass
class GatePolicy:
    thresholds: Mapping[str, float] = field(default_factory=dict)
    require_pass: Sequence[str] = ()


@dataclass
class MetricAggregate:
    metric: str
    value: float  # pass-rate (RATE), mean (MEAN), or 95th percentile (P95)
    ci_low: float
    ci_high: float
    n: int
    n_errors: int
    aggregation: Aggregation
    higher_is_better: bool = True


def _meets(value: float, threshold: float, higher_is_better: bool) -> bool:
    return value >= threshold if higher_is_better else value <= threshold


@dataclass
class GateVerdict:
    passed: bool
    metric_passed: dict[str, bool]
    reasons: list[str]


def decide_gate(aggregates: Sequence[MetricAggregate], policy: GatePolicy) -> GateVerdict:
    """Threshold gate: every metric with a threshold must meet it (in its own direction);
    ``require_pass`` metrics are hard blockers (and must have actually produced an aggregate)."""
    metric_passed: dict[str, bool] = {}
    reasons: list[str] = []
    overall = True

    by_name = {a.metric: a for a in aggregates}
    for agg in aggregates:
        threshold = policy.thresholds.get(agg.metric)
        if threshold is None:
            metric_passed[agg.metric] = True  # informational only
        else:
            passed = _meets(agg.value, threshold, agg.higher_is_better)
            metric_passed[agg.metric] = passed
            if not passed:
                op = ">=" if agg.higher_is_better else "<="
                reasons.append(
                    f"{agg.metric}={agg.value:.3f} fails {op} {threshold:.3f} "
                    f"(95% CI [{agg.ci_low:.3f}, {agg.ci_high:.3f}], n={agg.n})"
                )
                overall = False
        if agg.n_errors > 0:
            reasons.append(f"{agg.metric}: {agg.n_errors} item(s) errored")

    for name in policy.require_pass:
        if name not in by_name:
            reasons.append(f"required metric {name!r} did not run")
            metric_passed.setdefault(name, False)
            overall = False
        elif not metric_passed.get(name, False):
            overall = False

    return GateVerdict(overall, metric_passed, reasons)
