"""Gate policy and verdict logic.

A gate turns per-metric statistical aggregates into a pass/fail decision. In P0 this is
threshold-based (point estimate vs threshold, with a Wilson CI reported alongside). The
regression path (paired McNemar / bootstrap vs a frozen baseline, with BH-FDR) plugs in later
through the same ``GatePolicy`` fields (``min_effect``, ``significance_alpha``, ``fdr``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class GatePolicy:
    thresholds: Mapping[str, float] = field(default_factory=dict)
    require_pass: Sequence[str] = ()
    min_effect: float = 0.0
    significance_alpha: float = 0.05
    fdr: bool = True


@dataclass
class MetricAggregate:
    metric: str
    value: float  # pass-rate (binary) or mean (continuous)
    ci_low: float
    ci_high: float
    n: int
    n_errors: int
    binary: bool


@dataclass
class GateVerdict:
    passed: bool
    metric_passed: dict[str, bool]
    reasons: list[str]


def decide_gate(aggregates: Sequence[MetricAggregate], policy: GatePolicy) -> GateVerdict:
    """Threshold gate: every metric with a threshold must meet it; require_pass metrics are
    hard blockers (and must have actually run)."""
    metric_passed: dict[str, bool] = {}
    reasons: list[str] = []
    overall = True

    by_name = {a.metric: a for a in aggregates}
    for agg in aggregates:
        threshold = policy.thresholds.get(agg.metric)
        if threshold is None:
            metric_passed[agg.metric] = True  # informational only
            continue
        passed = agg.value >= threshold
        metric_passed[agg.metric] = passed
        if not passed:
            reasons.append(
                f"{agg.metric}={agg.value:.3f} < threshold {threshold:.3f} "
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
