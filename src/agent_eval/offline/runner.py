"""The offline entrypoint: run a Suite over a dataset and produce a gated, CI'd verdict.

Aggregation is per-metric (``Metric.aggregation``) and small-sample-correct:

- ``RATE`` (binary)  → pass rate with a **Wilson** interval (doesn't under-cover at small n).
- ``MEAN`` (graded)  → mean with a **cluster-robust** SE (set ``metadata['cluster_id']`` for
  non-iid items — RAG questions sharing a passage, turns in one scenario — so the CI isn't tight).
- ``P95`` (latency)  → 95th percentile with a **bootstrap** interval (latency is heavy-tailed).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from scipy.stats import norm

from agent_eval.core.contracts import Aggregation, EvalContext, MetricResult
from agent_eval.core.gate import GateVerdict, MetricAggregate, decide_gate
from agent_eval.core.metric import Metric
from agent_eval.core.suite import Suite, SuiteResult
from agent_eval.stats.clustered import clustered_se
from agent_eval.stats.intervals import percentile_ci, wilson_interval


def evaluate(suite: Suite, dataset: Iterable[EvalContext], alpha: float = 0.05) -> SuiteResult:
    contexts = list(dataset)
    # Run every metric on every context; keep each result paired with its context (for clustering).
    per_metric: dict[str, list[tuple[MetricResult, EvalContext]]] = {m.name: [] for m in suite.metrics}
    for ctx in contexts:
        for metric in suite.metrics:
            per_metric[metric.name].append((metric.score(ctx), ctx))

    aggregates = [_aggregate(m, per_metric[m.name], alpha) for m in suite.metrics]
    verdict: GateVerdict = decide_gate(aggregates, suite.gate)
    return SuiteResult(suite.agent_type, suite.category, len(contexts), aggregates, verdict)


def _criteria_breakdown(
    valid: Sequence[tuple[MetricResult, EvalContext]],
) -> dict[str, float] | None:
    """Mean per criterion over the items whose ``detail['criteria']`` carries it (else None)."""
    per_criterion: dict[str, list[float]] = {}
    for r, _ in valid:
        crit = r.detail.get("criteria")
        if isinstance(crit, Mapping):
            for name, score in crit.items():
                per_criterion.setdefault(str(name), []).append(float(score))
    if not per_criterion:
        return None
    return {name: sum(vs) / len(vs) for name, vs in per_criterion.items()}


def _aggregate(
    metric: Metric, paired: Sequence[tuple[MetricResult, EvalContext]], alpha: float
) -> MetricAggregate:
    valid = [(r, c) for r, c in paired if r.error is None]
    n_errors = len(paired) - len(valid)
    n = len(valid)
    agg, hib = metric.aggregation, metric.higher_is_better
    if n == 0:
        return MetricAggregate(metric.name, 0.0, 0.0, 0.0, 0, n_errors, agg, hib)

    if agg is Aggregation.RATE:
        successes = sum(1 for r, _ in valid if r.passed)
        value, lo, hi = wilson_interval(successes, n, alpha=alpha)
    elif agg is Aggregation.P95:
        value, lo, hi = percentile_ci([float(r.score) for r, _ in valid], q=0.95, alpha=alpha)
        lo = max(0.0, lo)  # latency/token magnitudes can't be negative
    else:  # MEAN
        scores = [float(r.score) for r, _ in valid]
        cluster_ids = [c.metadata.get("cluster_id", i) for i, (_, c) in enumerate(valid)]
        mean, se = clustered_se(scores, cluster_ids)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        half = 0.0 if se != se else z * se  # se is nan only for a single cluster
        value, lo, hi = mean, mean - half, mean + half
        if getattr(metric, "unit_interval", True):
            lo, hi = max(0.0, lo), min(1.0, hi)
        else:
            lo = max(0.0, lo)

    return MetricAggregate(
        metric.name, value, lo, hi, n, n_errors, agg, hib, _criteria_breakdown(valid)
    )
