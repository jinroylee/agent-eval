"""The offline entrypoint: run a Suite over a dataset and produce a gated, CI'd verdict.

Aggregation is small-sample-correct: binary metrics use a Wilson interval on the pass rate;
continuous metrics use a cluster-robust SE (clustered on ``metadata['cluster_id']`` when present)
so non-iid items don't get artificially tight error bars.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from scipy.stats import norm

from agent_eval.core.contracts import EvalContext, MetricResult
from agent_eval.core.gate import GateVerdict, MetricAggregate, decide_gate
from agent_eval.core.suite import Suite, SuiteResult
from agent_eval.stats.clustered import clustered_se
from agent_eval.stats.intervals import wilson_interval


def evaluate(suite: Suite, dataset: Iterable[EvalContext], alpha: float = 0.05) -> SuiteResult:
    contexts = list(dataset)
    # Run every metric on every context; keep results paired with their context for clustering.
    per_metric: dict[str, list[tuple[MetricResult, EvalContext]]] = {
        m.name: [] for m in suite.metrics
    }
    for ctx in contexts:
        for metric in suite.metrics:
            per_metric[metric.name].append((metric.score(ctx), ctx))

    aggregates = [
        _aggregate(name, paired, alpha) for name, paired in per_metric.items()
    ]
    verdict: GateVerdict = decide_gate(aggregates, suite.gate)
    return SuiteResult(suite.agent_type, suite.category, len(contexts), aggregates, verdict)


def _aggregate(
    name: str, paired: Sequence[tuple[MetricResult, EvalContext]], alpha: float
) -> MetricAggregate:
    valid = [(r, c) for r, c in paired if r.error is None]
    n_errors = len(paired) - len(valid)
    n = len(valid)
    if n == 0:
        return MetricAggregate(name, 0.0, 0.0, 0.0, 0, n_errors, binary=True)

    binary = all(r.passed is not None for r, _ in valid)
    if binary:
        successes = sum(1 for r, _ in valid if r.passed)
        value, lo, hi = wilson_interval(successes, n, alpha=alpha)
        return MetricAggregate(name, value, lo, hi, n, n_errors, binary=True)

    scores = [float(r.score) for r, _ in valid]
    cluster_ids = [c.metadata.get("cluster_id", i) for i, (_, c) in enumerate(valid)]
    mean, se = clustered_se(scores, cluster_ids)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    half = 0.0 if se != se else z * se  # se is nan only for a single cluster
    lo = max(0.0, mean - half)
    hi = min(1.0, mean + half)
    return MetricAggregate(name, mean, lo, hi, n, n_errors, binary=False)
