"""Runtime hallucination signals — label-free, computed over pre-sampled answers.

SelfCheckGPT-style consistency and a semantic-entropy proxy operate on candidate samples the agent
already produced (``metadata['samples']``), so the metric itself makes no LLM call on the hot path.
Low consistency / high entropy flags a likely hallucination -> the critic abstains/retries.

The semantic clustering here is lexical (token Jaccard); a production deployment can swap in an
NLI/embedding clusterer behind the same metric without changing the contract.
"""

from __future__ import annotations

import math

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric


def _tokens(s) -> set[str]:
    return set(str(s).lower().split())


def _jaccard(a, b) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cluster(items: list[str], threshold: float) -> list[list[str]]:
    clusters: list[list[str]] = []
    for item in items:
        for cluster in clusters:
            if _jaccard(item, cluster[0]) >= threshold:
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


class SelfCheckConsistency(BaseMetric):
    """Fraction of sampled answers consistent with the main answer (SelfCheckGPT, black-box)."""

    name = "selfcheck_consistency"
    tier = Tier.UNCERTAINTY
    requires = frozenset({"output"})
    cost_class = CostClass.CHEAP

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def _compute(self, ctx: EvalContext) -> MetricResult:
        samples = list(ctx.metadata.get("samples") or [])
        if not samples:
            return MetricResult(self.name, 1.0, passed=None, confidence=1.0, detail={"note": "no samples"})
        agree = [1.0 if _jaccard(ctx.output, s) >= self.threshold else 0.0 for s in samples]
        consistency = sum(agree) / len(agree)
        return MetricResult(self.name, consistency, passed=None, confidence=consistency)


class SemanticEntropy(BaseMetric):
    """Certainty = 1 - normalized entropy over meaning-clusters of the samples (semantic entropy)."""

    name = "semantic_entropy"
    tier = Tier.UNCERTAINTY
    requires = frozenset({"output"})
    cost_class = CostClass.CHEAP

    def __init__(self, sim_threshold: float = 0.6) -> None:
        self.sim_threshold = sim_threshold

    def _compute(self, ctx: EvalContext) -> MetricResult:
        items = [str(ctx.output)] + [str(s) for s in (ctx.metadata.get("samples") or [])]
        n = len(items)
        if n <= 1:
            return MetricResult(self.name, 1.0, passed=None, confidence=1.0)
        clusters = _cluster(items, self.sim_threshold)
        probs = [len(c) / n for c in clusters]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        certainty = 1.0 - entropy / math.log(n)
        return MetricResult(self.name, certainty, passed=None, confidence=certainty)


def register_uncertainty_metrics(registry) -> None:
    registry.register("selfcheck_consistency", lambda p: SelfCheckConsistency(**p))
    registry.register("semantic_entropy", lambda p: SemanticEntropy(**p))
