"""RAG retrieval metrics — pure, reference-grounded (no LLM).

Retrieval quality caps downstream answer quality, so Recall@k is the hard search-gate. Inputs are
ranked retrieved ids (``output`` or ``metadata['retrieved_ids']``) vs gold relevant ids
(``expected`` or ``metadata['relevant_ids']``). RetrievalSufficiency is the CRAG-style runtime gate
that triggers re-retrieval.
"""

from __future__ import annotations

import math

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric


def _retrieved(ctx: EvalContext) -> list:
    if isinstance(ctx.output, (list, tuple)):
        return list(ctx.output)
    return list(ctx.metadata.get("retrieved_ids", []))


def _relevant(ctx: EvalContext) -> set:
    rel = ctx.expected if isinstance(ctx.expected, (list, tuple)) else ctx.metadata.get("relevant_ids")
    if not rel:
        raise ValueError("no gold relevant ids (expected / metadata['relevant_ids'])")
    return set(rel)


class _RetrievalMetric(BaseMetric):
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE

    def __init__(self, k: int = 10) -> None:
        self.k = k


class RecallAtK(_RetrievalMetric):
    name = "recall_at_k"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        top = set(_retrieved(ctx)[: self.k])
        return MetricResult(self.name, len(top & rel) / len(rel), passed=None)


class PrecisionAtK(_RetrievalMetric):
    name = "precision_at_k"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        top = _retrieved(ctx)[: self.k]
        denom = len(top) if top else self.k
        return MetricResult(self.name, len(set(top) & rel) / denom, passed=None)


class HitAtK(_RetrievalMetric):
    name = "hit_at_k"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        hit = any(d in rel for d in _retrieved(ctx)[: self.k])
        return MetricResult(self.name, 1.0 if hit else 0.0, passed=hit)


class MRR(_RetrievalMetric):
    name = "mrr"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        rr = 0.0
        for i, d in enumerate(_retrieved(ctx)):
            if d in rel:
                rr = 1.0 / (i + 1)
                break
        return MetricResult(self.name, rr, passed=None)


class NdcgAtK(_RetrievalMetric):
    name = "ndcg_at_k"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        top = _retrieved(ctx)[: self.k]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top) if d in rel)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), self.k)))
        return MetricResult(self.name, dcg / idcg if idcg > 0 else 0.0, passed=None)


class RetrievalSufficiency(_RetrievalMetric):
    """CRAG-style gate: is the retrieved context good enough, or should we re-retrieve?"""

    name = "retrieval_sufficiency"
    cost_class = CostClass.CHEAP

    def __init__(self, k: int = 10, min_recall: float = 0.5) -> None:
        super().__init__(k)
        self.min_recall = min_recall

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        recall = len(set(_retrieved(ctx)[: self.k]) & rel) / len(rel)
        ok = recall >= self.min_recall
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail={"recall": recall})


def register_retrieval_metrics(registry) -> None:
    registry.register("recall_at_k", lambda p: RecallAtK(**p))
    registry.register("precision_at_k", lambda p: PrecisionAtK(**p))
    registry.register("hit_at_k", lambda p: HitAtK(**p))
    registry.register("mrr", lambda p: MRR(**p))
    registry.register("ndcg_at_k", lambda p: NdcgAtK(**p))
    registry.register("retrieval_sufficiency", lambda p: RetrievalSufficiency(**p))
