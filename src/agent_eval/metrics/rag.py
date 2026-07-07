"""RAG metrics — retrieval quality (reference-grounded) and answer groundedness (judge-based).

| metric            | GT? | reads (EvalContext)                                  |
|-------------------|-----|------------------------------------------------------|
| ``recall_at_k``   | yes | ``metadata['retrieved_ids']``, ``metadata['relevant_ids']`` |
| ``precision_at_k``| yes | ``metadata['retrieved_ids']``, ``metadata['relevant_ids']`` |
| ``ndcg_at_k``     | yes | ``+ metadata['relevance']`` (optional graded gains)  |
| ``faithfulness``  | no  | ``retrieved_context``, ``output``                    |
| ``consistency``   | no  | ``retrieved_context``, ``output``                    |

Retrieval metrics work on **ids**: ``retrieved_ids`` is the ranked list the retriever returned
(best first) and ``relevant_ids`` is the gold set. ``k`` is fixed per evaluation (a config param),
so the same operating point is measured every run.
"""

from __future__ import annotations

import math

from agent_eval.core.contracts import CostClass, EvalContext, MetaKey, MetricResult
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import BuildContext, MetricRegistry
from agent_eval.judges.backend import JudgeRequest
from agent_eval.metrics.common import JudgeMetric, resolve_judge

# --------------------------------------------------------------------------- retrieval (GT)


def _retrieved(ctx: EvalContext) -> list:
    ids = ctx.metadata.get(MetaKey.RETRIEVED_IDS)
    if ids is None:
        raise ValueError("metadata['retrieved_ids'] (ranked retrieved ids) is required")
    return list(ids)


def _relevant(ctx: EvalContext) -> set:
    rel = ctx.metadata.get(MetaKey.RELEVANT_IDS)
    if not rel:
        raise ValueError("metadata['relevant_ids'] (gold relevant ids) is required")
    return set(rel)


class _RetrievalMetric(BaseMetric):
    cost_class = CostClass.FREE
    requires = frozenset()  # ids live in metadata; checked in _compute

    def __init__(self, k: int = 5) -> None:
        self.k = k


class RecallAtK(_RetrievalMetric):
    """Fraction of the gold relevant ids that appear in the top-k retrieved. Caps answer quality."""

    name = "recall_at_k"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        top = set(_retrieved(ctx)[: self.k])
        return MetricResult(self.name, len(top & rel) / len(rel), passed=None)


class PrecisionAtK(_RetrievalMetric):
    """Fraction of the top-k retrieved that are relevant (how much noise the retriever returns)."""

    name = "precision_at_k"

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        top = _retrieved(ctx)[: self.k]
        denom = len(top) if top else self.k
        return MetricResult(self.name, len(set(top) & rel) / denom, passed=None)


class NdcgAtK(_RetrievalMetric):
    """Normalized DCG@k — rewards putting relevant docs *higher*. Binary gains by default; supply
    ``metadata['relevance']`` (``{id: gain}``) for graded relevance."""

    name = "ndcg_at_k"

    def _gain(self, ctx: EvalContext, doc_id, rel: set) -> float:
        graded = ctx.metadata.get(MetaKey.RELEVANCE)
        if graded:
            return float(graded.get(doc_id, 0.0))
        return 1.0 if doc_id in rel else 0.0

    def _compute(self, ctx: EvalContext) -> MetricResult:
        rel = _relevant(ctx)
        retrieved = _retrieved(ctx)[: self.k]
        gains = [self._gain(ctx, d, rel) for d in retrieved]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(
            (self._gain(ctx, d, rel) for d in set(_retrieved(ctx)) | rel), reverse=True
        )[: self.k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        return MetricResult(self.name, dcg / idcg if idcg else 0.0, passed=None)


# --------------------------------------------------------------------------- groundedness (non-GT, judge)
_FAITHFULNESS_RUBRIC = (
    "Judge whether the RESPONSE TO EVALUATE is faithful to the EVIDENCE: every factual claim in the "
    "response must be supported by (inferable from) the evidence. Penalize claims that are not "
    "supported by the evidence (hallucinations), even if they happen to be true."
)
_CONSISTENCY_RUBRIC = (
    "Judge whether the RESPONSE TO EVALUATE is consistent with the EVIDENCE: it must not contradict "
    "anything stated in the evidence. Penalize any claim that conflicts with the evidence. "
    "(Unlike faithfulness, omitting supported facts is fine — only contradictions are penalized.)"
)


class _ContextJudgeMetric(JudgeMetric):
    """A judge metric grading the response against the retrieved chunks."""

    requires = frozenset({"output", "retrieved_context"})
    _rubric = ""

    def build_request(self, ctx: EvalContext) -> JudgeRequest:
        return JudgeRequest(
            instruction=self._rubric,
            response=str(ctx.output),
            context=tuple(ctx.retrieved_context),
        )


class Faithfulness(_ContextJudgeMetric):
    name = "faithfulness"
    _rubric = _FAITHFULNESS_RUBRIC


class ResponseConsistency(_ContextJudgeMetric):
    name = "consistency"
    _rubric = _CONSISTENCY_RUBRIC


# --------------------------------------------------------------------------- registration
def _k(p: dict, ctx: BuildContext) -> int:
    """k is fixed per evaluation: a metric param, or ``defaults.k``, else 5."""
    return int(p.get("k", ctx.defaults.get("k", 5)))


def register(registry: MetricRegistry) -> None:
    registry.register("recall_at_k", lambda p, ctx: RecallAtK(_k(p, ctx)))
    registry.register("precision_at_k", lambda p, ctx: PrecisionAtK(_k(p, ctx)))
    registry.register("ndcg_at_k", lambda p, ctx: NdcgAtK(_k(p, ctx)))
    registry.register("faithfulness", lambda p, ctx: Faithfulness(resolve_judge(ctx), ctx.panel))
    registry.register("consistency", lambda p, ctx: ResponseConsistency(resolve_judge(ctx), ctx.panel))
