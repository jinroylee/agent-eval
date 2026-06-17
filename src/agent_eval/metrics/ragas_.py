"""RAGAS adapter — wrap RAGAS metrics (faithfulness, answer relevancy, context precision/recall).

Lazy like DeepEvalJudge: the heavy dependency and its LLM/embeddings are only touched when a
RagasMetric is actually constructed/run, so importing this module is always safe. Not unit-tested
live (needs the ``[ragas]`` extra + an LLM); validated when configured in a real deployment.
"""

from __future__ import annotations

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric


class RagasMetric(BaseMetric):
    """Adapt a single RAGAS metric (by name) to the Metric contract via SingleTurnSample."""

    tier = Tier.JUDGE
    requires = frozenset({"output"})
    cost_class = CostClass.EXPENSIVE

    def __init__(self, metric: str = "faithfulness", name: str | None = None) -> None:
        self.metric_name = metric
        self.name = name or f"ragas_{metric}"

    def _build(self):  # pragma: no cover - requires the optional dependency
        try:
            from ragas import metrics as ragas_metrics
        except ImportError as exc:
            raise ImportError("RagasMetric requires the '[ragas]' extra") from exc
        registry = {
            "faithfulness": "Faithfulness",
            "context_precision": "LLMContextPrecisionWithReference",
        }
        cls_name = registry.get(self.metric_name, self.metric_name)
        return getattr(ragas_metrics, cls_name)()

    def _compute(self, ctx: EvalContext) -> MetricResult:  # pragma: no cover
        from ragas.dataset_schema import SingleTurnSample

        sample = SingleTurnSample(
            user_input=str(ctx.input),
            response=str(ctx.output),
            retrieved_contexts=list(ctx.retrieved_context) or None,
            reference=str(ctx.expected) if ctx.expected is not None else None,
        )
        score = self._build().single_turn_score(sample)
        return MetricResult(self.name, float(score), passed=None)
