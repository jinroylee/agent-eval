"""JudgeMetric — an LLM-as-judge scorer (the JUDGE tier).

Confined to subjective quality where no oracle exists (never objective correctness). Supports a
single backend or a diverse-family PoLL panel. Certification is enforced separately at suite build
(see ``judges.governance``), so this metric stays a thin scorer.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric
from agent_eval.judges.backend import JudgeBackend
from agent_eval.judges.panel import poll_pointwise


class JudgeMetric(BaseMetric):
    name = "judge"
    tier = Tier.JUDGE
    requires = frozenset({"output"})
    cost_class = CostClass.EXPENSIVE

    def __init__(
        self,
        backend: JudgeBackend,
        criteria: str,
        panel: Sequence[JudgeBackend] | None = None,
    ) -> None:
        self.backend = backend
        self.criteria = criteria
        self.panel = list(panel or [])

    def _compute(self, ctx: EvalContext) -> MetricResult:
        if self.panel:
            score, confidence = poll_pointwise([self.backend, *self.panel], self.criteria, ctx)
            return MetricResult(self.name, score, passed=None, confidence=confidence)
        v = self.backend.evaluate(self.criteria, ctx)
        return MetricResult(
            self.name, v.score, passed=None, confidence=v.confidence, detail={"reason": v.reason}
        )
