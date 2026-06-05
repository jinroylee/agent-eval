"""agentevals adapter — LangGraph-native trajectory matchers (lazy/optional).

Wraps ``create_trajectory_match_evaluator`` (strict/unordered/subset/superset) behind the Metric
contract. Lazy like the other heavy adapters: importing this module never requires ``agentevals``.
The in-house ``trajectory_.TrajectoryMatch`` is the dependency-free default; use this when you want
agentevals' message-format semantics or its graph-trajectory variants.
"""

from __future__ import annotations

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric


class AgentEvalsTrajectoryMetric(BaseMetric):
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE

    def __init__(self, mode: str = "unordered", name: str | None = None) -> None:
        self.mode = mode
        self.name = name or f"agentevals_{mode}"

    def _evaluator(self):  # pragma: no cover - requires the optional dependency
        try:
            from agentevals.trajectory.match import create_trajectory_match_evaluator
        except ImportError as exc:
            raise ImportError("AgentEvalsTrajectoryMetric requires the '[agentevals]' extra") from exc
        return create_trajectory_match_evaluator(trajectory_match_mode=self.mode)

    def _compute(self, ctx: EvalContext) -> MetricResult:  # pragma: no cover
        result = self._evaluator()(
            outputs=list(ctx.trajectory), reference_outputs=list(ctx.expected or [])
        )
        score = 1.0 if result.get("score") else 0.0
        return MetricResult(self.name, score, passed=bool(result.get("score")),
                            detail={"comment": result.get("comment", "")})
