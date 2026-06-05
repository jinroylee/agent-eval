"""Judge backends — pluggable scorers behind one interface.

``FunctionJudge`` wraps a callable (used in tests and for simple rule-based judges).
``DeepEvalJudge`` adapts DeepEval's G-Eval (lazy import; needs the ``[deepeval]`` extra + an LLM).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from agent_eval.core.contracts import EvalContext
from agent_eval.judges.config import JudgeVerdict


@runtime_checkable
class JudgeBackend(Protocol):
    def evaluate(self, criteria: str, ctx: EvalContext) -> JudgeVerdict: ...


class FunctionJudge:
    """Wrap ``fn(criteria, ctx) -> float in [0, 1]`` (or a JudgeVerdict) as a JudgeBackend."""

    def __init__(self, fn: Callable[[str, EvalContext], float | JudgeVerdict]) -> None:
        self._fn = fn

    def evaluate(self, criteria: str, ctx: EvalContext) -> JudgeVerdict:
        out = self._fn(criteria, ctx)
        if isinstance(out, JudgeVerdict):
            return out
        return JudgeVerdict(float(out))


class DeepEvalJudge:
    """Adapter over DeepEval G-Eval (5-point Likert -> 0..1). Requires the optional dependency.

    Not unit-tested here (needs an LLM + credentials); validated via integration when configured.
    """

    def __init__(self, model: str, criteria: str, name: str = "geval") -> None:
        try:
            from deepeval.metrics import GEval  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("DeepEvalJudge requires the '[deepeval]' extra") from exc
        self.model = model
        self.criteria = criteria
        self.name = name

    def evaluate(self, criteria: str, ctx: EvalContext) -> JudgeVerdict:  # pragma: no cover
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        metric = GEval(
            name=self.name,
            criteria=criteria or self.criteria,
            model=self.model,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        )
        tc = LLMTestCase(input=str(ctx.input), actual_output=str(ctx.output))
        metric.measure(tc)
        return JudgeVerdict(float(metric.score), confidence=None, reason=str(metric.reason or ""))
