"""Deterministic, reference-grounded metrics — the FREE/cheap tier.

These never call an LLM: exact/regex/set/JSON-shape/numeric checks. They are the preferred
gate for objectively-checkable tasks and the first tier of the runtime critic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric


class ExactMatch(BaseMetric):
    name = "exact_match"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output", "expected"})
    cost_class = CostClass.FREE

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = ctx.output == ctx.expected
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


class RegexMatch(BaseMetric):
    name = "regex_match"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output"})
    cost_class = CostClass.FREE

    def __init__(self, pattern: str, flags: int = 0) -> None:
        self._re = re.compile(pattern, flags)

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = self._re.search(str(ctx.output)) is not None
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


class SetMatch(BaseMetric):
    """Order-insensitive equality of two collections (e.g. tool sets, tag sets)."""

    name = "set_match"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output", "expected"})
    cost_class = CostClass.FREE

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = set(_as_iterable(ctx.output)) == set(_as_iterable(ctx.expected))
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


class NumericTolerance(BaseMetric):
    name = "numeric_tolerance"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output", "expected"})
    cost_class = CostClass.FREE

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = float(tol)

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok = abs(float(ctx.output) - float(ctx.expected)) <= self.tol  # may raise -> trapped
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok)


class JsonShape(BaseMetric):
    """Output (JSON string or dict) must contain all required keys."""

    name = "json_shape"
    tier = Tier.DETERMINISTIC
    requires = frozenset({"output"})
    cost_class = CostClass.FREE

    def __init__(self, required_keys: Iterable[str]) -> None:
        self.required_keys = list(required_keys)

    def _compute(self, ctx: EvalContext) -> MetricResult:
        data = ctx.output if isinstance(ctx.output, dict) else json.loads(ctx.output)
        missing = [k for k in self.required_keys if k not in data]
        ok = not missing
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail={"missing": missing})


def _as_iterable(value) -> Iterable:
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]
