"""Orchestration trajectory metrics — deterministic tool-sequence matching + BFCL arg validity.

A trajectory is a list of steps ``{"tool": str, "args": dict}`` (in ``EvalContext.trajectory``);
the reference is in ``EvalContext.expected``. Subset mode is a clean hard safety gate (output uses
no tool outside the allowed set); superset mode requires all reference tools were used.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric

_TYPES: dict[str, Any] = {
    "str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict,
}


def _tools(steps) -> list:
    return [s.get("tool") for s in (steps or [])]


def _match(pred: list, ref: list, mode: str) -> bool:
    if mode == "strict":
        return pred == ref
    if mode == "unordered":
        return Counter(pred) == Counter(ref)
    if mode == "subset":
        return set(pred) <= set(ref)
    if mode == "superset":
        return set(ref) <= set(pred)
    raise ValueError(f"unknown trajectory match mode: {mode!r}")


class TrajectoryMatch(BaseMetric):
    name = "trajectory_match"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE

    def __init__(self, mode: str = "unordered") -> None:
        self.mode = mode

    def _compute(self, ctx: EvalContext) -> MetricResult:
        if ctx.expected is None:
            raise ValueError("trajectory_match requires a reference trajectory in 'expected'")
        ok = _match(_tools(ctx.trajectory), _tools(ctx.expected), self.mode)
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail={"mode": self.mode})


class ToolArgValidity(BaseMetric):
    """BFCL-style: each tool call names a known tool, supplies required args, and types match."""

    name = "tool_arg_validity"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE

    def _compute(self, ctx: EvalContext) -> MetricResult:
        schemas = ctx.metadata.get("tool_schemas") or {}
        steps = list(ctx.trajectory or [])
        if not steps:
            return MetricResult(self.name, 1.0, passed=True, detail={"note": "no tool calls"})
        invalid = []
        for i, step in enumerate(steps):
            ok, reason = _validate_call(step, schemas)
            if not ok:
                invalid.append({"step": i, "tool": step.get("tool"), "reason": reason})
        score = 1.0 - len(invalid) / len(steps)
        return MetricResult(self.name, score, passed=not invalid, detail={"invalid": invalid})


def _validate_call(step: Mapping[str, Any], schemas: Mapping[str, Any]) -> tuple[bool, str]:
    tool = step.get("tool")
    if not isinstance(tool, str) or tool not in schemas:
        return False, "unknown tool"
    args = step.get("args") or {}
    spec = schemas[tool]
    for arg, typename in spec.items():
        if arg not in args:
            return False, f"missing arg {arg!r}"
        expected = _TYPES.get(typename)
        if expected is not None and not isinstance(args[arg], expected):
            return False, f"arg {arg!r} expected {typename}"
    return True, ""


def register_trajectory_metrics(registry) -> None:
    registry.register("trajectory_match", lambda p: TrajectoryMatch(**p))
    registry.register("tool_arg_validity", lambda p: ToolArgValidity(**p))
