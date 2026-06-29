"""Fallback strategies invoked when the critic exhausts retries or hits a degenerate loop.

A strategy maps ``(ctx, assessment) -> (Decision, value)`` — e.g. abstain, escalate to a human, or
return a cached/degraded answer. Registered by name so a future config can reference one.
"""

from __future__ import annotations

from agent_eval.core.contracts import EvalContext
from agent_eval.runtime.critic import Assessment, Decision, FallbackFn


class FallbackRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, FallbackFn] = {}

    def register(self, name: str, fn: FallbackFn) -> None:
        self._strategies[name] = fn

    def get(self, name: str) -> FallbackFn | None:
        return self._strategies.get(name)


def _abstain(ctx: EvalContext, a: Assessment) -> tuple[Decision, None]:
    return Decision.ABSTAIN, None


def _escalate(ctx: EvalContext, a: Assessment) -> tuple[Decision, None]:
    return Decision.ESCALATE, None


def default_fallbacks() -> FallbackRegistry:
    reg = FallbackRegistry()
    reg.register("abstain", _abstain)
    reg.register("escalate", _escalate)
    return reg
