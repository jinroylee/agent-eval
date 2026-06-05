"""Fallback strategies invoked when the critic exhausts retries or hits a degenerate loop.

A strategy maps ``(ctx, assessment) -> (Decision, value)`` — e.g. abstain, escalate to a human, or
return a cached/degraded answer. Strategies are registered by name and referenced from config.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_eval.core.contracts import Decision, EvalContext

FallbackFn = Callable[[EvalContext, Any], tuple[Decision, Any]]


class FallbackRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, FallbackFn] = {}

    def register(self, name: str, fn: FallbackFn) -> None:
        self._strategies[name] = fn

    def get(self, name: str) -> FallbackFn | None:
        return self._strategies.get(name)


def default_fallbacks() -> FallbackRegistry:
    reg = FallbackRegistry()
    reg.register("abstain", lambda ctx, a: (Decision.ABSTAIN, None))
    reg.register("escalate", lambda ctx, a: (Decision.ESCALATE, None))
    return reg
