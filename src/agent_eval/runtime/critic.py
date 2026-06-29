"""Runtime critique — the in-flight counterpart to the offline gate (FOUNDATION).

> **Status:** this is a foundation for the planned runtime-critique mode, kept intentionally separate
> from the offline `predict`/`evaluate` path (nothing in the offline flow imports it). It is wired
> against the current metric contract so it stays green and ready to build on.

Where the offline runner scores a *dataset* and gates a release, the critic scores a *single
in-flight step* with the **same `Metric` objects** and turns the scores into a decision — so a
LangGraph node can self-correct before emitting a bad result. Design principles, preserved for the
build-out:

- **Cheap-first escalation** — run FREE/CHEAP grounded checks (e.g. ``ast_valid``) before the
  EXPENSIVE LLM judge, and short-circuit once a grounded check fails (don't pay for the judge).
- **Generator/verifier separation** — the critic only consumes *external* verifiers (parse/execution
  checks, retrieval, an independent judge); it never asks the generating model to grade its own
  correctness, which the literature shows doesn't reliably help.
- **Hard vs soft failure** — a failed binary/grounded check (``aggregation == RATE``, e.g.
  ``ast_valid``) is an objective error → **retry**; a low-confidence graded check (e.g.
  ``consistency``, ``faithfulness``) → **escalate** for review.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetricResult
from agent_eval.core.metric import Metric
from agent_eval.runtime.loop_guard import LoopGuard

_COST_ORDER = {CostClass.FREE: 0, CostClass.CHEAP: 1, CostClass.EXPENSIVE: 2}


class Decision(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


@dataclass
class CriticPolicy:
    """Per-metric acceptance thresholds (``tau``) + the retry budget.

    Binary checks default to "must pass" (1.0); graded checks default to 0.5. ``tau`` overrides per
    metric — in the planned design these are the operating points an offline study calibrates.
    """

    tau: Mapping[str, float] = field(default_factory=dict)
    max_retries: int = 2


@dataclass
class Assessment:
    accepted: bool
    failing: list[MetricResult]
    hard_fail: bool  # a grounded/binary check failed (objective error)
    confidence: float | None  # min confidence across graded checks (escalation signal)
    critique: str  # grounded, falsifiable feedback to inject on retry


@dataclass
class LoopResult:
    decision: Decision
    output: Any
    attempts: int
    history: list[Assessment]


FallbackFn = Callable[[EvalContext, Assessment], tuple[Decision, Any]]


def _is_binary(metric: Metric) -> bool:
    return metric.aggregation is Aggregation.RATE


def _threshold(policy: CriticPolicy, metric: Metric) -> float:
    if metric.name in policy.tau:
        return policy.tau[metric.name]
    return 1.0 if _is_binary(metric) else 0.5


class Critic:
    """Assess one in-flight step with a set of metrics, cheapest-first, and decide what to do."""

    def __init__(self, metrics: Sequence[Metric], policy: CriticPolicy | None = None) -> None:
        self.metrics = sorted(metrics, key=lambda m: _COST_ORDER.get(m.cost_class, 1))
        self.policy = policy or CriticPolicy()

    def assess(self, ctx: EvalContext) -> Assessment:
        failing: list[MetricResult] = []
        confidence: float | None = None
        hard_fail = False
        for metric in self.metrics:
            r = metric.score(ctx)
            if r.error is not None:
                failing.append(r)
                hard_fail = True
            elif _is_binary(metric):
                ok = r.passed if r.passed is not None else r.score >= _threshold(self.policy, metric)
                if not ok:
                    failing.append(r)
                    hard_fail = True
            else:
                conf = r.confidence if r.confidence is not None else r.score
                confidence = conf if confidence is None else min(confidence, conf)
                if conf < _threshold(self.policy, metric):
                    failing.append(r)
            if hard_fail:
                break  # cheap-first: don't run the expensive tiers once a grounded check fails
        return Assessment(not failing, failing, hard_fail, confidence, _build_critique(failing))

    def decide(self, ctx: EvalContext, attempt: int = 0) -> tuple[Decision, Assessment]:
        a = self.assess(ctx)
        if a.accepted:
            return Decision.ACCEPT, a
        if not a.hard_fail:
            return Decision.ESCALATE, a  # passed grounded checks, low confidence → escalate
        if attempt < self.policy.max_retries:
            return Decision.RETRY, a
        return Decision.FALLBACK, a


def critic_loop(
    generate: Callable[[int, str | None], Any],
    build_ctx: Callable[[Any], EvalContext],
    critic: Critic,
    fallback: FallbackFn | None = None,
) -> LoopResult:
    """Generate → grounded-verify → RETRY-with-critique (bounded) → FALLBACK / ESCALATE / ACCEPT.

    ``generate(attempt, critique)`` produces an output (the critique from the previous round is
    injected on retries); ``build_ctx(output)`` wraps it as an EvalContext to score.
    """
    guard = LoopGuard(critic.policy.max_retries)
    attempt = 0
    output = generate(attempt, None)
    guard.remember(output)
    history: list[Assessment] = []
    a = critic.assess(build_ctx(output))

    while True:
        history.append(a)
        if a.accepted:
            return LoopResult(Decision.ACCEPT, output, attempt, history)
        if not a.hard_fail:
            return LoopResult(Decision.ESCALATE, output, attempt, history)
        if guard.is_exhausted(attempt):
            break
        next_output = generate(attempt + 1, a.critique)
        if guard.is_repeat(next_output):
            break  # no progress / oscillation
        attempt += 1
        output = next_output
        guard.remember(output)
        a = critic.assess(build_ctx(output))

    if fallback is not None:
        decision, value = fallback(build_ctx(output), a)
        return LoopResult(decision, value, attempt, history)
    return LoopResult(Decision.FALLBACK, output, attempt, history)


def _build_critique(failing: list[MetricResult]) -> str:
    parts = []
    for r in failing:
        msg = r.error or r.detail.get("parse_error") or r.detail.get("reason") or ""
        parts.append(f"{r.metric}: {msg}".rstrip(": ").strip())
    return "; ".join(parts)
