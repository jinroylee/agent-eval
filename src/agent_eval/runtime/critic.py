"""The in-flight critic: tiered, cheap-first, with strict generator/verifier separation.

The critic NEVER lets the generating model grade its own correctness — the most replicated result
in the literature is that intrinsic self-correction on objective tasks doesn't reliably help; only a
SOUND EXTERNAL verifier does. So tier-1 is deterministic (parse/schema/execution), and only if a
step passes the grounded checks but lacks an oracle do we consult uncertainty/judge tiers.

``decide()`` makes a single accept/retry/fallback/escalate call; ``critic_loop()`` orchestrates the
bounded retry-with-injected-critique loop a LangGraph node will drive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_eval.core.contracts import Decision, EvalContext, MetricResult, Tier
from agent_eval.core.metric import Metric
from agent_eval.runtime.fallback import FallbackFn
from agent_eval.runtime.loop_guard import LoopGuard
from agent_eval.runtime.policy import CriticPolicy


@dataclass
class Assessment:
    accepted: bool
    failing: list[MetricResult]
    confidence: float | None
    hard_fail: bool  # a deterministic verifier failed (objective error)
    critique: str  # grounded, falsifiable feedback to inject on retry


@dataclass
class LoopResult:
    decision: Decision
    output: Any
    attempts: int
    history: list[Assessment]


class Critic:
    def __init__(self, metrics_by_tier: Mapping[Tier, Sequence[Metric]], policy: CriticPolicy) -> None:
        self.metrics_by_tier = {Tier(k): list(v) for k, v in metrics_by_tier.items()}
        self.policy = policy

    def assess(self, ctx: EvalContext) -> Assessment:
        failing: list[MetricResult] = []
        confidence: float | None = None
        hard_fail = False

        for tier in self.policy.ordered_tiers():
            for metric in self.metrics_by_tier.get(tier, []):
                r = metric.score(ctx)
                tau = self.policy.tau.get(metric.name)
                if tier is Tier.DETERMINISTIC:
                    ok = r.error is None and (
                        r.passed if r.passed is not None else r.score >= (tau if tau is not None else 1.0)
                    )
                    if not ok:
                        failing.append(r)
                        hard_fail = True
                else:
                    conf = r.confidence if r.confidence is not None else r.score
                    confidence = conf if confidence is None else min(confidence, conf)
                    if conf < (tau if tau is not None else 0.5):
                        failing.append(r)
            if tier is Tier.DETERMINISTIC and hard_fail:
                break  # cheap-first: don't run expensive tiers once a grounded check fails

        accepted = not hard_fail and not failing
        return Assessment(accepted, failing, confidence, hard_fail, _build_critique(failing))

    def decide(self, ctx: EvalContext, attempt: int = 0) -> tuple[Decision, Assessment]:
        a = self.assess(ctx)
        if a.accepted:
            return Decision.ACCEPT, a
        if not a.hard_fail:
            return Decision.ESCALATE, a  # passed grounded checks, low confidence -> escalate
        if attempt < self.policy.max_retries:
            return Decision.RETRY, a
        return Decision.FALLBACK, a


def critic_loop(
    generate: Callable[[int, str | None], Any],
    build_ctx: Callable[[Any], EvalContext],
    critic: Critic,
    fallback: FallbackFn | None = None,
) -> LoopResult:
    """Generate -> grounded-verify -> RETRY-with-critique (bounded) -> FALLBACK/ESCALATE/ACCEPT."""
    guard = LoopGuard(critic.policy.max_retries)
    attempt = 0
    output = generate(attempt, None)
    guard.remember(output)
    history: list[Assessment] = []
    ctx = build_ctx(output)
    a = critic.assess(ctx)

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
        ctx = build_ctx(output)
        a = critic.assess(ctx)

    if fallback is not None:
        decision, value = fallback(ctx, a)
        return LoopResult(decision, value, attempt, history)
    return LoopResult(Decision.FALLBACK, output, attempt, history)


def _build_critique(failing: list[MetricResult]) -> str:
    parts = []
    for r in failing:
        msg = r.error or r.detail.get("error") or r.detail.get("parse_error") or ""
        extra = ""
        if r.detail.get("unknown_tables"):
            extra += f" unknown tables {r.detail['unknown_tables']}"
        if r.detail.get("unknown_cols"):
            extra += f" unknown columns {r.detail['unknown_cols']}"
        parts.append(f"{r.metric}: {msg}{extra}".strip())
    return "; ".join(parts)
