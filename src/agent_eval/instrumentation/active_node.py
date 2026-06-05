"""Active (in-flight) instrumentation: a critic node that routes a LangGraph via Command.

``make_critic_node`` returns a node function you add to a raw StateGraph and reach via a conditional
edge. It runs the critic on the current state and emits ``Command(goto=...)`` to ACCEPT (continue),
RETRY (loop back, injecting the grounded critique + bumping the attempt counter), or FALL BACK /
escalate — the active counterpart to the observational path, both sharing the same Critic.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import Command

from agent_eval.core.contracts import Decision, EvalContext
from agent_eval.runtime.critic import Critic


def make_critic_node(
    critic: Critic,
    build_ctx: Callable[[dict], EvalContext],
    on_accept: str,
    on_retry: str,
    on_fallback: str,
    attempt_key: str = "_critic_attempt",
    critique_key: str = "_critique",
) -> Callable[[dict], Command]:
    def critic_node(state: dict) -> Command:
        attempt = state.get(attempt_key, 0)
        decision, assessment = critic.decide(build_ctx(state), attempt=attempt)
        if decision is Decision.ACCEPT:
            return Command(goto=on_accept)
        if decision is Decision.RETRY:
            return Command(
                goto=on_retry,
                update={attempt_key: attempt + 1, critique_key: assessment.critique},
            )
        return Command(goto=on_fallback)  # FALLBACK / ESCALATE / ABSTAIN

    return critic_node
