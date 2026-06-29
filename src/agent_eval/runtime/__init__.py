"""Runtime critique (foundation) — an in-flight critic over the same metrics as the offline gate.

Not used by the offline `predict`/`evaluate` path; this is the basis for the planned runtime mode.
See :mod:`agent_eval.runtime.critic` and ``docs/runtime-critic.md``.
"""

from agent_eval.runtime.critic import (
    Assessment,
    Critic,
    CriticPolicy,
    Decision,
    FallbackFn,
    LoopResult,
    critic_loop,
)
from agent_eval.runtime.fallback import FallbackRegistry, default_fallbacks
from agent_eval.runtime.loop_guard import LoopGuard

__all__ = [
    "Critic",
    "CriticPolicy",
    "Decision",
    "Assessment",
    "LoopResult",
    "critic_loop",
    "FallbackFn",
    "LoopGuard",
    "FallbackRegistry",
    "default_fallbacks",
]
