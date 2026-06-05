"""Orchestration per-step critic: validate each tool call before it runs.

Tier-1 deterministic = BFCL-style tool-arg validity (+ optional allowlist via subset matching).
Invalid call -> RETRY with the grounded reason injected; exhaustion -> fallback/escalate. This is
the hard safety layer in front of any softer LLM critique.
"""

from __future__ import annotations

from agent_eval.config.schema import ConfigModel
from agent_eval.core.contracts import Tier
from agent_eval.metrics.trajectory_ import ToolArgValidity
from agent_eval.runtime.critic import Critic
from agent_eval.runtime.policy import CriticPolicy


def build_orchestration_critic(cfg: ConfigModel) -> Critic:
    metrics = {Tier.DETERMINISTIC: [ToolArgValidity()]}
    rc = cfg.runtime_critic
    if rc is None:
        policy = CriticPolicy(tiers=[Tier.DETERMINISTIC])
    else:
        policy = CriticPolicy(
            tiers=rc.tiers or [Tier.DETERMINISTIC],
            tau=dict(rc.tau),
            max_retries=rc.max_retries,
            fallback=rc.fallback,
            latency_budget_ms=rc.latency_budget_ms,
        )
    return Critic(metrics, policy)
