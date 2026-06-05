"""Plain-agent runtime critic: a single-pass hallucination gate.

No retrieval or trajectory to verify, so the critic relies on a label-free uncertainty signal
(SelfCheckGPT-style consistency over sampled answers). Low consistency -> escalate/abstain.
Self-Refine, where used, is confined to style/format/safety — never to "fix" correctness.
"""

from __future__ import annotations

from agent_eval.config.schema import ConfigModel
from agent_eval.core.contracts import Tier
from agent_eval.metrics.uncertainty_ import SelfCheckConsistency
from agent_eval.runtime.critic import Critic
from agent_eval.runtime.policy import CriticPolicy


def build_plain_critic(cfg: ConfigModel, threshold: float = 0.6) -> Critic:
    metrics = {Tier.UNCERTAINTY: [SelfCheckConsistency(threshold=threshold)]}
    rc = cfg.runtime_critic
    if rc is None:
        policy = CriticPolicy(tiers=[Tier.UNCERTAINTY], tau={"selfcheck_consistency": 0.5})
    else:
        policy = CriticPolicy(
            tiers=rc.tiers or [Tier.UNCERTAINTY],
            tau=dict(rc.tau),
            max_retries=rc.max_retries,
            fallback=rc.fallback,
            latency_budget_ms=rc.latency_budget_ms,
        )
    return Critic(metrics, policy)
