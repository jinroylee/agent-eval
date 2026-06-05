"""RAG runtime critic: CRAG-style retrieval sufficiency gate + groundedness uncertainty.

Tier-1 deterministic = is the retrieved context sufficient (Recall@k >= floor)? If not, the critic
loop re-retrieves. Tier-2 uncertainty = semantic-entropy groundedness flag on the generated answer.
"""

from __future__ import annotations

from agent_eval.config.schema import ConfigModel
from agent_eval.core.contracts import Tier
from agent_eval.core.metric import Metric
from agent_eval.metrics.retrieval_ import RetrievalSufficiency
from agent_eval.metrics.uncertainty_ import SemanticEntropy
from agent_eval.runtime.critic import Critic
from agent_eval.runtime.policy import CriticPolicy


def build_rag_critic(cfg: ConfigModel, k: int = 10, min_recall: float = 0.5) -> Critic:
    metrics: dict[Tier, list[Metric]] = {
        Tier.DETERMINISTIC: [RetrievalSufficiency(k=k, min_recall=min_recall)],
        Tier.UNCERTAINTY: [SemanticEntropy()],
    }
    rc = cfg.runtime_critic
    if rc is None:
        policy = CriticPolicy(tiers=[Tier.DETERMINISTIC, Tier.UNCERTAINTY])
    else:
        policy = CriticPolicy(
            tiers=rc.tiers or [Tier.DETERMINISTIC, Tier.UNCERTAINTY],
            tau=dict(rc.tau),
            max_retries=rc.max_retries,
            fallback=rc.fallback,
            latency_budget_ms=rc.latency_budget_ms,
        )
    return Critic(metrics, policy)
