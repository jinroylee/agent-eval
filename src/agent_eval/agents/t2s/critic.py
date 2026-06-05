"""T2S runtime critic: deterministic tier-1 = AST validity + schema linking + dry-run execution.

The execution harness IS the verifier here — no LLM grades query correctness. On a soft failure the
critic loop retries with the (grounded) error injected; exhaustion falls back / escalates.
"""

from __future__ import annotations

from agent_eval.config.schema import ConfigModel
from agent_eval.core.contracts import Tier
from agent_eval.metrics.ast_query_ import AstValid, ExecValidity, SchemaLinking
from agent_eval.runtime.critic import Critic
from agent_eval.runtime.policy import CriticPolicy


def build_t2s_critic(cfg: ConfigModel, dialect: str = "sqlite") -> Critic:
    deterministic = [
        AstValid(dialect=dialect),
        SchemaLinking(dialect=dialect),
        ExecValidity(dialect=dialect),
    ]
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
    return Critic({Tier.DETERMINISTIC: deterministic}, policy)
