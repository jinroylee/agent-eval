"""Runtime-critic policy — the tiered decision configuration (calibrated offline, read at runtime)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agent_eval.core.contracts import Tier


@dataclass
class CriticPolicy:
    tiers: Sequence[Tier | str] = (Tier.DETERMINISTIC, Tier.UNCERTAINTY, Tier.JUDGE)
    tau: Mapping[str, float] = field(default_factory=dict)  # per-metric thresholds (calibrated offline)
    max_retries: int = 2
    fallback: str = "abstain"
    latency_budget_ms: float = 1500.0
    # Self-Refine is permitted only for these subjective aspects, never to fix hard correctness.
    self_refine_only_for: Sequence[str] = ("style", "format", "safety")

    def ordered_tiers(self) -> list[Tier]:
        return [t if isinstance(t, Tier) else Tier(t) for t in self.tiers]
