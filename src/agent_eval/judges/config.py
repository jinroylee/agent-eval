"""Judge configuration — a judge is a *pinned, certified measurement instrument*, not just a call.

Pinning model + prompt_version + rubric_id means a provider model update (which silently shifts
scores) invalidates the certification rather than corrupting a gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class Protocol(StrEnum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


@dataclass(frozen=True)
class JudgeVerdict:
    score: float  # normalized 0..1
    confidence: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class JudgeConfig:
    id: str
    model: str  # pinned
    prompt_version: str = "v1"  # pinned
    rubric_id: str = ""  # pinned
    criteria: str = ""  # the rubric text given to the judge
    protocol: Protocol = Protocol.POINTWISE
    panel: Sequence[str] = field(default_factory=tuple)  # other judge ids => PoLL
    confidence_policy: str = "logprob"
    certification_ref: str | None = None

    def fingerprint(self) -> str:
        """Identity that a certification is bound to; changes invalidate the cert (staleness)."""
        return f"{self.model}|{self.prompt_version}|{self.rubric_id}"
