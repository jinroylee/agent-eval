"""Certify a judge before trusting it in a gate.

A judge is a measurement instrument: validate it on a human-labeled meta-set BEFORE wiring it into
a gate (CALM/JudgeBench lesson). Certification requires (a) human-agreement kappa above a floor and
(b) a robustness/consistency floor. The record is bound to the judge's fingerprint so a model or
prompt change makes it stale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent_eval.core.contracts import EvalContext
from agent_eval.judges.backend import JudgeBackend
from agent_eval.stats.agreement import cohen_kappa


@dataclass(frozen=True)
class CertificationRecord:
    id: str
    judge_id: str
    fingerprint: str
    kappa: float
    bias_robustness: float
    n: int
    passed: bool


def certify_judge(
    judge_id: str,
    backend: JudgeBackend,
    criteria: str,
    samples: Sequence[tuple[EvalContext, float]],
    fingerprint: str = "",
    kappa_floor: float = 0.6,
    bias_floor: float = 0.8,
    repeats: int = 2,
    cert_id: str | None = None,
) -> CertificationRecord:
    """Score the meta-set, compute human agreement + run-to-run consistency, apply floors.

    ``samples`` are ``(EvalContext, human_label in [0, 1])``. ``repeats`` re-runs the judge per
    item to estimate stochastic consistency (a partial bias/robustness proxy).
    """
    judge_scores: list[float] = []
    consistencies: list[float] = []
    for ctx, _ in samples:
        runs = [backend.evaluate(criteria, ctx).score for _ in range(repeats)]
        judge_scores.append(sum(runs) / len(runs))
        consistencies.append(1.0 - (max(runs) - min(runs)))

    human_bin = [1 if h >= 0.5 else 0 for _, h in samples]
    judge_bin = [1 if s >= 0.5 else 0 for s in judge_scores]
    kappa = cohen_kappa(judge_bin, human_bin) if samples else 0.0
    bias_robustness = sum(consistencies) / len(consistencies) if consistencies else 0.0
    passed = bool(samples) and kappa >= kappa_floor and bias_robustness >= bias_floor
    return CertificationRecord(
        cert_id or f"cert::{judge_id}", judge_id, fingerprint, kappa, bias_robustness, len(samples), passed
    )
