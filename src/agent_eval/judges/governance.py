"""Judge governance: enforce certification at the gate, and PPI-correct judge-scored CIs.

Two safeguards from the research: (1) a judge may gate a release only if it is *certified and not
stale*, and (2) confidence intervals on judge-scored metrics are corrected against a small human
gold set via Prediction-Powered Inference, so they stay valid despite judge bias.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_eval.config.schema import ConfigModel
from agent_eval.core.errors import UncertifiedJudge
from agent_eval.judges.registry import JudgeRegistry
from agent_eval.stats.ppi import ppi_interval


def assert_gated_judges_certified(
    cfg: ConfigModel, category: str, judge_registry: JudgeRegistry
) -> None:
    """Raise UncertifiedJudge if any judge metric *used by the gate* lacks a valid, fresh cert.

    Judge metrics that are only informational (not in ``gate.thresholds`` / ``require_pass``) are
    not enforced — you can observe an uncertified judge, you just can't gate on it.
    """
    suite = cfg.suites[category]
    gated = set(suite.gate.thresholds) | set(suite.gate.require_pass)
    for spec in suite.metrics:
        if not isinstance(spec, dict) or spec.get("type") != "judge":
            continue
        name = spec.get("name") or "judge"
        if name not in gated:
            continue
        judge_id = (spec.get("params") or {}).get("judge")
        jc = cfg.judges.get(judge_id) if judge_id else None
        if jc is None:
            raise UncertifiedJudge(
                f"gated judge metric {name!r} references unknown judge {judge_id!r}"
            )
        fingerprint = f"{jc.model}|{jc.prompt_version}|{jc.rubric_id}"
        if not judge_registry.is_certified(jc.certification_ref, fingerprint):
            raise UncertifiedJudge(
                f"judge {judge_id!r} is not certified (or its certification is stale) "
                f"for gating suite {category!r}"
            )


def judge_gate_ci(
    judge_scores: Sequence[float],
    gold_labels: Sequence[float],
    gold_idx: Sequence[int],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """PPI-corrected ``(point, lo, hi)`` for a judge-scored metric against a human gold subset."""
    return ppi_interval(judge_scores, gold_labels, gold_idx, alpha=alpha)
