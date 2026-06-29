"""Panel-of-LLM-judges (PoLL).

A small, diverse panel of cheaper judges correlates with human ratings as well as (or better than)
a single large judge, while costing less and cancelling the self-preference bias a lone judge can't
see in itself (Verga et al., 2024). We average the panel's pointwise scores and report panel
agreement (``1 - spread``) as the confidence, which drives nothing automatically but flags items
where the judges disagree and a human should look.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_eval.judges.backend import JudgeBackend, JudgeRequest


def poll_pointwise(judges: Sequence[JudgeBackend], request: JudgeRequest) -> tuple[float, float]:
    """Average pointwise scores across the panel. Returns ``(mean_score, agreement)``.

    ``agreement = 1 - (max - min)`` over the panel's scores (1.0 for a single judge).
    """
    if not judges:
        raise ValueError("panel must be non-empty")
    scores = [j.evaluate(request).score for j in judges]
    mean = sum(scores) / len(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    return mean, 1.0 - spread
