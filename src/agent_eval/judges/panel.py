"""Panel-of-LLM-judges (PoLL) and pairwise position-bias mitigation.

A diverse-family panel correlates better with humans than a single large judge while costing less,
and cancels the self-preference bias a single judge cannot detect in itself. Pairwise verdicts are
run in both orderings and counted a tie if they flip (swap-and-average), neutralizing position bias.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_eval.core.contracts import EvalContext
from agent_eval.judges.backend import JudgeBackend


def poll_pointwise(
    judges: Sequence[JudgeBackend], criteria: str, ctx: EvalContext
) -> tuple[float, float]:
    """Average pointwise scores across a panel; confidence = 1 - score spread (panel agreement)."""
    if not judges:
        raise ValueError("panel must be non-empty")
    scores = [j.evaluate(criteria, ctx).score for j in judges]
    mean = sum(scores) / len(scores)
    spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
    return mean, 1.0 - spread


def pairwise_swap_average(
    compare: Callable[[str, object, object], int], criteria: str, a: object, b: object
) -> int:
    """Run a pairwise comparison in both orderings. Returns +1 (a), -1 (b), or 0 (flip => tie).

    ``compare(criteria, x, y)`` returns +1 if x is preferred, -1 if y is preferred.
    """
    ab = compare(criteria, a, b)  # +1 => a preferred
    ba = compare(criteria, b, a)  # +1 => b preferred (b is first here)
    if ab > 0 and ba < 0:
        return 1  # a wins under both orderings
    if ab < 0 and ba > 0:
        return -1  # b wins under both orderings
    return 0  # verdict flipped with position => tie
