"""Paired regression gate: block a merge only on a statistically significant, large-enough drop.

Binary correctness (execution/AST match) uses paired McNemar on discordant pairs; continuous
scores use a paired bootstrap. A drop must be (a) negative, (b) significant, and (c) at least
``min_effect`` in magnitude — so we neither ship spurious +1% gains nor flag noise as regressions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from agent_eval.stats.tests import mcnemar, paired_bootstrap


@dataclass
class RegressionResult:
    metric: str
    baseline_rate: float
    candidate_rate: float
    delta: float
    pvalue: float
    n_discordant: int
    significant_drop: bool


def mcnemar_regression(
    baseline_pass: Sequence[bool],
    candidate_pass: Sequence[bool],
    metric: str = "",
    alpha: float = 0.05,
    min_effect: float = 0.0,
) -> RegressionResult:
    bp, cp = list(baseline_pass), list(candidate_pass)
    if len(bp) != len(cp):
        raise ValueError("baseline and candidate must be aligned (paired)")
    b = sum(1 for x, y in zip(bp, cp, strict=True) if x and not y)  # regressions
    c = sum(1 for x, y in zip(bp, cp, strict=True) if (not x) and y)  # improvements
    res = mcnemar(b, c)
    base_rate = float(np.mean([1.0 if x else 0.0 for x in bp]))
    cand_rate = float(np.mean([1.0 if y else 0.0 for y in cp]))
    delta = cand_rate - base_rate
    sig = (delta < 0) and (res.n_discordant > 0) and (res.pvalue < alpha) and (abs(delta) >= min_effect)
    return RegressionResult(metric, base_rate, cand_rate, delta, float(res.pvalue), res.n_discordant, sig)


def bootstrap_regression(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    metric: str = "",
    alpha: float = 0.05,
    min_effect: float = 0.0,
    rng: np.random.Generator | None = None,
) -> RegressionResult:
    base = np.asarray(baseline_scores, dtype=float)
    cand = np.asarray(candidate_scores, dtype=float)
    if base.shape != cand.shape:
        raise ValueError("baseline and candidate must be aligned (paired)")
    diffs = cand - base
    res = paired_bootstrap(diffs, alpha=alpha, rng=rng)
    delta = res.mean_diff
    sig = (delta < 0) and (res.hi < 0) and (abs(delta) >= min_effect)  # CI entirely below 0
    return RegressionResult(
        metric, float(base.mean()), float(cand.mean()), float(delta), float(res.pvalue), diffs.size, sig
    )
