"""Agent-specific reliability and economic estimators.

- pass^k (all-k-pass): production cares about consistency, not best-of-k. Use this, not pass@k.
- Cost-of-Pass: expected dollars per *correct* answer; fuses accuracy and price (Erol et al., 2025).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from typing import NamedTuple

import numpy as np


class PassHatKResult(NamedTuple):
    estimate: float
    lo: float
    hi: float
    k: int


def pass_hat_k(
    successes_per_task: Sequence[int],
    trials_per_task: Sequence[int] | int,
    k: int,
    alpha: float = 0.05,
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
) -> PassHatKResult:
    """Unbiased pass^k = P(all k independent trials succeed), averaged over tasks.

    Per task with ``n`` trials and ``c`` successes, the unbiased estimate of all-k-pass is
    ``C(c, k) / C(n, k)`` (tau-bench). CI is a task-level bootstrap.
    """
    succ = np.asarray(list(successes_per_task), dtype=int)
    if isinstance(trials_per_task, int):
        trials = np.full(succ.shape, trials_per_task, dtype=int)
    else:
        trials = np.asarray(list(trials_per_task), dtype=int)
    if succ.shape != trials.shape:
        raise ValueError("successes_per_task and trials_per_task must align")
    if k < 1:
        raise ValueError("k must be >= 1")
    if np.any(trials < k):
        raise ValueError("every task must have at least k trials")

    per_task = np.array(
        [comb(int(c), k) / comb(int(n), k) for c, n in zip(succ, trials, strict=True)],
        dtype=float,
    )
    estimate = float(per_task.mean())

    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.integers(0, per_task.size, size=(n_boot, per_task.size))
    boot = per_task[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return PassHatKResult(estimate, lo, hi, k)


def cost_of_pass(success_indicators: Sequence[int], costs_usd: Sequence[float]) -> float:
    """Expected dollars per correct answer = mean cost / success rate.

    Returns ``inf`` when there are no successes (no price buys a correct answer).
    """
    succ = np.asarray(list(success_indicators), dtype=float)
    costs = np.asarray(list(costs_usd), dtype=float)
    if succ.shape != costs.shape:
        raise ValueError("success_indicators and costs_usd must align")
    if succ.size == 0:
        raise ValueError("inputs must be non-empty")
    total_success = float(succ.sum())
    if total_success <= 0:
        return float("inf")
    return float(costs.sum() / total_success)
