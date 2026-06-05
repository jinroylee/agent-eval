"""Significance tests for the regression gate and multi-version ranking.

Regression gates block a merge only on a *statistically significant* drop (paired McNemar for
binary correctness; paired bootstrap / Wilcoxon for continuous scores), with Benjamini-Hochberg
controlling false discoveries across the many metrics tested per PR.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy import stats as _sp
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.multitest import multipletests


class McnemarResult(NamedTuple):
    statistic: float
    pvalue: float
    n_discordant: int


def mcnemar(b: int, c: int, exact: bool | None = None) -> McnemarResult:
    """Paired McNemar test from discordant counts.

    ``b`` = baseline-correct & candidate-wrong, ``c`` = baseline-wrong & candidate-correct.
    Uses the exact binomial test for small samples (default), the chi-square form otherwise.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n_disc = b + c
    use_exact = (n_disc < 25) if exact is None else exact
    table = [[0, b], [c, 0]]
    res = _sm_mcnemar(table, exact=use_exact, correction=True)
    return McnemarResult(float(res.statistic), float(res.pvalue), n_disc)


def benjamini_hochberg(
    pvals: Sequence[float], alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control. Returns ``(reject_mask, qvalues)``."""
    arr = np.asarray(list(pvals), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    reject, qvals, _, _ = multipletests(arr, alpha=alpha, method="fdr_bh")
    return reject, qvals


class BootstrapResult(NamedTuple):
    mean_diff: float
    lo: float
    hi: float
    pvalue: float


def paired_bootstrap(
    diffs,
    alpha: float = 0.05,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Paired bootstrap on per-item differences (candidate - baseline) on a shared question list.

    Returns the mean difference, a ``(1 - alpha)`` percentile CI, and a two-sided p-value for
    H0: mean difference = 0.
    """
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        raise ValueError("diffs must be non-empty")
    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    p = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    return BootstrapResult(float(arr.mean()), lo, hi, min(p, 1.0))


class WilcoxonResult(NamedTuple):
    statistic: float
    pvalue: float


def wilcoxon(a, b) -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired continuous scores (non-parametric)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    res = _sp.wilcoxon(a, b)
    return WilcoxonResult(float(res.statistic), float(res.pvalue))


class BradleyTerryResult(NamedTuple):
    strengths: np.ndarray
    ci: np.ndarray | None  # shape (K, 2) lo/hi when n_boot > 0, else None


def bradley_terry(
    wins: np.ndarray,
    n_boot: int = 0,
    alpha: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> BradleyTerryResult:
    """Bradley-Terry strengths from a pairwise win matrix via the MM algorithm.

    ``wins[i, j]`` = number of times i beat j. Strengths are normalized to sum to 1. Declare a
    winner only when bootstrap CIs (n_boot > 0) separate.
    """
    wins = np.asarray(wins, dtype=float)
    k = wins.shape[0]
    if wins.shape != (k, k):
        raise ValueError("wins must be a square matrix")

    def _fit(w: np.ndarray) -> np.ndarray:
        wins_total = w.sum(axis=1)
        games = w + w.T
        p = np.ones(k)
        for _ in range(max_iter):
            prev = p.copy()
            for i in range(k):
                denom = 0.0
                for j in range(k):
                    if j != i and games[i, j] > 0:
                        denom += games[i, j] / (p[i] + p[j])
                if denom > 0:
                    p[i] = wins_total[i] / denom
            p = np.where(p <= 0, 1e-12, p)
            p /= p.sum()
            if np.max(np.abs(p - prev)) < tol:
                break
        return p

    strengths = _fit(wins)

    ci = None
    if n_boot > 0:
        rng = rng if rng is not None else np.random.default_rng()
        games = wins + wins.T
        samples = np.zeros((n_boot, k))
        for bdx in range(n_boot):
            resampled = np.zeros_like(wins)
            for i in range(k):
                for j in range(k):
                    if i < j and games[i, j] > 0:
                        wins_ij = rng.binomial(int(games[i, j]), wins[i, j] / games[i, j])
                        resampled[i, j] = wins_ij
                        resampled[j, i] = games[i, j] - wins_ij
            samples[bdx] = _fit(resampled)
        lo = np.quantile(samples, alpha / 2.0, axis=0)
        hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
        ci = np.stack([lo, hi], axis=1)

    return BradleyTerryResult(strengths, ci)
