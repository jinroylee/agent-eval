"""Prediction-Powered Inference (PPI++) for judge-scored metrics.

Many cheap LLM-judge labels + a small pinned human gold set => a valid CI for the true metric
that stays valid despite judge bias. The tuning parameter lambda down-weights the judge when it
is uninformative, so the estimator degrades gracefully to the gold-only interval.

Guarantees here are asymptotic (PPI++); finite-sample validity belongs to the original PPI.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ppi_interval(
    judge_labels,
    gold_labels,
    gold_idx,
    alpha: float = 0.05,
    lam: float | None = None,
) -> tuple[float, float, float]:
    """PPI++ confidence interval for the mean of the true label.

    ``judge_labels`` are cheap predictions for *all* items; ``gold_labels`` are human labels for
    the subset at positions ``gold_idx``. Returns ``(theta, lo, hi)``.
    """
    f_all = np.asarray(judge_labels, dtype=float)
    y = np.asarray(gold_labels, dtype=float)
    gold_idx = np.asarray(gold_idx, dtype=int)
    f_l = f_all[gold_idx]
    n = y.size
    m = f_all.size
    if f_l.shape != y.shape:
        raise ValueError("gold_labels must align with gold_idx")
    if n < 2:
        raise ValueError("need at least 2 gold labels")

    var_fl = float(np.var(f_l, ddof=1))
    cov = float(np.cov(f_l, y, ddof=1)[0, 1])
    if lam is None:
        lam = (cov / var_fl) if var_fl > 0 else 0.0
        lam = min(max(lam, 0.0), 1.0)  # clamp for stability

    theta = lam * float(f_all.mean()) + float(np.mean(y - lam * f_l))
    var_f_all = float(np.var(f_all, ddof=1)) if m > 1 else 0.0
    var_resid = float(np.var(y - lam * f_l, ddof=1))
    var_theta = (lam**2) * var_f_all / m + var_resid / n
    se = float(np.sqrt(var_theta))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return theta, theta - z * se, theta + z * se
