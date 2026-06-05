"""Cluster-robust standard error of a mean.

Eval items are often non-iid: RAG questions sharing a passage, turns within a multi-turn
scenario, tool-calls within one trajectory. Treating them as independent makes naive SEs far
too small (Miller, "Adding Error Bars to Evals", 2024). Cluster on the unit of randomization.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def clustered_se(values: Sequence[float], cluster_ids: Sequence) -> tuple[float, float]:
    """Return ``(mean, cluster_robust_se)`` for the mean of ``values``.

    With singleton clusters this reduces exactly to the iid SE ``s / sqrt(N)``; with positive
    within-cluster correlation it inflates appropriately.
    """
    vals = np.asarray(values, dtype=float)
    ids = np.asarray(list(cluster_ids))
    if vals.size == 0:
        raise ValueError("values must be non-empty")
    if vals.shape[0] != ids.shape[0]:
        raise ValueError("values and cluster_ids must align")
    n = vals.size
    mean = float(vals.mean())
    resid = vals - mean
    groups = np.unique(ids)
    g = groups.size
    if g < 2:
        return mean, float("nan")  # cannot estimate a between-cluster SE from one cluster
    sum_sg2 = sum(float(resid[ids == grp].sum()) ** 2 for grp in groups)
    var = (g / (g - 1)) * sum_sg2 / (n**2)
    return mean, float(np.sqrt(var))
