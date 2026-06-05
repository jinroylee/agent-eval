"""Inter-rater agreement metrics.

Cohen's kappa corrects raw agreement for chance — the standard for certifying that a judge agrees
with human labels beyond coincidence (a high raw-agreement judge can still have kappa ~0).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa between two raters' categorical labels."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("a and b must be non-empty and equal length")
    labels = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[lab] / n) * (cb[lab] / n) for lab in labels)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)
