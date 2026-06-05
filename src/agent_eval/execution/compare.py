"""Result-set comparison policy for execution-grounded T2S evaluation.

The semantics of "do two query results match" (row order, duplicates, NULLs, column order, float
tolerance) silently inflate or deflate Execution Accuracy if left implicit. We make every choice
explicit and configurable in one place (the documented silent-failure source for EX).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

Row = Sequence


@dataclass(frozen=True)
class ResultSetPolicy:
    row_order: str = "ignore"  # ignore | strict
    duplicates: str = "keep"  # keep | dedup
    nulls: str = "distinct"  # distinct | coalesce
    column_order: str = "ignore"  # ignore | strict
    float_tolerance: float = 1e-6


def _ndigits(tol: float) -> int | None:
    if tol is None or tol <= 0:
        return None
    return max(0, int(round(-math.log10(tol))))


def _norm_cell(value, ndigits: int | None):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and ndigits is not None:
        return round(value, ndigits)
    return value


def _sort_key(cell):
    # Canonical order for column-order-insensitive comparison; NULLs first, then by (type, value).
    return (cell is not None, type(cell).__name__, str(cell))


def _norm_row(row: Row, policy: ResultSetPolicy, ndigits: int | None) -> tuple:
    cells = [_norm_cell(c, ndigits) for c in row]
    if policy.column_order == "ignore":
        cells = sorted(cells, key=_sort_key)
    return tuple(cells)


def _normalize(rows: Sequence[Row], policy: ResultSetPolicy) -> list[tuple]:
    ndigits = _ndigits(policy.float_tolerance)
    return [_norm_row(r, policy, ndigits) for r in rows]


def result_sets_equal(rows_a: Sequence[Row], rows_b: Sequence[Row], policy: ResultSetPolicy) -> bool:
    a = _normalize(rows_a, policy)
    b = _normalize(rows_b, policy)
    if policy.row_order == "strict":
        if policy.duplicates == "dedup":
            a, b = _dedup_keep_order(a), _dedup_keep_order(b)
        return a == b
    if policy.duplicates == "dedup":
        return set(a) == set(b)
    return Counter(a) == Counter(b)


def _dedup_keep_order(rows: list[tuple]) -> list[tuple]:
    seen: set[tuple] = set()
    out: list[tuple] = []
    for r in rows:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def soft_f1(rows_pred: Sequence[Row], rows_gold: Sequence[Row], policy: ResultSetPolicy) -> float:
    """Cell-bag F1: partial credit for overlapping cell values (graded secondary metric)."""
    ndigits = _ndigits(policy.float_tolerance)
    pred = Counter(_norm_cell(c, ndigits) for row in rows_pred for c in row)
    gold = Counter(_norm_cell(c, ndigits) for row in rows_gold for c in row)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = sum((pred & gold).values())
    precision = tp / sum(pred.values())
    recall = tp / sum(gold.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
