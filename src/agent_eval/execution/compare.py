"""Result-set comparison policy for execution-grounded T2S evaluation.

The semantics of "do two query results match" (row order, duplicates, NULLs, column order, float
tolerance) silently inflate or deflate Execution Accuracy if left implicit. We make every choice
explicit and configurable in one place (the documented silent-failure source for EX).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Row = Sequence | Mapping


def _cells(row: Row) -> list:
    """The ordered cell *values* of one row, whichever shape it arrives in.

    Result sets reach us two ways: the T2S metrics pass canonical ``dict`` rows (``column -> value``),
    while the execution harness passes positional tuples. Comparison here is value-based (column
    identity is handled by ``column_order``), so we reduce both shapes to their list of cell values.
    """
    if isinstance(row, Mapping):
        return list(row.values())
    return list(row)


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
    cells = [_norm_cell(c, ndigits) for c in _cells(row)]
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


def _facts(row: Row) -> list[tuple]:
    """The ``(column, value)`` *facts* of one row — the atomic units ``soft_f1`` counts.

    Column identity comes from the dict key for canonical rows; a positional row falls back to the
    cell index. Either way a value is only ever credited under the column it actually appears in.
    """
    if isinstance(row, Mapping):
        return [(str(k), v) for k, v in row.items()]
    return [(str(i), v) for i, v in enumerate(row)]


def soft_f1(rows_pred: Sequence[Row], rows_gold: Sequence[Row], policy: ResultSetPolicy) -> float:
    """Fact-bag F1: partial credit for matching ``(column, value)`` facts across the result sets.

    Every cell is one atomic *fact* — a ``(column, normalized value)`` pair — and the score is the F1
    of the predicted fact multiset against the gold one. A true positive is a fact present in both (the
    same value **under the same column**), so precision drops on extra/wrong facts and recall on missing
    ones (e.g. an omitted column costs recall). Row order and duplicate rows wash out (it is a
    multiset); ``float_tolerance`` still merges near-equal numbers. Column identity is significant here:
    an aliased/renamed column no longer matches (unlike a bare value bag).
    """
    ndigits = _ndigits(policy.float_tolerance)
    pred = Counter((c, _norm_cell(v, ndigits)) for row in rows_pred for c, v in _facts(row))
    gold = Counter((c, _norm_cell(v, ndigits)) for row in rows_gold for c, v in _facts(row))
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
