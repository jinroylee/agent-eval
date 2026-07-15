"""T2S (text-to-SQL) metrics — result-set correctness + AST checks, plus judge groundedness.

| metric              | GT? | reads (EvalContext)                                                     |
|---------------------|-----|------------------------------------------------------------------------|
| ``soft_f1``         | yes | ``metadata['execution_result']``, ``metadata['gold_execution_result']``|
| ``component_match`` | yes | ``metadata['sql']``, ``metadata['gold_sql']``                          |
| ``ast_valid``       | no  | ``metadata['sql']``                                                    |
| ``t2s_faithfulness``| no  | ``metadata['execution_result']``, ``output``                          |
| ``t2s_consistency`` | no  | ``metadata['repeated_sql']`` (+ ``input``)                             |

Objective correctness is graded on the **result set** the query produced — but the query is **not
executed at eval time**. The predicted result set is supplied by the agent's own state
(``metadata['execution_result']``) and the gold result set is stored in the dataset
(``metadata['gold_execution_result']``). Each result set is canonically an **array of rows, one
``dict[str, Any]`` per row** (``column -> value``); loosely-typed encodings (a JSON string, a single
unwrapped row, or positional cells) are coerced to that shape on read (see ``_coerce_rows``).
``soft_f1`` compares those two row-sets (partial credit via
fact-bag F1 over ``(column, value)`` facts); the ``t2s_faithfulness`` judge is confined to whether the
``output`` reports the predicted result set — and it reads a **bounded statistical digest** of that set
(per-column aggregates over every row + a small sample; see ``_digest_lines``), never the raw rows, so
the prompt stays small no matter how large the result is. ``t2s_consistency`` is different: it judges
**repeated runs of the same query** — the N generated SQLs in ``metadata['repeated_sql']`` (filled via
``prediction.n_runs``) are compared pairwise for semantic equivalence. Result-set comparison
semantics (row order, duplicates, NULLs, float tolerance) are explicit via
``defaults.result_set_policy``. The AST metrics
(``component_match``, ``ast_valid``) parse the SQL text and need the ``[t2s]`` extra (``sqlglot``);
they first normalize escaped-whitespace artifacts from raw data (a literal ``\n``, ``\\n``, ``\t``)
that would otherwise make sqlglot choke on a stray backslash (see ``_clean_sql``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import sqlglot
from sqlglot import exp

from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetaKey, MetricResult
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import BuildContext, MetricRegistry
from agent_eval.execution.compare import ResultSetPolicy, soft_f1
from agent_eval.judges.backend import JudgeRequest
from agent_eval.metrics.common import JudgeMetric, SelfConsistencyMetric, resolve_judge

# The judge never sees the raw result set (it may be enormous). Instead it sees a bounded
# *statistical digest* whose size is O(columns), not O(rows): per-column aggregates + a small sample.
_SAMPLE_ROWS = 5  # rows shown verbatim, so the judge sees real structure/formatting
_MAX_DISTINCT = 20  # cap on distinct values enumerated per column (else only aggregates are shown)
_MAX_COLUMNS = 20  # cap on columns summarized (wide result sets)


# --------------------------------------------------------------------------- helpers
# Raw-data SQL often carries *literal* escaped-whitespace artifacts — a backslash-n (``\n``), ``\\n``,
# ``\r``, ``\t`` — left over from JSON/CSV encoding. Outside a string literal, sqlglot cannot tokenize
# the stray backslash and fails with "Unexpected token". We fold any run of backslashes before an
# ``n``/``r``/``t`` escape into a single space so the *structure* parses; the AST metrics only look at
# tables/projections/validity, so string-literal content does not matter. (Real newlines/tabs and
# backslashes *inside* string literals already parse, so they are left untouched.)
_ESCAPED_WS = re.compile(r"\\+[nrt]")


def _clean_sql(text: object) -> str:
    """Normalize escaped-whitespace artifacts out of raw SQL so sqlglot can parse its structure."""
    return _ESCAPED_WS.sub(" ", str(text)).strip()


def _sql(ctx: EvalContext) -> str:
    sql = ctx.metadata.get(MetaKey.SQL)
    if not sql:
        raise ValueError("metadata['sql'] (the generated SQL) is required")
    return _clean_sql(sql)


def _gold_sql(ctx: EvalContext) -> str:
    sql = ctx.metadata.get(MetaKey.GOLD_SQL)
    if not sql:
        raise ValueError("metadata['gold_sql'] (the gold SQL) is required")
    return _clean_sql(sql)


# The strict canonical type of a stored result set: an array of rows, each a ``column -> value`` map.
ResultRow = dict[str, Any]
ResultSet = list[ResultRow]


def _coerce_row(row: Any) -> ResultRow:
    """Coerce a single row into the canonical ``dict[str, Any]`` (``column -> value``).

    Accepts the canonical mapping as-is; casts the two common incompatible encodings rather than
    erroring — positional cells (``["Alice", 30]``) become ``{"col1": "Alice", "col2": 30}`` and a
    bare scalar becomes a one-column row ``{"col1": value}``.
    """
    if isinstance(row, Mapping):
        return {str(k): v for k, v in row.items()}
    if isinstance(row, (list, tuple)):
        return {f"col{i}": cell for i, cell in enumerate(row, start=1)}
    return {"col1": row}


def _coerce_rows(value: Any, field: str) -> ResultSet:
    """Coerce a stored result set to the strict canonical type ``list[dict[str, Any]]``.

    The metrics require rows of ``column -> value``; incompatible-but-recoverable encodings are cast
    with native Python rather than rejected: a whole set stored as a JSON *string* is parsed, a single
    unwrapped row ``dict`` is wrapped in a list, and positional/scalar rows are named (see
    :func:`_coerce_row`). Anything that still is not a list of rows is a genuine data error.
    """
    if isinstance(value, str):  # the whole set arrived as JSON text (e.g. a stringly-typed store)
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata['{field}'] is a string but not valid JSON: {exc}") from exc
    if isinstance(value, Mapping):  # a single row handed over without the enclosing list
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"metadata['{field}'] must be a list of rows (dict[str, Any]); got {type(value).__name__}"
        )
    return [_coerce_row(row) for row in value]


def _result(ctx: EvalContext) -> ResultSet:
    # An empty result set ([]) is valid; only a genuinely absent one (None) is an error.
    rows = ctx.metadata.get(MetaKey.EXECUTION_RESULT)
    if rows is None:
        raise ValueError("metadata['execution_result'] (the predicted SQL's result rows) is required")
    return _coerce_rows(rows, "execution_result")


def _gold_result(ctx: EvalContext) -> ResultSet:
    rows = ctx.metadata.get(MetaKey.GOLD_EXECUTION_RESULT)
    if rows is None:
        raise ValueError("metadata['gold_execution_result'] (the gold SQL's result rows) is required")
    return _coerce_rows(rows, "gold_execution_result")


def _policy(arg) -> ResultSetPolicy:
    if isinstance(arg, ResultSetPolicy):
        return arg
    if isinstance(arg, dict):
        return ResultSetPolicy(**arg)
    return ResultSetPolicy()


def _tables(node) -> set[str]:
    return {t.name.lower() for t in node.find_all(exp.Table) if t.name}


def _projections(node) -> set[str]:
    select = node.find(exp.Select)
    return {e.sql().lower() for e in (select.expressions if select else [])}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(x) -> str:
    """Compact numeric rendering: drop a trailing ``.0``, round long floats for readability."""
    f = float(x)
    return str(int(f)) if f.is_integer() else str(round(f, 6))


def _cell(c) -> str:
    return "NULL" if c is None else str(c)


def _distinct(values: list) -> list:
    """Distinct values, order-preserving (ignores unhashable-cell edge cases: cells are scalars)."""
    seen: set = set()
    out: list = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _column_summary(name: str, cells: list, max_distinct: int) -> str:
    """One line summarizing a column: type, counts, and either aggregates or enumerated values."""
    non_null = [c for c in cells if c is not None]
    n_null = len(cells) - len(non_null)
    distinct = _distinct(non_null)
    fields: list[str] = []
    if non_null and all(_is_number(c) for c in non_null):
        nums = [float(c) for c in non_null]
        fields += [
            "number",
            f"non-null={len(non_null)}",
            f"distinct={len(distinct)}",
            f"min={_num(min(nums))}",
            f"max={_num(max(nums))}",
            f"sum={_num(sum(nums))}",
            f"mean={_num(sum(nums) / len(nums))}",
        ]
        if 0 < len(distinct) <= max_distinct:
            fields.append("values=[" + ", ".join(_num(v) for v in distinct) + "]")
    else:
        shown = distinct[:max_distinct]
        more = len(distinct) - len(shown)
        vals = "values=[" + ", ".join(_cell(v) for v in shown) + "]"
        fields += ["text", f"non-null={len(non_null)}", f"distinct={len(distinct)}",
                   vals + (f" (+{more} more)" if more > 0 else "")]
    if n_null:
        fields.append(f"nulls={n_null}")
    return f"Column {name!r}: " + ", ".join(fields)


def _columns(rows: ResultSet) -> list[str]:
    """Ordered union of column names across rows (first-seen order; rows may be ragged)."""
    seen: set[str] = set()
    cols: list[str] = []
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                cols.append(name)
    return cols


def _digest_lines(
    rows: ResultSet,
    *,
    sample_rows: int = _SAMPLE_ROWS,
    max_distinct: int = _MAX_DISTINCT,
    max_columns: int = _MAX_COLUMNS,
) -> list[str]:
    """A **bounded** statistical digest of a result set for the judge to read.

    Instead of dumping (a truncated prefix of) the rows — which fails silently once the answer refers
    to a value past the cutoff — we summarize the *whole* set: per-column aggregates computed over
    every row, distinct values when few, and a small verbatim sample. The output size is O(columns),
    so it stays small whether the query returned 8 rows or 8 million. Rows are canonicalized to
    ``dict[str, Any]`` first, so columns are summarized under their real names.
    """
    rows = _coerce_rows(rows, "execution_result")
    if not rows:
        return ["(empty result set)"]
    cols = _columns(rows)
    n_rows, n_cols = len(rows), len(cols)
    if n_rows == 1 and n_cols == 1:
        return [f"Result digest — single value: {_cell(rows[0][cols[0]])}"]

    lines = [f"Result digest — full set: {n_rows} row(s), {n_cols} column(s) (summary, not raw rows)."]
    for name in cols[:max_columns]:
        lines.append(_column_summary(name, [row.get(name) for row in rows], max_distinct))
    if n_cols > max_columns:
        lines.append(f"... (+{n_cols - max_columns} more columns)")

    k = min(sample_rows, n_rows)
    lines.append(f"Sample rows (first {k} of {n_rows}):")
    lines += ["  " + " | ".join(_cell(row.get(c)) for c in cols) for row in rows[:k]]
    return lines


# --------------------------------------------------------------------------- result-set correctness (GT)
class SoftF1(BaseMetric):
    """Fact-bag F1 between the predicted and gold **result sets** (graded partial credit).

    Both result sets are supplied in the data — no query is executed here: the predicted rows in
    ``metadata['execution_result']`` and the gold rows in ``metadata['gold_execution_result']``, each
    an array of ``dict[str, Any]`` rows. Every cell is one atomic ``(column, value)`` fact and the
    score is the F1 of the predicted fact multiset vs the gold one — a value only earns credit under
    the **same column**, so an omitted column costs recall and an aliased/renamed column no longer
    matches. Row order and duplicate rows wash out; ``float_tolerance`` still merges near-equal numbers.
    """

    name = "soft_f1"
    cost_class = CostClass.FREE
    aggregation = Aggregation.MEAN

    def __init__(self, result_policy=None) -> None:
        self.policy = _policy(result_policy)

    def _compute(self, ctx: EvalContext) -> MetricResult:
        score = soft_f1(_result(ctx), _gold_result(ctx), self.policy)
        return MetricResult(self.name, score, passed=None)


class ComponentMatch(BaseMetric):
    """AST component overlap (tables + projections) between predicted and gold SQL — a diagnostic."""

    name = "component_match"
    cost_class = CostClass.FREE
    aggregation = Aggregation.MEAN

    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    def _compute(self, ctx: EvalContext) -> MetricResult:
        pred = sqlglot.parse_one(_sql(ctx), dialect=self.dialect)
        gold = sqlglot.parse_one(_gold_sql(ctx), dialect=self.dialect)
        t = _jaccard(_tables(pred), _tables(gold))
        s = _jaccard(_projections(pred), _projections(gold))
        return MetricResult(self.name, (t + s) / 2.0, passed=None, detail={"tables": t, "projections": s})


# --------------------------------------------------------------------------- AST validity (non-GT)
class AstValid(BaseMetric):
    """Does the generated SQL parse at all? A cheap binary sanity gate."""

    name = "ast_valid"
    cost_class = CostClass.FREE
    aggregation = Aggregation.RATE

    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    def _compute(self, ctx: EvalContext) -> MetricResult:
        sql = _sql(ctx)  # missing SQL is an error (can't run), not a parse failure
        try:
            sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as exc:
            return MetricResult(self.name, 0.0, passed=False, detail={"parse_error": str(exc)})
        return MetricResult(self.name, 1.0, passed=True)


# --------------------------------------------------------------------------- groundedness (non-GT, judge)
_T2S_FAITHFULNESS_RUBRIC = (
    "The EVIDENCE is a STATISTICAL DIGEST of the full result set the SQL query returned — row and "
    "column counts, per-column min/max/sum/mean and distinct values, and a small sample of rows (NOT "
    "the complete rows). Judge whether the RESPONSE TO EVALUATE faithfully reports this result: any "
    "count must match the reported row/value counts, any figure (total, average, minimum, maximum, "
    "count) must match the corresponding column aggregate, and any value it attributes to the result "
    "must be consistent with the enumerated or sampled values. Penalize figures or entities that "
    "contradict the digest or appear fabricated. If a specific value is not shown in the digest and "
    "cannot be confirmed or denied by it, do not penalize its use."
)
class _ExecutionJudgeMetric(JudgeMetric):
    """Judge the final NL ``output`` against the predicted result set (``metadata['execution_result']``).

    The result set is supplied in the data (the agent ran its own query); nothing is executed here.
    The judge reads a **bounded digest** of that result set (see :func:`_digest_lines`), never the raw
    rows, so the prompt stays small no matter how many rows the query returned.
    """

    requires = frozenset({"output"})
    _rubric = ""

    def __init__(
        self,
        backend,
        panel=(),
        *,
        sample_rows: int = _SAMPLE_ROWS,
        max_distinct: int = _MAX_DISTINCT,
        max_columns: int = _MAX_COLUMNS,
    ) -> None:
        super().__init__(backend, panel)
        self.sample_rows = sample_rows
        self.max_distinct = max_distinct
        self.max_columns = max_columns

    def build_request(self, ctx: EvalContext) -> JudgeRequest:
        digest = _digest_lines(
            _result(ctx),
            sample_rows=self.sample_rows,
            max_distinct=self.max_distinct,
            max_columns=self.max_columns,
        )
        return JudgeRequest(
            instruction=self._rubric,
            question=str(ctx.input),
            response=str(ctx.output),
            context=tuple(digest),
        )


class T2SFaithfulness(_ExecutionJudgeMetric):
    name = "t2s_faithfulness"
    _rubric = _T2S_FAITHFULNESS_RUBRIC


# --------------------------------------------------------------------------- self-consistency (non-GT, judge)
_SQL_CONSISTENCY_RUBRIC = (
    "The RESPONSE TO EVALUATE and the REFERENCE ANSWER are two SQL queries the same agent "
    "generated for the SAME user question on different runs. Judge whether they are semantically "
    "equivalent: run against the same database, would they return the same result (same rows, "
    "columns, filters, grouping, and aggregation)? Ignore formatting, letter case, aliases, and "
    "quoting; penalize different tables, columns, filters, aggregations, or limits."
)


class T2SConsistency(SelfConsistencyMetric):
    """Self-consistency: do N generated SQLs for the SAME query (``metadata['repeated_sql']``) agree?"""

    name = "t2s_consistency"
    runs_key = MetaKey.REPEATED_SQL
    _rubric = _SQL_CONSISTENCY_RUBRIC


# --------------------------------------------------------------------------- registration
def _dialect(p: dict, ctx: BuildContext) -> str:
    return p.get("dialect") or ctx.defaults.get("dialect", "sqlite")


def _result_policy(p: dict, ctx: BuildContext):
    return p.get("result_policy") or ctx.defaults.get("result_set_policy")


def _digest_params(p: dict) -> dict:
    """The digest-tuning params a config may set on the T2S judge metrics (all optional)."""
    return {k: p[k] for k in ("sample_rows", "max_distinct", "max_columns") if k in p}


def register(registry: MetricRegistry) -> None:
    registry.register("soft_f1", lambda p, ctx: SoftF1(_result_policy(p, ctx)))
    registry.register("component_match", lambda p, ctx: ComponentMatch(_dialect(p, ctx)))
    registry.register("ast_valid", lambda p, ctx: AstValid(_dialect(p, ctx)))
    registry.register(
        "t2s_faithfulness", lambda p, ctx: T2SFaithfulness(resolve_judge(ctx), ctx.panel, **_digest_params(p))
    )
    registry.register("t2s_consistency", lambda p, ctx: T2SConsistency(resolve_judge(ctx), ctx.panel))
