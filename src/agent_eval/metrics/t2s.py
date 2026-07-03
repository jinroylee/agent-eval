"""T2S (text-to-SQL) metrics — result-set correctness + AST checks, plus judge groundedness.

| metric              | GT? | reads (EvalContext)                                                     |
|---------------------|-----|------------------------------------------------------------------------|
| ``soft_f1``         | yes | ``metadata['execution_result']``, ``metadata['gold_execution_result']``|
| ``component_match`` | yes | ``metadata['sql']``, ``metadata['gold_sql']``                          |
| ``ast_valid``       | no  | ``metadata['sql']``                                                    |
| ``t2s_faithfulness``| no  | ``metadata['execution_result']``, ``output``                          |
| ``t2s_consistency`` | no  | ``metadata['execution_result']``, ``output``                          |

Objective correctness is graded on the **result set** the query produced — but the query is **not
executed at eval time**. The predicted result set is supplied by the agent's own state
(``metadata['execution_result']``) and the gold result set is stored in the dataset
(``metadata['gold_execution_result']``). ``soft_f1`` compares those two row-sets (partial credit via
cell-bag F1); the judge is confined to whether the natural-language ``output`` faithfully reports the
predicted result set. Result-set comparison semantics (row order, duplicates, NULLs, float tolerance)
are explicit via ``defaults.result_set_policy``. The AST metrics (``component_match``, ``ast_valid``)
parse the SQL text and need the ``[t2s]`` extra (``sqlglot``).
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from agent_eval.core.contracts import Aggregation, CostClass, EvalContext, MetaKey, MetricResult
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import BuildContext, MetricRegistry
from agent_eval.execution.compare import ResultSetPolicy, soft_f1
from agent_eval.judges.backend import JudgeRequest
from agent_eval.metrics.common import JudgeMetric, resolve_judge

_MAX_RESULT_ROWS = 50  # cap rows shown to the judge so the prompt stays bounded


# --------------------------------------------------------------------------- helpers
def _sql(ctx: EvalContext) -> str:
    sql = ctx.metadata.get(MetaKey.SQL)
    if not sql:
        raise ValueError("metadata['sql'] (the generated SQL) is required")
    return str(sql)


def _gold_sql(ctx: EvalContext) -> str:
    sql = ctx.metadata.get(MetaKey.GOLD_SQL)
    if not sql:
        raise ValueError("metadata['gold_sql'] (the gold SQL) is required")
    return str(sql)


def _rows(value) -> list[tuple]:
    """Coerce a stored result set into a list of row tuples (a bare scalar becomes a 1-cell row)."""
    return [tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in value]


def _result(ctx: EvalContext) -> list[tuple]:
    # An empty result set ([]) is valid; only a genuinely absent one (None) is an error.
    rows = ctx.metadata.get(MetaKey.EXECUTION_RESULT)
    if rows is None:
        raise ValueError("metadata['execution_result'] (the predicted SQL's result rows) is required")
    return _rows(rows)


def _gold_result(ctx: EvalContext) -> list[tuple]:
    rows = ctx.metadata.get(MetaKey.GOLD_EXECUTION_RESULT)
    if rows is None:
        raise ValueError("metadata['gold_execution_result'] (the gold SQL's result rows) is required")
    return _rows(rows)


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


def _format_rows(rows: list[tuple]) -> list[str]:
    """Render a result set as compact text lines for the judge to read."""
    if not rows:
        return ["(empty result set)"]
    lines = []
    for row in rows[:_MAX_RESULT_ROWS]:
        lines.append(" | ".join("NULL" if c is None else str(c) for c in row))
    if len(rows) > _MAX_RESULT_ROWS:
        lines.append(f"... ({len(rows) - _MAX_RESULT_ROWS} more rows)")
    return lines


# --------------------------------------------------------------------------- result-set correctness (GT)
class SoftF1(BaseMetric):
    """Cell-bag F1 between the predicted and gold **result sets** (graded partial credit).

    Both result sets are supplied in the data — no query is executed here: the predicted rows in
    ``metadata['execution_result']`` and the gold rows in ``metadata['gold_execution_result']``.
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
    "The EVIDENCE is the exact result set the SQL query returned. Judge whether the RESPONSE TO "
    "EVALUATE faithfully reports that result: numbers, names, and counts must match the evidence and "
    "nothing should be invented. Penalize fabricated or altered values."
)
_T2S_CONSISTENCY_RUBRIC = (
    "The EVIDENCE is the exact result set the SQL query returned. Judge whether the RESPONSE TO "
    "EVALUATE is consistent with it — it must not state anything that contradicts the result set."
)


class _ExecutionJudgeMetric(JudgeMetric):
    """Judge the final NL ``output`` against the predicted result set (``metadata['execution_result']``).

    The result set is supplied in the data (the agent ran its own query); nothing is executed here.
    """

    requires = frozenset({"output"})
    _rubric = ""

    def build_request(self, ctx: EvalContext) -> JudgeRequest:
        return JudgeRequest(
            instruction=self._rubric,
            question=str(ctx.input),
            response=str(ctx.output),
            context=tuple(_format_rows(_result(ctx))),
        )


class T2SFaithfulness(_ExecutionJudgeMetric):
    name = "t2s_faithfulness"
    _rubric = _T2S_FAITHFULNESS_RUBRIC


class T2SConsistency(_ExecutionJudgeMetric):
    name = "t2s_consistency"
    _rubric = _T2S_CONSISTENCY_RUBRIC


# --------------------------------------------------------------------------- registration
def _dialect(p: dict, ctx: BuildContext) -> str:
    return p.get("dialect") or ctx.defaults.get("dialect", "sqlite")


def _result_policy(p: dict, ctx: BuildContext):
    return p.get("result_policy") or ctx.defaults.get("result_set_policy")


def register(registry: MetricRegistry) -> None:
    registry.register("soft_f1", lambda p, ctx: SoftF1(_result_policy(p, ctx)))
    registry.register("component_match", lambda p, ctx: ComponentMatch(_dialect(p, ctx)))
    registry.register("ast_valid", lambda p, ctx: AstValid(_dialect(p, ctx)))
    registry.register("t2s_faithfulness", lambda p, ctx: T2SFaithfulness(resolve_judge(ctx), ctx.panel))
    registry.register("t2s_consistency", lambda p, ctx: T2SConsistency(resolve_judge(ctx), ctx.panel))
