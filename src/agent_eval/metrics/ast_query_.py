"""T2S (text-to-SQL) metrics — execution-grounded and AST-based.

These are the sound external verifier for objective query correctness: gate on Execution Accuracy,
not on an LLM judge (JudgeBench shows judges fail on objective tasks). The same checks power the
runtime critic's deterministic first tier (AST validity, schema linking, dry-run execution).
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric
from agent_eval.core.registry import MetricRegistry
from agent_eval.execution.compare import ResultSetPolicy, soft_f1
from agent_eval.execution.harness import ExecutionHarness
from agent_eval.execution.sandbox import readonly_connection


def _policy(result_policy) -> ResultSetPolicy:
    if isinstance(result_policy, ResultSetPolicy):
        return result_policy
    if isinstance(result_policy, dict):
        return ResultSetPolicy(**result_policy)
    return ResultSetPolicy()


def _require_db(ctx: EvalContext) -> str:
    db = ctx.metadata.get("db_ref")
    if not db:
        raise ValueError("metadata['db_ref'] is required for execution-grounded T2S metrics")
    return db


def _introspect_schema(db_path: str) -> dict[str, set[str]]:
    con = readonly_connection(db_path)
    try:
        tables = [
            r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        schema: dict[str, set[str]] = {}
        for t in tables:
            cols = {r[1].lower() for r in con.execute(f'PRAGMA table_info("{t}")').fetchall()}
            schema[t.lower()] = cols
        return schema
    finally:
        con.close()


class _ExecMetric(BaseMetric):
    """Base for metrics that need an ExecutionHarness."""

    tier = Tier.DETERMINISTIC
    cost_class = CostClass.CHEAP

    def __init__(self, dialect: str = "sqlite", result_policy=None, timeout_s: float = 5.0) -> None:
        self.dialect = dialect
        self.harness = ExecutionHarness(policy=_policy(result_policy), timeout_s=timeout_s)


class ExecutionAccuracy(_ExecMetric):
    """Denotation match: predicted and gold queries return the same result set. The gate metric."""

    name = "execution_accuracy"
    requires = frozenset({"output", "expected"})

    def _compute(self, ctx: EvalContext) -> MetricResult:
        ok, pred, gold = self.harness.execution_match(ctx.output, ctx.expected, _require_db(ctx))
        detail = {}
        if pred.error:
            detail["pred_error"] = pred.error
        if gold.error:
            detail["gold_error"] = gold.error
        return MetricResult(self.name, 1.0 if ok else 0.0, passed=ok, detail=detail)


class SoftF1(_ExecMetric):
    """Cell-bag F1 between predicted and gold result sets (graded partial credit)."""

    name = "soft_f1"
    requires = frozenset({"output", "expected"})

    def _compute(self, ctx: EvalContext) -> MetricResult:
        db = _require_db(ctx)
        gold = self.harness.run(ctx.expected, db)
        if gold.error:
            raise ValueError(f"gold query failed to execute: {gold.error}")
        pred = self.harness.run(ctx.output, db)
        if pred.error:
            return MetricResult(self.name, 0.0, passed=None, detail={"pred_error": pred.error})
        score = soft_f1(pred.rows or [], gold.rows or [], self.harness.policy)
        return MetricResult(self.name, score, passed=None)


class ComponentMatch(BaseMetric):
    """AST component overlap (tables + projections) — a diagnostic, not a gate."""

    name = "component_match"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE
    requires = frozenset({"output", "expected"})

    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    def _compute(self, ctx: EvalContext) -> MetricResult:
        pred = sqlglot.parse_one(ctx.output, dialect=self.dialect)
        gold = sqlglot.parse_one(ctx.expected, dialect=self.dialect)
        t = _jaccard(_tables(pred), _tables(gold))
        s = _jaccard(_projections(pred), _projections(gold))
        score = (t + s) / 2.0
        return MetricResult(self.name, score, passed=None, detail={"tables": t, "projections": s})


class AstValid(BaseMetric):
    """Runtime tier-1: does the predicted query parse at all?"""

    name = "ast_valid"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.FREE
    requires = frozenset({"output"})

    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    def _compute(self, ctx: EvalContext) -> MetricResult:
        try:
            sqlglot.parse_one(ctx.output, dialect=self.dialect)
        except Exception as exc:
            return MetricResult(self.name, 0.0, passed=False, detail={"parse_error": str(exc)})
        return MetricResult(self.name, 1.0, passed=True)


class SchemaLinking(BaseMetric):
    """Runtime tier-1: referenced tables/columns exist in the schema."""

    name = "schema_linking"
    tier = Tier.DETERMINISTIC
    cost_class = CostClass.CHEAP
    requires = frozenset({"output"})

    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    def _compute(self, ctx: EvalContext) -> MetricResult:
        schema = _introspect_schema(_require_db(ctx))
        ast = sqlglot.parse_one(ctx.output, dialect=self.dialect)
        ref_tables = _tables(ast)
        unknown_tables = sorted(ref_tables - set(schema))
        all_cols = set().union(*schema.values()) if schema else set()
        aliases = {a.alias.lower() for a in ast.find_all(exp.Alias) if a.alias}
        ref_cols = {c.name.lower() for c in ast.find_all(exp.Column) if c.name}
        unknown_cols = sorted(ref_cols - all_cols - aliases)
        ok = not unknown_tables and not unknown_cols
        return MetricResult(
            self.name,
            1.0 if ok else 0.0,
            passed=ok,
            detail={"unknown_tables": unknown_tables, "unknown_cols": unknown_cols},
        )


class ExecValidity(_ExecMetric):
    """Runtime tier-1: the predicted query executes without error (dry run)."""

    name = "exec_validity"
    requires = frozenset({"output"})

    def _compute(self, ctx: EvalContext) -> MetricResult:
        r = self.harness.run(ctx.output, _require_db(ctx))
        ok = r.error is None
        return MetricResult(
            self.name, 1.0 if ok else 0.0, passed=ok, detail={"error": r.error} if r.error else {}
        )


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


def register_t2s_metrics(registry: MetricRegistry) -> None:
    """Register the T2S metric types (call on a registry before building T2S suites)."""
    registry.register("execution_accuracy", lambda p: ExecutionAccuracy(**p))
    registry.register("soft_f1", lambda p: SoftF1(**p))
    registry.register("component_match", lambda p: ComponentMatch(**p))
    registry.register("ast_valid", lambda p: AstValid(**p))
    registry.register("schema_linking", lambda p: SchemaLinking(**p))
    registry.register("exec_validity", lambda p: ExecValidity(**p))
