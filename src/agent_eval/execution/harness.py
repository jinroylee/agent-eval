"""Dialect-aware execution harness: run predicted vs gold queries and compare result sets.

The harness IS the sound external verifier for T2S — execution grounds correctness, so the
runtime critic and offline gate both lean on it rather than on an LLM judging query correctness.
P1 implements the SQLite backend; Postgres/Cypher backends slot in behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval.execution.compare import ResultSetPolicy, result_sets_equal
from agent_eval.execution.sandbox import readonly_connection


@dataclass
class ExecResult:
    columns: list[str] | None
    rows: list[tuple] | None
    error: str | None = None


class ExecutionHarness:
    def __init__(
        self,
        policy: ResultSetPolicy | None = None,
        timeout_s: float = 5.0,
        max_ops: int = 5_000_000,
    ) -> None:
        self.policy = policy or ResultSetPolicy()
        self.timeout_s = timeout_s
        self.max_ops = max_ops

    def run(self, sql: str, db_path: str) -> ExecResult:
        try:
            con = readonly_connection(db_path, self.timeout_s, self.max_ops)
        except Exception as exc:
            return ExecResult(None, None, f"connect error: {exc}")
        try:
            cur = con.execute(sql)
            rows = [tuple(r) for r in cur.fetchall()]
            cols = [d[0] for d in cur.description] if cur.description else []
            return ExecResult(cols, rows, None)
        except Exception as exc:
            return ExecResult(None, None, str(exc))
        finally:
            con.close()

    def execution_match(
        self, pred_sql: str, gold_sql: str, db_path: str
    ) -> tuple[bool, ExecResult, ExecResult]:
        pred = self.run(pred_sql, db_path)
        gold = self.run(gold_sql, db_path)
        if pred.error or gold.error:
            return False, pred, gold
        return result_sets_equal(pred.rows or [], gold.rows or [], self.policy), pred, gold
