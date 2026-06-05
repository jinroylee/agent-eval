"""Known-good / known-bad tests for the T2S (text-to-SQL) metrics."""

import sqlite3

from agent_eval.core.contracts import EvalContext
from agent_eval.metrics.ast_query_ import (
    AstValid,
    ComponentMatch,
    ExecutionAccuracy,
    ExecValidity,
    SchemaLinking,
    SoftF1,
)


def _db(tmp_path):
    p = tmp_path / "e.sqlite"
    if not p.exists():  # build once, reuse across the several _ctx() calls in a test
        con = sqlite3.connect(p)
        con.executescript(
            """
            CREATE TABLE emp(id INTEGER, dept TEXT, salary REAL);
            INSERT INTO emp VALUES (1, 'A', 100.0), (2, 'A', 200.0), (3, 'B', NULL);
            """
        )
        con.commit()
        con.close()
    return str(p)


def _ctx(tmp_path, output, expected=None):
    return EvalContext(
        input="q", output=output, expected=expected, metadata={"db_ref": _db(tmp_path)}
    )


def test_execution_accuracy_match(tmp_path):
    ctx = _ctx(
        tmp_path,
        "SELECT id FROM emp WHERE dept = 'A' ORDER BY id",
        "SELECT id FROM emp WHERE dept = 'A'",
    )
    assert ExecutionAccuracy().score(ctx).score == 1.0


def test_execution_accuracy_mismatch(tmp_path):
    ctx = _ctx(tmp_path, "SELECT id FROM emp", "SELECT id FROM emp WHERE dept = 'A'")
    assert ExecutionAccuracy().score(ctx).score == 0.0


def test_execution_accuracy_invalid_pred_scores_zero(tmp_path):
    ctx = _ctx(tmp_path, "SELECT FROM oops", "SELECT id FROM emp")
    r = ExecutionAccuracy().score(ctx)
    assert r.score == 0.0 and r.passed is False
    assert "pred_error" in r.detail


def test_soft_f1_partial(tmp_path):
    ctx = _ctx(tmp_path, "SELECT id FROM emp WHERE dept = 'A'", "SELECT id FROM emp")
    f = SoftF1().score(ctx).score
    assert 0.0 < f < 1.0


def test_component_match(tmp_path):
    same = EvalContext(input="q", output="SELECT id FROM emp", expected="SELECT id FROM emp")
    assert ComponentMatch().score(same).score == 1.0
    diff = EvalContext(input="q", output="SELECT salary FROM emp", expected="SELECT id FROM emp")
    assert ComponentMatch().score(diff).score < 1.0


def test_ast_valid(tmp_path):
    assert AstValid().score(EvalContext(input="q", output="SELECT 1")).score == 1.0
    bad = AstValid().score(EvalContext(input="q", output="SELECT FROM WHERE"))
    assert bad.score == 0.0 and bad.passed is False


def test_schema_linking(tmp_path):
    ok = _ctx(tmp_path, "SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept ORDER BY n")
    assert SchemaLinking().score(ok).score == 1.0
    assert SchemaLinking().score(_ctx(tmp_path, "SELECT missing_col FROM emp")).score == 0.0
    assert SchemaLinking().score(_ctx(tmp_path, "SELECT id FROM nope")).score == 0.0


def test_exec_validity(tmp_path):
    assert ExecValidity().score(_ctx(tmp_path, "SELECT id FROM emp")).score == 1.0
    assert ExecValidity().score(_ctx(tmp_path, "SELECT id FROM nope")).score == 0.0
