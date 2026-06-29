"""T2S metrics: execution-grounded soft_f1, AST component_match/ast_valid, judge groundedness."""

import sqlite3

import pytest

from agent_eval.core.contracts import EvalContext, MetaKey
from agent_eval.judges.backend import FunctionJudge, lexical_overlap_judge
from agent_eval.metrics.t2s import AstValid, ComponentMatch, SoftF1, T2SFaithfulness


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "t.sqlite")
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE emp (id INTEGER, name TEXT, salary REAL);"
        "INSERT INTO emp VALUES (1,'Alice',150000),(2,'Bob',90000),(3,'Carol',120000);"
    )
    con.commit()
    con.close()
    return path


def _ctx(sql, gold_sql, db, output="", **extra):
    md = {MetaKey.SQL: sql, MetaKey.GOLD_SQL: gold_sql, MetaKey.DB_REF: db, **extra}
    return EvalContext(input="q", output=output, metadata=md)


def test_soft_f1_perfect_and_partial(db):
    same = "SELECT name FROM emp WHERE salary > 100000"
    assert SoftF1().score(_ctx(same, same, db)).score == 1.0
    # dropped filter -> superset of rows -> partial credit, not 0
    partial = SoftF1().score(_ctx("SELECT name FROM emp", same, db)).score
    assert 0.0 < partial < 1.0


def test_soft_f1_paraphrase_same_results(db):
    gold = "SELECT AVG(salary) FROM emp"
    pred = "SELECT AVG(salary) AS avg_salary FROM emp"
    assert SoftF1().score(_ctx(pred, gold, db)).score == 1.0


def test_component_match_dips_on_alias(db):
    gold = "SELECT AVG(salary) FROM emp"
    pred = "SELECT AVG(salary) AS avg_salary FROM emp"
    r = ComponentMatch().score(_ctx(pred, gold, db))
    assert 0.0 < r.score < 1.0  # same table, different projection text


def test_ast_valid(db):
    assert AstValid().score(_ctx("SELECT 1", "SELECT 1", db)).passed is True
    assert AstValid().score(_ctx("SELECT FROM WHERE", "SELECT 1", db)).passed is False


def test_t2s_metrics_require_sql_and_db():
    assert SoftF1().score(EvalContext(input="q")).error  # no sql
    assert AstValid().score(EvalContext(input="q")).error


def test_t2s_faithfulness_judges_answer_against_execution(db):
    judge = FunctionJudge(lexical_overlap_judge)
    sql = "SELECT name FROM emp WHERE salary > 100000"
    ctx = _ctx(sql, sql, db, output="Alice and Carol earn over 100000")
    r = T2SFaithfulness(judge).score(ctx)
    assert r.error is None and r.score > 0.0


def test_t2s_faithfulness_errors_on_bad_sql(db):
    judge = FunctionJudge(lexical_overlap_judge)
    ctx = _ctx("SELECT FROM nope", "SELECT 1", db, output="something")
    assert T2SFaithfulness(judge).score(ctx).error  # query can't execute -> no result to judge
