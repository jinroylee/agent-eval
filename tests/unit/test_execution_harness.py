"""Tests for the read-only SQLite ExecutionHarness and execution-match."""

import sqlite3

from agent_eval.execution.harness import ExecutionHarness


def _db(tmp_path):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.executescript(
        """
        CREATE TABLE t(id INTEGER, val REAL, name TEXT);
        INSERT INTO t VALUES (1, 1.5, 'a'), (2, 2.5, 'b'), (3, NULL, 'a');
        """
    )
    con.commit()
    con.close()
    return str(p)


def test_run_returns_rows(tmp_path):
    r = ExecutionHarness().run("SELECT id, name FROM t ORDER BY id", _db(tmp_path))
    assert r.error is None
    assert r.rows == [(1, "a"), (2, "b"), (3, "a")]


def test_run_invalid_sql_returns_error(tmp_path):
    r = ExecutionHarness().run("SELECT * FROM nonexistent", _db(tmp_path))
    assert r.error is not None and r.rows is None


def test_execution_match_equivalent_queries(tmp_path):
    db = _db(tmp_path)
    ok, _, _ = ExecutionHarness().execution_match(
        "SELECT name FROM t WHERE id <= 2", "SELECT name FROM t WHERE id IN (1, 2)", db
    )
    assert ok is True


def test_execution_match_different_queries(tmp_path):
    db = _db(tmp_path)
    ok, _, _ = ExecutionHarness().execution_match(
        "SELECT name FROM t WHERE id = 1", "SELECT name FROM t", db
    )
    assert ok is False


def test_readonly_connection_blocks_writes(tmp_path):
    r = ExecutionHarness().run("INSERT INTO t VALUES (4, 4.0, 'z')", _db(tmp_path))
    assert r.error is not None  # read-only connection rejects mutations
