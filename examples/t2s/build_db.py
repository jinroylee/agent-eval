"""Build the T2S example's fixtures: a SQLite database + a gold dataset with pre-computed results.

Run once before evaluating:  ``python examples/t2s/build_db.py``

The schema (employees + departments) exercises joins, GROUP BY/HAVING, NULLs, float aggregates, and
LIMIT, so the result-set comparison policy is genuinely tested. This script also **executes each gold
query once** and stores its result set in ``gold.jsonl`` (as ``gold_execution_result``). Evaluation
compares against those stored rows, so ``agent-eval evaluate`` never touches the database itself — the
only things that run SQL are this build step and the agent at predict time.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "sample.sqlite"
GOLD_PATH = HERE / "gold.jsonl"

SCHEMA_DDL = """
CREATE TABLE department (dept_id INTEGER PRIMARY KEY, name TEXT, budget REAL);
CREATE TABLE employee (
    emp_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary REAL, manager_id INTEGER
);
"""

INSERTS = """
INSERT INTO department VALUES (1, 'Engineering', 1000000.0), (2, 'Sales', 500000.5), (3, 'HR', 250000.0);
INSERT INTO employee VALUES
    (1, 'Alice', 1, 150000.0, NULL),
    (2, 'Bob', 1, 120000.0, 1),
    (3, 'Carol', 1, 110000.5, 1),
    (4, 'Dave', 2, 90000.0, 1),
    (5, 'Eve', 2, NULL, 4),
    (6, 'Frank', 3, 80000.0, 1),
    (7, 'Grace', 3, NULL, 6),
    (8, 'Heidi', 2, 95000.0, 4);
"""

# Gold (question, SQL) pairs — the ground-truth queries.
GOLD: list[dict[str, str]] = [
    {"id": "q01", "question": "List all employee names ordered by employee id.",
     "gold_sql": "SELECT name FROM employee ORDER BY emp_id"},
    {"id": "q02", "question": "Names of employees in the Engineering department.",
     "gold_sql": "SELECT e.name FROM employee e JOIN department d ON e.dept_id = d.dept_id "
                 "WHERE d.name = 'Engineering'"},
    {"id": "q03", "question": "Number of employees in each department, by department name.",
     "gold_sql": "SELECT d.name, COUNT(*) FROM employee e JOIN department d "
                 "ON e.dept_id = d.dept_id GROUP BY d.name"},
    {"id": "q04", "question": "Total salary paid in each department id.",
     "gold_sql": "SELECT dept_id, SUM(salary) FROM employee GROUP BY dept_id"},
    {"id": "q05", "question": "Average salary across all employees.",
     "gold_sql": "SELECT AVG(salary) FROM employee"},
    {"id": "q06", "question": "Names of employees who have no manager.",
     "gold_sql": "SELECT name FROM employee WHERE manager_id IS NULL"},
    {"id": "q07", "question": "The name of the department with the highest budget.",
     "gold_sql": "SELECT name FROM department ORDER BY budget DESC LIMIT 1"},
    {"id": "q08", "question": "Names of employees earning more than 100000.",
     "gold_sql": "SELECT name FROM employee WHERE salary > 100000"},
    {"id": "q09", "question": "Each employee name with their department name.",
     "gold_sql": "SELECT e.name, d.name FROM employee e JOIN department d ON e.dept_id = d.dept_id"},
    {"id": "q10", "question": "How many employees have no recorded salary?",
     "gold_sql": "SELECT COUNT(*) FROM employee WHERE salary IS NULL"},
    {"id": "q11", "question": "Department ids having more than 2 employees.",
     "gold_sql": "SELECT dept_id FROM employee GROUP BY dept_id HAVING COUNT(*) > 2"},
    {"id": "q12", "question": "Total budget across all departments.",
     "gold_sql": "SELECT SUM(budget) FROM department"},
]


def build_db(path: str | os.PathLike = DB_PATH) -> str:
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA_DDL + INSERTS)
        con.commit()
    finally:
        con.close()
    return str(path)


def _execute(con: sqlite3.Connection, sql: str) -> list[list]:
    """Run a query and return its rows as a list of lists (JSON-friendly)."""
    cur = con.execute(sql)
    return [list(row) for row in cur.fetchall()]


def write_gold(path: str | os.PathLike = GOLD_PATH, db_path: str | os.PathLike = DB_PATH) -> str:
    """Pre-compute each gold query's result set and write the gold dataset as JSONL.

    Each record is ``{id, question, gold_sql, gold_execution_result}``. The stored result set is what
    the offline metrics compare against, so the evaluation step needs no database.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        records = [{**row, "gold_execution_result": _execute(con, row["gold_sql"])} for row in GOLD]
    finally:
        con.close()
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return str(path)


if __name__ == "__main__":
    build_db()
    write_gold()
    print(f"Wrote {DB_PATH} and {GOLD_PATH}")
