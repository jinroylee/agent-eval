"""A tiny, self-contained T2S sample: a 2-table SQLite schema + gold NL->SQL pairs.

Covers joins, GROUP BY/HAVING, NULLs, float aggregates, and LIMIT so the result-set comparison
policy is genuinely exercised. ``build_db`` materializes the database; ``GOLD`` is the labeled set;
``broken_sql`` simulates a buggy agent (strips WHERE/HAVING/LIMIT) for the regression demo.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp

SCHEMA_DDL = """
CREATE TABLE department (dept_id INTEGER PRIMARY KEY, name TEXT, budget REAL);
CREATE TABLE employee (
    emp_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary REAL, manager_id INTEGER
);
"""

_INSERTS = """
INSERT INTO department VALUES (1, 'Engineering', 1000000.0), (2, 'Sales', 500000.50), (3, 'HR', 250000.0);
INSERT INTO employee VALUES
    (1, 'Alice', 1, 150000.0, NULL),
    (2, 'Bob', 1, 120000.0, 1),
    (3, 'Carol', 1, 110000.50, 1),
    (4, 'Dave', 2, 90000.0, 1),
    (5, 'Eve', 2, NULL, 4),
    (6, 'Frank', 3, 80000.0, 1),
    (7, 'Grace', 3, NULL, 6),
    (8, 'Heidi', 2, 95000.0, 4);
"""

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


def build_db(path: str) -> str:
    """Create a fresh SQLite database with the sample schema + data at ``path``."""
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA_DDL + _INSERTS)
        con.commit()
    finally:
        con.close()
    return path


def broken_sql(sql: str, dialect: str = "sqlite") -> str:
    """Simulate a buggy agent: strip WHERE/HAVING/LIMIT so filtered queries return wrong results."""
    tree = sqlglot.parse_one(sql, dialect=dialect)
    for cls in (exp.Where, exp.Having, exp.Limit):
        for node in list(tree.find_all(cls)):
            node.pop()
    return tree.sql(dialect=dialect)


def write_gold_csv(path: str) -> str:
    """Write GOLD to a CSV (question, gold_sql, id) for the tabular-adapter path."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "gold_sql"])
        writer.writeheader()
        writer.writerows(GOLD)
    return path


SAMPLE_DIR = Path(__file__).parent
