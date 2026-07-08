"""A toy *text-to-SQL* LangGraph agent — the agent under evaluation.

Deterministic (a question→SQL lookup, then it runs the query and summarizes the rows) so the example
needs no LLM. A couple of its queries are deliberately imperfect to show what each metric catches:

- ``q06`` qualifies a column (``e.name`` vs gold ``name``): identical *results* (soft_f1 = 1.0) but
  different projection *text* (component_match < 1.0). soft_f1 grades ``(column, value)`` facts over
  the result set, and sqlite names the column ``name`` either way, so the fact still matches — an
  *aliased/renamed* result column, by contrast, would lower soft_f1.
- ``q08`` drops a ``WHERE`` filter: it still parses (ast_valid = 1.0) but returns the wrong rows
  (soft_f1 < 1.0). Its NL answer faithfully reports *its own* (wrong) result — so t2s_faithfulness
  stays high while soft_f1 flags the error. That's the intended division of labor.

Required state fields (pointed at by the config's ``state_map``):
    sql              -> the generated SQL              (component_match / ast_valid)
    execution_result -> the rows that SQL returned     (soft_f1 / t2s_faithfulness / t2s_consistency)
    output           -> the final NL answer            (llm_judge)
    tokens           -> tokens spent                    (token_usage)

The agent runs its own query at predict time and captures the result rows in ``execution_result``;
evaluation compares those against the gold rows stored in the dataset and never touches a database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

DB_PATH = str(Path(__file__).parent / "sample.sqlite")

# What SQL the agent emits for each question (cf. gold in build_db.py).
PRED_SQL = {
    "List all employee names ordered by employee id.": "SELECT name FROM employee ORDER BY emp_id",
    "Names of employees in the Engineering department.":
        "SELECT e.name FROM employee e JOIN department d ON e.dept_id = d.dept_id "
        "WHERE d.name = 'Engineering'",
    "Number of employees in each department, by department name.":
        "SELECT d.name, COUNT(*) FROM employee e JOIN department d "
        "ON e.dept_id = d.dept_id GROUP BY d.name",
    "Total salary paid in each department id.": "SELECT dept_id, SUM(salary) FROM employee GROUP BY dept_id",
    "Average salary across all employees.": "SELECT AVG(salary) FROM employee",
    # table-qualified projection: identical results (soft_f1 = 1.0, sqlite still names the column
    # `name`) but different projection *text* -> component_match < 1.0
    "Names of employees who have no manager.": "SELECT e.name FROM employee e WHERE e.manager_id IS NULL",
    "The name of the department with the highest budget.":
        "SELECT name FROM department ORDER BY budget DESC LIMIT 1",
    # deliberate bug: dropped the WHERE filter -> parses fine but wrong rows
    "Names of employees earning more than 100000.": "SELECT name FROM employee",
    "Each employee name with their department name.":
        "SELECT e.name, d.name FROM employee e JOIN department d ON e.dept_id = d.dept_id",
    "How many employees have no recorded salary?": "SELECT COUNT(*) FROM employee WHERE salary IS NULL",
    "Department ids having more than 2 employees.":
        "SELECT dept_id FROM employee GROUP BY dept_id HAVING COUNT(*) > 2",
    "Total budget across all departments.": "SELECT SUM(budget) FROM department",
}


class T2SState(TypedDict, total=False):
    input: str  # the natural-language question
    sql: str  # the generated SQL
    execution_result: list[dict]  # the rows the query returned ({column: value} dicts, at predict time)
    output: str  # the final natural-language answer
    tokens: int


def _unique_columns(names: list[str]) -> list[str]:
    """De-duplicate column names so each row maps cleanly to a dict. SQL may repeat a name (e.g. two
    ``name`` columns from a join): the second ``name`` becomes ``name_2``, the third ``name_3``, ..."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def generate_sql_node(state: T2SState) -> dict:
    sql = PRED_SQL.get(state["input"], "SELECT 1")
    return {"sql": sql}


def answer_node(state: T2SState) -> dict:
    """Run the generated SQL and summarize the rows as a natural-language answer."""
    sql = state["sql"]
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = con.execute(sql)
        columns = _unique_columns([d[0] for d in cur.description])
        rows = cur.fetchall()
        con.close()
    except Exception as exc:
        return {"output": f"The query could not be executed: {exc}", "tokens": 12}

    if not rows:
        answer = "The query returned no rows."
    elif len(rows) == 1 and len(rows[0]) == 1:
        answer = f"The result is {rows[0][0]}."
    else:
        preview = "; ".join(", ".join("NULL" if c is None else str(c) for c in r) for r in rows[:10])
        answer = f"The query returned {len(rows)} row(s): {preview}."
    # Capture the result rows as {column: value} dicts so evaluation can score them without re-running.
    return {
        "output": answer,
        "execution_result": [dict(zip(columns, r, strict=True)) for r in rows],
        "tokens": len(sql.split()) + len(answer.split()),
    }


def build_graph():
    graph = StateGraph(T2SState)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "generate_sql")
    graph.add_edge("generate_sql", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


graph = build_graph()  # "examples/t2s/agent.py:graph"
