# Text-to-SQL (T2S) agent example

An agent that turns a question into SQL, runs it, and reports the answer in natural language.
Objective correctness is graded on the query's **result set** — but the query is **not executed at
eval time**: the agent captures its own result rows at predict time, and the gold result rows are
pre-computed into the dataset. A judge only checks whether the natural-language answer faithfully
reports what the query returned.

## Files

| File | What it is |
|---|---|
| [`build_db.py`](build_db.py) | Builds `sample.sqlite` and writes `gold.jsonl` with **pre-computed** gold results (run first) |
| [`agent.py`](agent.py) | NL→SQL→answer LangGraph (`graph`); captures its result rows in `execution_result` |
| [`gold.jsonl`](gold.jsonl) | `question`, gold `gold_sql`, and `gold_execution_result` (rows) — written by `build_db.py` |
| [`t2s.yaml`](t2s.yaml) | Config: dialect + result-set policy, prediction wiring, two suites |
| [`run_eval.py`](run_eval.py) | Programmatic predict→evaluate (the import-and-run alternative to the CLI) |

## Required graph state fields

| canonical field | graph state key | used by |
|---|---|---|
| `sql` | `sql` | `component_match`, `ast_valid` (the generated SQL text) |
| `execution_result` | `execution_result` | `soft_f1`, `t2s_faithfulness` (the rows the query returned) |
| `output` | `output` | `llm_judge`, `t2s_faithfulness` (the NL answer) |
| `sql` (run N times via `n_runs`) | `metadata['repeated_sql']` | `t2s_consistency` (do repeated runs agree?) |
| `tokens` | `tokens` | (available for `token_usage`) |

The gold dataset supplies `metadata['gold_sql']` (for the AST diagnostic) and the pre-computed
`metadata['gold_execution_result']` (the rows `soft_f1` compares against). **No database is opened
during `agent-eval evaluate`** — only `build_db.py` and the agent (at predict time) run SQL.

## Run it

```bash
uv run python examples/t2s/build_db.py             # one-time: build sample.sqlite + gold.jsonl
uv run agent-eval predict  -c examples/t2s/t2s.yaml
uv run agent-eval evaluate -c examples/t2s/t2s.yaml
# or, equivalently, the programmatic script:
uv run python examples/t2s/run_eval.py
```

Expected:

```
=== t2s/correctness  dataset=predictions  n=12 ===
soft_f1                    0.962        [0.888, 1.000]      0.80  PASS
component_match            0.958        [0.877, 1.000]         -  info
ast_valid                  1.000        [0.758, 1.000]      0.95  PASS
VERDICT: PASS

=== t2s/response  dataset=predictions  n=12 ===
t2s_faithfulness           0.650        [0.581, 0.719]         -  info
t2s_consistency            1.000        [1.000, 1.000]         -  info
llm_judge                  0.500        [0.500, 0.500]         -  info
```

This example is built to show the **division of labor** between metrics. The agent deliberately:

- **qualifies a column** on one query (`e.name` vs gold `name`) → the same result set, so `soft_f1`
  stays 1.0 (it grades `(column, value)` facts and sqlite still names the column `name`), but the
  projection *text* differs so `component_match` dips below 1.0;
- **drops a `WHERE` filter** on one query → it still parses (`ast_valid` = 1.0) but returns the wrong
  rows (`soft_f1` < 1.0). Its NL answer faithfully reports its *own* (wrong) result, so
  `t2s_faithfulness` doesn't flag it — `soft_f1` does. Correctness and groundedness are separate
  questions, measured separately.

The `t2s_*` judge scores are low only because the offline stub compares tokens literally; a real
Claude judge (uncomment `judge:` in the config) scores a faithful answer near 1.0. Until then they
are left informational (no threshold).
