# Text-to-SQL (T2S) agent example

An agent that turns a question into SQL, runs it, and reports the answer in natural language.
Objective correctness is gated on **execution** (never on a judge); a judge only checks whether the
natural-language answer faithfully reports what the query returned.

## Files

| File | What it is |
|---|---|
| [`build_db.py`](build_db.py) | Creates `sample.sqlite` + `gold.csv` (run this first) |
| [`agent.py`](agent.py) | NL→SQL→answer LangGraph (`graph`) |
| [`gold.csv`](gold.csv) | `question`, gold `gold_sql`, `db` path (written by `build_db.py`) |
| [`t2s.yaml`](t2s.yaml) | Config: dialect + result-set policy, prediction wiring, two suites |

## Required graph state fields

| canonical field | graph state key | used by |
|---|---|---|
| `sql` | `sql` | `soft_f1`, `component_match`, `ast_valid`, `t2s_*` (the generated SQL) |
| `output` | `output` | `t2s_faithfulness`, `t2s_consistency`, `llm_judge` (the NL answer) |
| `tokens` | `tokens` | (available for `token_usage`) |

The gold dataset supplies `metadata['gold_sql']` and `metadata['db_ref']` (the database the metrics
execute against).

## Run it

```bash
uv run python examples/t2s/build_db.py             # one-time: build sample.sqlite + gold.csv
uv run agent-eval predict  -c examples/t2s/t2s.yaml
uv run agent-eval evaluate -c examples/t2s/t2s.yaml
```

Expected:

```
=== t2s/correctness  dataset=predictions  n=12 ===
soft_f1                    0.962        [0.888, 1.000]      0.80  PASS
component_match            0.931        [0.837, 1.000]         -  info
ast_valid                  1.000        [0.758, 1.000]      0.95  PASS
VERDICT: PASS

=== t2s/response  dataset=predictions  n=12 ===
t2s_faithfulness           0.423        [0.337, 0.509]         -  info
t2s_consistency            0.423        [0.337, 0.509]         -  info
llm_judge                  0.500        [0.500, 0.500]         -  info
```

This example is built to show the **division of labor** between metrics. The agent deliberately:

- **paraphrases** two queries (aliased aggregates) → same results so `soft_f1` stays 1.0, but the
  AST differs so `component_match` dips below 1.0;
- **drops a `WHERE` filter** on one query → it still parses (`ast_valid` = 1.0) but returns the wrong
  rows (`soft_f1` < 1.0). Its NL answer faithfully reports its *own* (wrong) result, so
  `t2s_faithfulness` doesn't flag it — `soft_f1` does. Correctness and groundedness are separate
  questions, measured separately.

The `t2s_*` judge scores are low only because the offline stub compares tokens literally; a real
Claude judge (uncomment `judge:` in the config) scores a faithful answer near 1.0. Until then they
are left informational (no threshold).
