# The metric catalog

Every metric, organized by agent type, with the **`EvalContext` fields it requires**. Fields written
`metadata['x']` are keys in `EvalContext.metadata` (constants in `agent_eval.core.contracts.MetaKey`);
the prediction harness fills them from your graph state via the config's `state_map` (see
[langgraph-integration.md](langgraph-integration.md)).

Build a registry with every family registered:

```python
from agent_eval.core.registry import default_registry
registry = default_registry()   # 14 metric types (T2S included if `sqlglot` is installed)
```

Legend: **GT** = needs ground truth · **agg** = how scores aggregate (rate/mean/p95) · ↓ = lower is better.

---

## 1. Common — apply to every agent type

They score the **final response** (`output`), so they work for plain, RAG, and T2S agents alike.

| metric | GT? | reads | agg | what |
|---|---|---|---|---|
| `llm_judge` | optional | `input`, `output` (+ `expected`) | mean | LLM-as-a-judge on overall quality |
| `bertscore` | yes | `output`, `expected` | mean | semantic similarity to the gold answer |
| `p95_latency` | no | `metadata['latency_ms']` | p95 ↓ | tail end-to-end latency (ms) |
| `token_usage` | no | `metadata['tokens']` | mean ↓ | mean tokens spent per query |

### `llm_judge`

Scores the response on **Completeness, Clarity, Usefulness, Relevance, and Friendliness** (one
holistic 1–5 score, normalized to `[0, 1]`). With `expected` set it grades against that reference;
without it, intrinsic quality. Needs a judge backend — see [judges.md](judges.md); it falls back to a
deterministic offline stub when none is configured. Params: `criteria` (override the rubric),
`scale` (default `(1, 5)`). Supports a **PoLL panel** (multiple judges averaged) — set
`judge.factory` to a function returning a list.

### `bertscore`

BERTScore F1 between `output` and `expected` (with `rescale_with_baseline` for an interpretable
range). Needs the `[bertscore]` extra. Catches correct answers worded differently, where exact match
would fail. Params: `lang`, `model_type`, `rescale_with_baseline`, `scorer` (inject for testing).

### `p95_latency` / `token_usage`

Performance metrics; **lower is better**, so their gate threshold is an upper bound
(`p95_latency: 2000.0` means ≤ 2000 ms). `latency_ms` is filled automatically by the harness;
`tokens` is filled when your graph state exposes a token count (mapped via `state_map`).

---

## 2. RAG — retrieval quality + answer groundedness

Retrieval metrics work on **ids**: `metadata['retrieved_ids']` is the ranked list the retriever
returned (best first), `metadata['relevant_ids']` is the gold set. `k` is fixed per evaluation
(`defaults.k` in the config, or a metric param).

| metric | GT? | reads | agg | what |
|---|---|---|---|---|
| `recall_at_k` | yes | `metadata['retrieved_ids']`, `metadata['relevant_ids']` | mean | fraction of gold ids in the top-k (the **retrieval gate**) |
| `precision_at_k` | yes | same | mean | fraction of the top-k that are relevant |
| `ndcg_at_k` | yes | same (+ `metadata['relevance']` for graded gains) | mean | rank-weighted relevance |
| `faithfulness` | no | `retrieved_context`, `output` | mean | is every claim supported by the retrieved chunks? |
| `consistency` | no | `retrieved_context`, `output` | mean | does the answer avoid contradicting the chunks? |

`recall_at_k` caps everything downstream, so it's the usual hard gate. `faithfulness` (claims must be
*supported*) and `consistency` (claims must not *contradict*) are judge-based — distinct rubrics, both
reading the retrieved chunk texts in `retrieved_context`. `ndcg_at_k` uses binary relevance by
default; pass `metadata['relevance']` (`{id: gain}`) for graded relevance.

---

## 3. T2S — text-to-SQL correctness + groundedness

Needs the `[t2s]` extra (`sqlglot`) for the AST metrics. The generated SQL lives in `metadata['sql']`
and the gold SQL in `metadata['gold_sql']`. The query is **not executed at eval time**: the predicted
result set is supplied by the agent's own state in `metadata['execution_result']`, and the gold result
set is stored in the dataset in `metadata['gold_execution_result']`. The final natural-language answer
is `output`.

Each result set is canonically an **array of rows, one `dict[str, Any]` per row** (`column -> value`),
e.g. `[{"name": "Alice", "dept": "Engineering"}, ...]`. Loosely-typed encodings are coerced to that
shape on read rather than rejected — a whole set stored as a JSON *string* is parsed, a single
unwrapped row `dict` is wrapped, and positional/scalar rows are named `col1, col2, …`. `soft_f1` grades
the multiset of **`(column, value)` facts**: each cell is one fact, and a true positive is the same
value **under the same column** — so an omitted column costs recall and an aliased/renamed column no
longer matches (column identity is significant). The judge digest labels columns by their real name.

| metric | GT? | reads | agg | what |
|---|---|---|---|---|
| `soft_f1` | yes | `metadata['execution_result']`, `metadata['gold_execution_result']` | mean | cell-F1 of the predicted vs gold result sets (the **correctness gate**) |
| `component_match` | yes | `metadata['sql']`, `metadata['gold_sql']` | mean | AST overlap (tables + projections) — diagnostic |
| `ast_valid` | no | `metadata['sql']` | rate | does the SQL parse? |
| `t2s_faithfulness` | no | `metadata['execution_result']`, `output` | mean | does the NL answer report the query result faithfully? |
| `t2s_consistency` | no | `metadata['execution_result']`, `output` | mean | does the NL answer avoid contradicting the result? |

**Objective correctness is gated on the result set, never on a judge.** `soft_f1` compares the
predicted and gold result sets — both pre-computed and supplied in the data, so no database is touched
at eval time (partial credit via fact-bag F1 over `(column, value)` facts). The judge is confined to whether the NL `output`
faithfully reports what the query returned, and it reads a **bounded statistical digest** of the
result set (per-column aggregates + a small sample), never the raw rows, so it scales to any result
size. Result-set comparison is explicit and configurable — `defaults.result_set_policy` controls row
order, duplicates, NULLs, and float tolerance (the documented silent-failure source for execution
metrics). Params: `soft_f1` takes `result_policy`; `component_match` / `ast_valid` take `dialect`; the
judge metrics take `sample_rows` / `max_distinct` / `max_columns`.

> The T2S example ([`examples/t2s`](../examples/t2s/)) is built to show the division of labor:
> a paraphrased query keeps `soft_f1` at 1.0 while `component_match` dips; a dropped `WHERE` keeps
> `ast_valid` at 1.0 while `soft_f1` falls. Correctness and groundedness are separate questions.

---

## A note on the "consistency" naming

The final metric set lists *consistency* under both RAG and T2S. They share the idea (the answer must
not contradict the evidence) but read different evidence, so the registry exposes them under distinct
type names: `consistency` (RAG, evidence = retrieved chunks) and `t2s_consistency` (T2S, evidence =
the query's result set, as a bounded digest). Same for `faithfulness` / `t2s_faithfulness`.
