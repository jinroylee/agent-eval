# The four built-in agent types (and the metric catalog)

Each agent type is just a **registry** (which metric types are available) + a **critic builder** +
example config. None of them are special-cased in the core — they're the same `Metric`/`Suite`/
`Critic` machinery wired with different defaults. A fifth type is a new registry + builder away
(see [customizing.md](customizing.md)).

```python
from agent_eval.agents.plain.suites import plain_registry
from agent_eval.agents.t2s.suites import t2s_registry
from agent_eval.agents.rag.suites import rag_registry
from agent_eval.agents.orchestration.suites import orchestration_registry
```

## Metric catalog

`requires` = `EvalContext` fields the metric needs (`metadata[...]` keys noted in *italics*).

### Deterministic — always available (`default_registry()`)

| type | params | kind | reads | what |
|---|---|---|---|---|
| `exact_match` | — | binary | output, expected | `output == expected` |
| `regex_match` | `pattern`, `flags` | binary | output | `re.search(pattern, output)` |
| `set_match` | — | binary | output, expected | order-insensitive set equality |
| `numeric_tolerance` | `tol` | binary | output, expected | `|float(output)-float(expected)| <= tol` |
| `json_shape` | `required_keys` | binary | output | output (JSON/dict) has all keys |

### T2S — `t2s_registry()` (needs `--extra t2s`; execution metrics read *`db_ref`*)

| type | params | kind | what |
|---|---|---|---|
| `execution_accuracy` | `dialect`, `result_policy`, `timeout_s` | binary (gate) | predicted vs gold **result-set** match |
| `soft_f1` | … | graded | cell-bag F1 of result sets (partial credit) |
| `component_match` | `dialect` | graded | AST overlap (tables + projections) — diagnostic |
| `ast_valid` | `dialect` | binary | predicted query parses (runtime tier-1) |
| `schema_linking` | `dialect` | binary | referenced tables/columns exist (runtime) |
| `exec_validity` | … | binary | predicted query executes without error (runtime) |

### RAG retrieval — `rag_registry()` (reads `output`=retrieved ids or *`retrieved_ids`*, `expected`=relevant or *`relevant_ids`*)

| type | params | kind | what |
|---|---|---|---|
| `recall_at_k` | `k` | graded (gate) | fraction of relevant retrieved in top-k |
| `precision_at_k` | `k` | graded | fraction of top-k that are relevant |
| `hit_at_k` | `k` | binary | any relevant in top-k |
| `mrr` | — | graded | mean reciprocal rank |
| `ndcg_at_k` | `k` | graded | graded-relevance ranking quality |
| `retrieval_sufficiency` | `k`, `min_recall` | binary | CRAG gate: Recall@k ≥ floor (else re-retrieve) |

### Uncertainty — `rag_registry()` / `plain_registry()` (read *`samples`* = list of sampled answers)

| type | params | kind | what |
|---|---|---|---|
| `selfcheck_consistency` | `threshold` | graded (confidence) | fraction of samples consistent with the main answer |
| `semantic_entropy` | `sim_threshold` | graded (confidence) | 1 − normalized entropy over meaning-clusters |

### Orchestration — `orchestration_registry()`

| type | params | kind | reads | what |
|---|---|---|---|---|
| `trajectory_match` | `mode` (`strict`/`unordered`/`subset`/`superset`) | binary | trajectory, expected | tool-sequence match; **subset = no-forbidden-tool gate** |
| `tool_arg_validity` | — | graded | trajectory, *`tool_schemas`* | each call: tool exists, required args present, types match |
| `latency_budget` | `max_ms` | binary | *`latency_ms`* | step latency within budget |
| `token_budget` | `max_tokens` | binary | *`tokens`* | step tokens within budget |

### Judge — constructed in code (`metrics/judge_.py`), see [judges.md](judges.md)

`JudgeMetric(backend, criteria, panel=…)` — tier JUDGE, graded + confidence. Confined to subjective
quality; certified & pinned before it may gate.

---

## Plain

Single-shot LLM response. No retrieval or trajectory to verify, so the **response** category carries
the load (a certified judge for subjective quality; deterministic checks for format/compliance), and
the runtime critic is a label-free **hallucination gate**.

- Registry: `plain_registry()` (deterministic + uncertainty).
- Critic: `build_plain_critic(cfg, threshold=0.6)` → `SelfCheckConsistency` (UNCERTAINTY) → low
  consistency *escalates*.
- Example: [`examples/toy/toy.yaml`](../examples/toy/toy.yaml) (exact-match QA gate).

## T2S (text-to-SQL / text-to-Cypher)

Objective correctness with an oracle (execution), so **gate on execution + AST, not a judge**
(JudgeBench: judges fail on objective tasks). The LLM judge is confined to NL-intent/explanation.

- Registry: `t2s_registry()`. Critic: `build_t2s_critic(cfg)` → AST validity → schema linking →
  dry-run execution; retry with the grounded error injected.
- Result-set comparison is explicit/configurable (`defaults.result_set_policy`: row order,
  duplicates, NULLs, column order, float tolerance) — the documented silent-failure source for EX.
- Example: [`examples/t2s/t2s.yaml`](../examples/t2s/t2s.yaml) (+ `gold.csv`, `sample.sqlite`).

```bash
uv run agent-eval evaluate  -c examples/t2s/t2s.yaml
uv run agent-eval calibrate -c examples/t2s/t2s.yaml --category search   # writes runtime_critic.tau
```

## RAG

Two categories: **search** (retrieval quality — caps everything downstream) and **response**
(faithfulness/groundedness). Retrieval gate is **Recall@k**; the runtime critic is CRAG-style
re-retrieval + a semantic-entropy groundedness flag.

- Registry: `rag_registry()`. Critic: `build_rag_critic(cfg, k=10, min_recall=0.5)` →
  `RetrievalSufficiency` (re-retrieve) + `SemanticEntropy`.
- Example: [`examples/rag/rag.yaml`](../examples/rag/rag.yaml).
- For LLM faithfulness/answer-relevancy, configure a `RagasMetric` (lazy; needs `--extra ragas` +
  an LLM) or a `JudgeMetric` in the response suite.

## Orchestration

A planner/router over sub-agents/tools. **search** = plan/tool-use correctness (trajectory match +
tool-arg validity); **scenario** = end-to-end reliability via **pass^k**; per-step critic validates
each tool call before it runs.

- Registry: `orchestration_registry()`. Critic: `build_orchestration_critic(cfg)` → `ToolArgValidity`.
- `subset` trajectory match is a clean **hard safety gate** (used no tool outside the allowed set).
- Example: [`examples/orchestration/orchestration.yaml`](../examples/orchestration/orchestration.yaml).

```python
# reliability over repeated end-to-end runs (not pass@k):
from agent_eval.metrics.perf_ import scenario_pass_hat_k
scenario_pass_hat_k(per_task_runs=[[True,True,True,False,True], ...], k=3).estimate
```
