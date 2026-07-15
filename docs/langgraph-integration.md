# Evaluating your LangGraph agent

This is the worked guide to pointing the framework at *your* compiled LangGraph and getting a gated
verdict. The only thing the framework needs from your agent is a small **state contract**: which
fields of your graph's final state hold the things the metrics score.

```
 your gold dataset ──► agent-eval predict ──► predictions.jsonl ──► agent-eval evaluate ──► verdict
                       (runs your graph)                            (scores + gate)
```

Install the extra that lets the harness drive a graph:

```bash
uv sync --extra langgraph        # add --extra t2s for text-to-SQL metrics
```

## 1. The state contract

The harness invokes your compiled graph once per input and reads the **final state dict**. You tell
it which state keys map onto the canonical fields the metrics read, via the config's `state_map`
(`canonical field: your state key`). The canonical fields and the metrics that read them:

| canonical field | where it goes | read by |
|---|---|---|
| `output` | `EvalContext.output` | `llm_judge`, `bertscore`, `faithfulness`, `t2s_faithfulness` |
| `retrieved_context` | `EvalContext.retrieved_context` | `faithfulness` |
| `retrieved_ids` | `metadata['retrieved_ids']` | `recall_at_k`, `precision_at_k`, `ndcg_at_k` |
| `sql` | `metadata['sql']` | `soft_f1`, `component_match`, `ast_valid`, `t2s_*` |
| `tokens` | `metadata['tokens']` | `token_usage` |

`latency_ms` is filled automatically (the harness times each run). Anything not in the canonical set
lands in `metadata` under its own name. With `prediction.n_runs > 1` the harness runs the graph N
times per input and stores the repeated generations in `metadata['repeated_outputs']` /
`['repeated_sql']` — that is what the self-consistency metrics (`consistency`, `t2s_consistency`)
read. The full field reference is in [metrics.md](metrics.md).

Expose whatever a metric needs as a key in your graph's state. For example, a RAG agent should put
the ranked ids it retrieved and the chunk texts into its state so retrieval and groundedness can be
scored:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RagState(TypedDict, total=False):
    input: str                    # the question (the harness sets this)
    retrieved_ids: list[str]      # ranked doc ids   → recall/precision/ndcg
    retrieved_context: list[str]  # chunk texts      → faithfulness
    output: str                   # the final answer → llm_judge
    tokens: int

# ... add nodes ...
graph = builder.compile()         # the config points at this object
```

## 2. The gold dataset

A file (JSONL or CSV/Excel/Parquet) with your inputs and whatever ground truth you have. Only the
GT metrics need labels; reference-free metrics need none.

```jsonl
{"question": "What is the capital of France?", "relevant": ["d1","d2"], "reference": "Paris is the capital of France."}
```

## 3. The config

```yaml
version: 1
agent_type: rag
defaults: {k: 3}

datasets:
  gold:                                    # your labels
    adapter: jsonl
    path: data/gold.jsonl
    field_map: {input: question, expected: reference, relevant_ids: relevant}
  predictions:                             # written by `predict`, read by `evaluate`
    adapter: jsonl
    path: data/predictions.jsonl

prediction:
  graph: my_pkg.agent:graph                # "module:attr" or "path/to/agent.py:graph"
  source: gold                             # dataset of inputs to run the agent on
  target: predictions                      # dataset to write predictions into
  input_key: input                         # graph state key the question is passed under
  state_map:                               # canonical field <- your graph state key
    retrieved_ids: retrieved_ids
    retrieved_context: retrieved_context
    output: output
  n_runs: 3                     # repeated runs → metadata['repeated_outputs'] (consistency)

suites:
  retrieval:
    dataset: predictions
    metrics: [recall_at_k, precision_at_k, ndcg_at_k]
    gate: {thresholds: {recall_at_k: 0.7}, require_pass: [recall_at_k]}
  response:
    dataset: predictions
    metrics: [faithfulness, consistency, llm_judge]
    gate: {thresholds: {faithfulness: 0.6}, require_pass: [faithfulness]}
```

`field_map` (target ← source) maps your gold columns onto canonical fields; `state_map` (target ←
graph key) does the same for your agent's output. Both share one vocabulary — top-level fields
(`input`, `output`, `expected`, `retrieved_context`) and metadata keys (everything else). Full
reference: [configuration.md](configuration.md).

## 4. Run it

```bash
uv run agent-eval predict  -c eval.yaml      # runs your graph → predictions.jsonl
uv run agent-eval evaluate -c eval.yaml      # scores + gates (exit non-zero on failure)
```

`predict` merges each gold record with the prediction your agent produced and writes a ready-to-score
predictions file. `evaluate` then runs the suites and prints a gated report.

## Resolving the graph

`prediction.graph` is `module:attr` (an importable package) or `path/to/file.py:attr` (a script —
its directory is put on `sys.path` so it can import siblings). The attribute must be either:

- a **compiled graph** (anything with `.invoke`), or
- a **zero-arg factory** returning one.

`run_once` calls `graph.invoke({input_key: question})` and reads the returned state dict. If your
graph needs a richer input shape than a single key, wrap it in a small factory that adapts it.

## Driving it from Python

Everything the CLI does is available as a library:

```python
from agent_eval.config.loader import load_config, build_suite, build_dataset_spec
from agent_eval.core.registry import default_registry
from agent_eval.datasets.base import load_dataset
from agent_eval.harness.predict import predict
from agent_eval.offline.runner import evaluate

cfg = load_config("eval.yaml")
predict(cfg)                                                   # fill predictions
suite = build_suite(cfg, "retrieval", default_registry())
result = evaluate(suite, load_dataset(build_dataset_spec(cfg, "predictions")))
print(result.verdict.passed)
```

## Not using LangGraph?

The harness is a convenience. If you already have predictions (from any agent, in any framework),
skip `predict` entirely: write a predictions file with the canonical fields and run `evaluate`
directly. `output` is just "the final response"; `metadata` carries the rest.
