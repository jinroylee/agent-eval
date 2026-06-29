# Examples — evaluate a LangGraph agent end to end

Three complete, runnable examples, one per agent type. Each shows the **same workflow**:

```
 gold dataset  ──►  agent-eval predict  ──►  predictions  ──►  agent-eval evaluate  ──►  gated verdict
 (your labels)      (runs your graph)       (jsonl)            (scores + thresholds)
```

| Example | Agent type | Metrics it exercises |
|---|---|---|
| [`plain/`](plain/) | A single-shot Q&A agent | common: `llm_judge`, `p95_latency`, `token_usage` |
| [`rag/`](rag/) | Retrieve-then-answer | common + RAG: `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `faithfulness`, `consistency` |
| [`t2s/`](t2s/) | Text-to-SQL | common + T2S: `soft_f1`, `component_match`, `ast_valid`, `t2s_faithfulness`, `t2s_consistency` |

## Setup

```bash
uv sync --extra langgraph --extra t2s      # langgraph runs the agents; t2s adds sqlglot
```

## Run any example

```bash
# 1. Run the agent over the gold questions to produce predictions:
uv run agent-eval predict  -c examples/rag/rag.yaml

# 2. Score the predictions and gate on thresholds (exit non-zero on failure):
uv run agent-eval evaluate -c examples/rag/rag.yaml
```

(For T2S, first build the sample database: `uv run python examples/t2s/build_db.py`.)

## The two pieces you write for your own agent

1. **A gold dataset** — your questions plus whatever ground truth you have (reference answers,
   relevant doc ids, gold SQL). See each example's `gold.*` file.
2. **A config** — points at your compiled LangGraph, maps your **graph state fields** to the metrics'
   inputs (`state_map`), and lists the metrics + thresholds. See each example's `*.yaml`.

The agent under test is an ordinary compiled LangGraph (`agent.py` in each example). The only
contract is the **state fields** the metrics need — each example documents them at the top of its
`agent.py` and wires them in the config's `state_map`. See
[`../docs/langgraph-integration.md`](../docs/langgraph-integration.md) for the full field reference.

## Judges: offline by default, real Claude when you want it

The judge-based metrics (`llm_judge`, `faithfulness`, `consistency`, `t2s_*`) fall back to a
**deterministic token-overlap stub** so every example runs with no API key. To use a real Claude
judge, uncomment the `judge:` block in any config (it points at
[`judges.py`](judges.py)) and:

```bash
uv pip install anthropic && export ANTHROPIC_API_KEY=...
```
