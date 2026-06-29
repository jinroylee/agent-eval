# agent-eval — documentation

An offline **evaluation framework for LangGraph agents**: run your agent over a labeled dataset,
score it with a focused metric set, and **gate your CI** on statistically-defensible thresholds.

```
 gold dataset ──► agent-eval predict ──► predictions ──► agent-eval evaluate ──► gated verdict
```

Three agent types, one metric core:

- **Common** — `llm_judge`, `bertscore`, `p95_latency`, `token_usage`
- **RAG** — `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `faithfulness`, `consistency`
- **T2S** — `soft_f1`, `component_match`, `ast_valid`, `t2s_faithfulness`, `t2s_consistency`

## Install

```bash
uv sync --extra langgraph --extra t2s      # run agents + text-to-SQL metrics
```

## Quickstart

```bash
uv run agent-eval predict  -c examples/rag/rag.yaml
uv run agent-eval evaluate -c examples/rag/rag.yaml
```

## Where to go next

| Doc | What it covers |
|---|---|
| [concepts.md](concepts.md) | The mental model: metric core, `EvalContext`, GT vs non-GT, aggregation, the gate. **Read first.** |
| [metrics.md](metrics.md) | The metric catalog (Common / RAG / T2S) and the **graph-state fields each metric requires**. |
| [langgraph-integration.md](langgraph-integration.md) | Evaluate your own LangGraph agent: the state contract, `state_map`, `agent-eval predict`. |
| [offline-evaluation.md](offline-evaluation.md) | Datasets, running the gate, the statistics, reporting, CI. |
| [configuration.md](configuration.md) | The complete YAML reference. |
| [judges.md](judges.md) | LLM-as-a-judge: the rubrics, PoLL, and wiring a real Claude judge. |
| [customizing.md](customizing.md) | Add a metric, a dataset adapter, a judge backend, or a plugin. |
| [repository-structure.md](repository-structure.md) | The physical layout and a "where does X live?" map. |
| [runtime-critic.md](runtime-critic.md) | **Foundation** for a future in-flight critique mode that reuses the same metrics (not part of the offline gate). |
| [final_metric_list.md](final_metric_list.md) | The agreed target metric set this framework implements. |

The bundled [`examples/`](../examples/) are the fastest way in — one runnable example per agent type.
`research/` holds the original SOTA research report that informed the method choices (a point-in-time
record; the live metric set is whatever [metrics.md](metrics.md) lists).
