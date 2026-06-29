# agent-eval

An offline **evaluation framework for LangGraph agents**. Run your agent over a labeled dataset,
score it with a focused set of metrics, and **gate your CI** on statistically-defensible thresholds.

It covers three agent types with one shared metric core:

- **Common** (every agent) — `llm_judge` (LLM-as-a-judge, with PoLL), `bertscore`, `p95_latency`, `token_usage`
- **RAG** — `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `faithfulness`, `consistency`
- **T2S** (text-to-SQL) — `soft_f1`, `component_match`, `ast_valid`, `t2s_faithfulness`, `t2s_consistency`

Everything is **config-first (YAML)** with a Python escape hatch, bring-your-own dataset
(JSONL/CSV/Excel/Parquet), and a built-in harness that runs your LangGraph agent to fill in
predicted answers.

## How it works

```
 gold dataset  ──►  agent-eval predict  ──►  predictions  ──►  agent-eval evaluate  ──►  gated verdict
 (your labels)      (runs your graph)        (jsonl)            (metrics + thresholds)   (CI exit code)
```

1. **`predict`** invokes your compiled LangGraph on each input and captures its outputs into a
   predictions dataset (a `state_map` says which graph-state fields the metrics read).
2. **`evaluate`** scores the predictions per metric — with small-sample-correct confidence intervals
   — and applies your pass/fail gate (exit non-zero on failure, for CI).

## Install

Requires **Python 3.12** ([`uv`](https://docs.astral.sh/uv/)):

```bash
uv sync                                  # core: metrics + offline gate + CLI
uv sync --extra langgraph                # + run your agent with `agent-eval predict`
uv sync --extra t2s                      # + text-to-SQL metrics (sqlglot)
uv sync --extra bertscore                # + BERTScore (bert-score)
```

## 60-second quickstart

```bash
uv sync --extra langgraph --extra t2s
uv run agent-eval predict  -c examples/rag/rag.yaml   # run the toy RAG agent → predictions
uv run agent-eval evaluate -c examples/rag/rag.yaml   # score + gate
```

```
=== rag/retrieval  dataset=predictions  n=5 ===
metric                     value                95% CI       thr  gate
recall_at_k                1.000        [1.000, 1.000]      0.70  PASS
precision_at_k             0.533        [0.373, 0.693]         -  info
ndcg_at_k                  1.000        [1.000, 1.000]         -  info
VERDICT: PASS
```

Three runnable, self-contained examples live in [`examples/`](examples/) — one per agent type.

## Documentation

Start with [`docs/README.md`](docs/README.md). Highlights:

| Doc | What it covers |
|---|---|
| [concepts.md](docs/concepts.md) | The mental model: metric core, `EvalContext`, GT vs non-GT, aggregation, the gate. **Read first.** |
| [metrics.md](docs/metrics.md) | The full metric catalog (Common / RAG / T2S) and the **graph-state fields each one requires**. |
| [langgraph-integration.md](docs/langgraph-integration.md) | Wire your own LangGraph agent: the state contract, `state_map`, and `agent-eval predict`. |
| [offline-evaluation.md](docs/offline-evaluation.md) | Datasets, running the gate, reading verdicts, the statistics, CI integration. |
| [configuration.md](docs/configuration.md) | The complete YAML reference. |
| [judges.md](docs/judges.md) | LLM-as-a-judge: the rubrics, PoLL, and wiring a real Claude judge. |
| [customizing.md](docs/customizing.md) | Add a metric, a dataset adapter, or a judge backend. |
