# agent-eval — documentation

A reusable **evaluation + runtime-critique** framework for **LangGraph** agents.

It does two jobs over one shared metric core:

- **Offline (CI/CD gate).** Holistically score an agent before deploy and **block the release** if it
  fails statistically-defensible thresholds. `agent-eval evaluate` exits non-zero and emits JUnit.
- **Runtime (in-flight critic).** Inside a node or tool-call, decide **accept / retry / fallback /
  escalate / abstain** when a step misses requirements.

It covers four agent types — **Plain, RAG, T2S (text-to-SQL/Cypher), Orchestration** — across four
categories — **search, response, latency, scenario** — and can evaluate a LangGraph at any level:
**whole-graph → subgraph → node → tool-call**. Everything is **config-first (YAML)** with a **Python
plugin escape hatch**, and bring-your-own dataset (CSV/Excel/Parquet/JSONL).

> The method/library choices are grounded in a verified SOTA research report:
> [`research/2026-06-05-agent-eval-sota-research.md`](research/2026-06-05-agent-eval-sota-research.md).

---

## Install

Requires **Python 3.12**. The project uses [`uv`](https://docs.astral.sh/uv/).

```bash
# core only (pure metrics + offline gate + stats + CLI)
uv sync

# add the optional extras you need:
uv sync --extra t2s         # text-to-SQL/Cypher: sqlglot, sqlalchemy
uv sync --extra agentevals  # LangGraph instrumentation + agentevals: langgraph, langchain-core
uv sync --extra t2s --extra agentevals   # both (what the test suite uses)
```

Other extras: `deepeval` (G-Eval judge), `ragas` (RAG LLM metrics), `inspect` (sandboxed scenario
suites), `obs` (OpenTelemetry export). These power **lazy adapters** — install only when you wire a
real LLM/backend.

## 60-second quickstart

```bash
# Score a tiny "Plain" agent (exact-match QA) and gate on a threshold:
uv run agent-eval evaluate --config examples/toy/toy.yaml
```
```
=== plain/response  dataset=qa  n=6 ===
metric                     value              95% CI    thr  gate
exact_match                0.667      [0.300, 0.903]   0.60  PASS
nonempty                   1.000      [0.610, 1.000]      -  info
VERDICT: PASS
```

Run the other built-in examples the same way:

```bash
uv run agent-eval evaluate -c examples/rag/rag.yaml            # RAG retrieval gate (Recall@k)
uv run agent-eval evaluate -c examples/t2s/t2s.yaml            # text-to-SQL (Execution Accuracy)
uv run agent-eval evaluate -c examples/orchestration/orchestration.yaml   # tool-trajectory gate
```

Calibrate the runtime critic's thresholds from an offline run (writes them back into the config):

```bash
uv run agent-eval calibrate -c examples/t2s/t2s.yaml --category search
```

## Where to go next

| Doc | What it covers |
|---|---|
| [concepts.md](concepts.md) | The mental model: the metric core, two modes, three tiers, multi-level, the calibration link, and the core data types. **Read this first.** |
| [configuration.md](configuration.md) | The complete YAML reference — every section and field. |
| [offline-evaluation.md](offline-evaluation.md) | Datasets, running the gate, reading verdicts, the statistics, regression vs a baseline, calibration, and CI integration. |
| [runtime-critic.md](runtime-critic.md) | The in-flight critic: policy, tiers, the retry→fallback loop, loop guards, and the built-in per-agent-type critics. |
| [langgraph-integration.md](langgraph-integration.md) | **The full worked example** — evaluate *and* critique a real LangGraph agent (observational + active, at the level you choose). |
| [agent-types.md](agent-types.md) | The four built-in agent types and their metrics + example configs. |
| [judges.md](judges.md) | LLM-as-judge governance: certify, pin, panel-of-judges, PPI, the `UncertifiedJudge` gate, and wiring a real G-Eval judge. |
| [statistics.md](statistics.md) | The statistics toolkit and *why* (Wilson vs CLT, clustered SEs, pass^k, Cost-of-Pass, PPI, anytime-valid drift). |
| [customizing.md](customizing.md) | **Extending the framework**: add a metric, a dataset adapter, a fallback, a judge backend, or a whole new agent type — via config, in-code, or as a plugin. |

The implementation plan / phasing lives in [`plan.md`](plan.md).
