# Configuration reference (YAML)

One file describes how to run your agent, where the data lives, which judge to use, and what to score
and gate. It is validated against a pydantic schema (`agent_eval.config.schema`). Environment
variables are interpolated (`${VAR}` / `$VAR`).

```yaml
version: 1
agent_type: rag                  # plain | rag | t2s — informational (drives nothing)

defaults:                        # shared metric defaults
  k: 5                           # fixed retrieval depth for recall/precision/ndcg
  dialect: sqlite                # T2S SQL dialect
  system_prompt: null            # judge persona for ALL judge metrics (see docs/judges.md)
  result_set_policy:             # T2S result-set comparison semantics
    row_order: ignore            #   ignore | strict
    duplicates: keep             #   keep | dedup
    nulls: distinct              #   distinct | coalesce
    float_tolerance: 1.0e-6

judge:                           # optional — omit to use the deterministic offline stub
  factory: my_pkg.judges:claude_judge   # "module:attr" or "file.py:attr" returning a JudgeBackend
                                        # (or a list of them = a PoLL panel)

datasets:
  gold:
    adapter: jsonl               # jsonl | tabular
    path: data/gold.jsonl
    format: null                 # tabular only: csv | tsv | excel | parquet (inferred if null)
    field_map:                   # target field <- source column
      input: question
      expected: reference
      relevant_ids: relevant
    include_unmapped: true       # keep unmapped source columns in metadata
  predictions:
    adapter: jsonl
    path: data/predictions.jsonl

prediction:                      # how `agent-eval predict` fills predictions (optional)
  graph: my_pkg.agent:graph      # compiled LangGraph, or a zero-arg factory returning one
  source: gold                   # dataset of inputs to run the agent on
  target: predictions            # dataset to write predictions into (must be jsonl)
  input_key: input               # graph state key the question is passed under
  n_runs: 1                      # >1: run the graph N times per input and store the repeated
                                 #     generations (repeated_outputs / repeated_sql) for consistency
  state_map:                     # canonical field <- graph final-state key
    output: output
    retrieved_ids: retrieved_ids
    retrieved_context: retrieved_context

suites:                          # one or more named evaluations
  retrieval:
    dataset: predictions         # which dataset to score (defaults to the first dataset)
    metrics:                     # bare type strings, or {type, name, params}
      - recall_at_k
      - {type: precision_at_k, name: precision_at_k, params: {k: 10}}
    gate:
      thresholds: {recall_at_k: 0.7}   # metric → threshold (direction per metric.higher_is_better)
      require_pass: [recall_at_k]      # hard ship-blockers (must run AND pass)
```

## Sections

| Section | Purpose |
|---|---|
| `agent_type` | A label (`plain` / `rag` / `t2s`). Informational — the metric set is whatever the suites list. |
| `defaults` | Shared params injected into metric factories: `k`, `dialect`, `result_set_policy`, `system_prompt` (judge persona for every judge metric). A metric's own `params` override these. |
| `judge` | Resolves the LLM judge backend for judge-based metrics. See [judges.md](judges.md). Omitted ⇒ deterministic stub. |
| `datasets` | Named datasets. `field_map` projects source columns onto canonical fields (target ← source). |
| `prediction` | Wires `agent-eval predict` to your LangGraph. See [langgraph-integration.md](langgraph-integration.md). |
| `suites` | Named groups of metrics + a gate. `agent-eval evaluate` runs all of them (or `--category <name>`). |

## Metric specs

A metric in a suite is either a bare type string (`recall_at_k`) or a dict:

```yaml
- {type: recall_at_k, name: recall_at_10, params: {k: 10}}
- type: llm_judge
  params:
    system_prompt: "You are a meticulous Korean-speaking evaluator."   # judge persona override
    criteria:                              # per-criteria judging (one judge call per criterion);
      - Accuracy                           #   a plain string rubric = one holistic call instead
      - {name: Tone, description: "polite, professional"}
```

`type` selects the metric (see [metrics.md](metrics.md) for types + params); `name` is the reporting
name (defaults to `type`); `params` are constructor arguments. Judge-based metrics receive the
configured judge automatically — you don't pass it in `params`.

## The gate

`thresholds` maps a metric name to a number. The comparison direction follows the metric:
`recall_at_k: 0.7` means **≥ 0.7**, while `p95_latency: 2000` means **≤ 2000** (lower-is-better
metrics). `require_pass` lists metrics that must have run *and* passed — a metric that errored on
every item, or isn't in the suite, fails the gate. Metrics with no threshold are reported as `info`.

## CLI

```bash
agent-eval predict  -c eval.yaml                 # run the agent → predictions
agent-eval evaluate -c eval.yaml                 # all suites
agent-eval evaluate -c eval.yaml --category retrieval   # one suite
agent-eval evaluate -c eval.yaml --dataset other_predictions   # override the dataset
```

Exit codes for `evaluate`: **0** all gates passed, **1** a gate failed, **2** config/usage error.
