# Configuration reference (YAML)

A config file fully describes an agent's evaluation: its datasets, the suites (metrics + gate) per
category, the optional judges, and the runtime-critic policy. The **same file** is read by the
offline runner and (after `calibrate`) by the runtime critic.

Load and validate it in Python with:

```python
from agent_eval.config.loader import load_config
cfg = load_config("examples/t2s/t2s.yaml")   # -> a validated ConfigModel (pydantic)
```

`${ENV_VAR}` / `$ENV_VAR` anywhere in the file are expanded from the environment before parsing
(handy for dataset paths, API keys, DB references).

## Top-level shape

```yaml
version: 1
agent_type: t2s              # free-form label: plain | rag | t2s | orchestration | <your own>
defaults: {}                 # free-form dict you can read in custom code (e.g. result_set_policy)
datasets: {}                 # name -> dataset spec
judges: {}                   # id   -> judge spec (LLM-as-judge)
suites: {}                   # category -> suite spec
runtime_critic: {}           # the in-flight critic policy (optional)
eval_targets: []             # optional list of default EvalTargets
```

## `datasets`

Each named dataset maps your raw records into `EvalContext`s.

```yaml
datasets:
  gold:
    adapter: tabular         # tabular | jsonl
    path: examples/t2s/gold.csv
    format: csv              # csv | tsv | excel | parquet | jsonl  (inferred from extension if omitted)
    include_unmapped: true   # keep columns you didn't map, under metadata (default true)
    frozen_slice: null       # optional path to a frozen baseline slice (for regression)
    field_map:               # TARGET -> SOURCE column/key
      input: question
      output: pred
      expected: gold
      db_ref: db             # non-EvalContext target -> goes to metadata["db_ref"]
```

**field_map** is the key idea: keys are *targets*, values are your *source* columns/keys.

- Targets that are `EvalContext` fields — `input`, `output`, `expected`, `retrieved_context`,
  `trajectory` — populate the context directly.
- Any **other** target (e.g. `db_ref`, `schema`, `samples`, `tool_schemas`, `cluster_id`) lands in
  `metadata[target]` — that's how metrics get extra inputs.
- If you omit `field_map`, columns whose names already match `EvalContext` fields are used as-is.
- With `include_unmapped: true`, every unmapped column is preserved under `metadata` (nothing lost).

Tabular files use pandas (`csv`/`tsv`/`excel`/`parquet`); empty cells become `None` (not `NaN`).
JSONL accepts one JSON object per line or a top-level JSON array. List-valued fields (e.g. a
`retrieved` list or a `trajectory` list) come through natively in JSONL.

## `suites`

A suite is the metrics + gate for one **category** (`search`, `response`, `latency`, `scenario`).

```yaml
suites:
  search:
    dataset: gold                       # which configured dataset to evaluate (CLI can override)
    target: {level: tool, selector: run_query, attach: observe}   # which graph level (optional)
    metrics:
      - execution_accuracy              # bare string: type == name, no params
      - {type: soft_f1, name: soft_f1}  # explicit form
      - {type: recall_at_k, name: recall_at_k, params: {k: 5}}    # with constructor params
    gate:
      thresholds: {execution_accuracy: 0.80, recall_at_k: 0.7}
      require_pass: [execution_accuracy]   # hard ship-blockers (must run AND pass)
      min_effect: 0.02                     # practical-significance floor (regression)
      significance_alpha: 0.05
      fdr: true                            # Benjamini-Hochberg across metrics
```

### Metric specs — two forms

- **Bare string** `"recall_at_k"` → `type` and `name` both equal the string, no params.
- **Object** `{type, name, params}`:
  - `type` — the registered metric type (see [agent-types.md](agent-types.md) for the catalog).
  - `name` — the reporting name (and the key used in `gate.thresholds`). Lets you run the same type
    twice with different params under different names.
  - `params` — passed to the metric's constructor (e.g. `{k: 5}`, `{tol: 1e-6}`, `{mode: subset}`).

### Gate semantics

- A metric **with** a threshold must meet it (`value >= threshold`) or the suite fails.
- A metric **without** a threshold is **informational** (reported, never blocks).
- `require_pass` metrics are hard blockers — they must have actually run *and* passed.
- The reported `value` is a pass-rate (binary metrics, with a **Wilson** CI) or a mean (graded
  metrics, with a cluster-robust CI). See [statistics.md](statistics.md).

## `judges`

LLM-as-judge configs. A judge is **pinned** by `model|prompt_version|rubric_id`, and may only gate a
release if it has a valid, non-stale certification (see [judges.md](judges.md)).

```yaml
judges:
  nl_intent_judge:
    model: "claude-judge@2026-01"   # pinned
    prompt_version: v3              # pinned
    rubric_id: t2s_nl_intent        # pinned
    protocol: pointwise             # pointwise | pairwise
    panel: [prometheus2_local]      # other judge ids => panel-of-judges (PoLL)
    confidence_policy: logprob
    certification_ref: cert_t2s_nl_2026_06   # must resolve to a passing, fresh cert to gate
```

Reference a judge from a suite metric:

```yaml
suites:
  response:
    metrics:
      - {type: judge, name: nl_intent_fidelity, params: {judge: nl_intent_judge}}
    gate: {thresholds: {nl_intent_fidelity: 0.7}}
```

## `runtime_critic`

The in-flight critic policy. `tau` is the per-metric threshold table — **written by
`agent-eval calibrate`** from an offline run, so you usually start it empty.

```yaml
runtime_critic:
  tiers: [deterministic, uncertainty, judge]   # cheap-to-strong escalation order
  max_retries: 2
  latency_budget_ms: 1500
  fallback: abstain                            # FallbackStrategy id (abstain | escalate | <custom>)
  self_refine_only_for: [style, format, safety]  # never "fix" correctness with self-refine
  tau: {}                                       # metric_name -> threshold (filled by calibrate)
```

See [runtime-critic.md](runtime-critic.md) for how these map to behavior.

## `eval_targets`

An optional list of default `EvalTarget`s (graph/subgraph/node/tool) for the LangGraph
instrumentation layer to attach to. Each suite can also carry its own `target`.

```yaml
eval_targets:
  - {level: graph, selector: "*",          attach: observe}
  - {level: node,  selector: generate_sql, attach: active}
  - {level: tool,  selector: run_query,    attach: observe}
```

## A note on round-tripping

`agent-eval calibrate` (and `write_calibrated_tau`) rewrite the file via a YAML dump, which
**does not preserve comments**. Keep the canonical, commented config in version control and treat
the `tau` block as generated — or call `calibrate` against a copy and review the `tau` diff.

## Minimal complete example

```yaml
version: 1
agent_type: plain
datasets:
  qa:
    adapter: jsonl
    path: ${DATA_DIR}/qa.jsonl
    field_map: {input: question, output: answer, expected: gold}
suites:
  response:
    dataset: qa
    metrics: [{type: exact_match, name: exact_match}]
    gate: {thresholds: {exact_match: 0.6}, require_pass: [exact_match]}
```
```bash
DATA_DIR=./data uv run agent-eval evaluate -c that_file.yaml
```
