# Concepts & architecture

Read this first — every other doc builds on these ideas.

## The one big idea: one metric core, two CLI steps

```
          config (YAML)                      METRIC CORE
                │                 Metric protocol · Registry · Suite · Gate
   EvalContext(input, output,          each Metric: EvalContext → MetricResult(score 0..1, …)
     expected, retrieved_context,                   │
     metadata)                                      │
        │                                           ▼
        ├── agent-eval predict ──► fills predictions by running your LangGraph agent
        └── agent-eval evaluate ─► scores predictions → aggregate w/ CIs → GATE verdict (CI exit code)
```

A **metric** scores one **unit of work** (an `EvalContext`) and returns a normalized
`MetricResult`. The offline **runner** aggregates those results over a dataset with proper
statistics and applies a **gate** → a pass/fail verdict. That's the whole framework.

## The core data types

`from agent_eval.core.contracts import ...` — four small, frozen types.

### `EvalContext` — one item to score

```python
@dataclass(frozen=True)
class EvalContext:
    input: Any                      # the user message / question
    output: Any = None              # the final response to the user  ← common metrics score this
    expected: Any = None            # gold final response (GT metrics only)
    retrieved_context: Sequence[str] = ()   # RAG: retrieved chunk texts
    metadata: Mapping = {}          # agent-specific fields: retrieved_ids, sql, db_ref, latency_ms, …
```

The **invariant** that keeps metrics uniform across agent types: `output` is *always the final
response shown to the user*, so the common metrics (judge, BERTScore) score it for every agent type.
Agent-specific artifacts live in `metadata` under documented keys (`MetaKey.*`) — a RAG agent's
ranked ids in `metadata['retrieved_ids']`, a T2S agent's SQL in `metadata['sql']`. See
[metrics.md](metrics.md) for exactly which fields each metric reads.

### `MetricResult` — a normalized score

```python
@dataclass(frozen=True)
class MetricResult:
    metric: str
    score: float                    # normalized 0..1 (raw magnitude for latency/tokens)
    passed: bool | None = None      # set for binary metrics; None for graded
    confidence: float | None = None # judge/panel agreement
    cost: Cost = Cost()             # tokens / usd / latency_ms (auto-captures latency)
    detail: Mapping = {}            # sub-scores, reasons, parse errors
    error: str | None = None        # set ⇒ the metric could NOT RUN (≠ a low score)
```

`error` vs a low `score` matters: a missing required field or an exception sets `error` (the item is
excluded from the denominator and counted separately); a genuine bad result just scores low. Metrics
never crash a run — exceptions are trapped into `error`.

### `Metric` — the one contract everything implements

```python
class Metric(Protocol):
    name: str
    requires: frozenset[str]        # EvalContext fields it needs (validated before running)
    cost_class: CostClass           # FREE | CHEAP | EXPENSIVE  (EXPENSIVE ⇒ calls an LLM)
    aggregation: Aggregation        # RATE | MEAN | P95  (how per-item scores roll up)
    higher_is_better: bool          # gate direction (False for latency/tokens)
    unit_interval: bool             # are scores bounded to [0, 1]?
    def score(self, ctx) -> MetricResult: ...
```

You almost never implement this by hand — subclass `BaseMetric` and write `_compute`. See
[customizing.md](customizing.md).

### `Suite` and `GatePolicy` — what to run, and how to gate

```python
@dataclass
class Suite:
    agent_type: str                 # plain | rag | t2s (informational)
    category: str                   # e.g. retrieval | response | correctness
    metrics: Sequence[Metric]
    gate: GatePolicy

@dataclass
class GatePolicy:
    thresholds: Mapping[str, float] # metric_name → threshold (direction per metric.higher_is_better)
    require_pass: Sequence[str]     # hard ship-blockers (must run AND pass)
```

## GT vs non-GT (the key metric axis)

Each metric is either **ground-truth** (needs a reference/label) or **reference-free**:

- **GT** — `bertscore`, `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `soft_f1`, `component_match`.
  These read `expected` or a gold field in `metadata` (`relevant_ids`, `gold_sql`).
- **non-GT** — `faithfulness`, `consistency`, `t2s_faithfulness`, `t2s_consistency`, `ast_valid`,
  `p95_latency`, `token_usage`. No label required — they judge the output against retrieved evidence,
  the executed query result, or measure cost/latency.
- `llm_judge` works **either way**: with a reference (`expected` set) it grades against it, without
  one it grades intrinsic quality.

The design rule: **gate objective tasks on grounded checks, not on a judge** — T2S correctness is
gated on query *execution* (`soft_f1`), and judges are confined to subjective quality and
groundedness. Never let the model that produced an answer grade its own correctness.

## How scores aggregate

Every metric declares an `aggregation`, and the runner rolls up its per-item scores accordingly:

| Aggregation | Used by | Statistic | Interval |
|---|---|---|---|
| `RATE` | binary metrics (`ast_valid`) | pass rate | **Wilson** (doesn't under-cover at small n) |
| `MEAN` | graded metrics (`recall_at_k`, `soft_f1`, `faithfulness`, judge) | mean | **cluster-robust** SE |
| `P95` | `p95_latency` | 95th percentile | **bootstrap** (latency is heavy-tailed) |

For non-iid items (RAG questions sharing a passage), set `metadata['cluster_id']` so the
cluster-robust interval doesn't report an artificially tight error bar. See
[offline-evaluation.md](offline-evaluation.md).

## How the pieces map to packages

```
core/         contracts · metric (BaseMetric) · registry (+ BuildContext) · suite · gate · errors
metrics/      common · rag · t2s          # the three metric families
judges/       backend (LLMJudge, FunctionJudge) · panel (PoLL)
execution/    harness · sandbox · compare(result-set policy)   # T2S
stats/        intervals (Wilson, bootstrap) · clustered (cluster-robust SE)
datasets/     base (field_map → canonical) · jsonl · tabular
harness/      predict — run a LangGraph agent to fill predictions
config/       schema (pydantic) · loader
offline/      runner (evaluate) · report (JSON/JUnit/text)
cli/          main (predict · evaluate · version)
runtime/      critic · loop_guard · fallback   # foundation for a future in-flight critique mode
```

The same `Metric` objects that power the offline gate can also drive an **in-flight critic** that
scores a single step and decides accept/retry/fallback — a foundation for a future runtime mode, kept
separate from the offline path. See [runtime-critic.md](runtime-critic.md).
