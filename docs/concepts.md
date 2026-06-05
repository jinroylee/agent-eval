# Concepts & architecture

Read this first — every other doc builds on these ideas.

## The one big idea: one metric core, two entrypoints

```
                 config (YAML, per agent type)  ──▶  METRIC CORE
                                                      Metric protocol · Registry · Suite
   EvalContext(input, output, expected,              tier: DETERMINISTIC < UNCERTAINTY < JUDGE
     retrieved_context, trajectory, metadata)        → MetricResult(score 0..1, confidence, cost, detail)
                          │                                  │
          ┌───────────────┴───────────────┐   same metric    │   plugins behind ONE interface
          ▼                               ▼   objects         ▼
   OFFLINE RUNNER                   RUNTIME CRITIC
   dataset → stats → GATE verdict   node/tool wrapper → ACCEPT/RETRY/FALLBACK/ESCALATE/ABSTAIN
          │                               ▲
          └── calibrate writes τ ──────────┘   the SAME thresholds drive both modes
```

A **metric** is a small object that scores one **unit of work** (`EvalContext`) and returns a
normalized `MetricResult`. The *same* metric objects are used by:

- the **offline runner** — aggregates results over a dataset with proper statistics into a pass/fail
  gate verdict, and
- the **runtime critic** — runs the metric on a single in-flight step and turns the score into a
  decision.

Because both modes share metrics *and* thresholds (the offline `calibrate` step writes the runtime
critic's thresholds), your CI gate and your in-flight critic can't silently disagree.

## The core data types

These five types are the whole vocabulary. (`from agent_eval.core.contracts import ...`)

### `EvalContext` — one unit of work to score, at any graph level

```python
@dataclass(frozen=True)
class EvalContext:
    input: Any                      # the question / step input / tool args
    output: Any                     # what's under test: an answer, SQL, retrieved ids, a tool result
    expected: Any = None            # gold / reference (offline)
    retrieved_context: Sequence[str] = ()   # RAG passages
    trajectory: Sequence[Mapping] = ()       # orchestration: list of {"tool","args"} steps
    metadata: Mapping = {}          # db_ref, schema, samples, cluster_id, level, selector, …
```

The same object describes a whole-graph run, a single node's I/O, or one tool call — only which
fields are populated differs. Optional fields cover all four agent types.

### `MetricResult` — a normalized score

```python
@dataclass(frozen=True)
class MetricResult:
    metric: str
    score: float                    # normalized 0..1 (bool → 0/1, graded e.g. nDCG)
    passed: bool | None = None      # set for binary metrics; None for graded
    confidence: float | None = None # for UNCERTAINTY/JUDGE tiers (drives runtime escalation)
    cost: Cost = Cost()             # tokens / usd / latency_ms (auto-captures latency)
    detail: Mapping = {}            # sub-scores, errors, repair hints
    error: str | None = None        # set ⇒ the metric could NOT RUN (≠ scored a failure)
```

`error` vs a low `score` is an important distinction: a missing required field or an exception sets
`error` (the item is excluded from the denominator and counted separately); a genuine bad result
just scores low. Metrics never crash a run — exceptions are trapped into `error`.

### `Metric` — the one contract everything implements

```python
class Metric(Protocol):
    name: str
    tier: Tier                      # DETERMINISTIC | UNCERTAINTY | JUDGE
    modes: frozenset[Mode]          # {OFFLINE, RUNTIME}
    requires: frozenset[str]        # EvalContext fields it needs (validated before running)
    cost_class: CostClass           # FREE | CHEAP | EXPENSIVE  (runtime tiering)
    def score(self, ctx) -> MetricResult: ...
    async def ascore(self, ctx) -> MetricResult: ...
```

You almost never implement this by hand — you subclass `BaseMetric` and write one method
(`_compute`). See [customizing.md](customizing.md). OSS libraries (DeepEval, RAGAS, agentevals) are
wrapped as metrics behind this same interface.

### `Suite` and `GatePolicy` — what to run, and how to gate

```python
@dataclass
class Suite:
    agent_type: str
    category: str                   # search | response | latency | scenario
    metrics: Sequence[Metric]
    gate: GatePolicy
    target: EvalTarget | None       # which graph level (graph/subgraph/node/tool)

@dataclass
class GatePolicy:
    thresholds: Mapping[str, float] # metric_name → min score to pass
    require_pass: Sequence[str]     # hard ship-blockers (must run AND pass)
    min_effect: float = 0.0         # practical-significance floor for regressions
    significance_alpha: float = 0.05
    fdr: bool = True
```

### `EvalTarget` — the graph level

```python
@dataclass(frozen=True)
class EvalTarget:
    level: Level                    # GRAPH | SUBGRAPH | NODE | TOOL
    selector: str = "*"             # node/tool/subgraph name; "*" == whole graph
    attach: str = "observe"         # "observe" (non-intrusive) | "active" (inject a critic)
```

## The three tiers

Every metric declares a `tier`, which is also the runtime critic's **cheap-to-strong escalation
order**:

1. **DETERMINISTIC** — grounded, reproducible, usually free: exact/regex/set match, SQL execution &
   AST checks, retrieval Recall@k, tool-arg validity. *Preferred for objective correctness.*
2. **UNCERTAINTY** — label-free confidence with no oracle: SelfCheckGPT consistency, semantic
   entropy. Cheap-ish.
3. **JUDGE** — an LLM-as-judge for subjective quality where no oracle exists. Expensive; certified &
   pinned before it may gate a release.

The design rule (from the research): **gate objective tasks on deterministic checks, not judges**,
and never let the model that produced an answer grade its own correctness.

## The two modes in one sentence each

- **Offline:** `evaluate(suite, dataset)` runs each metric on every `EvalContext`, aggregates per
  metric with small-sample-correct statistics (Wilson intervals; clustered SEs), and applies the
  `GatePolicy` → a `SuiteResult` with a pass/fail `GateVerdict`. See
  [offline-evaluation.md](offline-evaluation.md).
- **Runtime:** a `Critic` runs the tiers cheapest-first on one step and returns a `Decision`;
  `critic_loop` orchestrates bounded retry-with-injected-critique → fallback. See
  [runtime-critic.md](runtime-critic.md).

## The calibration link

The offline runner can compute, per metric, the **operating point** (`tau`) that bounds the
false-fail rate on a known-good calibration set, and write it into the *same* config the runtime
critic reads (`agent-eval calibrate`). That's the mechanism that keeps offline and runtime
consistent — there's no separate place to get the thresholds wrong.

## How the pieces map to packages

```
core/            contracts · metric (BaseMetric) · registry · suite · gate · errors
metrics/         deterministic_ · ast_query_(T2S) · retrieval_ · uncertainty_ · trajectory_ · perf_ · judge_
                 + lazy adapters: deepeval (via judges), ragas_ , agentevals_
judges/          config · backend · panel(PoLL) · certify · registry · governance
stats/           intervals · clustered · tests · ppi · sequential · agents · agreement
offline/         runner(evaluate) · gate · regression · calibrate · report
runtime/         critic · policy · loop_guard · fallback
execution/       harness · sandbox · compare(result-set policy)        # T2S
instrumentation/ topology · observe · active_node · context_builder    # LangGraph
obs/             boundary · adapters(OTel/LangSmith/Phoenix/Langfuse) · drift
datasets/        base(DatasetAdapter) · jsonl · tabular
config/          schema(pydantic) · loader
agents/          plain · t2s · rag · orchestration            # per-type registries + critics + (t2s) sample
cli/             main(evaluate · calibrate · version)
```
