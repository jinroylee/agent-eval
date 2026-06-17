# Repository structure

A physical map of the repo: every package and module under `src/agent_eval/`, plus `tests/`,
`examples/`, and `docs/`. For the *mental* model (one metric core, two modes, three tiers,
multi-level attachment) read [concepts.md](concepts.md) first — this doc is the layout that model
lives in.

## Top level

```
agent-eval/
├── src/agent_eval/     the framework package (everything below)
├── tests/              unit · integration · (property/meta scaffolds)
├── examples/           runnable example configs, one per agent type
├── docs/               this documentation set
├── pyproject.toml      PEP 621 metadata, optional extras, tool config (uv)
├── uv.lock             locked dependency set
└── README.md           repo entry point
```

## The `agent_eval` package at a glance

```
src/agent_eval/
├── core/            the spine: contracts, the Metric protocol, registry, suite, gate
├── metrics/         one module per metric family (deterministic, retrieval, T2S, judge, …)
├── stats/           small-sample-correct statistics (CIs, regression tests, PPI, drift)
├── offline/         entrypoint #1 — the CI/CD gate (evaluate · regression · calibrate · report)
├── runtime/         entrypoint #2 — the in-flight critic (decide · loop · fallback · guards)
├── judges/          LLM-as-judge governance (config · backends · panel · certify · gate)
├── execution/       T2S sandboxed query execution + result-set comparison
├── instrumentation/ LangGraph attachment (topology · observe · active critic node)
├── obs/             observability boundary + online drift monitor
├── config/          the config-first surface (pydantic schema + YAML loader)
├── datasets/        bring-your-own data adapters (tabular · jsonl)
├── agents/          per-agent-type presets (registries + critic builders)
├── cli/             the `agent-eval` command-line app
└── plugins/         entry-point discovery for third-party metrics
```

**Dependency direction.** `core/` imports nothing else in the package — it's the bottom. Everything
else depends on `core/`. `metrics/` and `stats/` are leaves built on `core/`; `offline/` and
`runtime/` are the two thin entrypoints that orchestrate metrics; `agents/` wires opinionated presets
on top; `cli/` sits at the very top. You add capability at the edges (a metric, an adapter, an agent
type) without touching `core/` — see [customizing.md](customizing.md).

---

## `core/` — the spine

The universal vocabulary every other module imports. Depends on nothing else in the package.

```
core/contracts.py   EvalContext · MetricResult · Cost · Decision · EvalTarget + enums (Tier, Mode, Level, CostClass)
core/metric.py      the Metric Protocol + BaseMetric (requires-check, sync/async bridge, normalization, cost capture, error trapping)
core/registry.py    MetricRegistry + default_registry(); builds metrics from config specs by `type`; loads plugin entry points
core/suite.py       Suite (agent_type, category, metrics, gate, target) + SuiteResult
core/gate.py        GatePolicy (thresholds · require_pass · min_effect · significance_alpha · fdr) + MetricAggregate · GateVerdict
core/errors.py      typed errors: AgentEvalError base + ConfigError · SelectorNotFound · UncertifiedJudge · MissingRequirement · CriticLoop
```

## `metrics/` — one module per family

Each module exposes its metrics and (except the in-code ones) a `register_*` helper. OSS libraries
are wrapped behind the same `Metric` contract as lazy adapters. See the catalog in
[agent-types.md](agent-types.md).

```
metrics/ast_query_.py    T2S: execution_accuracy · soft_f1 · component_match · ast_valid · schema_linking · exec_validity  (uses execution/)
metrics/retrieval_.py    RAG retrieval: recall_at_k · precision_at_k · retrieval_sufficiency
metrics/trajectory_.py   orchestration: trajectory_match · tool_arg_validity
metrics/uncertainty_.py  label-free hallucination signals: selfcheck_consistency · semantic_entropy
metrics/perf_.py         latency_budget · token_budget metrics + attribute / latency_percentiles / scenario_pass_hat_k helpers
metrics/judge_.py        JudgeMetric — LLM-as-judge / PoLL panel (JUDGE tier; constructed in code)
metrics/ragas_.py        lazy RAGAS adapter (ragas_faithfulness · ragas_context_precision); needs the [ragas] extra
metrics/agentevals_.py   lazy agentevals trajectory-matcher adapter; needs the [agentevals] extra
metrics/__init__.py      empty — metrics are imported from their modules
```

> The previous `deterministic_.py` (exact/regex/set/numeric/json_shape) was removed; `default_registry()`
> is now an empty base. Add a deterministic check via `BaseMetric` ([customizing.md](customizing.md)).

## `stats/` — statistics (pure numpy/scipy/statsmodels)

Why each piece exists is in [statistics.md](statistics.md).

```
stats/intervals.py    Wilson · Clopper-Pearson · beta-binomial · bootstrap-percentile CIs
stats/clustered.py    cluster-robust SE for non-iid items (clustered_se)
stats/tests.py        regression tests: mcnemar · paired_bootstrap · wilcoxon · benjamini_hochberg (FDR) · bradley_terry
stats/ppi.py          Prediction-Powered Inference interval (valid CIs for judge-scored metrics)
stats/agreement.py    cohen_kappa (judge↔human agreement, for certification)
stats/agents.py       pass^k reliability + Cost-of-Pass
stats/sequential.py   anytime-valid confidence sequence (peeking-safe online drift)
```

## `offline/` — entrypoint #1: the CI/CD gate

```
offline/runner.py      evaluate(suite, dataset): run metrics → aggregate (stats) → apply gate → SuiteResult
offline/regression.py  paired champion-vs-challenger tests on a frozen slice (mcnemar_regression · bootstrap_regression)
offline/calibrate.py   derive runtime `tau` from an offline run and write it back into the config (the offline→runtime link)
offline/report.py      JSON · JUnit XML · human-text reports (CI exit codes)
```

See [offline-evaluation.md](offline-evaluation.md).

## `runtime/` — entrypoint #2: the in-flight critic

```
runtime/critic.py     Critic (tiered assess/decide) + critic_loop (generate → verify → retry-with-critique → fallback)
runtime/policy.py     CriticPolicy (tiers · tau · max_retries · fallback · latency_budget)
runtime/fallback.py   FallbackRegistry + built-in strategies (abstain · escalate)
runtime/loop_guard.py retry-cap / oscillation / no-progress guards
```

See [runtime-critic.md](runtime-critic.md).

## `judges/` — LLM-as-judge governance

```
judges/config.py      JudgeConfig (pinned by model|prompt|rubric) · JudgeVerdict · Protocol
judges/backend.py     JudgeBackend interface + FunctionJudge (tests) + DeepEvalJudge (lazy G-Eval)
judges/panel.py       PoLL: poll_pointwise · pairwise_swap_average (position-bias mitigation)
judges/certify.py     certify_judge — Cohen's-kappa + consistency certification against human labels
judges/registry.py    JudgeRegistry (configs + certifications; is_certified by fingerprint)
judges/governance.py  gate enforcement (assert_gated_judges_certified → UncertifiedJudge) + judge_gate_ci (PPI)
```

See [judges.md](judges.md).

## `execution/` — T2S sandboxed query execution (needs the `[t2s]` extra)

```
execution/harness.py  ExecutionHarness: run / compare predicted vs gold queries
execution/sandbox.py  read-only connection + statement timeouts (ephemeral teardown)
execution/compare.py  ResultSetPolicy (row order · duplicates · NULLs · float tolerance) + soft_f1
```

## `instrumentation/` — LangGraph attachment (needs the `[agentevals]` extra)

```
instrumentation/topology.py         introspect · validate_selector (fail fast on a bad selector)
instrumentation/observe.py          non-intrusive capture via astream_events → one EvalContext per level
instrumentation/active_node.py      make_critic_node: inject a critic that routes the graph via Command (retry/fallback)
instrumentation/context_builder.py  build the identical EvalContext at graph/subgraph/node/tool (for_graph · for_node · for_tool)
```

See [langgraph-integration.md](langgraph-integration.md).

## `obs/` — observability boundary + drift

```
obs/boundary.py   TraceBoundary + Span (internal span abstraction; OTel GenAI semantic conventions)
obs/adapters.py   swappable backends: MemoryBackend · JsonlBackend · lazy OTelBackend (→ LangSmith/Phoenix/Langfuse)
obs/drift.py      DriftMonitor: anytime-valid online drift / auto-rollback signal
```

## `config/` and `datasets/` — the config-first surface + data

```
config/schema.py    pydantic ConfigModel — validates the YAML
config/loader.py    load_config (+ ${ENV} expansion) · build_suite · build_dataset_spec · write_calibrated_tau
datasets/base.py    DatasetSpec · record_to_context (field_map semantics) · load_dataset dispatch
datasets/tabular.py CSV / TSV / Excel / Parquet via pandas
datasets/jsonl.py   JSONL / JSON-array adapter
```

See [configuration.md](configuration.md).

## `agents/` — per-agent-type presets

A "new agent type" is just a registry + a critic builder + a config — not a core change. Each type
ships two small files: `suites.py` (which metrics are available) and `critic.py` (which runtime tiers).

```
agents/plain/          suites.py (plain_registry) · critic.py (build_plain_critic — self-consistency)
agents/rag/            suites.py (rag_registry)   · critic.py (build_rag_critic — sufficiency + groundedness)
agents/t2s/            suites.py (t2s_registry)   · critic.py (build_t2s_critic — AST/schema/dry-run) · sample/ (fixture hook)
agents/orchestration/  suites.py (orchestration_registry) · critic.py (build_orchestration_critic — tool-arg validity)
```

See [agent-types.md](agent-types.md).

## `cli/` and `plugins/`

```
cli/main.py        Typer app: `agent-eval evaluate` · `calibrate` · `version`
plugins/__init__.py entry-point discovery hook (third-party metrics via the `agent_eval.metrics` group)
```

---

## `tests/`

```
tests/unit/          per-module unit tests (metrics, stats, core, config, judges, runtime, offline, …)
tests/integration/   end-to-end: test_cli_evaluate · test_instrumentation (real StateGraph) · test_t2s_slice
tests/property/      scaffold for Hypothesis stats property-tests — currently only __init__.py
tests/meta/          scaffold for judge-certification meta-eval — currently only __init__.py
```

The test suite needs the optional extras: `uv sync --extra t2s --extra agentevals`, then `uv run pytest`.

## `examples/` — runnable example configs (one per agent type)

```
examples/rag/            rag.yaml + rag.jsonl                  (Recall@k retrieval gate)
examples/t2s/            t2s.yaml + gold.csv + sample.sqlite   (Execution Accuracy gate)
examples/orchestration/  orchestration.yaml + orchestration.jsonl  (tool-trajectory subset gate)
```

Run any of them: `uv run agent-eval evaluate -c examples/<type>/<file>.yaml`. (A plain example was
removed along with the deterministic metrics; Plain's offline gate is a certified judge.)

## `docs/`

```
docs/README.md                 documentation entry point + 60-second quickstart
docs/concepts.md               the mental model — read first
docs/repository-structure.md   this file
docs/configuration.md          the complete YAML reference
docs/offline-evaluation.md     the CI/CD gate: datasets, stats, regression, calibration, CI
docs/runtime-critic.md         the in-flight critic: policy, tiers, retry→fallback loop
docs/langgraph-integration.md  the full worked LangGraph example (observe + active)
docs/agent-types.md            the four agent types + the metric catalog
docs/judges.md                 LLM-as-judge governance (certify, pin, panel, PPI)
docs/statistics.md             the statistics toolkit and why each piece exists
docs/customizing.md            extending the framework (metric, adapter, fallback, judge, agent type)
docs/plan.md                   the original architecture + phasing plan
docs/research/                 the SOTA research report + the Korean metric-selection summary/explainer
```

---

## "Where does X live?" — quick reference

| I want to… | Look in | Doc |
|---|---|---|
| Understand the core data types | `core/contracts.py` | [concepts.md](concepts.md) |
| Add or change a metric | `metrics/` + `core/registry.py` | [customizing.md](customizing.md) |
| Change the gate / aggregation logic | `core/gate.py` · `offline/runner.py` | [offline-evaluation.md](offline-evaluation.md) |
| Tune thresholds / calibrate `tau` | `offline/calibrate.py` | [offline-evaluation.md](offline-evaluation.md) |
| Work on the runtime critic | `runtime/` | [runtime-critic.md](runtime-critic.md) |
| Add a new agent type | `agents/<type>/` | [customizing.md](customizing.md) §7 |
| Wire a real LangGraph agent | `instrumentation/` | [langgraph-integration.md](langgraph-integration.md) |
| Configure a run (YAML) | `config/` | [configuration.md](configuration.md) |
| Work on the statistics | `stats/` | [statistics.md](statistics.md) |
| Govern an LLM judge | `judges/` | [judges.md](judges.md) |
| Touch T2S query execution | `execution/` | [agent-types.md](agent-types.md) |
| Export spans / monitor drift | `obs/` | [statistics.md](statistics.md) |
| Add the CLI a command | `cli/main.py` | — |
