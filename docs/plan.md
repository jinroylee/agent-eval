# Plan: `agent-eval-codex` — a reusable LangGraph agent evaluation + runtime-critique framework, purely built with codex

## Context (why we're building this)

The client builds AI agents on **LangGraph** and needs an evaluation capability that (1) **gates CI/CD** before deploy (holistic, threshold-based, blocks bad releases) and (2) acts as an **in-flight critic** at runtime — inside a node or tool-call, decide **retry / fallback / fail / escalate**. "Evaluation" is too narrow: this is an eval **+ critique** framework.

It must be a **reusable framework**, not scripts for today's agents: future LangGraph agents reuse it by adding config (+ optional plugins), not by rewriting code. It must cover **4 agent types** (Orchestration, RAG, T2S = text-to-SQL/Cypher, Plain) across **4 categories** (search / response / latency / scenario-E2E), prefer **SOTA methods backed by papers**, be **highly generic but very customizable** (bring-your-own dataset; evaluate the graph at **any level** — whole-graph → subgraph → node → tool-call), and stay **standalone** (no coupling to any sibling repo).

**Source of truth for every method/library choice:** `docs/research/2026-06-05-agent-eval-sota-research.md` (287 KB; 33-agent deep-research run with per-claim adversarial verification — 8 dimensions, 20-row metric matrix, 16 key papers). This plan operationalizes it.

**Intended outcome of this plan:** the architecture + a phased, verifiable build. **No code yet** — endpoint is research → architecture → phased plan.

### Decisions locked (with the user)
| Decision | Choice |
|---|---|
| Architecture | **A — unified Metric core + two thin entrypoints** (offline runner + runtime critic share one metric set & calibrated thresholds) |
| Tooling stance | **Thin swappable layer over best-in-class OSS** (wrap; don't reinvent; no lock-in) |
| Runtime critic | **Tiered: cheap deterministic gates → LLM-judge escalation** |
| Judge model | No hard constraint (frontier judges allowed); diverse-family panel for gates |
| Graph "levels" | **LangGraph execution-graph granularity** (graph / subgraph / node / tool) |
| Datasets first | **Tabular (CSV/Excel/Parquet) + JSON/JSONL** (adapter extensible to HF/LangSmith) |
| Customization | **Config-first (YAML) + Python plugin escape hatch** |
| First slice | **T2S** (highest-signal tracer bullet: unambiguous execution ground truth) |

### The 5 research principles the design enforces
1. **Separate generator from verifier** — intrinsic self-correction doesn't reliably fix objective tasks; only a *sound external verifier* helps. The critic never lets a model grade its own correctness.
2. **Gate objective tasks on deterministic checks, not judges** (JudgeBench) — T2S gates on execution/AST; LLM judge confined to NL-intent.
3. **Confidence-gated cheap→strong escalation** (Trust-or-Escalate) for the runtime critic.
4. **Treat the judge as a measurement instrument you certify & pin** (CALM bias audit + human-correlation; PPI for valid CIs).
5. **Gate on reliability & economics** — pass^k (not pass@k), p95/p99 (not means), Cost-of-Pass.

---

## Architecture

**One metric core, two entrypoints, three tiers, multi-level attachment.**

```
                 config (YAML, per agent type)  ──▶  METRIC CORE
                                                      Metric protocol · Registry · Suite
   EvalContext(input, output, expected,              tier: DETERMINISTIC < UNCERTAINTY < JUDGE
     retrieved_context, trajectory, metadata)        → MetricResult(score 0..1, confidence, cost, detail)
                          │                                  │
          ┌───────────────┴───────────────┐   same metric    │   plugins behind ONE interface:
          ▼                               ▼   objects         ▼   DeepEval · RAGAS/RAGChecker · agentevals/BFCL
   OFFLINE RUNNER                   RUNTIME CRITIC                · Inspect · custom AST-query · uncertainty
   dataset → stats → CI GATE        LangGraph node/tool wrapper
   (significance + min effect)      ACCEPT/RETRY/FALLBACK/ESCALATE/ABSTAIN
          │                               ▲
          └── calibrate.py writes τ ───────┘   KEY LINK: offline-calibrated thresholds are what
              into the SAME config            the runtime critic reads → offline & runtime can't drift
```

**Multi-level instrumentation** attaches the same core at any LangGraph granularity, two ways:
- **Observational (non-intrusive, works on any graph unmodified):** consume `astream_events(version="v2")`, key spans by `metadata["langgraph_node"]`, score post-hoc at graph / subgraph / node / tool.
- **Active (in-flight critic):** raw `StateGraph` + `add_conditional_edges` + `Command(goto/update)` to inject a critic node and loop back on retry; `ToolNode` wrapper + `handle_tool_errors` to validate tool args pre-execution and `Command`-escalate.
- **Topology introspection** (`compiled.get_graph().nodes/.edges`) validates the user's `selector` (fail fast with `SelectorNotFound`) and lets users attach metrics declaratively.

---

## Module layout (`src/` package `agent_eval`)

```
core/            contracts.py (EvalContext, MetricResult, Cost, Decision, EvalTarget, enums) · metric.py (Protocol+BaseMetric)
                 registry.py · suite.py · gate.py (GatePolicy) · errors.py
metrics/         deterministic_.py · deepeval_.py · ragas_.py · agentevals_.py · ast_query_.py (first-class) · judge_.py · uncertainty_.py · perf_.py
stats/           intervals.py · clustered.py · tests.py · ppi.py · sequential.py · agents.py
judges/          config.py (JudgeConfig) · registry.py · certify.py (bias audit + κ floor) · panel.py (PoLL, pairwise swap-avg)
offline/         dataset.py · runner.py (evaluate) · gate.py (+BH-FDR) · regression.py · calibrate.py (KEY LINK) · report.py (JSON/JUnit, exit codes)
runtime/         critic.py (decide) · policy.py (CriticPolicy) · loop_guard.py · fallback.py
instrumentation/ target.py (EvalTarget) · topology.py · observe.py (astream_events) · active_node.py · active_tool.py · context_builder.py
execution/       harness.py · sql_backend.py · cypher_backend.py · compare.py (result-set policy) · sandbox.py (read-only, timeouts)
obs/             boundary.py (internal span abstraction) · adapters.py (LangSmith/Phoenix/Langfuse, swappable)
config/          schema.py (pydantic) · loader.py (validate/merge, threshold round-trip)
datasets/        base.py (DatasetAdapter) · tabular.py (CSV/Excel/Parquet) · jsonl.py
agents/t2s/      suites.py · critic.py · sample/ (tiny dataset + sqlite fixture + golden queries)
plugins/         entry-point discovery (agent_eval.metrics / .datasets / .fallbacks)
cli/             main.py (evaluate | calibrate | critic-dryrun | certify-judge | introspect | report)
```

In-house = connective tissue only (contracts, registry, stats, instrumentation, ExecutionHarness `compare.py`, critic policy, config, dataset adapters). Everything else delegates to a named OSS function.

## Core interfaces (`core/contracts.py`, `core/metric.py`)

```python
class Tier(Enum):  DETERMINISTIC; UNCERTAINTY; JUDGE      # = runtime escalation order
class Mode(Enum):  OFFLINE; RUNTIME
class Level(Enum): GRAPH; SUBGRAPH; NODE; TOOL
class CostClass(Enum): FREE; CHEAP; EXPENSIVE
class Decision(Enum):  ACCEPT; RETRY; FALLBACK; ESCALATE; ABSTAIN

@dataclass(frozen=True)
class Cost: tokens:int=0; usd:float=0.0; latency_ms:float=0.0

@dataclass(frozen=True)
class EvalContext:                 # universal currency at EVERY level
    input:Any; output:Any; expected:Any=None
    retrieved_context:Seq[str]=(); trajectory:Seq[Mapping]=()   # RAG / orchestration
    metadata:Mapping=field(default_factory=dict)               # level, selector, langgraph_node, dialect, schema, db_ref, run_id…

@dataclass(frozen=True)
class MetricResult: metric:str; score:float; passed:bool|None; confidence:float|None=None
                    cost:Cost=Cost(); detail:Mapping=…; error:str|None=None   # error set => couldn't RUN

@dataclass(frozen=True)
class EvalTarget: level:Level; selector:str="*"; attach:str="observe"   # "observe" | "active"

class Metric(Protocol):            # OSS libs are wrappers behind this
    name:str; tier:Tier; modes:frozenset[Mode]; requires:frozenset[str]; cost_class:CostClass
    def score(ctx)->MetricResult; async def ascore(ctx)->MetricResult
# BaseMetric: requires-checking, sync↔async bridge, normalization, cost capture, error trapping.
```

`Suite(agent_type, category, metrics, gate:GatePolicy, target:EvalTarget)` · `GatePolicy(thresholds, min_effect, significance_alpha, fdr=True, require_pass)` · `JudgeConfig(id, model, prompt_version, rubric_id, protocol, panel, confidence_policy, certification_ref)` (suite-build **refuses** an uncertified/stale judge in a gate) · `CriticPolicy(tiers, tau, max_retries, fallback, latency_budget_ms, self_refine_only_for=[style,format,safety])`.

## Multi-level instrumentation (the key feature)

User declares targets in YAML; the framework validates against topology and attaches observationally or actively:
```yaml
eval_targets:
  - {level: graph,    selector: "*",            attach: observe}   # E2E, non-intrusive
  - {level: node,     selector: generate_sql,   attach: active}    # inject critic, loop on retry
  - {level: tool,     selector: run_query,      attach: active}    # validate args pre-exec
  - {level: subgraph, selector: retrieval,      attach: observe}   # score at state boundary
```
`context_builder.py` constructs the **identical `EvalContext`** at each level — only field population differs (TOOL: args→input, result→output; NODE: node in/out; GRAPH/SUBGRAPH: task input + boundary state). This tool-vs-node asymmetry is the highest-risk correctness point → isolated in one module with its own fixtures.

## Stats module (small-sample-correct by default; never CLT below a few hundred)

| Function | Source |
|---|---|
| `wilson_interval`, `clopper_pearson` | statsmodels `proportion_confint` |
| `mcnemar`, `wilcoxon`, `benjamini_hochberg` | statsmodels / scipy |
| `percentile_ci` (p95/p99, bootstrap), `clustered_se`, `paired_bootstrap`, `bradley_terry`, `beta_binomial_ci` | in-house (numpy/scipy) |
| `ppi_interval` (PPI++, asymptotic), `anytime_valid_sequence` (online drift), `pass_hat_k`, `cost_of_pass` | in-house |

Regression gate = **paired** McNemar (binary EX) / paired bootstrap or Wilcoxon (continuous) on a **frozen** golden slice; block only on a *statistically significant* drop ≥ `min_effect`; BH-FDR across metrics; PPI wraps judge-scored metrics; anytime-valid sequences for online peeking.

## Config schema (worked T2S example, condensed)
```yaml
version: 1
agent_type: t2s
defaults: {dialect: sqlite, result_set_policy: {row_order: ignore, duplicates: keep, nulls: distinct, float_tolerance: 1e-6}}
datasets:
  t2s_gold: {adapter: tabular, path: agents/t2s/sample/t2s_gold.csv, format: csv,
             field_map: {input: question, expected: gold_sql, schema: schema_ddl, db_ref: db_path},
             frozen_slice: agents/t2s/sample/t2s_gold.frozen.json}
judges:
  nl_intent_judge: {model: "claude-judge@2026-01", prompt_version: v3, rubric_id: t2s_nl_intent,
                    protocol: pointwise, panel: [prometheus2_local], certification_ref: cert_t2s_nl_2026_06}
suites:
  search:   {target: {level: tool, selector: run_query, attach: observe},
             metrics: [execution_accuracy, test_suite_accuracy, soft_f1, component_match],
             gate: {thresholds: {execution_accuracy: 0.80, test_suite_accuracy: 0.75}, require_pass: [execution_accuracy], min_effect: 0.02}}
  response: {target: {level: node, selector: explain_answer, attach: observe},
             metrics: [{name: nl_intent_fidelity, type: judge, params: {judge: nl_intent_judge}}], gate: {thresholds: {nl_intent_fidelity: 0.7}}}   # NL only, never correctness
  latency:  {target: {level: graph, selector: "*"}, metrics: [r_ves, p95_latency_ms, cost_of_pass], gate: {thresholds: {r_ves: 0.6}}}
  scenario: {target: {level: graph, selector: "*"}, metrics: [{name: pass_hat_k, params: {k: 5}}], gate: {thresholds: {pass_hat_k: 0.9}, require_pass: [pass_hat_k]}}
runtime_critic: {tiers: [deterministic, uncertainty, judge], max_retries: 2, latency_budget_ms: 1500, fallback: cached_or_escalate,
                 tau: {execution_accuracy: 1.0, schema_linking: 1.0, self_consistency: 0.66}}   # tau WRITTEN BY calibrate.py
```
A future agent type = a sibling YAML (+ optional plugin); `core/` unchanged.

## T2S vertical slice (P1 — tracer bullet, doubles as the integration test)
1. **Sample data + fixtures** (`agents/t2s/sample/`): ~20 NL→SQL items over a 2–3 table SQLite schema; include a join, GROUP BY, NULLs, a float aggregate (exercises `result_set_policy`), and a semantically-equivalent alternative (CTE vs subquery); add `*.frozen.json` baseline.
2. **ExecutionHarness (SQL)**: ephemeral in-memory SQLite, read-only tx + statement timeout (`sandbox.py`), `sqlglot` parse/transpile; compute **Execution Accuracy** (gate), **Test-Suite Accuracy** (distilled DBs when present, else single-DB EX + warning), **Soft-F1** & **R-VES** (port BIRD mini-dev), component/exact-set match (diagnostic) via `compare.py` policy.
3. **AST plugin** (`metrics/ast_query_.py`): wrap harness outputs as DETERMINISTIC `Metric`s; parse failures → `MetricResult(error=…)`.
4. **Offline gate E2E**: tabular adapter → `search` suite → Wilson CIs → **McNemar** vs frozen slice → BH-FDR → verdict (`require_pass: [execution_accuracy]`); `report.py` → JSON/JUnit + non-zero exit on failure.
5. **NL-intent judge** (response): DeepEval `GEval` scoring explanation/intent **only**; refused in gate unless certified (P1 ships a stub cert + loud "uncertified" banner).
6. **Runtime critic**: tier-1 deterministic = AST validity + schema-linking + dry-run/`EXPLAIN` + result sanity → soft-fail RETRY with harness error injected (bounded) → tier-2 self-consistency over execution outputs → FALLBACK/ESCALATE; wired via `active_node.py`.
7. **Calibration round-trip** (`calibrate.py`): derive per-metric `tau` operating points (bound false-fail rate) → write into the same YAML → critic reads them.

## Testing the framework itself
- **Per-metric known-good/known-bad fixtures** + a "missing requirement" case asserting `MetricResult(error=…)`; stub LLM clients for deterministic judge unit tests.
- **Property tests (Hypothesis) for stats** — Wilson coverage at small n (the Bowyer guard), BH monotonicity, `pass_hat_k ≤ pass@k`, PPI degrades to gold-only under noise, anytime-valid Type-I bounded under peeking.
- **Judge-certification meta-eval** — assert gate **refuses** uncertified/stale judges (`UncertifiedJudge`).
- **Runtime-critic decision tests** — injected failures → exact `Decision`; **loop-guard** (oscillation/no-progress/retry-cap) + latency-budget short-circuit.
- **Instrumentation tests** — `astream_events` keys by `langgraph_node`; `SelectorNotFound` on bogus selector; same `EvalContext` at NODE vs TOOL; active routing via `Command`.
- **Calibration round-trip** — offline `tau` → reload → critic consumes it.

## Phasing (each phase independently shippable + verifiable)
| Phase | Build | Definition of Done / Verify |
|---|---|---|
| **P0 — Core spine** | `core/`, `stats/`, `config/`, `datasets/`, `plugins/`, `cli/` skeleton | `evaluate()` runs a deterministic metric over JSONL with Wilson CIs; **stats property tests pass**; `agent-eval evaluate` works on a toy dataset |
| **P1 — T2S slice E2E** | the 7 steps above | integration test green: gate passes on good agent / fails on broken variant; critic RETRY→FALLBACK; `tau` flows offline→runtime |
| **P2 — Judge governance** | `judges/` (certify, PoLL), real `JudgeMetric`, PPI in gate | `UncertifiedJudge` raised on uncertified gate; PPI CI degrades to gold-only under noise |
| **P3 — Plain + RAG** | `ragas_.py`, DeepEval DAG/G-Eval, uncertainty metrics, CRAG runtime critic, clustered SEs | RAG retrieval gate (Recall@k); runtime hallucination flag; clustered-SE test on passage-grouped fixture |
| **P4 — Orchestration** | `agentevals_.py` (match + LLM-judge), BFCL AST, per-step critic, token/latency attribution, Bradley-Terry | subset = no-forbidden-tool gate; per-node tokens reconcile to graph total; pass^k scenario gate |
| **P5 — Obs + drift** | `obs/` boundary+adapters, `sequential.py` on sampled traffic | one boundary → swappable backend; "inject drift → alert" without peeking-inflated false positives |

## Dependencies & packaging (uv / PEP 621, `src/` layout, optional extras)
- **core:** pydantic, pyyaml, pandas, numpy, scipy, statsmodels, typer.
- **extras:** `[t2s]` sqlglot, sqlalchemy, (psycopg, neo4j+testcontainers); `[deepeval]`, `[ragas]`, `[agentevals]` (langgraph, langchain-core), `[inspect]`, `[obs]` (opentelemetry-sdk, openinference) + `[langsmith]`/`[phoenix]`/`[langfuse]`; `[dev]` pytest, hypothesis, ruff, mypy.
- Pure offline-metrics users stay lean (LangGraph only via graph-interception extras). **Target Python 3.12** for dependency-wheel compatibility (host has 3.14, which DeepEval/RAGAS/LangGraph may not fully support yet — verify at P0). `git init` needed.
- **Future-agent onboarding:** new `agents/<type>/` YAML; bespoke metric/adapter/fallback shipped as an entry-point plugin referenced by `type:` in YAML. Registry resolves `type`→factory; `core/` never changes.

## Risks / open questions
- **Multi-level `EvalContext` (tool vs node asymmetry)** — isolated in `context_builder.py` + fixtures; gated in P1/P4.
- **Judge family-diversity** — PoLL/self-preference needs a *different* family than the agent's generator; certification refuses same-family judges; plan a self-hosted judge (Prometheus-2/Lynx) as anchor.
- **OTel `gen_ai.*` experimental** — depend on internal `obs/boundary.py`, never a backend SDK in core; re-verify at P5.
- **LangGraph stream-events drift** — depend only on `astream_events(version="v2")`; capability check fails fast; v3/transformers = verify-at-build-time.
- **ExecutionHarness safety** — read-only tx, timeouts, allowlist, ephemeral teardown; never touch prod DB without explicit read-replica config.
- **Result-set comparison semantics** — explicit, configurable, unit-tested (NULL/float fixtures); gold queries themselves can be wrong.
- **Calibration cold-start** — conservative `tau` defaults + loud "uncalibrated" warning until `calibrate.py` runs.
- **Consensus ≠ correctness** — pair agreement signals with a grounded verifier on objective tasks.
- **Python 3.14 vs eval-lib wheels** — pin to 3.12 if deps lag (verify P0).

## Critical files (highest-leverage)
- `core/contracts.py` — universal currency every module depends on.
- `instrumentation/context_builder.py` — identical `EvalContext` at graph/subgraph/node/tool (highest-risk correctness).
- `offline/calibrate.py` — the key link (offline τ → runtime config; prevents drift).
- `execution/compare.py` — explicit result-set comparison policy (documented silent-failure source).
- `runtime/critic.py` — tiered decision engine w/ generator-verifier separation, loop guards, latency budget.
- Reference (do not modify): `docs/research/2026-06-05-agent-eval-sota-research.md`.

## End-to-end verification (how we'll know it works)
1. **P0 gate:** `uv run pytest tests/property` (stats coverage) + `agent-eval evaluate --config <toy>` exits 0 with a printed verdict + Wilson CIs.
2. **P1 tracer bullet:** `uv run pytest tests/integration/test_t2s_slice.py` — asserts (a) offline gate **passes** on the known-good T2S agent and **fails** on a deliberately broken variant (dropped WHERE); (b) runtime critic returns **RETRY → FALLBACK** on an unparseable/empty-result query; (c) `calibrate.py` writes `tau` and `runtime.Critic` consumes it.
3. **CI hook:** `agent-eval evaluate` returns a non-zero exit code on gate failure (drop into CI as the deploy gate); `report.py` emits JUnit XML.
4. **Per-phase:** each phase's "Verify" row above is its acceptance test; full suite nightly, anchor subset per-PR.