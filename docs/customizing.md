# Customizing & extending the framework

Everything below is additive — you never edit `core/`. There are three customization surfaces:

1. **Config (YAML)** — pick metrics, thresholds, datasets, critic policy, graph levels. No code.
2. **In-code** — register a custom metric/adapter/fallback/judge on a registry you build.
3. **Plugins (entry points)** — ship a metric in your own package; the CLI auto-discovers it.

## 1. Customize via config only

Most tuning is config: which metrics run, their `params`, gate `thresholds`/`require_pass`,
`min_effect`, which dataset, and the `runtime_critic` policy. See
[configuration.md](configuration.md). Example — run the same metric twice with different params and
gate on both:

```yaml
suites:
  search:
    dataset: ds
    metrics:
      - {type: recall_at_k, name: recall@5,  params: {k: 5}}
      - {type: recall_at_k, name: recall@20, params: {k: 20}}
    gate: {thresholds: {recall@5: 0.6, recall@20: 0.9}, require_pass: [recall@5]}
```

## 2. Add a custom metric

Subclass `BaseMetric` and implement **one** method, `_compute`. You get requirement-checking, the
sync/async bridge, score normalization to `[0,1]`, automatic latency capture, and
exception-trapping (a metric never crashes a run) for free.

```python
# my_pkg/metrics.py
from agent_eval.core.contracts import CostClass, EvalContext, MetricResult, Tier
from agent_eval.core.metric import BaseMetric

class KeywordCoverage(BaseMetric):
    name = "keyword_coverage"          # default reporting name (config can override per-instance)
    tier = Tier.DETERMINISTIC          # also the runtime escalation tier
    requires = frozenset({"output"})   # missing field -> MetricResult(error=...), not a crash
    cost_class = CostClass.FREE        # FREE | CHEAP | EXPENSIVE (runtime tiering)

    def __init__(self, keywords: list[str], min_coverage: float = 1.0):
        self.keywords = [k.lower() for k in keywords]
        self.min_coverage = min_coverage

    def _compute(self, ctx: EvalContext) -> MetricResult:
        text = str(ctx.output).lower()
        present = [k for k in self.keywords if k in text]
        coverage = len(present) / len(self.keywords) if self.keywords else 1.0
        return MetricResult(
            self.name, coverage,
            passed=coverage >= self.min_coverage,           # set passed -> binary (Wilson CI); leave None -> graded
            detail={"missing": [k for k in self.keywords if k not in text]},
        )
    # For a genuinely async metric (e.g. an API call), override `async def _acompute(self, ctx)`.
```

Conventions: return `score` in `[0,1]`; set `passed` for binary metrics (gets a Wilson CI), leave it
`None` for graded ones (gets a mean + cluster-robust CI); raise for "can't run" cases (it's trapped
into `error`); read extra inputs from `ctx.metadata[...]` and declare nothing exotic in `requires`
(it only checks the standard `EvalContext` fields).

**Use it directly:**

```python
KeywordCoverage(["paris", "france"]).score(EvalContext("q", "Paris is in France")).score  # 1.0
```

**Register it on a registry** (so config can build it by `type`):

```python
from agent_eval.core.registry import default_registry
reg = default_registry()
reg.register("keyword_coverage", lambda params: KeywordCoverage(**params))

m = reg.build({"type": "keyword_coverage", "name": "facts", "params": {"keywords": ["paris"]}})
# then: build_suite(cfg, "response", reg) / evaluate(...)
```

## 3. Ship a metric as a plugin (entry points)

So `agent-eval evaluate` and `default_registry().load_plugins()` auto-discover your metric — no
wiring in user code. In your package's `pyproject.toml`:

```toml
[project.entry-points."agent_eval.metrics"]
keyword_coverage = "my_pkg.metrics:keyword_coverage_factory"
```
```python
# my_pkg/metrics.py
def keyword_coverage_factory(params: dict):
    return KeywordCoverage(**params)
```

The entry-point **name** is the metric `type`; the value is a **factory** `(params) -> Metric`.
After `pip/uv install` of your package, any config can use `{type: keyword_coverage, params: {...}}`
and the CLI picks it up (its registry calls `load_plugins()`).

## 4. Add a custom dataset / adapter

If your data is CSV/Excel/Parquet/JSONL, you need no code — just `field_map` (see
[configuration.md](configuration.md)). For another source (a DB, an API, a HuggingFace dataset),
implement an adapter: anything iterable that yields `EvalContext`s. Reuse `record_to_context` to get
the same `field_map` semantics:

```python
from agent_eval.datasets.base import record_to_context
from agent_eval.offline.runner import evaluate

class HFAdapter:
    def __init__(self, name, split, field_map):
        self.name, self.split, self.field_map = name, split, field_map
    def __iter__(self):
        import datasets
        for row in datasets.load_dataset(self.name, split=self.split):
            yield record_to_context(dict(row), self.field_map)

data = HFAdapter("squad", "validation", {"input": "question", "expected": "answers"})
evaluate(suite, data)            # the runner accepts any Iterable[EvalContext]
```

The built-in `load_dataset(spec)` dispatches only `jsonl`/`tabular`. For a custom adapter, either
pass it straight to `evaluate` (as above), or export your source to JSONL once and point a config at
it. (A pluggable adapter registry is a natural future addition — the `EvalContext` contract is the
only thing the runner depends on.)

## 5. Add a custom fallback strategy

```python
from agent_eval.runtime.fallback import default_fallbacks
from agent_eval.core.contracts import Decision

reg = default_fallbacks()                          # built-ins: "abstain", "escalate"
reg.register("cached_answer", lambda ctx, assessment: (Decision.FALLBACK, cache.get(ctx.input)))

from agent_eval.runtime.critic import critic_loop
critic_loop(generate, build_ctx, critic, fallback=reg.get("cached_answer"))
```

## 6. Add a custom judge backend

Implement one method; everything else (panel, certification, PPI, the gate) is unchanged:

```python
from agent_eval.judges.config import JudgeVerdict
from agent_eval.metrics.judge_ import JudgeMetric

class AzureJudge:
    def __init__(self, deployment): self.deployment = deployment
    def evaluate(self, criteria, ctx) -> JudgeVerdict:
        raw = call_azure(self.deployment, criteria=criteria, input=ctx.input, output=ctx.output)
        return JudgeVerdict(score=parse_0_to_1(raw), confidence=None, reason=raw)

metric = JudgeMetric(backend=AzureJudge("gpt-4o"), criteria="helpfulness, correctness")
```

Certify and gate it exactly as in [judges.md](judges.md).

## 7. Add a whole new agent type

A "new agent type" is **not a core change** — it's a *registry* (which metrics are available) + a
*critic builder* (which tiers) + a *config*. Here's a complete **Summarizer** type:

```python
# my_pkg/summarizer.py
from agent_eval.core.contracts import Tier
from agent_eval.core.registry import default_registry
from agent_eval.config.loader import build_suite
from agent_eval.metrics.uncertainty_ import SelfCheckConsistency, register_uncertainty_metrics
from agent_eval.runtime.critic import Critic
from agent_eval.runtime.policy import CriticPolicy
from my_pkg.metrics import KeywordCoverage           # the custom metric from §2

def summarizer_registry():
    reg = default_registry()
    register_uncertainty_metrics(reg)                 # selfcheck_consistency, semantic_entropy
    reg.register("keyword_coverage", lambda p: KeywordCoverage(**p))
    return reg

def build_summarizer_critic(cfg) -> Critic:
    rc = cfg.runtime_critic
    policy = CriticPolicy(
        tiers=(rc.tiers if rc else ["uncertainty"]),
        tau=(dict(rc.tau) if rc else {"selfcheck_consistency": 0.6}),
        max_retries=(rc.max_retries if rc else 1),
        fallback=(rc.fallback if rc else "abstain"),
    )
    return Critic({Tier.UNCERTAINTY: [SelfCheckConsistency()]}, policy)
```

```yaml
# summarizer.yaml
version: 1
agent_type: summarizer
datasets:
  docs: {adapter: jsonl, path: data/summaries.jsonl,
         field_map: {input: document, output: summary, expected: reference}}
suites:
  response:
    dataset: docs
    metrics:
      - {type: keyword_coverage, name: must_include, params: {keywords: ["revenue", "risks"], min_coverage: 1.0}}
      - {type: selfcheck_consistency, name: consistency}     # informational (no threshold)
    gate:
      thresholds: {must_include: 0.9}
      require_pass: [must_include]
runtime_critic:
  tiers: [uncertainty]
  tau: {selfcheck_consistency: 0.6}
```

```python
from agent_eval.config.loader import load_config, build_dataset_spec
from agent_eval.datasets.base import load_dataset
from agent_eval.offline.runner import evaluate

cfg = load_config("summarizer.yaml")
reg = summarizer_registry()
# OFFLINE gate:
suite = build_suite(cfg, "response", reg)
result = evaluate(suite, load_dataset(build_dataset_spec(cfg, "docs")))
print(result.verdict.passed)
# RUNTIME critic (e.g. inside a LangGraph node — see langgraph-integration.md):
critic = build_summarizer_critic(cfg)
```

To make the new type usable from the **`agent-eval` CLI** too, register your metric(s) as plugins
(§3) so the CLI's registry discovers them; the CLI builds suites from any `agent_type`/category in
the config. (Your bespoke critic is wired in your application code, as in
[langgraph-integration.md](langgraph-integration.md).)

## What you can extend, at a glance

| Extend | How | Surfaces in config? |
|---|---|---|
| Metric | subclass `BaseMetric` + register / entry-point plugin | ✅ by `type` |
| Dataset source | implement an adapter (`__iter__ → EvalContext`) | code (or export to JSONL) |
| Fallback strategy | `FallbackRegistry.register(name, fn)` | ✅ `runtime_critic.fallback` |
| Judge backend | implement `evaluate(criteria, ctx) → JudgeVerdict` | code |
| Agent type | a registry + a critic builder + a config | ✅ `agent_type` + suites |
| Backend (observability) | implement `export(span)` | code (`TraceBoundary.register`) |

If you find yourself wanting to change `core/`, you probably don't need to — open an issue describing
the use case; the `EvalContext` / `Metric` contracts are designed to absorb new metrics and agent
types without touching the spine.
