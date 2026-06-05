# Offline evaluation (the CI/CD gate)

The offline mode answers: *is this agent good enough to deploy?* — with statistics, not vibes.

## The flow

1. You have a **dataset** of labeled examples (gold answers / relevant ids / gold queries / …).
2. Your agent produces **outputs** for each (you run it; or your dataset already has predictions).
3. A **suite** of metrics scores each `(input, output, expected)` triple.
4. The runner **aggregates** per metric with small-sample-correct confidence intervals.
5. A **gate** turns that into a pass/fail verdict; the CLI exits non-zero on failure.

## Run it from the CLI

```bash
uv run agent-eval evaluate --config examples/rag/rag.yaml
# -c <yaml>            required
# --category <name>    run one suite only (default: all suites in the file)
# --dataset <name>     override which configured dataset to use
```

Exit codes: **0** = all gates passed, **1** = a gate failed, **2** = config/usage error. Drop it
straight into CI as the deploy gate.

## Run it from Python

```python
from agent_eval.core.contracts import EvalContext
from agent_eval.core.gate import GatePolicy
from agent_eval.core.suite import Suite
from agent_eval.metrics.deterministic_ import ExactMatch
from agent_eval.offline.runner import evaluate

dataset = [
    EvalContext(input="2+2?", output="4", expected="4"),
    EvalContext(input="capital of France?", output="Paris", expected="Paris"),
    EvalContext(input="sky color?", output="Blue", expected="blue"),   # mismatch (case)
]
suite = Suite(
    agent_type="plain", category="response",
    metrics=[ExactMatch()],
    gate=GatePolicy(thresholds={"exact_match": 0.6}, require_pass=["exact_match"]),
)
result = evaluate(suite, dataset)

print(result.verdict.passed)              # True / False
for agg in result.aggregates:
    print(agg.metric, agg.value, (agg.ci_low, agg.ci_high), f"n={agg.n}", f"errors={agg.n_errors}")
```

`evaluate(suite, dataset, alpha=0.05)` returns a `SuiteResult`:

```python
SuiteResult(agent_type, category, n_items, aggregates: list[MetricAggregate], verdict: GateVerdict)
MetricAggregate(metric, value, ci_low, ci_high, n, n_errors, binary)
GateVerdict(passed: bool, metric_passed: dict[str,bool], reasons: list[str])
```

Building the dataset from a config (so you reuse your `field_map`) is one call:

```python
from agent_eval.config.loader import load_config, build_suite, build_dataset_spec
from agent_eval.core.registry import default_registry
from agent_eval.datasets.base import load_dataset

cfg = load_config("my.yaml")
suite = build_suite(cfg, "response", default_registry())
data = load_dataset(build_dataset_spec(cfg, cfg.suites["response"].dataset))
result = evaluate(suite, data)
```

## How aggregation works (and why)

- **Binary metrics** (those returning `passed=True/False`, e.g. `exact_match`, `execution_accuracy`,
  `hit_at_k`, `trajectory_match`) are summarized as a **pass rate** with a **Wilson** confidence
  interval — which, unlike the normal-approximation CLT, does not under-cover at small *n*.
- **Graded metrics** (those returning a float `score` with `passed=None`, e.g. `recall_at_k`,
  `soft_f1`, `semantic_entropy`, a judge score) are summarized as a **mean** with a
  **cluster-robust** CI. If your items aren't independent — RAG questions sharing a passage,
  multi-turn scenarios — set `metadata["cluster_id"]` so the SE isn't artificially tight.
- **Errors** (a metric that couldn't run — missing field, exception, bad gold) are counted in
  `n_errors` and excluded from `n`. They show up in the gate reasons.

See [statistics.md](statistics.md) for the full toolbox.

## Regression gates (champion vs. challenger)

Don't gate only on a point threshold — block a merge only when a drop is **statistically
significant** *and* exceeds a minimum effect size. The framework gives you paired tests on a frozen
baseline slice:

```python
from agent_eval.offline.regression import mcnemar_regression, bootstrap_regression

# binary correctness (e.g. per-item execution-accuracy pass/fail), paired by item:
r = mcnemar_regression(baseline_pass, candidate_pass, metric="execution_accuracy", min_effect=0.02)
r.significant_drop      # True only if candidate < baseline, significant, and |delta| >= min_effect
r.delta, r.pvalue, r.n_discordant

# continuous scores (e.g. G-Eval), paired by item:
bootstrap_regression(baseline_scores, candidate_scores, metric="geval", min_effect=0.03)
```

This is the principle behind `GatePolicy.min_effect` / `significance_alpha` / `fdr` — wire these
into your own gate runner when you compare two releases on the same frozen dataset.

## Reporting (JSON / JUnit / text)

```python
from agent_eval.offline import report
report.to_json(result)     # durable machine artifact
report.to_junit([result])  # JUnit XML for CI dashboards; failing thresholds become <failure>
report.to_text(result)     # the human table the CLI prints
```

A failed threshold becomes a JUnit `<failure>`, so CI surfaces exactly which metric blocked the
release and why.

## Calibrating the runtime critic from the offline run

The offline run can produce the runtime critic's thresholds, bounding the false-fail rate on a
known-good calibration set, and write them into the same config:

```bash
uv run agent-eval calibrate -c examples/t2s/t2s.yaml --category search --max-false-fail 0.05
# -> writes runtime_critic.tau into the config
```

In Python:

```python
from agent_eval.offline.calibrate import calibrate_and_write, calibrate_tau, collect_scores
tau = calibrate_and_write("my.yaml", "search", max_false_fail=0.05)   # {metric: threshold}
```

`tau[metric]` is the `max_false_fail` empirical quantile of that metric's scores on the calibration
set. For a metric that's perfect on good data (e.g. execution accuracy), `tau` is `1.0` (must pass).
This is the only place the runtime thresholds come from — see
[runtime-critic.md](runtime-critic.md).

## CI integration sketch

```yaml
# .github/workflows/eval.yml (sketch)
- run: uv sync --extra t2s
- run: uv run agent-eval evaluate -c eval/t2s.yaml   # non-zero exit fails the job
```

Keep a **frozen** golden slice in version control for apples-to-apples regression comparisons; add a
rolling supplement from production failures for realism (version every dataset and link it to the
agent release).
